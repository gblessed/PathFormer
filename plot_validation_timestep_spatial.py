import argparse
import os

import deepmimo as dm
import matplotlib.pyplot as plt
import pandas as pd
import torch
from tqdm import tqdm
import numpy as np

from dataset.dataloaders import PreTrainMySeqDataLoader
from models import PathDecoder
from utils.utils import generate_paths_no_env_batch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot validation power/delay over RX x-y for a chosen generated time step."
    )
    parser.add_argument(
        "--scenario",
        required=True,
        help="DeepMIMO scenario name, e.g. city_19_oklahoma_3p5_s",
    )
    parser.add_argument(
        "--time-step",
        type=int,
        choices=[1, 2],
        default=1,
        help="Path time step to plot: 1 for first path, 2 for second path.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints2",
        help="Directory containing scenario checkpoints.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="Optional explicit checkpoint path. Overrides --checkpoint-dir.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output image path.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Validation batch size.",
    )
    parser.add_argument(
        "--max-generate",
        type=int,
        default=26,
        help="Autoregressive generation length.",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=512,
        help="Model hidden dimension used by the checkpoint.",
    )
    parser.add_argument(
        "--n-layers",
        type=int,
        default=8,
        help="Number of decoder layers used by the checkpoint.",
    )
    parser.add_argument(
        "--n-heads",
        type=int,
        default=8,
        help="Number of attention heads used by the checkpoint.",
    )
    return parser.parse_args()


def default_checkpoint_path(args):
    experiment = f"snoise_enc_direct_{args.scenario}_interacaction_all_inter_str_dec_all_repeat"
    return os.path.join(args.checkpoint_dir, f"{experiment}_best_model_checkpoint.pth")


def load_best_checkpoint(model, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    epoch = checkpoint["epoch"]
    best_val_loss = checkpoint["best_val_loss"]
    best_val_loss = best_val_loss.item() if hasattr(best_val_loss, "item") else best_val_loss
    print(f"Loaded checkpoint from epoch {epoch} with val_loss={best_val_loss:.4f}")


def load_scenario_dataset(scenario_name):
    return dm.load(scenario_name)


def get_num_transmitters(dataset):
    if isinstance(getattr(dataset, "n_ue", None), int):
        return 1
    return len(dataset)


def select_transmitter_dataset(dataset, tx_index):
    if isinstance(getattr(dataset, "n_ue", None), int):
        if tx_index != 0:
            raise IndexError(f"Scenario has only one transmitter, but got tx_index={tx_index}.")
        return dataset
    return dataset[tx_index]


def build_val_loader(dataset, args):
    val_data = PreTrainMySeqDataLoader(
        dataset,
        train=False,
        split_by="user",
        sort_by="power",
        normalizers=None,
        apply_normalizers=[],
        pad_value=0,
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=val_data,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=val_data.collate_fn,
    )
    return val_loader


def collect_timestep_predictions(model, val_loader, time_step, max_generate):
    model.eval()
    target_idx = time_step
    pred_idx = time_step - 1
    total_samples = 0
    used_samples = 0
    skipped_samples = 0
    collected = {
        "x": [],
        "y": [],
        "gt_delay_us": [],
        "pred_delay_us": [],
        "gt_power_db": [],
        "pred_power_db": [],
    }

    with torch.no_grad():
        for prompts, paths, path_lengths, interactions, env, env_prop, path_padding_mask in tqdm(
            val_loader,
            desc=f"Collecting validation step-{time_step} predictions",
            leave=True,
        ):
            batch_size = prompts.size(0)
            total_samples += batch_size
            prompts = prompts.to(device)
            paths = paths.to(device)
            path_lengths = path_lengths.to(device)

            generated, _, _ = generate_paths_no_env_batch(
                model,
                prompts,
                max_steps=max_generate,
            )
            generated = generated.to(device)

            valid_mask = (path_lengths.squeeze(-1) * 25).round().long() >= time_step
            valid_count = int(valid_mask.sum().item())
            batch_skipped = batch_size - valid_count
            used_samples += valid_count
            skipped_samples += batch_skipped

            if batch_skipped > 0:
                print(
                    f"Batch skip report: {batch_skipped} sample(s) had fewer than "
                    f"{time_step} valid path(s)."
                    f"{valid_mask.shape}"
                )

            if not valid_mask.any():
                continue

            prompts = prompts[valid_mask]
            paths = paths[valid_mask]
            generated = generated[valid_mask]

            gt_step = paths[:, target_idx, :5]
            pred_step = generated[:, pred_idx, :5]

            collected["x"].extend(prompts[:, 3].detach().cpu().numpy().tolist())
            collected["y"].extend(prompts[:, 4].detach().cpu().numpy().tolist())
            collected["gt_delay_us"].extend(gt_step[:, 0].detach().cpu().numpy().tolist())
            collected["pred_delay_us"].extend(pred_step[:, 0].detach().cpu().numpy().tolist())
            collected["gt_power_db"].extend((gt_step[:, 1] / 0.01).detach().cpu().numpy().tolist())
            collected["pred_power_db"].extend((pred_step[:, 1] / 0.01).detach().cpu().numpy().tolist())

    df = pd.DataFrame(collected)
    print(
        f"Validation summary for time step {time_step}: "
        f"{used_samples}/{total_samples} sample(s) used, "
        f"{skipped_samples} skipped."
    )
    if df.empty:
        raise ValueError(f"No validation samples had at least {time_step} valid paths.")
    return df


def print_per_timestep_rmse(model, val_loader, max_generate):
    model.eval()
    delay_sq_errors = {step: [] for step in range(1, max_generate + 1)}
    power_sq_errors = {step: [] for step in range(1, max_generate + 1)}
    valid_counts = {step: 0 for step in range(1, max_generate + 1)}

    with torch.no_grad():
        for prompts, paths, path_lengths, interactions, env, env_prop, path_padding_mask in tqdm(
            val_loader,
            desc="Computing per-time-step RMSE",
            leave=True,
        ):
            prompts = prompts.to(device)
            paths = paths.to(device)
            path_lengths = path_lengths.to(device)

            generated, _, _ = generate_paths_no_env_batch(
                model,
                prompts,
                max_steps=max_generate,
            )
            generated = generated.to(device)

            valid_path_counts = (path_lengths.squeeze(-1) * 25).round().long()
            for step in range(1, max_generate + 1):
                valid_mask = valid_path_counts >= step
                if not valid_mask.any():
                    continue

                gt_step = paths[valid_mask, step, :5]
                pred_step = generated[valid_mask, step - 1, :5]

                delay_errors = (pred_step[:, 0] - gt_step[:, 0]) ** 2
                power_errors = ((pred_step[:, 1] / 0.01) - (gt_step[:, 1] / 0.01)) ** 2

                delay_sq_errors[step].extend(delay_errors.detach().cpu().numpy().tolist())
                power_sq_errors[step].extend(power_errors.detach().cpu().numpy().tolist())
                valid_counts[step] += int(valid_mask.sum().item())

    print("\nPer-time-step RMSE on valid validation samples:")
    printed_any = False
    for step in range(1, max_generate + 1):
        if valid_counts[step] == 0:
            continue
        printed_any = True
        delay_rmse = float(np.sqrt(np.mean(delay_sq_errors[step])))
        power_rmse = float(np.sqrt(np.mean(power_sq_errors[step])))
        print(
            f"  Step {step:02d}: "
            f"count={valid_counts[step]:6d}, "
            f"delay_rmse={delay_rmse:.4f} us, "
            f"power_rmse={power_rmse:.4f} dB"
        )
    if not printed_any:
        print("  No valid samples found for any time step.")


def plot_spatial_distribution(df, scenario, time_step, output_path, tx_label):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)

    power_vmin = min(df["gt_power_db"].min(), df["pred_power_db"].min())
    power_vmax = max(df["gt_power_db"].max(), df["pred_power_db"].max())
    delay_vmin = min(df["gt_delay_us"].min(), df["pred_delay_us"].min())
    delay_vmax = max(df["gt_delay_us"].max(), df["pred_delay_us"].max())

    power_gt = axes[0, 0].scatter(
        df["x"], df["y"], c=df["gt_power_db"], s=12, cmap="viridis",
        vmin=power_vmin, vmax=power_vmax, linewidths=0,
    )
    axes[0, 0].set_title(f"Ground Truth Step-{time_step} Power (dB)")

    axes[0, 1].scatter(
        df["x"], df["y"], c=df["pred_power_db"], s=12, cmap="viridis",
        vmin=power_vmin, vmax=power_vmax, linewidths=0,
    )
    axes[0, 1].set_title(f"Predicted Step-{time_step} Power (dB)")

    delay_gt = axes[1, 0].scatter(
        df["x"], df["y"], c=df["gt_delay_us"], s=12, cmap="plasma",
        vmin=delay_vmin, vmax=delay_vmax, linewidths=0,
    )
    axes[1, 0].set_title(f"Ground Truth Step-{time_step} Delay (us)")

    axes[1, 1].scatter(
        df["x"], df["y"], c=df["pred_delay_us"], s=12, cmap="plasma",
        vmin=delay_vmin, vmax=delay_vmax, linewidths=0,
    )
    axes[1, 1].set_title(f"Predicted Step-{time_step} Delay (us)")

    for ax in axes.flat:
        ax.set_xlabel("RX x (m)")
        ax.set_ylabel("RX y (m)")

    fig.colorbar(power_gt, ax=[axes[0, 0], axes[0, 1]], shrink=0.9, label="Power (dB)")
    fig.colorbar(delay_gt, ax=[axes[1, 0], axes[1, 1]], shrink=0.9, label="Delay (us)")
    fig.suptitle(
        f"{scenario} ({tx_label}): Validation Spatial Distribution for Step {time_step}",
        fontsize=15,
    )

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {output_path}")


def main():
    args = parse_args()
    checkpoint_path = args.checkpoint_path or default_checkpoint_path(args)

    model = PathDecoder(
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
    ).to(device)
    load_best_checkpoint(model, checkpoint_path)

    scenario_dataset = load_scenario_dataset(args.scenario)
    num_txs = get_num_transmitters(scenario_dataset)
    print(f'Scenario "{args.scenario}" has {num_txs} transmitter(s).')

    for tx_index in range(num_txs):
        tx_dataset = select_transmitter_dataset(scenario_dataset, tx_index)
        tx_label = f"tx_{tx_index}"
        print(f"\n=== Processing {args.scenario} {tx_label} ===")
        val_loader = build_val_loader(tx_dataset, args)
        print_per_timestep_rmse(model, val_loader, args.max_generate)
        df = collect_timestep_predictions(
            model,
            val_loader,
            time_step=args.time_step,
            max_generate=args.max_generate,
        )

        if args.output:
            root, ext = os.path.splitext(args.output)
            ext = ext or ".png"
            output_path = f"{root}_{tx_label}{ext}"
        else:
            output_path = os.path.join(
                "plots",
                f"{args.scenario}_{tx_label}_step_{args.time_step}_spatial_power_delay.png",
            )

        plot_spatial_distribution(df, args.scenario, args.time_step, output_path, tx_label)


if __name__ == "__main__":
    main()
