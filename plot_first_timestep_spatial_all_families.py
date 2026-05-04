import argparse
import os
import time
import warnings
from pathlib import Path

import deepmimo as dm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset.dataloaders import PreTrainMySeqDataLoader
from models import PathDecoder
from multiscenario_direct_training_first_step_residual import (
    FirstStepResidualDataset,
    FirstStepResidualPathDecoder,
    build_first_step_assignments,
    generate_paths_first_step_residual_batch,
    load_best_checkpoint as load_residual_checkpoint,
)
from multiscenario_direct_training_first_step_residual_corridor import (
    build_first_step_assignments_with_corridor,
)
from scene_feature_utils import SceneFeatureBank
from utils.utils import (
    count_parameters,
    generate_paths_no_env_batch,
    load_best_checkpoint as load_direct_checkpoint,
)


warnings.filterwarnings("ignore", category=UserWarning)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot first-timestep metric values over test-user x-y locations for direct, residual, and corridor models."
    )
    parser.add_argument("--scenario", type=str, required=True, help="Scenario to evaluate.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-generate", type=int, default=1, help="Number of autoregressive steps to generate. Defaults to 1 for first-step analysis.")
    parser.add_argument("--n-clusters", type=int, default=25)
    parser.add_argument("--nearest-k", type=int, default=5)
    parser.add_argument("--corridor-k", type=int, default=5)
    parser.add_argument("--corridor-bins", type=int, default=8)
    parser.add_argument("--use-material-features", action="store_true")
    parser.add_argument("--no-material-features", dest="use_material_features", action="store_false")
    parser.add_argument("--checkpoint-root-direct", type=str, default="/home/blessedg/Pathformer/base_no_env")
    parser.add_argument("--checkpoint-root-residual", type=str, default="/home/blessedg/Pathformer/checkpoints_first_step_residual")
    parser.add_argument("--checkpoint-root-corridor", type=str, default="/home/blessedg/Pathformer/checkpoints_first_step_residual_corridor")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--summary-csv", type=str, default=None)
    parser.add_argument("--point-size", type=float, default=10.0)
    parser.set_defaults(use_material_features=True)
    return parser.parse_args()


def default_output_dir(scenario: str) -> Path:
    return Path("/home/blessedg/Pathformer/logs") / f"first_timestep_spatial_{scenario}"


def build_direct_val_loader(dataset, batch_size, pad_value):
    val_data = PreTrainMySeqDataLoader(
        dataset,
        train=False,
        split_by="user",
        sort_by="power",
        normalizers=None,
        apply_normalizers=[],
        pad_value=pad_value,
        include_aod=True,
    )
    val_loader = DataLoader(
        dataset=val_data,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=val_data.collate_fn,
    )
    return val_data, val_loader


def build_residual_val_loader(dataset, batch_size, pad_value, n_clusters):
    base_train = PreTrainMySeqDataLoader(
        dataset,
        train=True,
        split_by="user",
        sort_by="power",
        normalizers=None,
        apply_normalizers=[],
        pad_value=pad_value,
        include_aod=True,
    )
    base_val = PreTrainMySeqDataLoader(
        dataset,
        train=False,
        split_by="user",
        sort_by="power",
        normalizers=None,
        apply_normalizers=[],
        pad_value=pad_value,
        include_aod=True,
    )
    _, _, val_aug_prompts, val_baselines = build_first_step_assignments(
        base_train,
        base_val,
        n_clusters=n_clusters,
    )
    val_data = FirstStepResidualDataset(base_val, val_aug_prompts, val_baselines)
    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=val_data.collate_fn,
    )
    return val_data, val_loader


def build_corridor_val_loader(dataset, batch_size, pad_value, args):
    base_train = PreTrainMySeqDataLoader(
        dataset,
        train=True,
        split_by="user",
        sort_by="power",
        normalizers=None,
        apply_normalizers=[],
        pad_value=pad_value,
        include_aod=True,
    )
    base_val = PreTrainMySeqDataLoader(
        dataset,
        train=False,
        split_by="user",
        sort_by="power",
        normalizers=None,
        apply_normalizers=[],
        pad_value=pad_value,
        include_aod=True,
    )
    scene_bank = SceneFeatureBank.from_dataset(dataset, use_material_features=args.use_material_features)
    _, _, val_aug_prompts, val_baselines = build_first_step_assignments_with_corridor(
        base_train,
        base_val,
        scene_bank,
        n_clusters=args.n_clusters,
        nearest_k=args.nearest_k,
        corridor_k=args.corridor_k,
        corridor_bins=args.corridor_bins,
    )
    val_data = FirstStepResidualDataset(base_val, val_aug_prompts, val_baselines)
    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=val_data.collate_fn,
    )
    return val_data, val_loader


def checkpoint_specs_for_scenario(scenario, args):
    return [
        (
            "direct",
            os.path.join(args.checkpoint_root_direct, f"multiscenario_direct_{scenario}_best_model_checkpoint.pth"),
        ),
        (
            "first_step_residual",
            os.path.join(args.checkpoint_root_residual, f"first_step_residual_{scenario}_best_model_checkpoint.pth"),
        ),
        (
            "first_step_residual_corridor",
            os.path.join(args.checkpoint_root_corridor, f"first_step_residual_corridor_{scenario}_best_model_checkpoint.pth"),
        ),
    ]


def maybe_sync_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def wrap_deg(values_deg: np.ndarray) -> np.ndarray:
    return (values_deg + 180.0) % 360.0 - 180.0


def metric_transform(metric_name: str, values: np.ndarray) -> np.ndarray:
    if metric_name == "delay_us":
        return values.astype(np.float64)
    if metric_name == "power_db":
        return (values / 0.01).astype(np.float64)
    if metric_name in {"phase_deg", "aoa_az_deg", "aoa_el_deg", "aod_az_deg", "aod_el_deg"}:
        return wrap_deg(np.rad2deg(values.astype(np.float64)))
    raise ValueError(f"Unknown metric: {metric_name}")


def collect_direct_outputs(model, val_loader, max_generate):
    preds = []
    gts = []
    rx_xy = []
    tx_xy = []
    total_samples = 0
    total_time = 0.0

    model.eval()
    with torch.no_grad():
        for prompts, paths, path_lengths, interactions, env, env_prop, path_padding_mask in val_loader:
            rx_xy.append(prompts[:, 3:5].detach().cpu().numpy())
            tx_xy.append(prompts[:, 0:2].detach().cpu().numpy())
            prompts = prompts.to(device)
            maybe_sync_cuda()
            t0 = time.perf_counter()
            generated, _, _ = generate_paths_no_env_batch(model, prompts, max_steps=max_generate)
            maybe_sync_cuda()
            total_time += time.perf_counter() - t0

            pred_dim = generated.shape[-1]
            first_pred = generated[:, 0, :pred_dim].detach().cpu().numpy()
            first_gt = paths[:, 1, :pred_dim].detach().cpu().numpy()
            preds.append(first_pred)
            gts.append(first_gt)
            total_samples += prompts.size(0)

    return (
        np.concatenate(gts, axis=0),
        np.concatenate(preds, axis=0),
        np.concatenate(rx_xy, axis=0),
        np.concatenate(tx_xy, axis=0),
        total_time / max(total_samples, 1),
    )


def collect_residual_outputs(model, val_loader, max_generate):
    preds = []
    gts = []
    rx_xy = []
    tx_xy = []
    total_samples = 0
    total_time = 0.0

    model.eval()
    with torch.no_grad():
        for prompts, paths, path_lengths, interactions, env, env_prop, path_padding_mask, first_step_baselines in val_loader:
            rx_xy.append(prompts[:, 3:5].detach().cpu().numpy())
            tx_xy.append(prompts[:, 0:2].detach().cpu().numpy())
            prompts = prompts.to(device)
            first_step_baselines = first_step_baselines.to(device)
            maybe_sync_cuda()
            t0 = time.perf_counter()
            generated, _, _ = generate_paths_first_step_residual_batch(
                model,
                prompts,
                first_step_baselines,
                max_steps=max_generate,
            )
            maybe_sync_cuda()
            total_time += time.perf_counter() - t0

            pred_dim = generated.shape[-1]
            first_pred = generated[:, 0, :pred_dim].detach().cpu().numpy()
            first_gt = paths[:, 1, :pred_dim].detach().cpu().numpy()
            preds.append(first_pred)
            gts.append(first_gt)
            total_samples += prompts.size(0)

    return (
        np.concatenate(gts, axis=0),
        np.concatenate(preds, axis=0),
        np.concatenate(rx_xy, axis=0),
        np.concatenate(tx_xy, axis=0),
        total_time / max(total_samples, 1),
    )


def evaluate_model_family(dataset, model_family, checkpoint_path, args):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Missing checkpoint for {model_family}: {checkpoint_path}")

    pad_value = 0
    if model_family == "direct":
        _, val_loader = build_direct_val_loader(dataset, args.batch_size, pad_value)
        model = PathDecoder(hidden_dim=512, n_layers=8, n_heads=8, include_aod=True).to(device)
        print(f"{model_family} parameters: {count_parameters(model)}")
        load_direct_checkpoint(model, checkpoint_path=checkpoint_path)
        gt, pred, rx_xy, tx_xy, avg_time = collect_direct_outputs(model, val_loader, args.max_generate)
    elif model_family == "first_step_residual":
        _, val_loader = build_residual_val_loader(dataset, args.batch_size, pad_value, args.n_clusters)
        model = FirstStepResidualPathDecoder(prompt_dim=10, hidden_dim=512, n_layers=8, n_heads=8, include_aod=True).to(device)
        print(f"{model_family} parameters: {count_parameters(model)}")
        load_residual_checkpoint(model, checkpoint_path)
        gt, pred, rx_xy, tx_xy, avg_time = collect_residual_outputs(model, val_loader, args.max_generate)
    elif model_family == "first_step_residual_corridor":
        val_data, val_loader = build_corridor_val_loader(dataset, args.batch_size, pad_value, args)
        prompt_dim = int(val_data.augmented_prompts[0].numel())
        model = FirstStepResidualPathDecoder(prompt_dim=prompt_dim, hidden_dim=512, n_layers=8, n_heads=8, include_aod=True).to(device)
        print(f"{model_family} parameters: {count_parameters(model)}")
        load_residual_checkpoint(model, checkpoint_path)
        gt, pred, rx_xy, tx_xy, avg_time = collect_residual_outputs(model, val_loader, args.max_generate)
    else:
        raise ValueError(f"Unknown model family: {model_family}")

    return {
        "groundtruth": gt,
        "prediction": pred,
        "prediction_dim": int(pred.shape[1]),
        "rx_xy": rx_xy,
        "tx_xy": tx_xy,
        "avg_inference_time_per_sample_sec": avg_time,
        "n_eval": int(gt.shape[0]),
        "checkpoint_path": checkpoint_path,
    }


def format_family_label(family: str) -> str:
    return {
        "direct": "Direct",
        "first_step_residual": "First-step Residual",
        "first_step_residual_corridor": "Corridor Residual",
    }[family]


def group_indices_by_tx(tx_xy: np.ndarray):
    unique_tx, inverse = np.unique(tx_xy, axis=0, return_inverse=True)
    groups = []
    for tx_idx in range(unique_tx.shape[0]):
        sample_indices = np.where(inverse == tx_idx)[0]
        groups.append((tx_idx, unique_tx[tx_idx], sample_indices))
    return groups


def plot_spatial_metric(metric_name, title, base_result, family_results, output_path, point_size, sample_indices, tx_position, tx_idx):
    panels = [("groundtruth", "Ground Truth", base_result["groundtruth"])]
    for family in ["direct", "first_step_residual", "first_step_residual_corridor"]:
        if family in family_results:
            panels.append((family, format_family_label(family), family_results[family]["prediction"]))

    idx = None
    for name, metric_idx, _ in [
        ("delay_us", 0, "Delay (us)"),
        ("power_db", 1, "Power (dB)"),
        ("phase_deg", 2, "Phase (deg)"),
        ("aoa_az_deg", 3, "AoA Azimuth (deg)"),
        ("aoa_el_deg", 4, "AoA Elevation (deg)"),
        ("aod_az_deg", 5, "AoD Azimuth (deg)"),
        ("aod_el_deg", 6, "AoD Elevation (deg)"),
    ]:
        if name == metric_name:
            idx = metric_idx
            break
    if idx is None:
        raise ValueError(f"Unknown metric {metric_name}")

    transformed = []
    for key, panel_title, values in panels:
        transformed.append(metric_transform(metric_name, values[sample_indices, idx]))

    all_values = np.concatenate([vals[np.isfinite(vals)] for vals in transformed if vals.size > 0])
    vmin = float(np.nanmin(all_values))
    vmax = float(np.nanmax(all_values))
    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-6

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.5), sharex=True, sharey=True)
    axes = axes.ravel()
    rx_xy = base_result["rx_xy"][sample_indices]

    scatter = None
    for ax, ((panel_key, panel_title, _), values) in zip(axes, zip(panels, transformed)):
        scatter = ax.scatter(
            rx_xy[:, 0],
            rx_xy[:, 1],
            c=values,
            s=point_size,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            linewidths=0.0,
        )
        ax.scatter(
            [tx_position[0]],
            [tx_position[1]],
            marker="^",
            s=max(36.0, point_size * 3.0),
            color="red",
            edgecolors="black",
            linewidths=0.5,
        )
        ax.set_title(panel_title)
        ax.set_xlabel("RX x-position (m)")
        ax.set_ylabel("RX y-position (m)")
        ax.grid(alpha=0.2)

    for ax in axes[len(panels):]:
        ax.axis("off")

    cbar_ax = fig.add_axes([0.18, 0.07, 0.64, 0.025])
    cbar = fig.colorbar(
        scatter,
        cax=cbar_ax,
        orientation="horizontal",
    )
    cbar.set_label(title)
    fig.suptitle(
        f"First-step {title} over Test User Locations (TX {tx_idx}: x={tx_position[0]:.1f}, y={tx_position[1]:.1f})",
        y=0.965,
    )
    fig.tight_layout(rect=[0.0, 0.12, 1.0, 0.94])
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir(args.scenario)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = Path(args.summary_csv) if args.summary_csv else output_dir / "inference_time_summary.csv"

    dataset = dm.load(args.scenario)

    metric_specs = [
        ("delay_us", 0, "Delay (us)"),
        ("power_db", 1, "Power (dB)"),
        ("phase_deg", 2, "Phase (deg)"),
        ("aoa_az_deg", 3, "AoA Azimuth (deg)"),
        ("aoa_el_deg", 4, "AoA Elevation (deg)"),
        ("aod_az_deg", 5, "AoD Azimuth (deg)"),
        ("aod_el_deg", 6, "AoD Elevation (deg)"),
    ]

    family_results = {}
    timing_rows = []
    base_result = None
    for model_family, checkpoint_path in checkpoint_specs_for_scenario(args.scenario, args):
        print(f"\nEvaluating {model_family} on {args.scenario}")
        result = evaluate_model_family(dataset, model_family, checkpoint_path, args)
        family_results[model_family] = result
        if base_result is None:
            base_result = result
        timing_rows.append(
            {
                "scenario": args.scenario,
                "model_family": model_family,
                "avg_inference_time_per_sample_sec": result["avg_inference_time_per_sample_sec"],
                "avg_inference_time_per_sample_ms": result["avg_inference_time_per_sample_sec"] * 1000.0,
                "n_eval": result["n_eval"],
                "checkpoint_path": result["checkpoint_path"],
                "max_generate": args.max_generate,
            }
        )

    for metric_name, idx, title in metric_specs:
        valid_family_results = {
            family: result
            for family, result in family_results.items()
            if result["prediction_dim"] > idx
        }
        if not valid_family_results or base_result["prediction_dim"] <= idx:
            print(f"Skipping {metric_name}: no evaluated model predicts this metric.")
            continue
        tx_groups = group_indices_by_tx(base_result["tx_xy"])
        for tx_idx, tx_position, sample_indices in tx_groups:
            output_path = output_dir / f"{metric_name}_spatial_comparison_tx{tx_idx}.pdf"
            plot_spatial_metric(
                metric_name,
                title,
                base_result,
                valid_family_results,
                output_path,
                args.point_size,
                sample_indices,
                tx_position,
                tx_idx,
            )
            print(f"Saved {output_path}")

    pd.DataFrame(timing_rows).to_csv(summary_csv, index=False)
    print(f"\nSaved inference-time summary to {summary_csv}")
    print(pd.DataFrame(timing_rows).to_string(index=False))


if __name__ == "__main__":
    main()
