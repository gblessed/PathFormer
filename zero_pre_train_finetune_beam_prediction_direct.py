import argparse
import os
import warnings

import deepmimo as dm
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.dataloaders import PreTrainMySeqDataLoader
from models import PathDecoder
from multiscenario_direct_training_first_step_residual import (
    FirstStepResidualDataset,
    FirstStepResidualPathDecoder,
    build_first_step_assignments,
    generate_paths_first_step_residual_batch,
    load_best_checkpoint as load_residual_checkpoint,
    resolve_scenarios,
)
from multiscenario_direct_training_first_step_residual_corridor import (
    build_first_step_assignments_with_corridor,
)
from scene_feature_utils import SceneFeatureBank
from utils.utils import (
    ChannelParameters,
    MyChannelComputer,
    compute_single_array_response_torch,
    count_parameters,
    generate_paths_no_env_batch,
    load_best_checkpoint as load_direct_checkpoint,
)

warnings.filterwarnings("ignore", category=UserWarning)

csv_log_file = "zero_shot_beam_prediction_aod_first_path.csv"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Zero-shot beam prediction using first-path AOD for direct, first-step residual, and corridor models."
    )
    parser.add_argument("scenarios", nargs="*", help="Optional scenario names to run.")
    parser.add_argument("--scenario", dest="scenario_flag", action="append")
    parser.add_argument("--scenario-file", type=str)
    parser.add_argument("--shard-index", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=None)
    parser.add_argument("--csv-log-file", type=str, default=csv_log_file)
    parser.add_argument(
        "--checkpoint-root-direct",
        type=str,
        default="/home/blessedg/Pathformer/base_no_env",
    )
    parser.add_argument(
        "--checkpoint-root-residual",
        type=str,
        default="/home/blessedg/Pathformer/checkpoints_first_step_residual",
    )
    parser.add_argument(
        "--checkpoint-root-corridor",
        type=str,
        default="/home/blessedg/Pathformer/checkpoints_first_step_residual_corridor",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-generate", type=int, default=1)
    parser.add_argument("--n-clusters", type=int, default=25)
    parser.add_argument("--nearest-k", type=int, default=5)
    parser.add_argument("--corridor-k", type=int, default=5)
    parser.add_argument("--corridor-bins", type=int, default=8)
    parser.add_argument("--use-material-features", action="store_true")
    parser.add_argument("--no-material-features", dest="use_material_features", action="store_false")
    parser.set_defaults(use_material_features=True)
    return parser.parse_args()


def default_scenarios():
    return [
        "city_0_newyork_3p5_s",
        "city_1_losangeles_3p5",
        "city_2_chicago_3p5",
        "city_3_houston_3p5",
        "city_4_phoenix_3p5",
        "city_5_philadelphia_3p5",
        "city_6_miami_3p5",
        "city_7_sandiego_3p5",
        "city_8_dallas_3p5",
        "city_9_sanfrancisco_3p5",
        "city_10_austin_3p5",
        "city_11_santaclara_3p5",
        "city_12_fortworth_3p5",
        "city_13_columbus_3p5",
        "city_17_seattle_3p5_s",
        "city_18_denver_3p5",
        "city_19_oklahoma_3p5_s",
        "city_16_sanfrancisco_3p5_lwm",
        "city_23_beijing_3p5",
        "city_31_barcelona_3p5",
        "city_35_san_francisco_3p5",
        "city_47_chicago_3p5",
        "city_89_nairobi_3p5",
        "city_91_xiangyang_3p5",
        "city_92_sãopaulo_3p5",
        "boston5g_3p5",
        "city_86_ankara_3p5",
        "city_72_capetown_3p5",
        "city_84_baoding_3p5",
        "city_95_delhi_3p5",
        "city_96_osaka_3p5",
        "city_88_tongshan_3p5",
    ]


def make_dft_codebook(B=8):
    params = ChannelParameters()
    az_t = np.linspace(-np.pi, np.pi, B, endpoint=False, dtype=np.float32)
    el_t = np.linspace(-np.pi, np.pi, B, endpoint=False, dtype=np.float32)
    az_new = []
    el_new = []
    for az in az_t:
        for el in el_t:
            az_new.append(az)
            el_new.append(el)
    az_new = torch.tensor(az_new).unsqueeze(1)
    el_new = torch.tensor(el_new).unsqueeze(1)
    array_response = compute_single_array_response_torch(params.bs_antenna, az_new, el_new)
    return array_response.squeeze(2).T


def compute_beam_label_from_channel(H, S):
    if not torch.is_tensor(H):
        H = torch.from_numpy(H)
    if not torch.is_tensor(S):
        S = torch.from_numpy(S)
    H = H.to(torch.complex64)
    S = S.to(torch.complex64)
    Y = S.conj().T @ H
    prx = torch.sum(torch.abs(Y) ** 2, dim=2)
    best = torch.argmax(prx, dim=1)
    return best, prx


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
    train_aug_prompts, train_baselines, val_aug_prompts, val_baselines = build_first_step_assignments(
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
    train_aug_prompts, train_baselines, val_aug_prompts, val_baselines = build_first_step_assignments_with_corridor(
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


@torch.no_grad()
def zero_shot_beam_eval_first_path_aod(model, model_family, val_loader, S, params, pad_value, max_generate=1):
    model.eval()
    total = 0
    correct = 0
    aod_az_errors = []
    aod_el_errors = []
    mycomputer = MyChannelComputer()

    for batch in tqdm(val_loader, desc=f"Zero-shot beam eval [{model_family}]", leave=False):
        if model_family == "direct":
            prompts, paths, path_lengths, interactions, env, env_prop, path_padding_mask = batch
            prompts = prompts.to(device)
            generated, _, _ = generate_paths_no_env_batch(model, prompts, max_steps=max_generate)
            first_pred = generated[:, 0, :]
        else:
            prompts, paths, path_lengths, interactions, env, env_prop, path_padding_mask, first_step_baselines = batch
            prompts = prompts.to(device)
            first_step_baselines = first_step_baselines.to(device)
            generated, _, _ = generate_paths_first_step_residual_batch(
                model,
                prompts,
                first_step_baselines,
                max_steps=max_generate,
            )
            first_pred = generated[:, 0, :]

        paths = paths.to(device)
        paths_out = paths[:, 1:, :]

        aod_az_pred = first_pred[:, 5].to(device)
        aod_el_pred = first_pred[:, 6].to(device)
        aod_az_gt_first = paths_out[:, 0, 5]
        aod_el_gt_first = paths_out[:, 0, 6]

        aod_az_dist = (torch.rad2deg(aod_az_pred) - torch.rad2deg(aod_az_gt_first) + 180.0) % 360.0 - 180.0
        aod_el_dist = (torch.rad2deg(aod_el_pred) - torch.rad2deg(aod_el_gt_first) + 180.0) % 360.0 - 180.0
        aod_az_errors.append(torch.mean(aod_az_dist ** 2).sqrt().item())
        aod_el_errors.append(torch.mean(aod_el_dist ** 2).sqrt().item())

        # compute_single_array_response_torch expects theta=elevation, phi=azimuth.
        # Keep this on CPU to avoid fragile complex CUDA matmul issues during evaluation.
        a_pred = compute_single_array_response_torch(
            params.bs_antenna,
            aod_el_pred.float().cpu().unsqueeze(1),
            aod_az_pred.float().cpu().unsqueeze(1),
        ).squeeze(2).cpu()

        scores = torch.abs(S.conj().T @ a_pred.T) ** 2
        beam_pred = torch.argmax(scores, dim=0)

        delay_gt = paths_out[:, :, 0].detach().cpu().numpy()
        power_gt = paths_out[:, :, 1].detach().cpu().numpy()
        phase_gt = paths_out[:, :, 2].detach().cpu().numpy()
        aod_az_gt = paths_out[:, :, 5].detach().cpu().numpy()
        aod_el_gt = paths_out[:, :, 6].detach().cpu().numpy()

        mask = delay_gt == pad_value
        power_gt = np.where(mask, 0, power_gt)
        power_linear_gt = 10 ** ((power_gt / 0.01) / 10)
        delay_sec_gt = np.where(mask, np.nan, delay_gt / 1e6)
        phase_gt = np.where(mask, np.nan, phase_gt)
        aod_az_gt = np.where(mask, np.nan, aod_az_gt)
        aod_el_gt = np.where(mask, np.nan, aod_el_gt)

        H_gt = mycomputer.compute_channels(
            power_linear_gt,
            delay_sec_gt,
            phase_gt,
            aod_az_gt,
            aod_el_gt,
            kwargs=None,
        )[:, 0, :, :]

        beam_gt, _ = compute_beam_label_from_channel(H_gt, S)
        correct += (beam_pred.cpu() == beam_gt.cpu()).sum().item()
        total += prompts.size(0)

    acc = correct / max(total, 1)
    return {
        "beam_acc": acc,
        "first_path_aod_az_rmse_deg": float(np.mean(aod_az_errors)) if aod_az_errors else 0.0,
        "first_path_aod_el_rmse_deg": float(np.mean(aod_el_errors)) if aod_el_errors else 0.0,
        "n_eval": total,
    }


def checkpoint_specs_for_scenario(scenario):
    return [
        {
            "model_family": "direct",
            "checkpoint_path": os.path.join(
                "/home/blessedg/Pathformer/base_no_env",
                f"multiscenario_direct_{scenario}_best_model_checkpoint.pth",
            ),
        },
        {
            "model_family": "first_step_residual",
            "checkpoint_path": os.path.join(
                "/home/blessedg/Pathformer/checkpoints_first_step_residual",
                f"first_step_residual_{scenario}_best_model_checkpoint.pth",
            ),
        },
        {
            "model_family": "first_step_residual_corridor",
            "checkpoint_path": os.path.join(
                "/home/blessedg/Pathformer/checkpoints_first_step_residual_corridor",
                f"first_step_residual_corridor_{scenario}_best_model_checkpoint.pth",
            ),
        },
    ]


def evaluate_scenario_family(dataset, scenario, model_family, checkpoint_path, S, args):
    if not os.path.exists(checkpoint_path):
        print(f"Skipping {scenario} [{model_family}] because checkpoint is missing: {checkpoint_path}")
        return None

    pad_value = 0
    if model_family == "direct":
        _, val_loader = build_direct_val_loader(dataset, batch_size=args.batch_size, pad_value=pad_value)
        model = PathDecoder(hidden_dim=512, n_layers=8, n_heads=8, include_aod=True).to(device)
        print(f"{model_family} parameters: {count_parameters(model)}")
        load_direct_checkpoint(model, checkpoint_path=checkpoint_path)
    elif model_family == "first_step_residual":
        val_data, val_loader = build_residual_val_loader(
            dataset,
            batch_size=args.batch_size,
            pad_value=pad_value,
            n_clusters=args.n_clusters,
        )
        model = FirstStepResidualPathDecoder(prompt_dim=10, hidden_dim=512, n_layers=8, n_heads=8).to(device)
        print(f"{model_family} parameters: {count_parameters(model)}")
        load_residual_checkpoint(model, checkpoint_path)
    elif model_family == "first_step_residual_corridor":
        val_data, val_loader = build_corridor_val_loader(
            dataset,
            batch_size=args.batch_size,
            pad_value=pad_value,
            args=args,
        )
        prompt_dim = int(val_data.augmented_prompts[0].numel())
        model = FirstStepResidualPathDecoder(prompt_dim=prompt_dim, hidden_dim=512, n_layers=8, n_heads=8).to(device)
        print(f"{model_family} parameters: {count_parameters(model)}")
        load_residual_checkpoint(model, checkpoint_path)
    else:
        raise ValueError(f"Unknown model family: {model_family}")

    metrics = zero_shot_beam_eval_first_path_aod(
        model=model,
        model_family=model_family,
        val_loader=val_loader,
        S=S,
        params=ChannelParameters(),
        pad_value=pad_value,
        max_generate=args.max_generate,
    )
    metrics.update(
        {
            "scenario": scenario,
            "model_family": model_family,
            "checkpoint_path": checkpoint_path,
        }
    )
    print(
        f"{scenario} [{model_family}] | beam_acc={metrics['beam_acc']:.4f}, "
        f"aod_az_rmse={metrics['first_path_aod_az_rmse_deg']:.4f}, "
        f"aod_el_rmse={metrics['first_path_aod_el_rmse_deg']:.4f}"
    )
    return metrics


def main():
    args = parse_args()
    if not args.scenarios and not args.scenario_flag and not args.scenario_file:
        scenarios = default_scenarios()
    else:
        scenarios = resolve_scenarios(args)

    if not scenarios:
        print("No scenarios selected for evaluation.")
        return

    print(f"Running zero-shot first-path AOD beam prediction for {len(scenarios)} scenario(s): {scenarios}")
    os.makedirs(os.path.dirname(args.csv_log_file) or ".", exist_ok=True)

    S = make_dft_codebook()
    all_rows = []

    for scenario in scenarios:
        print(f"\nEvaluating scenario: {scenario}")
        dataset = dm.load(scenario)
        for spec in checkpoint_specs_for_scenario(scenario):
            row = evaluate_scenario_family(
                dataset=dataset,
                scenario=scenario,
                model_family=spec["model_family"],
                checkpoint_path=spec["checkpoint_path"],
                S=S,
                args=args,
            )
            if row is None:
                continue
            all_rows.append(row)
            pd.DataFrame([row]).to_csv(
                args.csv_log_file,
                mode="a",
                index=False,
                header=not os.path.exists(args.csv_log_file),
            )

    if all_rows:
        print(f"\nSaved {len(all_rows)} evaluation rows to {args.csv_log_file}")
    else:
        print("\nNo rows were written. Check whether the expected checkpoints exist.")


if __name__ == "__main__":
    main()
