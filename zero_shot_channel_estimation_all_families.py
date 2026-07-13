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
    count_parameters,
    generate_paths_no_env_batch,
    load_best_checkpoint as load_direct_checkpoint,
)


warnings.filterwarnings("ignore", category=UserWarning)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_CSV_LOG = "/home/blessedg/Pathformer/logs/zero_shot_channel_estimation_all_families.csv"


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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Zero-shot channel estimation from path-prediction checkpoints for direct, first-step residual, and corridor models."
    )
    parser.add_argument("scenarios", nargs="*", help="Optional scenario names to run.")
    parser.add_argument("--scenario", dest="scenario_flag", action="append")
    parser.add_argument("--scenario-file", type=str)
    parser.add_argument("--shard-index", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=None)
    parser.add_argument("--csv-log-file", type=str, default=DEFAULT_CSV_LOG)
    parser.add_argument("--checkpoint-root-direct", type=str, default="/home/blessedg/Pathformer/base_no_env")
    parser.add_argument("--checkpoint-root-residual", type=str, default="/home/blessedg/Pathformer/checkpoints_first_step_residual")
    parser.add_argument("--checkpoint-root-corridor", type=str, default="/home/blessedg/Pathformer/checkpoints_first_step_residual_corridor")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-generate", type=int, default=26)
    parser.add_argument("--n-clusters", type=int, default=25)
    parser.add_argument("--nearest-k", type=int, default=5)
    parser.add_argument("--corridor-k", type=int, default=5)
    parser.add_argument("--corridor-bins", type=int, default=8)
    parser.add_argument("--use-material-features", action="store_true")
    parser.add_argument("--no-material-features", dest="use_material_features", action="store_false")
    parser.add_argument(
        "--phase-source",
        choices=["predicted", "groundtruth"],
        default="predicted",
        help="Phase used when constructing the predicted channel.",
    )
    parser.set_defaults(use_material_features=True)
    return parser.parse_args()


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
        {
            "model_family": "direct",
            "checkpoint_path": os.path.join(
                args.checkpoint_root_direct,
                f"multiscenario_direct_{scenario}_best_model_checkpoint.pth",
            ),
        },
        {
            "model_family": "first_step_residual",
            "checkpoint_path": os.path.join(
                args.checkpoint_root_residual,
                f"first_step_residual_{scenario}_best_model_checkpoint.pth",
            ),
        },
        {
            "model_family": "first_step_residual_corridor",
            "checkpoint_path": os.path.join(
                args.checkpoint_root_corridor,
                f"first_step_residual_corridor_{scenario}_best_model_checkpoint.pth",
            ),
        },
    ]


def _power_scaled_to_linear(power_scaled):
    return 10 ** ((np.clip(power_scaled, -15000.0, 500.0) / 0.01) / 10)


def _batch_channel_inputs(paths_out, generated, path_lengths, phase_source, pad_value):
    paths_np = paths_out.detach().cpu().numpy()
    pred_np = generated.detach().cpu().numpy()
    B = paths_np.shape[0]
    T_gen = pred_np.shape[1]

    delay_gt = np.full((B, T_gen), np.nan, dtype=np.float32)
    power_gt = np.full((B, T_gen), np.nan, dtype=np.float32)
    phase_gt = np.full((B, T_gen), np.nan, dtype=np.float32)
    aod_az_gt = np.full((B, T_gen), np.nan, dtype=np.float32)
    aod_el_gt = np.full((B, T_gen), np.nan, dtype=np.float32)

    delay_pred = np.full((B, T_gen), np.nan, dtype=np.float32)
    power_pred = np.full((B, T_gen), np.nan, dtype=np.float32)
    phase_pred = np.full((B, T_gen), np.nan, dtype=np.float32)
    aod_az_pred = np.full((B, T_gen), np.nan, dtype=np.float32)
    aod_el_pred = np.full((B, T_gen), np.nan, dtype=np.float32)

    valid_paths = 0
    for b in range(B):
        n_valid = int(round(float(path_lengths[b].item()) * 25))
        n_valid = max(0, min(n_valid, paths_np.shape[1], T_gen))
        # print(f"generated len: {T_gen}")
        if n_valid == 0:
            continue
        gt_b = paths_np[b, :n_valid, :]
        pred_b = pred_np[b, :n_valid, :]
        valid_mask = gt_b[:, 0] != pad_value
        n_valid = int(valid_mask.sum())
        if n_valid == 0:
            continue
        gt_b = gt_b[:n_valid]
        pred_b = pred_b[:n_valid]
        valid_paths += n_valid

        delay_gt[b, :n_valid] = gt_b[:, 0] / 1e6
        power_gt[b, :n_valid] = _power_scaled_to_linear(gt_b[:, 1])
        phase_gt[b, :n_valid] = np.rad2deg(gt_b[:, 2])
        aod_az_gt[b, :n_valid] = gt_b[:, 5]
        aod_el_gt[b, :n_valid] = gt_b[:, 6]

        delay_pred[b, :n_valid] = pred_b[:, 0] / 1e6
        power_pred[b, :n_valid] = _power_scaled_to_linear(pred_b[:, 1])
        phase_rad = gt_b[:, 2] if phase_source == "groundtruth" else pred_b[:, 2]
        phase_pred[b, :n_valid] = np.rad2deg(phase_rad)
        aod_az_pred[b, :n_valid] = pred_b[:, 5]
        aod_el_pred[b, :n_valid] = pred_b[:, 6]

    return (
        (power_gt, delay_gt, phase_gt, aod_az_gt, aod_el_gt),
        (power_pred, delay_pred, phase_pred, aod_az_pred, aod_el_pred),
        valid_paths,
    )


def _channel_metrics(gt_ch, pred_ch):
    gt = gt_ch[:, 0, :, :]
    pred = pred_ch[:, 0, :, :]
    scale = 1e6
    gt_s = gt * scale
    pred_s = pred * scale
    mse_per_sample = np.mean(np.abs(gt_s - pred_s) ** 2, axis=(1, 2))
    norm_per_sample = np.mean(np.abs(gt_s) ** 2, axis=(1, 2))
    nmse = mse_per_sample / (np.maximum(norm_per_sample, 1e-6) + 1e-10)
    nmse_db = 10.0 * np.log10(nmse + 1e-10)
    nmse_log = np.log10(nmse + 1e-10)
    scores = 1.0 - ((nmse_db - (-20.0)) / (0.0 - (-20.0)))
    scores = np.clip(scores, 0.0, 1.0)
    return nmse_log, nmse_db, scores


@torch.no_grad()
def zero_shot_channel_eval(model, model_family, val_loader, pad_value, max_generate, phase_source):
    model.eval()
    computer = MyChannelComputer()
    params = ChannelParameters()
    nmse_logs = []
    nmse_dbs = []
    scores = []
    n_samples = 0
    n_valid_paths = 0

    for batch in tqdm(val_loader, desc=f"Zero-shot channel [{model_family}]", leave=False):
        if model_family == "direct":
            prompts, paths, path_lengths, interactions, env, env_prop, path_padding_mask = batch
            prompts = prompts.to(device)
            generated, _, _ = generate_paths_no_env_batch(model, prompts, max_steps=max_generate)
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

        paths_out = paths[:, 1:, :]
        gt_inputs, pred_inputs, valid_paths = _batch_channel_inputs(
            paths_out,
            generated,
            path_lengths,
            phase_source=phase_source,
            pad_value=pad_value,
        )
        if valid_paths == 0:
            continue

        gt_ch = computer.compute_channels(*gt_inputs, params=params)
        pred_ch = computer.compute_channels(*pred_inputs, params=params)
        batch_logs, batch_dbs, batch_scores = _channel_metrics(gt_ch, pred_ch)
        nmse_logs.extend(batch_logs.tolist())
        nmse_dbs.extend(batch_dbs.tolist())
        scores.extend(batch_scores.tolist())
        n_samples += paths.size(0)
        n_valid_paths += valid_paths

    return {
        "avg_ch_nmse_log10": float(np.mean(nmse_logs)) if nmse_logs else 0.0,
        "std_ch_nmse_log10": float(np.std(nmse_logs)) if nmse_logs else 0.0,
        "avg_ch_nmse_dB": float(np.mean(nmse_dbs)) if nmse_dbs else 0.0,
        "std_ch_nmse_dB": float(np.std(nmse_dbs)) if nmse_dbs else 0.0,
        "avg_ch_score": float(np.mean(scores)) if scores else 0.0,
        "std_ch_score": float(np.std(scores)) if scores else 0.0,
        "n_eval": int(n_samples),
        "n_valid_paths": int(n_valid_paths),
    }


def evaluate_scenario_family(dataset, scenario, model_family, checkpoint_path, args):
    if not os.path.exists(checkpoint_path):
        print(f"Skipping {scenario} [{model_family}] because checkpoint is missing: {checkpoint_path}")
        return None

    pad_value = 0
    if model_family == "direct":
        _, val_loader = build_direct_val_loader(dataset, args.batch_size, pad_value)
        model = PathDecoder(hidden_dim=512, n_layers=8, n_heads=8, include_aod=True).to(device)
        print(f"{model_family} parameters: {count_parameters(model)}")
        load_direct_checkpoint(model, checkpoint_path=checkpoint_path)
    elif model_family == "first_step_residual":
        _, val_loader = build_residual_val_loader(dataset, args.batch_size, pad_value, args.n_clusters)
        model = FirstStepResidualPathDecoder(prompt_dim=10, hidden_dim=512, n_layers=8, n_heads=8).to(device)
        print(f"{model_family} parameters: {count_parameters(model)}")
        load_residual_checkpoint(model, checkpoint_path)
    elif model_family == "first_step_residual_corridor":
        val_data, val_loader = build_corridor_val_loader(dataset, args.batch_size, pad_value, args)
        prompt_dim = int(val_data.augmented_prompts[0].numel())
        model = FirstStepResidualPathDecoder(prompt_dim=prompt_dim, hidden_dim=512, n_layers=8, n_heads=8).to(device)
        print(f"{model_family} parameters: {count_parameters(model)}")
        load_residual_checkpoint(model, checkpoint_path)
    else:
        raise ValueError(f"Unknown model family: {model_family}")

    metrics = zero_shot_channel_eval(
        model=model,
        model_family=model_family,
        val_loader=val_loader,
        pad_value=pad_value,
        max_generate=args.max_generate,
        phase_source=args.phase_source,
    )
    metrics.update(
        {
            "scenario": scenario,
            "model_family": model_family,
            "phase_source": args.phase_source,
            "checkpoint_path": checkpoint_path,
        }
    )
    print(
        f"{scenario} [{model_family}] | phase={args.phase_source} | "
        f"nmse_dB={metrics['avg_ch_nmse_dB']:.4f}, score={metrics['avg_ch_score']:.4f}"
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

    print(
        f"Running zero-shot channel estimation for {len(scenarios)} scenario(s), "
        f"phase_source={args.phase_source}: {scenarios}"
    )
    os.makedirs(os.path.dirname(args.csv_log_file) or ".", exist_ok=True)

    all_rows = []
    for scenario in scenarios:
        print(f"\nEvaluating scenario: {scenario}")
        dataset = dm.load(scenario)
        for spec in checkpoint_specs_for_scenario(scenario, args):
            row = evaluate_scenario_family(
                dataset=dataset,
                scenario=scenario,
                model_family=spec["model_family"],
                checkpoint_path=spec["checkpoint_path"],
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
