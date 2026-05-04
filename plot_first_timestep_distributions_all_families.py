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
from utils.utils import count_parameters, generate_paths_no_env_batch, load_best_checkpoint as load_direct_checkpoint


warnings.filterwarnings("ignore", category=UserWarning)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot first-timestep ground-truth vs predicted distributions for direct, residual, and corridor models."
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
    parser.set_defaults(use_material_features=True)
    return parser.parse_args()


def default_output_dir(scenario: str) -> Path:
    return Path("/home/blessedg/Pathformer/logs") / f"first_timestep_distributions_{scenario}"


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
    total_samples = 0
    total_time = 0.0

    model.eval()
    with torch.no_grad():
        for prompts, paths, path_lengths, interactions, env, env_prop, path_padding_mask in val_loader:
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

    return np.concatenate(gts, axis=0), np.concatenate(preds, axis=0), total_time / max(total_samples, 1)


def collect_residual_outputs(model, val_loader, max_generate):
    preds = []
    gts = []
    total_samples = 0
    total_time = 0.0

    model.eval()
    with torch.no_grad():
        for prompts, paths, path_lengths, interactions, env, env_prop, path_padding_mask, first_step_baselines in val_loader:
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

    return np.concatenate(gts, axis=0), np.concatenate(preds, axis=0), total_time / max(total_samples, 1)


def evaluate_model_family(dataset, model_family, checkpoint_path, args):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Missing checkpoint for {model_family}: {checkpoint_path}")

    pad_value = 0
    if model_family == "direct":
        _, val_loader = build_direct_val_loader(dataset, args.batch_size, pad_value)
        model = PathDecoder(hidden_dim=512, n_layers=8, n_heads=8, include_aod=True).to(device)
        print(f"{model_family} parameters: {count_parameters(model)}")
        load_direct_checkpoint(model, checkpoint_path=checkpoint_path)
        gt, pred, avg_time = collect_direct_outputs(model, val_loader, args.max_generate)
    elif model_family == "first_step_residual":
        _, val_loader = build_residual_val_loader(dataset, args.batch_size, pad_value, args.n_clusters)
        model = FirstStepResidualPathDecoder(prompt_dim=10, hidden_dim=512, n_layers=8, n_heads=8, include_aod=True).to(device)
        print(f"{model_family} parameters: {count_parameters(model)}")
        load_residual_checkpoint(model, checkpoint_path)
        gt, pred, avg_time = collect_residual_outputs(model, val_loader, args.max_generate)
    elif model_family == "first_step_residual_corridor":
        val_data, val_loader = build_corridor_val_loader(dataset, args.batch_size, pad_value, args)
        prompt_dim = int(val_data.augmented_prompts[0].numel())
        model = FirstStepResidualPathDecoder(prompt_dim=prompt_dim, hidden_dim=512, n_layers=8, n_heads=8, include_aod=True).to(device)
        print(f"{model_family} parameters: {count_parameters(model)}")
        load_residual_checkpoint(model, checkpoint_path)
        gt, pred, avg_time = collect_residual_outputs(model, val_loader, args.max_generate)
    else:
        raise ValueError(f"Unknown model family: {model_family}")

    return {
        "groundtruth": gt,
        "prediction": pred,
        "prediction_dim": int(pred.shape[1]),
        "avg_inference_time_per_sample_sec": avg_time,
        "n_eval": int(gt.shape[0]),
        "checkpoint_path": checkpoint_path,
    }


def plot_metric(metric_name, title, gt_values, family_values, output_path):
    plt.figure(figsize=(7.2, 4.6))

    combined = [gt_values]
    combined.extend([vals for vals in family_values.values()])
    combined = [vals[np.isfinite(vals)] for vals in combined if vals.size > 0]

    if metric_name in {"phase_deg", "aoa_az_deg", "aoa_el_deg", "aod_az_deg", "aod_el_deg"}:
        bins = np.linspace(-180.0, 180.0, 61)
    else:
        all_vals = np.concatenate(combined)
        bins = np.histogram_bin_edges(all_vals, bins=50)

    plt.hist(gt_values, bins=bins, density=True, histtype="step", linewidth=2.4, color="black", label="groundtruth")

    color_map = {
        "direct": "#1f77b4",
        "first_step_residual": "#ff7f0e",
        "first_step_residual_corridor": "#2ca02c",
    }
    label_map = {
        "direct": "direct",
        "first_step_residual": "first-step residual",
        "first_step_residual_corridor": "corridor residual",
    }

    for family, values in family_values.items():
        plt.hist(
            values,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=2.0,
            color=color_map[family],
            label=label_map[family],
        )

    plt.title(title)
    plt.xlabel(title)
    plt.ylabel("Density")
    plt.grid(alpha=0.2)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


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
    for model_family, checkpoint_path in checkpoint_specs_for_scenario(args.scenario, args):
        print(f"\nEvaluating {model_family} on {args.scenario}")
        result = evaluate_model_family(dataset, model_family, checkpoint_path, args)
        family_results[model_family] = result
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
        gt_source = None
        family_values = {}
        for family, result in family_results.items():
            if result["prediction_dim"] <= idx:
                continue
            if gt_source is None:
                gt_source = result["groundtruth"][:, idx]
            family_values[family] = metric_transform(metric_name, result["prediction"][:, idx])
        if gt_source is None or not family_values:
            print(f"Skipping {metric_name}: no evaluated model predicts this metric.")
            continue
        gt_values = metric_transform(metric_name, gt_source)
        output_path = output_dir / f"{metric_name}_distribution_comparison.pdf"
        plot_metric(metric_name, f"First-step {title}", gt_values, family_values, output_path)
        print(f"Saved {output_path}")

    pd.DataFrame(timing_rows).to_csv(summary_csv, index=False)
    print(f"\nSaved inference-time summary to {summary_csv}")
    print(pd.DataFrame(timing_rows).to_string(index=False))


if __name__ == "__main__":
    main()
