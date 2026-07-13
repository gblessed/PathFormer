import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dm = None
PreTrainMySeqDataLoader = None
PathDecoder = None
load_best_checkpoint = None
FirstStepResidualDataset = None
FirstStepResidualPathDecoder = None
build_first_step_assignments = None
generate_paths_first_step_residual_batch = None
build_first_step_assignments_with_corridor = None
SceneFeatureBank = None
generate_paths_no_env_batch = None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot delay and power vs path index for one validation sample across three model families."
    )
    parser.add_argument("--scenario", required=True, help="Scenario name to load with DeepMIMO.")
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Validation-sample index to visualize.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="plots/model_family_single_sample",
        help="Directory where plots will be saved.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=25,
        help="Maximum number of autoregressive path steps to generate.",
    )
    parser.add_argument(
        "--direct-checkpoint-dir",
        type=str,
        default="base_no_env",
        help="Directory containing direct-model checkpoints.",
    )
    parser.add_argument(
        "--residual-checkpoint-dir",
        type=str,
        default="checkpoints_first_step_residual",
        help="Directory containing first-step residual checkpoints.",
    )
    parser.add_argument(
        "--corridor-checkpoint-dir",
        type=str,
        default="checkpoints_first_step_residual_corridor",
        help="Directory containing corridor checkpoints.",
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=25,
        help="Cluster count used for residual and corridor prompt augmentation.",
    )
    parser.add_argument(
        "--nearest-k",
        type=int,
        default=5,
        help="Neighborhood size for corridor features.",
    )
    parser.add_argument(
        "--corridor-k",
        type=int,
        default=5,
        help="Corridor neighbor count for corridor features.",
    )
    parser.add_argument(
        "--corridor-bins",
        type=int,
        default=8,
        help="Number of corridor angular bins.",
    )
    parser.add_argument(
        "--use-material-features",
        action="store_true",
        help="Use material features for the corridor prompt, matching training defaults.",
    )
    parser.add_argument(
        "--no-material-features",
        dest="use_material_features",
        action="store_false",
        help="Disable material features for the corridor prompt.",
    )
    parser.set_defaults(use_material_features=True)
    return parser.parse_args()


def _resolve_checkpoint(checkpoint_dir, filename):
    checkpoint_path = Path(checkpoint_dir) / filename
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    return str(checkpoint_path)


def _init_runtime_imports():
    global dm
    global PreTrainMySeqDataLoader
    global PathDecoder
    global load_best_checkpoint
    global FirstStepResidualDataset
    global FirstStepResidualPathDecoder
    global build_first_step_assignments
    global generate_paths_first_step_residual_batch
    global build_first_step_assignments_with_corridor
    global SceneFeatureBank
    global generate_paths_no_env_batch

    if dm is not None:
        return

    import deepmimo as dm_module

    from dataset.dataloaders import PreTrainMySeqDataLoader as loader_cls
    from models import PathDecoder as path_decoder_cls
    from multiscenario_direct_training import load_best_checkpoint as load_best_checkpoint_fn
    from multiscenario_direct_training_first_step_residual import (
        FirstStepResidualDataset as residual_dataset_cls,
        FirstStepResidualPathDecoder as residual_decoder_cls,
        build_first_step_assignments as build_assignments_fn,
        generate_paths_first_step_residual_batch as generate_residual_fn,
    )
    from multiscenario_direct_training_first_step_residual_corridor import (
        build_first_step_assignments_with_corridor as build_corridor_assignments_fn,
    )
    from scene_feature_utils import SceneFeatureBank as scene_feature_bank_cls
    from utils.utils import generate_paths_no_env_batch as generate_direct_fn

    dm = dm_module
    PreTrainMySeqDataLoader = loader_cls
    PathDecoder = path_decoder_cls
    load_best_checkpoint = load_best_checkpoint_fn
    FirstStepResidualDataset = residual_dataset_cls
    FirstStepResidualPathDecoder = residual_decoder_cls
    build_first_step_assignments = build_assignments_fn
    generate_paths_first_step_residual_batch = generate_residual_fn
    build_first_step_assignments_with_corridor = build_corridor_assignments_fn
    SceneFeatureBank = scene_feature_bank_cls
    generate_paths_no_env_batch = generate_direct_fn


def _extract_ground_truth(base_val, sample_index):
    prompt, paths, num_paths, interactions, env, env_prop = base_val[sample_index]
    gt_len = int(round(float(num_paths.item()) * 25))
    gt_paths = paths[1 : 1 + gt_len, :7].cpu().numpy()
    return {
        "prompt": prompt,
        "paths": gt_paths,
        "gt_len": gt_len,
        "interactions": interactions,
        "env": env,
        "env_prop": env_prop,
    }


def _load_direct_prediction(dataset, sample_index, scenario, checkpoint_dir, max_steps, target_len):
    model = PathDecoder(hidden_dim=512, n_layers=8, n_heads=8, include_aod=True).to(device)
    checkpoint_path = _resolve_checkpoint(
        checkpoint_dir,
        f"multiscenario_direct_{scenario}_best_model_checkpoint.pth",
        # f"{scenario}_best_model_checkpoint.pth",

        # f"multiscenario_direct_{scenario}_latest_checkpoint.pth",

    )
    load_best_checkpoint(model, checkpoint_path)

    prompt, *_ = dataset[sample_index]
    prompts = prompt.unsqueeze(0).to(device)
    generated, _, _ = generate_paths_no_env_batch(model, prompts, max_steps=max_steps)
    pred_paths = generated[0, :target_len, :7].cpu().numpy()
    return pred_paths


def _load_residual_prediction(base_train, base_val, sample_index, scenario, checkpoint_dir, max_steps, n_clusters, target_len):
    _, _, val_aug_prompts, val_baselines = build_first_step_assignments(base_train, base_val, n_clusters=n_clusters)
    val_data = FirstStepResidualDataset(base_val, val_aug_prompts, val_baselines)

    prompt, _, _, _, _, _, _, baseline = val_data.collate_fn([val_data[sample_index]])

    model = FirstStepResidualPathDecoder(prompt_dim=10, hidden_dim=512, n_layers=8, n_heads=8).to(device)
    checkpoint_path = _resolve_checkpoint(
        checkpoint_dir,
        f"first_step_residual_{scenario}_best_model_checkpoint.pth",
    )
    load_best_checkpoint(model, checkpoint_path)

    generated, _, _ = generate_paths_first_step_residual_batch(
        model,
        prompt.to(device),
        baseline.to(device),
        max_steps=max_steps,
    )
    pred_paths = generated[0, :target_len, :7].cpu().numpy()
    return pred_paths


def _load_corridor_prediction(
    dataset,
    base_train,
    base_val,
    sample_index,
    scenario,
    checkpoint_dir,
    max_steps,
    n_clusters,
    nearest_k,
    corridor_k,
    corridor_bins,
    use_material_features,
    target_len,
):
    scene_bank = SceneFeatureBank.from_dataset(dataset, use_material_features=use_material_features)
    _, _, val_aug_prompts, val_baselines = build_first_step_assignments_with_corridor(
        base_train,
        base_val,
        scene_bank,
        n_clusters=n_clusters,
        nearest_k=nearest_k,
        corridor_k=corridor_k,
        corridor_bins=corridor_bins,
    )
    val_data = FirstStepResidualDataset(base_val, val_aug_prompts, val_baselines)
    prompt, _, _, _, _, _, _, baseline = val_data.collate_fn([val_data[sample_index]])

    prompt_dim = int(val_aug_prompts[0].numel())
    model = FirstStepResidualPathDecoder(
        prompt_dim=prompt_dim,
        hidden_dim=512,
        n_layers=8,
        n_heads=8,
    ).to(device)
    checkpoint_path = _resolve_checkpoint(
        checkpoint_dir,
        f"first_step_residual_corridor_{scenario}_best_model_checkpoint.pth",
    )
    load_best_checkpoint(model, checkpoint_path)

    generated, _, _ = generate_paths_first_step_residual_batch(
        model,
        prompt.to(device),
        baseline.to(device),
        max_steps=max_steps,
    )
    pred_paths = generated[0, :target_len, :7].cpu().numpy()
    return pred_paths


def _plot_metric(output_path, title, ylabel, gt_values, series):
    plt.figure(figsize=(9, 5))
    gt_x = np.arange(1, len(gt_values) + 1)
    plt.plot(gt_x, gt_values, marker="o", linewidth=2.4, color="black", label=f"Ground truth")

    for label, values, color in series:
        x = np.arange(1, len(values) + 1)
        plt.plot(x, values, marker="o", linewidth=2.0, alpha=0.9, color=color, label=f"{label}")

    plt.xlabel("Path Index")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()


def main():
    args = parse_args()
    _init_runtime_imports()

    dataset = dm.load(args.scenario)
    base_train = PreTrainMySeqDataLoader(
        dataset,
        train=True,
        split_by="user",
        sort_by="power",
        normalizers=None,
        apply_normalizers=[],
        pad_value=0,
        include_aod=True,
    )
    base_val = PreTrainMySeqDataLoader(
        dataset,
        train=False,
        split_by="user",
        sort_by="power",
        normalizers=None,
        apply_normalizers=[],
        pad_value=0,
        include_aod=True,
    )

    if args.sample_index < 0 or args.sample_index >= len(base_val):
        raise IndexError(f"--sample-index must be in [0, {len(base_val) - 1}] for scenario {args.scenario}.")

    gt = _extract_ground_truth(base_val, args.sample_index)

    target_len = min(gt["gt_len"], args.max_steps)

    direct_paths = _load_direct_prediction(
        dataset=base_val,
        sample_index=args.sample_index,
        scenario=args.scenario,
        checkpoint_dir=args.direct_checkpoint_dir,
        max_steps=args.max_steps,
        target_len=target_len,
    )
    residual_paths = _load_residual_prediction(
        base_train=base_train,
        base_val=base_val,
        sample_index=args.sample_index,
        scenario=args.scenario,
        checkpoint_dir=args.residual_checkpoint_dir,
        max_steps=args.max_steps,
        n_clusters=args.n_clusters,
        target_len=target_len,
    )
    corridor_paths = _load_corridor_prediction(
        dataset=dataset,
        base_train=base_train,
        base_val=base_val,
        sample_index=args.sample_index,
        scenario=args.scenario,
        checkpoint_dir=args.corridor_checkpoint_dir,
        max_steps=args.max_steps,
        n_clusters=args.n_clusters,
        nearest_k=args.nearest_k,
        corridor_k=args.corridor_k,
        corridor_bins=args.corridor_bins,
        use_material_features=args.use_material_features,
        target_len=target_len,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.scenario}_sample_{args.sample_index}"

    delay_plot = output_dir / f"{stem}_delay.pdf"
    power_plot = output_dir / f"{stem}_power.pdf"

    _plot_metric(
        output_path=delay_plot,
        title=f"{args.scenario} sample {args.sample_index}: delay by path index",
        ylabel="Delay (us)",
        gt_values=gt["paths"][:target_len, 0],
        series=[
            ("Direct", direct_paths[:, 0], "#1f77b4"),
            ("Residual", residual_paths[:, 0], "#ff7f0e"),
            ("Corridor", corridor_paths[:, 0], "#2ca02c"),
        ],
    )
    _plot_metric(
        output_path=power_plot,
        title=f"{args.scenario} sample {args.sample_index}: power by path index",
        ylabel="Power",
        gt_values=gt["paths"][:target_len, 1] / 0.01,
        series=[
            ("Direct", direct_paths[:, 1] / 0.01, "#1f77b4"),
            ("Residual", residual_paths[:, 1] / 0.01, "#ff7f0e"),
            ("Corridor", corridor_paths[:, 1] / 0.01, "#2ca02c"),
        ],
    )

    print(f"Saved delay plot to {delay_plot}")
    print(f"Saved power plot to {power_plot}")
    print(f"Using ground-truth path length {target_len} for all plotted model outputs.")


if __name__ == "__main__":
    main()
