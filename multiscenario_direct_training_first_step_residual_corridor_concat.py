import argparse
import hashlib
import os
import warnings

import deepmimo as dm
import pandas as pd
import torch
from torch.utils.data import ConcatDataset, DataLoader

from dataset.dataloaders import PreTrainMySeqDataLoader
from multiscenario_direct_training_first_step_residual import (
    FirstStepResidualDataset,
    FirstStepResidualPathDecoder,
    evaluate_model,
    get_resume_checkpoint_path,
    load_best_checkpoint,
    resolve_scenarios,
    train_with_interactions,
)
from multiscenario_direct_training_first_step_residual_corridor import (
    build_first_step_assignments_with_corridor,
)
from scene_feature_utils import SceneFeatureBank

warnings.filterwarnings("ignore", category=UserWarning)

csv_log_file = "muldims_weighted_first_step_residual_corridor_concat_results.csv"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train/evaluate corridor-aware first-step residual PathDecoder on a concatenated multi-scenario dataset."
    )
    parser.add_argument("scenarios", nargs="*", help="Optional scenario names to run.")
    parser.add_argument("--scenario", dest="scenario_flag", action="append")
    parser.add_argument("--scenario-file", type=str)
    parser.add_argument("--shard-index", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=None)
    parser.add_argument("--csv-log-file", type=str, default=csv_log_file)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints_first_step_residual_corridor_concat")
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--noise-prob", type=float, default=0.0)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--n-clusters", type=int, default=25)
    parser.add_argument("--nearest-k", type=int, default=5)
    parser.add_argument("--corridor-k", type=int, default=5)
    parser.add_argument("--corridor-bins", type=int, default=8)
    parser.add_argument("--use-material-features", action="store_true")
    parser.add_argument("--no-material-features", dest="use_material_features", action="store_false")
    parser.set_defaults(use_material_features=True)
    return parser.parse_args()
4

def _scenario_group_name(scenarios):
    ordered = sorted(scenarios)
    digest = hashlib.md5("||".join(ordered).encode("utf-8")).hexdigest()[:8]
    if len(ordered) <= 3:
        prefix = "_".join(s.replace("/", "_") for s in ordered)
    else:
        prefix = f"{len(ordered)}scenarios"
    return f"{prefix}_{digest}"


def _make_concat_collate_fn(reference_dataset):
    def _collate_fn(batch):
        base_items = [item[0] for item in batch]
        aug_prompts = torch.stack([item[1] for item in batch], dim=0)
        first_step_baselines = torch.stack([item[2] for item in batch], dim=0)
        _, paths, path_lengths, interactions, env, env_prop, path_padding_mask = reference_dataset.base_dataset.collate_fn(base_items)
        return aug_prompts, paths, path_lengths, interactions, env, env_prop, path_padding_mask, first_step_baselines

    return _collate_fn


def build_concat_datasets(scenarios, args, config):
    train_wrappers = []
    val_wrappers = []
    prompt_dim = None

    for scenario in scenarios:
        print(f"Preparing scenario: {scenario}")
        dataset = dm.load(scenario)
        base_train = PreTrainMySeqDataLoader(
            dataset,
            train=True,
            split_by="user",
            sort_by="power",
            normalizers=None,
            apply_normalizers=[],
            pad_value=config["PAD_VALUE"],
            include_aod=True,
        )
        base_val = PreTrainMySeqDataLoader(
            dataset,
            train=False,
            split_by="user",
            sort_by="power",
            normalizers=None,
            apply_normalizers=[],
            pad_value=config["PAD_VALUE"],
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

        if prompt_dim is None:
            prompt_dim = int(train_aug_prompts[0].numel())
        elif prompt_dim != int(train_aug_prompts[0].numel()):
            raise ValueError(
                f"Prompt dimension mismatch across scenarios: expected {prompt_dim}, got {int(train_aug_prompts[0].numel())} for {scenario}"
            )

        train_data = FirstStepResidualDataset(base_train, train_aug_prompts, train_baselines)
        val_data = FirstStepResidualDataset(base_val, val_aug_prompts, val_baselines)
        print(f"  train samples={len(train_data)} | val samples={len(val_data)}")
        train_wrappers.append(train_data)
        val_wrappers.append(val_data)

    combined_train = ConcatDataset(train_wrappers)
    combined_val = ConcatDataset(val_wrappers)
    collate_fn = _make_concat_collate_fn(train_wrappers[0])
    return combined_train, combined_val, collate_fn, prompt_dim


def run_scenarios(scenarios, args):
    if not scenarios:
        raise ValueError("At least one scenario is required.")

    experiment_suffix = args.experiment_name or _scenario_group_name(scenarios)
    config = {
        "BATCH_SIZE": 128,
        "PAD_VALUE": 0,
        "LR": 2e-5,
        "epochs": 300,
        "interaction_weight": 0.01,
        "experiment": f"first_step_residual_corridor_concat_{experiment_suffix}",
        "hidden_dim": 512*2,
        "n_layers": 8+4,
        "n_heads": 8,
        "time_step_weighted": False,
        "TARGET_NOISE_PROB": args.noise_prob,
        "TARGET_NOISE_PARAMS": None,
    }

    train_data, val_data, collate_fn, prompt_dim = build_concat_datasets(scenarios, args, config)
    train_loader = DataLoader(train_data, batch_size=config["BATCH_SIZE"], shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_data, batch_size=config["BATCH_SIZE"], shuffle=False, collate_fn=collate_fn)

    model = FirstStepResidualPathDecoder(
        prompt_dim=prompt_dim,
        hidden_dim=config["hidden_dim"],
        n_layers=config["n_layers"],
        n_heads=config["n_heads"],
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["LR"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=25, T_mult=1, eta_min=1e-8)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.checkpoint_dir, f"{config['experiment']}_best_model_checkpoint.pth")
    resume_checkpoint_path = get_resume_checkpoint_path(checkpoint_path)

    if not args.skip_train:
        train_with_interactions(
            model,
            train_loader,
            val_loader,
            config,
            train_data.datasets[0].base_dataset,
            optimizer,
            scheduler,
            checkpoint_path,
            resume_checkpoint_path=resume_checkpoint_path,
        )

    _, best_loss = load_best_checkpoint(model, checkpoint_path)
    results = evaluate_model(model, val_loader)
    (
        avg_delay,
        std_delay,
        avg_power,
        std_power,
        avg_phase,
        std_phase,
        avg_az,
        std_az,
        avg_el,
        std_el,
        avg_aod_az,
        std_aod_az,
        avg_aod_el,
        std_aod_el,
        avg_path_length_rmse,
        std_path_length_rmse,
        avg_interaction_accuracy,
        std_interaction_accuracy,
        avg_interaction_f1,
        std_interaction_f1,
        avg_delay_mae,
        std_delay_mae,
        avg_power_mae,
        std_power_mae,
        avg_phase_mae,
        std_phase_mae,
        avg_az_mae,
        std_az_mae,
        avg_el_mae,
        std_el_mae,
        avg_aod_az_mae,
        std_aod_az_mae,
        avg_aod_el_mae,
        std_aod_el_mae,
        avg_path_length_mae,
        std_path_length_mae,
    ) = results

    row = {
        "scenario_group": experiment_suffix,
        "num_scenarios": len(scenarios),
        "scenarios": "|".join(scenarios),
        "n_clusters": args.n_clusters,
        "nearest_k": args.nearest_k,
        "corridor_k": args.corridor_k,
        "corridor_bins": args.corridor_bins,
        "prompt_dim": prompt_dim,
        "use_material_features": args.use_material_features,
        "noise_prob": args.noise_prob,
        "train_samples": len(train_data),
        "val_samples": len(val_data),
        "delay_rmse": avg_delay,
        "delay_rmse_std": std_delay,
        "power_rmse": avg_power,
        "power_rmse_std": std_power,
        "phase_rmse": avg_phase,
        "phase_rmse_std": std_phase,
        "az_rmse": avg_az,
        "az_rmse_std": std_az,
        "el_rmse": avg_el,
        "el_rmse_std": std_el,
        "aod_az_rmse": avg_aod_az,
        "aod_az_rmse_std": std_aod_az,
        "aod_el_rmse": avg_aod_el,
        "aod_el_rmse_std": std_aod_el,
        "path_length_rmse": avg_path_length_rmse,
        "path_length_rmse_std": std_path_length_rmse,
        "interaction_accuracy": avg_interaction_accuracy,
        "interaction_accuracy_std": std_interaction_accuracy,
        "interaction_f1": avg_interaction_f1,
        "interaction_f1_std": std_interaction_f1,
        "delay_mae": avg_delay_mae,
        "delay_mae_std": std_delay_mae,
        "power_mae": avg_power_mae,
        "power_mae_std": std_power_mae,
        "phase_mae": avg_phase_mae,
        "phase_mae_std": std_phase_mae,
        "avg_az_mae": avg_az_mae,
        "avg_az_mae_std": std_az_mae,
        "avg_el_mae": avg_el_mae,
        "avg_el_mae_std": std_el_mae,
        "avg_aod_az_mae": avg_aod_az_mae,
        "avg_aod_az_mae_std": std_aod_az_mae,
        "avg_aod_el_mae": avg_aod_el_mae,
        "avg_aod_el_mae_std": std_aod_el_mae,
        "path_length_mae": avg_path_length_mae,
        "path_length_mae_std": std_path_length_mae,
        "best_val_loss": best_loss.item() if hasattr(best_loss, "item") else best_loss,
    }
    pd.DataFrame([row]).to_csv(args.csv_log_file, mode="a", index=False, header=not os.path.exists(args.csv_log_file))
    print(f"✓ Combined results saved to {args.csv_log_file}")


def main():
    args = parse_args()
    scenarios = resolve_scenarios(args)
    if not scenarios:
        print("No scenarios selected for this run.")
        return
    print(f"Training on {len(scenarios)} concatenated scenario(s): {scenarios}")
    run_scenarios(scenarios, args)


if __name__ == "__main__":
    main()
