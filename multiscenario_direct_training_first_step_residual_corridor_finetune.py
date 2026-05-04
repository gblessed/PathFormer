import argparse
import os
import warnings

import deepmimo as dm
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset.dataloaders import PreTrainMySeqDataLoader
from multiscenario_direct_training_first_step_residual import (
    FirstStepResidualDataset,
    FirstStepResidualPathDecoder,
    evaluate_model,
    get_resume_checkpoint_path,
    load_best_checkpoint,
    train_with_interactions,
)
from multiscenario_direct_training_first_step_residual_corridor import (
    build_first_step_assignments_with_corridor,
)
from scene_feature_utils import SceneFeatureBank

warnings.filterwarnings("ignore", category=UserWarning)

csv_log_file = "muldims_weighted_first_step_residual_corridor_finetune_results.csv"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune or evaluate a corridor-aware first-step residual PathDecoder on a single target scenario, starting from a pretrained multi-environment checkpoint."
    )
    parser.add_argument("--scenario", type=str, required=True, help="Target scenario to finetune/evaluate on.")
    parser.add_argument(
        "--pretrained-checkpoint",
        type=str,
        required=True,
        help="Path to the pretrained multi-environment checkpoint to load before finetuning/evaluation.",
    )
    parser.add_argument("--csv-log-file", type=str, default=csv_log_file)
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints_first_step_residual_corridor_finetune",
        help="Directory to store finetuned checkpoints.",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Optional experiment suffix for the finetuned checkpoint name.",
    )
    parser.add_argument("--noise-prob", type=float, default=0.0)
    parser.add_argument("--skip-train", action="store_true", help="Only evaluate the loaded checkpoint on the target scenario.")
    parser.add_argument("--n-clusters", type=int, default=25)
    parser.add_argument("--nearest-k", type=int, default=5)
    parser.add_argument("--corridor-k", type=int, default=5)
    parser.add_argument("--corridor-bins", type=int, default=8)
    parser.add_argument("--use-material-features", action="store_true")
    parser.add_argument("--no-material-features", dest="use_material_features", action="store_false")
    parser.set_defaults(use_material_features=True)
    return parser.parse_args()


def build_single_scenario_datasets(scenario, args, config):
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

    prompt_dim = int(train_aug_prompts[0].numel())
    train_data = FirstStepResidualDataset(base_train, train_aug_prompts, train_baselines)
    val_data = FirstStepResidualDataset(base_val, val_aug_prompts, val_baselines)
    train_loader = DataLoader(train_data, batch_size=config["BATCH_SIZE"], shuffle=True, collate_fn=train_data.collate_fn)
    val_loader = DataLoader(val_data, batch_size=config["BATCH_SIZE"], shuffle=False, collate_fn=val_data.collate_fn)
    return base_train, train_data, val_data, train_loader, val_loader, prompt_dim


def run_scenario(args):
    if not os.path.exists(args.pretrained_checkpoint):
        raise FileNotFoundError(f"Pretrained checkpoint not found: {args.pretrained_checkpoint}")

    scenario = args.scenario
    experiment_suffix = args.experiment_name or scenario
    config = {
        "BATCH_SIZE": 128,
        "PAD_VALUE": 0,
        "LR": 2e-5,
        "epochs": 100,
        "interaction_weight": 0.01,
        "experiment": f"first_step_residual_corridor_finetune_{experiment_suffix}",
        "hidden_dim": 512 * 2,
        "n_layers": 8 + 4,
        "n_heads": 8,
        "time_step_weighted": False,
        "TARGET_NOISE_PROB": args.noise_prob,
        "TARGET_NOISE_PARAMS": None,
    }

    base_train, train_data, val_data, train_loader, val_loader, prompt_dim = build_single_scenario_datasets(
        scenario,
        args,
        config,
    )

    model = FirstStepResidualPathDecoder(
        prompt_dim=prompt_dim,
        hidden_dim=config["hidden_dim"],
        n_layers=config["n_layers"],
        n_heads=config["n_heads"],
    ).to(device)

    _, pretrained_best_loss = load_best_checkpoint(model, args.pretrained_checkpoint)
    print(
        f"Loaded pretrained multi-environment checkpoint: {args.pretrained_checkpoint} "
        f"(best_val_loss={pretrained_best_loss.item() if hasattr(pretrained_best_loss, 'item') else pretrained_best_loss})"
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["LR"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=25, T_mult=1, eta_min=1e-8)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    finetuned_checkpoint_path = os.path.join(
        args.checkpoint_dir,
        f"{config['experiment']}_best_model_checkpoint.pth",
    )
    resume_checkpoint_path = get_resume_checkpoint_path(finetuned_checkpoint_path)

    checkpoint_used_for_eval = args.pretrained_checkpoint
    best_loss_for_eval = pretrained_best_loss

    if not args.skip_train:
        train_with_interactions(
            model,
            train_loader,
            val_loader,
            config,
            base_train,
            optimizer,
            scheduler,
            finetuned_checkpoint_path,
            resume_checkpoint_path=resume_checkpoint_path,
        )
        _, best_loss_for_eval = load_best_checkpoint(model, finetuned_checkpoint_path)
        checkpoint_used_for_eval = finetuned_checkpoint_path

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
        "scenario": scenario,
        "pretrained_checkpoint": args.pretrained_checkpoint,
        "checkpoint_used_for_eval": checkpoint_used_for_eval,
        "skip_train": args.skip_train,
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
        "best_val_loss": best_loss_for_eval.item() if hasattr(best_loss_for_eval, "item") else best_loss_for_eval,
    }

    pd.DataFrame([row]).to_csv(
        args.csv_log_file,
        mode="a",
        index=False,
        header=not os.path.exists(args.csv_log_file),
    )
    print(f"✓ Finetune/eval results saved to {args.csv_log_file}")


def main():
    args = parse_args()
    print(f"Target scenario: {args.scenario}")
    if args.skip_train:
        print("skip_train=True: evaluating the loaded pretrained checkpoint directly on the target scenario.")
    else:
        print("Finetuning on the target scenario before evaluation.")
    run_scenario(args)


if __name__ == "__main__":
    main()
