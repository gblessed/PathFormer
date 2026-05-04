import argparse
import os
import random
import warnings

import deepmimo as dm
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
from dataset.dataloaders import PreTrainMySeqDataLoaderAblate
from multiscenario_direct_training_first_step_residual import (
    FirstStepResidualDataset,
    FirstStepResidualPathDecoder,
    all_scenarios,
    evaluate_model,
    load_best_checkpoint,
    resolve_scenarios,
    train_with_interactions,
)
from scene_feature_utils import SceneFeatureBank

warnings.filterwarnings("ignore", category=UserWarning)

csv_log_file = "muldims_weighted_first_step_residual_corridor_results.csv"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _extract_first_step_metadata(seq_dataset, scene_bank, nearest_k, corridor_k, corridor_bins, radii):
    samples = []
    for idx in range(len(seq_dataset)):
        prompt, paths, *_ = seq_dataset[idx]
        tx_pos = np.asarray(seq_dataset.dataset_filtered["tx_pos"][idx], dtype=np.float32)
        rx_pos = np.asarray(seq_dataset.dataset_filtered["rx_pos"][idx], dtype=np.float32)
        tx_key = tuple(tx_pos.tolist())
        first_target = paths[1, :2].numpy().astype(np.float32) if paths.shape[0] > 1 else np.zeros(2, dtype=np.float32)
        scene_features = scene_bank.build_feature_vector(
            tx_pos=tx_pos,
            rx_pos=rx_pos,
            nearest_k=nearest_k,
            corridor_k=corridor_k,
            radii=radii,
            corridor_bins=corridor_bins,
        )
        samples.append(
            {
                "tx_key": tx_key,
                "tx_pos": tx_pos,
                "rx_pos": rx_pos,
                "prompt": prompt.numpy().astype(np.float32),
                "first_target": first_target,
                "scene_features": scene_features,
            }
        )
    return samples


def _compute_cluster_stats(targets, labels, centers):
    stds = np.zeros_like(centers, dtype=np.float32)
    for k in range(centers.shape[0]):
        members = targets[labels == k]
        if len(members) > 0:
            stds[k] = members.std(axis=0).astype(np.float32)
    return stds


def _fit_standardizer(tensor_list):
    arr = np.stack([t.numpy() for t in tensor_list], axis=0).astype(np.float32)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def _apply_standardizer(tensor_list, mean, std):
    out = []
    for tensor in tensor_list:
        arr = tensor.numpy().astype(np.float32)
        out.append(torch.from_numpy(((arr - mean) / std).astype(np.float32)))
    return out


def build_first_step_assignments_with_corridor(train_data, val_data, scene_bank, n_clusters, nearest_k=5, corridor_k=5, corridor_bins=8, radii=(25.0, 50.0, 100.0), seed=42):
    train_meta = _extract_first_step_metadata(train_data, scene_bank, nearest_k, corridor_k, corridor_bins, radii)
    val_meta = _extract_first_step_metadata(val_data, scene_bank, nearest_k, corridor_k, corridor_bins, radii)

    train_aug_prompts = [None] * len(train_meta)
    train_baselines = [None] * len(train_meta)
    val_aug_prompts = [None] * len(val_meta)
    val_baselines = [None] * len(val_meta)

    train_groups = {}
    for idx, sample in enumerate(train_meta):
        train_groups.setdefault(sample["tx_key"], {"indices": [], "rx_pos": [], "targets": []})
        train_groups[sample["tx_key"]]["indices"].append(idx)
        train_groups[sample["tx_key"]]["rx_pos"].append(sample["rx_pos"])
        train_groups[sample["tx_key"]]["targets"].append(sample["first_target"])

    val_groups = {}
    for idx, sample in enumerate(val_meta):
        val_groups.setdefault(sample["tx_key"], {"indices": [], "rx_pos": []})
        val_groups[sample["tx_key"]]["indices"].append(idx)
        val_groups[sample["tx_key"]]["rx_pos"].append(sample["rx_pos"])

    for tx_key, group in train_groups.items():
        rx_pos = np.stack(group["rx_pos"], axis=0).astype(np.float32)
        targets = np.stack(group["targets"], axis=0).astype(np.float32)
        k_eff = min(n_clusters, len(targets))
        kmeans = KMeans(n_clusters=k_eff, random_state=seed, n_init=10)
        labels = kmeans.fit_predict(targets)
        centers = kmeans.cluster_centers_.astype(np.float32)
        stds = _compute_cluster_stats(targets, labels, centers)

        for local_idx, dataset_idx in enumerate(group["indices"]):
            sample = train_meta[dataset_idx]
            baseline = centers[labels[local_idx]]
            std = stds[labels[local_idx]]
            prompt_aug = np.concatenate([sample["prompt"], baseline, std, sample["scene_features"]], axis=0).astype(np.float32)
            train_aug_prompts[dataset_idx] = torch.from_numpy(prompt_aug)
            train_baselines[dataset_idx] = torch.from_numpy(baseline.astype(np.float32))

        if tx_key not in val_groups:
            continue

        val_rx = np.stack(val_groups[tx_key]["rx_pos"], axis=0).astype(np.float32)
        dists = np.sum((val_rx[:, None, :] - rx_pos[None, :, :]) ** 2, axis=2)
        nearest_train_idx = np.argmin(dists, axis=1)
        assigned_labels = labels[nearest_train_idx]

        for local_idx, dataset_idx in enumerate(val_groups[tx_key]["indices"]):
            sample = val_meta[dataset_idx]
            baseline = centers[assigned_labels[local_idx]]
            std = stds[assigned_labels[local_idx]]
            prompt_aug = np.concatenate([sample["prompt"], baseline, std, sample["scene_features"]], axis=0).astype(np.float32)
            val_aug_prompts[dataset_idx] = torch.from_numpy(prompt_aug)
            val_baselines[dataset_idx] = torch.from_numpy(baseline.astype(np.float32))

    for idx in range(len(train_aug_prompts)):
        if train_aug_prompts[idx] is None:
            sample = train_meta[idx]
            zeros = np.zeros(4, dtype=np.float32)
            prompt_aug = np.concatenate([sample["prompt"], zeros, sample["scene_features"]], axis=0).astype(np.float32)
            train_aug_prompts[idx] = torch.from_numpy(prompt_aug)
            train_baselines[idx] = torch.zeros(2, dtype=torch.float32)

    for idx in range(len(val_aug_prompts)):
        if val_aug_prompts[idx] is None:
            sample = val_meta[idx]
            zeros = np.zeros(4, dtype=np.float32)
            prompt_aug = np.concatenate([sample["prompt"], zeros, sample["scene_features"]], axis=0).astype(np.float32)
            val_aug_prompts[idx] = torch.from_numpy(prompt_aug)
            val_baselines[idx] = torch.zeros(2, dtype=torch.float32)

    prompt_mean, prompt_std = _fit_standardizer(train_aug_prompts)
    train_aug_prompts = _apply_standardizer(train_aug_prompts, prompt_mean, prompt_std)
    val_aug_prompts = _apply_standardizer(val_aug_prompts, prompt_mean, prompt_std)

    return train_aug_prompts, train_baselines, val_aug_prompts, val_baselines


def parse_args():
    parser = argparse.ArgumentParser(description="Train/evaluate corridor-aware first-step residual PathDecoder across scenarios.")
    parser.add_argument("scenarios", nargs="*", help="Optional scenario names to run.")
    parser.add_argument("--scenario", dest="scenario_flag", action="append")
    parser.add_argument("--scenario-file", type=str)
    parser.add_argument("--shard-index", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=None)
    parser.add_argument("--csv-log-file", type=str, default=csv_log_file)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints_first_step_residual_corridor")
    parser.add_argument("--noise-prob", type=float, default=0.0)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--n-clusters", type=int, default=25)
    parser.add_argument("--nearest-k", type=int, default=5)
    parser.add_argument("--corridor-k", type=int, default=5)
    parser.add_argument("--corridor-bins", type=int, default=8)
    parser.add_argument("--use-material-features", action="store_true")
    parser.add_argument("--no-material-features", dest="use_material_features", action="store_false")
    parser.add_argument("--seed", type=int, default=42)
    parser.set_defaults(use_material_features=True)
    return parser.parse_args()


def run_scenario(scenario, args):
    set_seed(args.seed)
    dataset = dm.load(scenario)
    config = {
        "BATCH_SIZE": 128,
        "PAD_VALUE": 0,
        "LR": 2e-5,
        "epochs": 300,
        "interaction_weight": 0.01,
        "experiment": f"first_step_residual_corridor_{scenario}_{args.train_ratio}_seed{args.seed}",
        "hidden_dim": 512,
        "n_layers": 8,
        "n_heads": 8,
        "time_step_weighted": False,
        "TARGET_NOISE_PROB": args.noise_prob,
        "TARGET_NOISE_PARAMS": None,
    }

    base_train = PreTrainMySeqDataLoaderAblate(dataset, train=True, split_by="user", sort_by="power", normalizers=None, apply_normalizers=[], pad_value=config["PAD_VALUE"], train_ratio=args.train_ratio, seed=args.seed, include_aod=True)
    base_val = PreTrainMySeqDataLoaderAblate(dataset, train=False, split_by="user", sort_by="power", normalizers=None, apply_normalizers=[], pad_value=config["PAD_VALUE"], train_ratio=args.train_ratio, seed=args.seed, include_aod=True)

    scene_bank = SceneFeatureBank.from_dataset(dataset, use_material_features=args.use_material_features)
    train_aug_prompts, train_baselines, val_aug_prompts, val_baselines = build_first_step_assignments_with_corridor(
        base_train,
        base_val,
        scene_bank,
        n_clusters=args.n_clusters,
        nearest_k=args.nearest_k,
        corridor_k=args.corridor_k,
        corridor_bins=args.corridor_bins,
        seed=args.seed,
    )

    train_data = FirstStepResidualDataset(base_train, train_aug_prompts, train_baselines)
    val_data = FirstStepResidualDataset(base_val, val_aug_prompts, val_baselines)
    train_loader = DataLoader(train_data, batch_size=config["BATCH_SIZE"], shuffle=True, collate_fn=train_data.collate_fn)
    val_loader = DataLoader(val_data, batch_size=config["BATCH_SIZE"], shuffle=False, collate_fn=val_data.collate_fn)

    prompt_dim = int(train_aug_prompts[0].numel())
    model = FirstStepResidualPathDecoder(prompt_dim=prompt_dim, hidden_dim=config["hidden_dim"], n_layers=config["n_layers"], n_heads=config["n_heads"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["LR"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=25, T_mult=1, eta_min=1e-8)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.checkpoint_dir, f"{config['experiment']}_best_model_checkpoint.pth")

    if not args.skip_train:
        train_with_interactions(model, train_loader, val_loader, config, base_train, optimizer, scheduler, checkpoint_path)

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
        "scenario": scenario,
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "n_clusters": args.n_clusters,
        "nearest_k": args.nearest_k,
        "corridor_k": args.corridor_k,
        "corridor_bins": args.corridor_bins,
        "prompt_dim": prompt_dim,
        "use_material_features": args.use_material_features,
        "noise_prob": args.noise_prob,
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
    print(f"✓ Results for {scenario} saved to {args.csv_log_file}")


def main():
    args = parse_args()
    scenarios = resolve_scenarios(args)
    print(f"Running {len(scenarios)} scenario(s): {scenarios}")
    for scenario in scenarios:
        run_scenario(scenario, args)


if __name__ == "__main__":
    main()
