import argparse
from pathlib import Path

import deepmimo as dm
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from dataset.dataloaders import PreTrainMySeqDataLoader
from models import PathDecoder


DEFAULT_SCENARIOS = [
    "city_47_chicago_3p5",
]


def _extract_first_timestep_sample(item):
    prompt = item[0]
    paths = item[1]
    if paths.shape[0] <= 1:
        return None

    tx_key = tuple(prompt[:3].numpy().tolist())
    tx_pos = prompt[:3].numpy().astype(np.float32)
    rx_pos = prompt[3:].numpy().astype(np.float32)
    first_step = paths[1, :2].numpy().astype(np.float32)
    return {
        "tx_key": tx_key,
        "tx_pos": tx_pos,
        "rx_pos": rx_pos,
        "prompt": prompt.numpy().astype(np.float32),
        "target": first_step,
    }


def _build_grouped_samples(seq_data):
    grouped = {}
    for item in seq_data:
        sample = _extract_first_timestep_sample(item)
        if sample is None:
            continue
        tx_key = sample["tx_key"]
        grouped.setdefault(tx_key, []).append(sample)

    for tx_key, samples in grouped.items():
        grouped[tx_key] = {
            "tx_pos": np.stack([s["tx_pos"] for s in samples], axis=0).astype(np.float32),
            "rx_pos": np.stack([s["rx_pos"] for s in samples], axis=0).astype(np.float32),
            "prompt": np.stack([s["prompt"] for s in samples], axis=0).astype(np.float32),
            "target": np.stack([s["target"] for s in samples], axis=0).astype(np.float32),
        }
    return grouped


def _compute_cluster_stats(targets, labels, centers):
    n_clusters = centers.shape[0]
    stds = np.zeros_like(centers, dtype=np.float32)
    for k in range(n_clusters):
        members = targets[labels == k]
        if len(members) == 0:
            continue
        stds[k] = members.std(axis=0).astype(np.float32)
    return stds


def _build_prompt_features(base_prompt, center_features, std_features):
    return np.concatenate([base_prompt, center_features, std_features], axis=1).astype(np.float32)


def _build_train_examples(train_group, n_clusters, random_state):
    targets = train_group["target"]
    n_train = targets.shape[0]
    k_eff = min(n_clusters, n_train)
    if k_eff <= 0:
        raise ValueError("No train samples available for this TX group.")

    kmeans = KMeans(n_clusters=k_eff, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(targets)
    centers = kmeans.cluster_centers_.astype(np.float32)
    stds = _compute_cluster_stats(targets, labels, centers)

    center_features = centers[labels]
    std_features = stds[labels]
    deltas = targets - center_features
    prompts = _build_prompt_features(train_group["prompt"], center_features, std_features)

    return {
        "prompts": prompts,
        "targets": targets,
        "centers": center_features,
        "stds": std_features,
        "deltas": deltas.astype(np.float32),
        "cluster_ids": labels.astype(np.int64),
        "kmeans_centers": centers,
        "kmeans_stds": stds,
        "train_rx_pos": train_group["rx_pos"],
    }


def _build_val_examples(train_kmeans_data, val_group):
    if val_group["target"].shape[0] == 0:
        return None

    dists = np.sum(
        (val_group["rx_pos"][:, None, :] - train_kmeans_data["train_rx_pos"][None, :, :]) ** 2,
        axis=2,
    )
    nearest_train_idx = np.argmin(dists, axis=1)
    assigned_cluster_ids = train_kmeans_data["cluster_ids"][nearest_train_idx]
    center_features = train_kmeans_data["kmeans_centers"][assigned_cluster_ids]
    std_features = train_kmeans_data["kmeans_stds"][assigned_cluster_ids]
    prompts = _build_prompt_features(val_group["prompt"], center_features, std_features)

    return {
        "prompts": prompts,
        "targets": val_group["target"],
        "centers": center_features.astype(np.float32),
        "cluster_ids": assigned_cluster_ids.astype(np.int64),
    }


def _fit_standardizer(x):
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1e-6, std)
    return mean.astype(np.float32), std.astype(np.float32)


def _apply_standardizer(x, mean, std):
    return ((x - mean) / std).astype(np.float32)


def _build_decoder_io(batch_size, device):
    paths_in = torch.zeros(batch_size, 1, 5, dtype=torch.float32, device=device)
    interactions_in = -torch.ones(batch_size, 1, 4, dtype=torch.float32, device=device)
    return paths_in, interactions_in


def _train_model(train_prompts, train_delta, hidden_dim, n_layers, n_heads, lr, weight_decay, batch_size, epochs, device):
    model = PathDecoder(
        prompt_dim=train_prompts.shape[1],
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        n_heads=n_heads,
        max_T=4,
        prefix_len=4,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    ds = TensorDataset(
        torch.from_numpy(train_prompts),
        torch.from_numpy(train_delta),
    )
    loader = DataLoader(ds, batch_size=min(batch_size, len(ds)), shuffle=True)

    model.train()
    pbar = tqdm(range(epochs), desc="Residual train", leave=False)
    for epoch in pbar:
        epoch_loss = 0.0
        n_batches = 0
        for prompts_b, delta_b in loader:
            prompts_b = prompts_b.to(device)
            delta_b = delta_b.to(device)
            paths_in, interactions_in = _build_decoder_io(prompts_b.size(0), device)
            outputs = model(prompts_b, paths_in, interactions_in)
            delay_pred = outputs[0][:, 0]
            power_pred = outputs[1][:, 0]
            pred = torch.stack([delay_pred, power_pred], dim=1)
            loss = F.mse_loss(pred, delta_b)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item())
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        pbar.set_postfix(epoch=epoch + 1, loss=f"{avg_loss:.6f}")
    return model


@torch.no_grad()
def _predict(model, prompts, device):
    model.eval()
    xb = torch.from_numpy(prompts).to(device)
    paths_in, interactions_in = _build_decoder_io(xb.size(0), device)
    outputs = model(xb, paths_in, interactions_in)
    delay_pred = outputs[0][:, 0]
    power_pred = outputs[1][:, 0]
    return torch.stack([delay_pred, power_pred], dim=1).cpu().numpy().astype(np.float32)


def _compute_metrics(pred_arr, gt_arr):
    errors = pred_arr - gt_arr
    delay_rmse = float(np.sqrt(np.mean(errors[:, 0] ** 2)))
    power_rmse = float(np.sqrt(np.mean((errors[:, 1] / 0.01) ** 2)))
    joint_rmse = float(np.sqrt(np.mean(np.sum(errors ** 2, axis=1))))
    return delay_rmse, power_rmse, joint_rmse


def evaluate_scenario(
    scenario,
    n_clusters=5,
    sort_by="power",
    split_by="user",
    train_ratio=0.8,
    random_state=42,
    hidden_dim=128,
    n_layers=4,
    n_heads=4,
    lr=1e-3,
    weight_decay=1e-4,
    batch_size=256,
    epochs=40,
    normalize_delta=True,
):
    dm.download(scenario)
    dataset = dm.load(scenario)

    train_data = PreTrainMySeqDataLoader(
        dataset,
        train=True,
        split_by=split_by,
        sort_by=sort_by,
        train_ratio=train_ratio,
    )
    val_data = PreTrainMySeqDataLoader(
        dataset,
        train=False,
        split_by=split_by,
        sort_by=sort_by,
        train_ratio=train_ratio,
    )

    train_by_tx = _build_grouped_samples(train_data)
    val_by_tx = _build_grouped_samples(val_data)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_prompt_list = []
    train_delta_list = []
    val_prompt_list = []
    val_center_list = []
    val_target_list = []
    kmeans_pred_list = []
    skipped_no_tx_match = 0
    print(f"Txs: {train_by_tx.keys()}")
    for tx_key, train_group in train_by_tx.items():
        train_kmeans_data = _build_train_examples(train_group, n_clusters=n_clusters, random_state=random_state)
        train_prompt_list.append(train_kmeans_data["prompts"])
        train_delta_list.append(train_kmeans_data["deltas"])

        val_group = val_by_tx.get(tx_key)
        if val_group is None:
            continue
        val_examples = _build_val_examples(train_kmeans_data, val_group)
        if val_examples is None:
            continue

        val_prompt_list.append(val_examples["prompts"])
        val_center_list.append(val_examples["centers"])
        val_target_list.append(val_examples["targets"])
        kmeans_pred_list.append(val_examples["centers"])

    for tx_key, val_group in val_by_tx.items():
        if tx_key not in train_by_tx:
            skipped_no_tx_match += int(val_group["target"].shape[0])

    if not train_prompt_list:
        raise ValueError("No train samples with a first timestep were found.")
    if not val_prompt_list:
        raise ValueError("No evaluable validation samples with a first timestep were found.")

    train_prompts = np.concatenate(train_prompt_list, axis=0)
    train_delta = np.concatenate(train_delta_list, axis=0)
    val_prompts = np.concatenate(val_prompt_list, axis=0)
    val_centers = np.concatenate(val_center_list, axis=0)
    val_targets = np.concatenate(val_target_list, axis=0)
    kmeans_preds = np.concatenate(kmeans_pred_list, axis=0)

    prompt_mean, prompt_std = _fit_standardizer(train_prompts)
    print(f"train_prompts: {train_prompts[:, :3].std(axis=0)}")
    train_prompts_norm = _apply_standardizer(train_prompts, prompt_mean, prompt_std)
    val_prompts_norm = _apply_standardizer(val_prompts, prompt_mean, prompt_std)
    # print(f"prompt_mean: {prompt_mean} prompt_std: {prompt_std}")
    if normalize_delta:
        delta_mean, delta_std = _fit_standardizer(train_delta)
        train_delta_norm = _apply_standardizer(train_delta, delta_mean, delta_std)
    else:
        delta_mean = np.zeros((1, train_delta.shape[1]), dtype=np.float32)
        delta_std = np.ones((1, train_delta.shape[1]), dtype=np.float32)
        train_delta_norm = train_delta.astype(np.float32)
    # print(f"train_prompts: {train_prompts[:, :3].std(axis=0)}")

    model = _train_model(
        train_prompts=train_prompts_norm,
        train_delta=train_delta_norm,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        n_heads=n_heads,
        lr=lr,
        weight_decay=weight_decay,
        batch_size=batch_size,
        epochs=epochs,
        device=device,
    )

    pred_delta_norm = _predict(model, val_prompts_norm, device)
    pred_delta = pred_delta_norm * delta_std + delta_mean
    pred_targets = val_centers + pred_delta

    delay_rmse, power_rmse, joint_rmse = _compute_metrics(pred_targets, val_targets)
    base_delay_rmse, base_power_rmse, base_joint_rmse = _compute_metrics(kmeans_preds, val_targets)

    return {
        "scenario": scenario,
        "n_clusters": int(n_clusters),
        "epochs": int(epochs),
        "hidden_dim": int(hidden_dim),
        "n_layers": int(n_layers),
        "n_heads": int(n_heads),
        "normalize_delta": bool(normalize_delta),
        "prompt_dim": int(train_prompts_norm.shape[1]),
        "n_train_first_timestep": int(train_prompts.shape[0]),
        "n_val_first_timestep": int(val_targets.shape[0]),
        "n_eval": int(val_targets.shape[0]),
        "n_skipped_no_tx_match": int(skipped_no_tx_match),
        "kmeans_delay_rmse": base_delay_rmse,
        "kmeans_power_rmse": base_power_rmse,
        "kmeans_joint_rmse": base_joint_rmse,
        "residual_delay_rmse": delay_rmse,
        "residual_power_rmse": power_rmse,
        "residual_joint_rmse": joint_rmse,
        "delay_rmse_gain_vs_kmeans": base_delay_rmse - delay_rmse,
        "power_rmse_gain_vs_kmeans": base_power_rmse - power_rmse,
        "joint_rmse_gain_vs_kmeans": base_joint_rmse - joint_rmse,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="First-timestep residual model on top of TX-aware KMeans retrieval using PathDecoder."
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        default=",".join(DEFAULT_SCENARIOS),
        help="Comma-separated scenario names",
    )
    parser.add_argument("--n-clusters", type=int, default=5)
    parser.add_argument("--sort-by", type=str, default="power", choices=["power", "delay"])
    parser.add_argument("--split-by", type=str, default="user")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--normalize-delta", action="store_true")
    parser.add_argument("--no-normalize-delta", dest="normalize_delta", action="store_false")
    parser.set_defaults(normalize_delta=True)
    parser.add_argument(
        "--output-csv",
        type=str,
        default="kmeans_first_timestep_residual_results.csv",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]

    rows = []
    for scenario in scenarios:
        try:
            print(f"\nEvaluating scenario: {scenario}")
            row = evaluate_scenario(
                scenario=scenario,
                n_clusters=args.n_clusters,
                sort_by=args.sort_by,
                split_by=args.split_by,
                train_ratio=args.train_ratio,
                hidden_dim=args.hidden_dim,
                n_layers=args.n_layers,
                n_heads=args.n_heads,
                lr=args.lr,
                weight_decay=args.weight_decay,
                batch_size=args.batch_size,
                epochs=args.epochs,
                normalize_delta=args.normalize_delta,
            )
            rows.append(row)
            print(
                f"{scenario} | "
                f"kmeans_delay_rmse={row['kmeans_delay_rmse']:.4f}, "
                f"residual_delay_rmse={row['residual_delay_rmse']:.4f}, "
                f"kmeans_power_rmse={row['kmeans_power_rmse']:.4f}, "
                f"residual_power_rmse={row['residual_power_rmse']:.4f}, "
                f"n_eval={row['n_eval']}"
            )
        except Exception as exc:
            print(f"Failed scenario {scenario}: {exc}")
            rows.append(
                {
                    "scenario": scenario,
                    "n_clusters": int(args.n_clusters),
                    "error": str(exc),
                }
            )

    output_path = Path(args.output_csv)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    main()
