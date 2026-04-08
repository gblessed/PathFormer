import argparse
from pathlib import Path

import deepmimo as dm
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from dataset.dataloaders import PreTrainMySeqDataLoader
from models import PathDecoder
from utils.utils import generate_paths_no_env_batch, masked_loss


DEFAULT_SCENARIOS = [
    "city_47_chicago_3p5",
]

PAD_VALUE = 0.0
MAX_GENERATE_STEPS = 25


def _extract_full_sample(item):
    prompt = item[0]
    paths = item[1]
    path_length = item[2]
    interactions = item[3]
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
        "first_target": first_step,
        "paths": paths.numpy().astype(np.float32),
        "interactions": interactions.numpy().astype(np.float32),
        "path_length": path_length.numpy().astype(np.float32),
    }


def _build_grouped_samples(seq_data):
    grouped = {}
    for item in seq_data:
        sample = _extract_full_sample(item)
        if sample is None:
            continue
        tx_key = sample["tx_key"]
        grouped.setdefault(tx_key, []).append(sample)

    grouped_out = {}
    for tx_key, samples in grouped.items():
        grouped_out[tx_key] = {
            "tx_pos": np.stack([s["tx_pos"] for s in samples], axis=0).astype(np.float32),
            "rx_pos": np.stack([s["rx_pos"] for s in samples], axis=0).astype(np.float32),
            "prompt": np.stack([s["prompt"] for s in samples], axis=0).astype(np.float32),
            "first_target": np.stack([s["first_target"] for s in samples], axis=0).astype(np.float32),
            "paths": [s["paths"] for s in samples],
            "interactions": [s["interactions"] for s in samples],
            "path_length": np.stack([s["path_length"] for s in samples], axis=0).astype(np.float32),
        }
    return grouped_out


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


def _pad_sequence_batch(paths_list, interactions_list):
    n_samples = len(paths_list)
    max_len = MAX_GENERATE_STEPS + 1

    padded_paths = np.full((n_samples, max_len, 5), PAD_VALUE, dtype=np.float32)
    padded_interactions = -np.ones((n_samples, max_len, 4), dtype=np.float32)
    path_padding_mask = np.zeros((n_samples, max_len), dtype=bool)

    for idx, (paths, interactions) in enumerate(zip(paths_list, interactions_list)):
        seq_len = min(paths.shape[0], max_len)
        padded_paths[idx, :seq_len] = paths[:seq_len]
        padded_interactions[idx, :seq_len] = interactions[:seq_len]
        path_padding_mask[idx, :seq_len] = True
    
    return padded_paths, padded_interactions, path_padding_mask


def _compute_cluster_timestep_means(padded_paths, path_padding_mask, labels, centers):
    n_clusters = centers.shape[0]
    max_len = padded_paths.shape[1]
    timestep_means = np.zeros((n_clusters, max_len, 2), dtype=np.float32)
    timestep_means[:, 0, :] = 0.0

    for cluster_id in range(n_clusters):
        cluster_mask = labels == cluster_id
        if not np.any(cluster_mask):
            continue

        cluster_paths = padded_paths[cluster_mask, :, :2]
        cluster_valid = path_padding_mask[cluster_mask]

        timestep_means[cluster_id, 1, :] = centers[cluster_id]
        for t in range(1, max_len):
            valid_t = cluster_valid[:, t]
            if np.any(valid_t):
                timestep_means[cluster_id, t, :] = cluster_paths[valid_t, t, :].mean(axis=0).astype(np.float32)
            elif t > 1:
                timestep_means[cluster_id, t, :] = timestep_means[cluster_id, t - 1, :]
            else:
                timestep_means[cluster_id, t, :] = centers[cluster_id]

    return timestep_means


def _apply_cluster_residual_transform(paths, timestep_means, path_padding_mask):
    paths_residual = paths.copy()
    paths_residual[:, 1:, :2] = paths_residual[:, 1:, :2] - timestep_means[:, 1:, :]

    invalid_mask = ~path_padding_mask
    paths_residual[invalid_mask] = PAD_VALUE
    paths_residual[:, 0, :] = 0.0
    return paths_residual


def _build_train_examples(train_group, n_clusters, random_state):
    first_targets = train_group["first_target"]
    n_train = first_targets.shape[0]
    k_eff = min(n_clusters, n_train)
    if k_eff <= 0:
        raise ValueError("No train samples available for this TX group.")

    kmeans = KMeans(n_clusters=k_eff, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(first_targets)
    centers = kmeans.cluster_centers_.astype(np.float32)
    stds = _compute_cluster_stats(first_targets, labels, centers)

    center_features = centers[labels]
    std_features = stds[labels]
    prompts = _build_prompt_features(train_group["prompt"], center_features, std_features)
    padded_paths, padded_interactions, path_padding_mask = _pad_sequence_batch(
        train_group["paths"],
        train_group["interactions"],
    )
    cluster_timestep_means = _compute_cluster_timestep_means(
        padded_paths,
        path_padding_mask,
        labels,
        centers,
    )
    sample_timestep_means = cluster_timestep_means[labels]

    padded_paths = _apply_cluster_residual_transform(
        padded_paths,
        sample_timestep_means,
        path_padding_mask,
    )

    return {
        "prompts": prompts,
        "paths": padded_paths,
        "interactions": padded_interactions,
        "path_padding_mask": path_padding_mask,
        "path_length": train_group["path_length"],
        "cluster_ids": labels.astype(np.int64),
        "kmeans_centers": centers,
        "kmeans_stds": stds,
        "cluster_timestep_means": cluster_timestep_means,
        "train_rx_pos": train_group["rx_pos"],
    }


def _build_val_examples(train_kmeans_data, val_group):
    if val_group["first_target"].shape[0] == 0:
        return None

    dists = np.sum(
        (val_group["rx_pos"][:, None, :] - train_kmeans_data["train_rx_pos"][None, :, :]) ** 2,
        axis=2,
    )
    nearest_train_idx = np.argmin(dists, axis=1)
    assigned_cluster_ids = train_kmeans_data["cluster_ids"][nearest_train_idx]
    center_features = train_kmeans_data["kmeans_centers"][assigned_cluster_ids]
    std_features = train_kmeans_data["kmeans_stds"][assigned_cluster_ids]
    timestep_means = train_kmeans_data["cluster_timestep_means"][assigned_cluster_ids]
    prompts = _build_prompt_features(val_group["prompt"], center_features, std_features)
    padded_paths, padded_interactions, path_padding_mask = _pad_sequence_batch(
        val_group["paths"],
        val_group["interactions"],
    )

    return {
        "prompts": prompts,
        "paths": padded_paths,
        "interactions": padded_interactions,
        "path_padding_mask": path_padding_mask,
        "path_length": val_group["path_length"],
        "centers": center_features.astype(np.float32),
        "stds": std_features.astype(np.float32),
        "timestep_means": timestep_means.astype(np.float32),
        "first_targets": val_group["first_target"],
        "cluster_ids": assigned_cluster_ids.astype(np.int64),
    }


def _fit_standardizer(x):
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1e-6, std)
    return mean.astype(np.float32), std.astype(np.float32)


def _apply_standardizer(x, mean, std):
    return ((x - mean) / std).astype(np.float32)


def _train_model(
    train_prompts,
    train_paths,
    train_path_lengths,
    train_interactions,
    train_path_padding_mask,
    hidden_dim,
    n_layers,
    n_heads,
    lr,
    weight_decay,
    batch_size,
    epochs,
    device,
):
    model = PathDecoder(
        prompt_dim=train_prompts.shape[1],
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        n_heads=n_heads,
        max_T=max(train_paths.shape[1] - 1, 1),
        prefix_len=4,
        pad_value=PAD_VALUE,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    ds = TensorDataset(
        torch.from_numpy(train_prompts),
        torch.from_numpy(train_paths),
        torch.from_numpy(train_path_lengths),
        torch.from_numpy(train_interactions),
        torch.from_numpy(train_path_padding_mask),
    )
    loader = DataLoader(ds, batch_size=min(batch_size, len(ds)), shuffle=True)

    model.train()
    pbar = tqdm(range(epochs), desc="Residual seq train", leave=False)
    for epoch in pbar:
        epoch_loss = 0.0
        n_batches = 0
        for prompts_b, paths_b, path_lengths_b, interactions_b, path_padding_mask_b in loader:
            prompts_b = prompts_b.to(device)
            paths_b = paths_b.to(device)
            path_lengths_b = path_lengths_b.to(device)
            interactions_b = interactions_b.to(device)
            path_padding_mask_b = path_padding_mask_b.to(device)

            paths_in = paths_b[:, :-1, :]
            interactions_in = interactions_b[:, :-1, :]
            paths_out = paths_b[:, 1:, :]
            # print(paths_out[0])
            interactions_out = interactions_b[:, 1:, :]

            (
                delay_pred,
                power_pred,
                phase_sin_pred,
                phase_cos_pred,
                phase_pred,
                az_sin_pred,
                az_cos_pred,
                az_pred,
                el_sin_pred,
                el_cos_pred,
                el_pred,
                path_length_pred,
                interaction_logits,
            ) = model(prompts_b, paths_in, interactions_in)

            total_loss, *_ = masked_loss(
                delay_pred,
                power_pred,
                phase_sin_pred,
                phase_cos_pred,
                phase_pred,
                az_sin_pred,
                az_cos_pred,
                az_pred,
                el_sin_pred,
                el_cos_pred,
                el_pred,
                path_length_pred,
                interaction_logits,
                paths_out,
                path_lengths_b,
                interactions_out,
                pad_value=PAD_VALUE,
                interaction_weight=0.1,
                path_padding_mask=path_padding_mask_b,
            )

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_loss += float(total_loss.item())
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        pbar.set_postfix(epoch=epoch + 1, loss=f"{avg_loss:.6f}")
    return model


@torch.no_grad()
def _generate_batched(model, prompts, timestep_means, batch_size, device):
    model.eval()
    generated_list = []
    pathcount_list = []

    for start in range(0, prompts.shape[0], batch_size):
        end = min(start + batch_size, prompts.shape[0])
        prompts_b = torch.from_numpy(prompts[start:end]).to(device)
        generated_b, pathcount_b, _ = generate_paths_no_env_batch(
            model,
            prompts_b,
            max_steps=MAX_GENERATE_STEPS,
        )
        generated_b = generated_b.numpy().astype(np.float32)
        generated_b[:, :, :2] = generated_b[:, :, :2] + timestep_means[start:end, 1 : 1 + MAX_GENERATE_STEPS, :]
        generated_list.append(generated_b)
        pathcount_list.append(pathcount_b.detach().cpu().numpy().astype(np.float32))

    return np.concatenate(generated_list, axis=0), np.concatenate(pathcount_list, axis=0)


def _compute_first_step_metrics(pred_arr, gt_arr):
    errors = pred_arr - gt_arr
    delay_rmse = float(np.sqrt(np.mean(errors[:, 0] ** 2)))
    power_rmse = float(np.sqrt(np.mean((errors[:, 1] / 0.01) ** 2)))
    joint_rmse = float(np.sqrt(np.mean(np.sum(errors[:, :2] ** 2, axis=1))))
    return delay_rmse, power_rmse, joint_rmse


def _compute_sequence_metrics(pred_paths, pred_path_lengths, gt_paths, gt_path_lengths):
    delay_sse = 0.0
    power_sse = 0.0
    joint_sse = 0.0
    n_steps = 0
    path_length_se = []

    for idx in range(gt_paths.shape[0]):
        n_valid = int(round(float(gt_path_lengths[idx].squeeze()) * MAX_GENERATE_STEPS))
        # n_valid = 1

        if n_valid <= 0:
            continue

        gt_seq = gt_paths[idx, 1:1 + n_valid, :2]
        pred_seq = pred_paths[idx, :n_valid, :2]
        err = pred_seq - gt_seq

        delay_sse += float(np.sum(err[:, 0] ** 2))
        power_sse += float(np.sum((err[:, 1] / 0.01) ** 2))
        joint_sse += float(np.sum(np.sum(err ** 2, axis=1)))
        n_steps += n_valid

        pred_len = float(np.squeeze(pred_path_lengths[idx]))
        gt_len = float(np.squeeze(gt_path_lengths[idx]))
        path_length_se.append((pred_len - gt_len) ** 2)

    if n_steps == 0:
        raise ValueError("No valid validation timesteps were available for evaluation.")

    delay_rmse = float(np.sqrt(delay_sse / n_steps))
    power_rmse = float(np.sqrt(power_sse / n_steps))
    joint_rmse = float(np.sqrt(joint_sse / n_steps))
    path_length_rmse = float(np.sqrt(np.mean(path_length_se))) if path_length_se else 0.0
    return delay_rmse, power_rmse, joint_rmse, path_length_rmse, int(n_steps)


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
    train_path_list = []
    train_path_length_list = []
    train_interaction_list = []
    train_path_padding_mask_list = []

    val_prompt_list = []
    val_center_list = []
    val_std_list = []
    val_timestep_mean_list = []
    val_first_target_list = []
    val_path_list = []
    val_path_length_list = []

    skipped_no_tx_match = 0
    for tx_key, train_group in train_by_tx.items():
        train_kmeans_data = _build_train_examples(
            train_group,
            n_clusters=n_clusters,
            random_state=random_state,
        )
        train_prompt_list.append(train_kmeans_data["prompts"])
        train_path_list.append(train_kmeans_data["paths"])
        train_path_length_list.append(train_kmeans_data["path_length"])
        train_interaction_list.append(train_kmeans_data["interactions"])
        train_path_padding_mask_list.append(train_kmeans_data["path_padding_mask"])

        val_group = val_by_tx.get(tx_key)
        if val_group is None:
            continue

        val_examples = _build_val_examples(train_kmeans_data, val_group)
        if val_examples is None:
            continue

        val_prompt_list.append(val_examples["prompts"])
        val_center_list.append(val_examples["centers"])
        val_std_list.append(val_examples["stds"])
        val_timestep_mean_list.append(val_examples["timestep_means"])
        val_first_target_list.append(val_examples["first_targets"])
        val_path_list.append(val_examples["paths"])
        val_path_length_list.append(val_examples["path_length"])

    for tx_key, val_group in val_by_tx.items():
        if tx_key not in train_by_tx:
            skipped_no_tx_match += int(val_group["first_target"].shape[0])

    if not train_prompt_list:
        raise ValueError("No train samples with a first timestep were found.")
    if not val_prompt_list:
        raise ValueError("No evaluable validation samples with a first timestep were found.")

    train_prompts = np.concatenate(train_prompt_list, axis=0)
    train_paths = np.concatenate(train_path_list, axis=0)
    train_path_lengths = np.concatenate(train_path_length_list, axis=0)
    train_interactions = np.concatenate(train_interaction_list, axis=0)
    train_path_padding_mask = np.concatenate(train_path_padding_mask_list, axis=0)

    val_prompts = np.concatenate(val_prompt_list, axis=0)
    val_centers = np.concatenate(val_center_list, axis=0)
    val_stds = np.concatenate(val_std_list, axis=0)
    val_timestep_means = np.concatenate(val_timestep_mean_list, axis=0)
    val_first_targets = np.concatenate(val_first_target_list, axis=0)
    val_paths = np.concatenate(val_path_list, axis=0)
    val_path_lengths = np.concatenate(val_path_length_list, axis=0)

    prompt_mean, prompt_std = _fit_standardizer(train_prompts)
    train_prompts_norm = _apply_standardizer(train_prompts, prompt_mean, prompt_std)
    val_prompts_norm = _apply_standardizer(val_prompts, prompt_mean, prompt_std)

    model = _train_model(
        train_prompts=train_prompts_norm,
        train_paths=train_paths,
        train_path_lengths=train_path_lengths,
        train_interactions=train_interactions,
        train_path_padding_mask=train_path_padding_mask,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        n_heads=n_heads,
        lr=lr,
        weight_decay=weight_decay,
        batch_size=batch_size,
        epochs=epochs,
        device=device,
    )

    pred_paths, pred_path_lengths = _generate_batched(
        model,
        val_prompts_norm,
        val_timestep_means,
        batch_size=batch_size,
        device=device,
    )

    residual_delay_rmse, residual_power_rmse, residual_joint_rmse, residual_path_length_rmse, n_eval_steps = (
        _compute_sequence_metrics(pred_paths, pred_path_lengths, val_paths, val_path_lengths)
    )
    base_delay_rmse, base_power_rmse, base_joint_rmse = _compute_first_step_metrics(val_centers, val_first_targets)

    return {
        "scenario": scenario,
        "n_clusters": int(n_clusters),
        "epochs": int(epochs),
        "hidden_dim": int(hidden_dim),
        "n_layers": int(n_layers),
        "n_heads": int(n_heads),
        "normalize_delta": bool(normalize_delta),
        "prompt_dim": int(train_prompts_norm.shape[1]),
        "n_train_samples": int(train_prompts.shape[0]),
        "n_val_samples": int(val_paths.shape[0]),
        "n_eval": int(val_paths.shape[0]),
        "n_eval_steps": int(n_eval_steps),
        "n_skipped_no_tx_match": int(skipped_no_tx_match),
        "kmeans_delay_rmse": base_delay_rmse,
        "kmeans_power_rmse": base_power_rmse,
        "kmeans_joint_rmse": base_joint_rmse,
        "residual_delay_rmse": residual_delay_rmse,
        "residual_power_rmse": residual_power_rmse,
        "residual_joint_rmse": residual_joint_rmse,
        "residual_path_length_rmse": residual_path_length_rmse,
        "delay_rmse_gain_vs_kmeans": base_delay_rmse - residual_delay_rmse,
        "power_rmse_gain_vs_kmeans": base_power_rmse - residual_power_rmse,
        "joint_rmse_gain_vs_kmeans": base_joint_rmse - residual_joint_rmse,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Full-sequence residual model conditioned on first-timestep TX-aware KMeans retrieval."
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
        # try:
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
            f"n_eval={row['n_eval']}, "
            f"n_eval_steps={row['n_eval_steps']}"
        )
        # except Exception as exc:
        #     print(f"Failed scenario {scenario}: {exc}")
        #     rows.append(
        #         {
        #             "scenario": scenario,
        #             "n_clusters": int(args.n_clusters),
        #             "error": str(exc),
        #         }
        #     )

    output_path = Path(args.output_csv)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    main()
