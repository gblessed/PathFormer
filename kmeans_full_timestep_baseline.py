import argparse
from pathlib import Path

import deepmimo as dm
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from dataset.dataloaders import PreTrainMySeqDataLoader


DEFAULT_SCENARIOS = [
    "city_47_chicago_3p5",
]

PAD_VALUE = 0.0
MAX_GENERATE_STEPS = 25


def _extract_full_sample(item):
    prompt = item[0]
    paths = item[1]
    path_length = item[2]
    if paths.shape[0] <= 1:
        return None

    tx_key = tuple(prompt[:3].numpy().tolist())
    rx_pos = prompt[3:].numpy().astype(np.float32)
    first_step = paths[1, :2].numpy().astype(np.float32)
    return {
        "tx_key": tx_key,
        "rx_pos": rx_pos,
        "first_target": first_step,
        "paths": paths.numpy().astype(np.float32),
        "path_length": path_length.numpy().astype(np.float32),
    }


def _build_grouped_samples(seq_data):
    grouped = {}
    for item in seq_data:
        sample = _extract_full_sample(item)
        if sample is None:
            continue
        grouped.setdefault(sample["tx_key"], []).append(sample)

    grouped_out = {}
    for tx_key, samples in grouped.items():
        grouped_out[tx_key] = {
            "rx_pos": np.stack([s["rx_pos"] for s in samples], axis=0).astype(np.float32),
            "first_target": np.stack([s["first_target"] for s in samples], axis=0).astype(np.float32),
            "paths": [s["paths"] for s in samples],
            "path_length": np.stack([s["path_length"] for s in samples], axis=0).astype(np.float32),
        }
    return grouped_out


def _pad_sequence_batch(paths_list):
    n_samples = len(paths_list)
    max_len = MAX_GENERATE_STEPS + 1

    padded_paths = np.full((n_samples, max_len, 5), PAD_VALUE, dtype=np.float32)
    path_padding_mask = np.zeros((n_samples, max_len), dtype=bool)

    for idx, paths in enumerate(paths_list):
        seq_len = min(paths.shape[0], max_len)
        padded_paths[idx, :seq_len] = paths[:seq_len]
        path_padding_mask[idx, :seq_len] = True

    return padded_paths, path_padding_mask


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


def _compute_cluster_timestep_stds(padded_paths, path_padding_mask, labels, centers, timestep_means):
    n_clusters = centers.shape[0]
    max_len = padded_paths.shape[1]
    timestep_stds = np.zeros((n_clusters, max_len, 2), dtype=np.float32)

    for cluster_id in range(n_clusters):
        cluster_mask = labels == cluster_id
        if not np.any(cluster_mask):
            continue

        cluster_paths = padded_paths[cluster_mask, :, :2]
        cluster_valid = path_padding_mask[cluster_mask]

        for t in range(1, max_len):
            valid_t = cluster_valid[:, t]
            if np.any(valid_t):
                timestep_stds[cluster_id, t, :] = cluster_paths[valid_t, t, :].std(axis=0).astype(np.float32)
            else:
                timestep_stds[cluster_id, t, :] = 0.0

    return timestep_stds


def _cluster_timestep_rows(scenario, tx_key, labels, path_padding_mask, timestep_means, timestep_stds):
    rows = []
    n_clusters = timestep_means.shape[0]
    max_len = timestep_means.shape[1]

    for cluster_id in range(n_clusters):
        cluster_mask = labels == cluster_id
        if not np.any(cluster_mask):
            continue

        cluster_valid = path_padding_mask[cluster_mask]
        for t in range(1, max_len):
            n_valid = int(cluster_valid[:, t].sum())
            rows.append(
                {
                    "scenario": scenario,
                    "tx_key": str(tx_key),
                    "cluster_id": int(cluster_id),
                    "timestep": int(t),
                    "n_valid": n_valid,
                    "delay_mean": float(timestep_means[cluster_id, t, 0]),
                    "power_mean": float(timestep_means[cluster_id, t, 1]),
                    "delay_std": float(timestep_stds[cluster_id, t, 0]),
                    "power_std": float(timestep_stds[cluster_id, t, 1]),
                }
            )
    return rows


def _compute_sequence_metrics(pred_paths, gt_paths, gt_path_lengths):
    delay_sse = 0.0
    power_sse = 0.0
    joint_sse = 0.0
    n_steps = 0

    for idx in range(gt_paths.shape[0]):
        n_valid = int(round(float(np.squeeze(gt_path_lengths[idx])) * MAX_GENERATE_STEPS))
        if n_valid <= 0:
            continue

        gt_seq = gt_paths[idx, 1:1 + n_valid, :2]
        pred_seq = pred_paths[idx, 1:1 + n_valid, :2]
        err = pred_seq - gt_seq

        delay_sse += float(np.sum(err[:, 0] ** 2))
        power_sse += float(np.sum((err[:, 1] / 0.01) ** 2))
        joint_sse += float(np.sum(np.sum(err ** 2, axis=1)))
        n_steps += n_valid

    if n_steps == 0:
        raise ValueError("No valid validation timesteps were available for evaluation.")

    delay_rmse = float(np.sqrt(delay_sse / n_steps))
    power_rmse = float(np.sqrt(power_sse / n_steps))
    joint_rmse = float(np.sqrt(joint_sse / n_steps))
    return delay_rmse, power_rmse, joint_rmse, int(n_steps)


def evaluate_scenario(
    scenario,
    n_clusters=5,
    sort_by="power",
    split_by="user",
    train_ratio=0.8,
    random_state=42,
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

    pred_list = []
    gt_path_list = []
    gt_path_length_list = []
    skipped_no_tx_match = 0
    cluster_timestep_rows = []

    for tx_key, train_group in train_by_tx.items():
        n_train = train_group["first_target"].shape[0]
        k_eff = min(n_clusters, n_train)
        if k_eff <= 0:
            continue

        kmeans = KMeans(n_clusters=k_eff, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(train_group["first_target"])
        centers = kmeans.cluster_centers_.astype(np.float32)

        train_paths_padded, train_path_padding_mask = _pad_sequence_batch(train_group["paths"])
        cluster_timestep_means = _compute_cluster_timestep_means(
            train_paths_padded,
            train_path_padding_mask,
            labels,
            centers,
        )
        cluster_timestep_stds = _compute_cluster_timestep_stds(
            train_paths_padded,
            train_path_padding_mask,
            labels,
            centers,
            cluster_timestep_means,
        )
        cluster_timestep_rows.extend(
            _cluster_timestep_rows(
                scenario,
                tx_key,
                labels,
                train_path_padding_mask,
                cluster_timestep_means,
                cluster_timestep_stds,
            )
        )

        val_group = val_by_tx.get(tx_key)
        if val_group is None:
            continue

        dists = np.sum(
            (val_group["rx_pos"][:, None, :] - train_group["rx_pos"][None, :, :]) ** 2,
            axis=2,
        )
        nearest_train_idx = np.argmin(dists, axis=1)
        assigned_cluster_ids = labels[nearest_train_idx]
        pred_paths = cluster_timestep_means[assigned_cluster_ids]

        val_paths_padded, _ = _pad_sequence_batch(val_group["paths"])
        pred_list.append(pred_paths)
        gt_path_list.append(val_paths_padded)
        gt_path_length_list.append(val_group["path_length"])

    for tx_key, val_group in val_by_tx.items():
        if tx_key not in train_by_tx:
            skipped_no_tx_match += int(val_group["first_target"].shape[0])

    if not pred_list:
        raise ValueError("No evaluable validation samples with a first timestep were found.")

    pred_arr = np.concatenate(pred_list, axis=0)
    gt_paths = np.concatenate(gt_path_list, axis=0)
    gt_path_lengths = np.concatenate(gt_path_length_list, axis=0)

    delay_rmse, power_rmse, joint_rmse, n_eval_steps = _compute_sequence_metrics(
        pred_arr,
        gt_paths,
        gt_path_lengths,
    )

    return {
        "scenario": scenario,
        "n_clusters": int(n_clusters),
        "n_train_samples": int(sum(group["first_target"].shape[0] for group in train_by_tx.values())),
        "n_val_samples": int(sum(group["first_target"].shape[0] for group in val_by_tx.values())),
        "n_eval": int(gt_paths.shape[0]),
        "n_eval_steps": int(n_eval_steps),
        "n_skipped_no_tx_match": int(skipped_no_tx_match),
        "delay_rmse": delay_rmse,
        "power_rmse": power_rmse,
        "joint_rmse": joint_rmse,
    }, cluster_timestep_rows


def parse_args():
    parser = argparse.ArgumentParser(
        description="KMeans full-timestep baseline using cluster-wise per-timestep mean delay/power sequences."
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
    parser.add_argument(
        "--output-csv",
        type=str,
        default="kmeans_full_timestep_baseline_results.csv",
    )
    parser.add_argument(
        "--cluster-stats-csv",
        type=str,
        default="full_timestep_user_kmeans_cluster_stats.csv",
        help="CSV path for saving per-cluster per-timestep mean/std statistics.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]

    rows = []
    cluster_stats_rows = []
    for scenario in scenarios:
        try:
            print(f"\nEvaluating scenario: {scenario}")
            row, stats_rows = evaluate_scenario(
                scenario=scenario,
                n_clusters=args.n_clusters,
                sort_by=args.sort_by,
                split_by=args.split_by,
                train_ratio=args.train_ratio,
            )
            rows.append(row)
            cluster_stats_rows.extend(stats_rows)
            print(
                f"{scenario} | "
                f"delay_rmse={row['delay_rmse']:.4f}, "
                f"power_rmse={row['power_rmse']:.4f}, "
                f"joint_rmse={row['joint_rmse']:.4f}, "
                f"n_eval={row['n_eval']}, "
                f"n_eval_steps={row['n_eval_steps']}"
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

    cluster_stats_path = Path(args.cluster_stats_csv)
    pd.DataFrame(cluster_stats_rows).to_csv(cluster_stats_path, index=False)
    print(f"Saved cluster timestep stats to {cluster_stats_path}")


if __name__ == "__main__":
    main()
