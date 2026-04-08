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


def _extract_first_timestep_sample(item):
    prompt = item[0]
    paths = item[1]
    if paths.shape[0] <= 1:
        return None

    tx_key = tuple(prompt[:3].numpy().tolist())
    rx_pos = prompt[3:].numpy().astype(np.float32)
    first_step = paths[1, :2].numpy().astype(np.float32)
    return {
        "tx_key": tx_key,
        "rx_pos": rx_pos,
        "target": first_step,
    }


def _build_train_index(train_data):
    train_by_tx = {}
    for item in train_data:
        sample = _extract_first_timestep_sample(item)
        if sample is None:
            continue
        tx_key = sample["tx_key"]
        if tx_key not in train_by_tx:
            train_by_tx[tx_key] = {"rx_pos": [], "targets": []}
        train_by_tx[tx_key]["rx_pos"].append(sample["rx_pos"])
        train_by_tx[tx_key]["targets"].append(sample["target"])

    for tx_key, group in train_by_tx.items():
        group["rx_pos"] = np.stack(group["rx_pos"], axis=0).astype(np.float32)
        group["targets"] = np.stack(group["targets"], axis=0).astype(np.float32)

    return train_by_tx


def _build_val_index(val_data):
    val_by_tx = {}
    for item in val_data:
        sample = _extract_first_timestep_sample(item)
        if sample is None:
            continue
        tx_key = sample["tx_key"]
        if tx_key not in val_by_tx:
            val_by_tx[tx_key] = {"rx_pos": [], "targets": []}
        val_by_tx[tx_key]["rx_pos"].append(sample["rx_pos"])
        val_by_tx[tx_key]["targets"].append(sample["target"])

    for tx_key, group in val_by_tx.items():
        group["rx_pos"] = np.stack(group["rx_pos"], axis=0).astype(np.float32)
        group["targets"] = np.stack(group["targets"], axis=0).astype(np.float32)

    return val_by_tx


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

    train_by_tx = _build_train_index(train_data)
    val_by_tx = _build_val_index(val_data)

    preds = []
    gts = []
    skipped_no_tx_match = 0

    for tx_key, train_group in train_by_tx.items():
        n_train = train_group["targets"].shape[0]
        k_eff = min(n_clusters, n_train)
        if k_eff == 0:
            continue

        kmeans = KMeans(n_clusters=k_eff, random_state=random_state, n_init=10)
        train_cluster_ids = kmeans.fit_predict(train_group["targets"])
        cluster_centers = kmeans.cluster_centers_.astype(np.float32)

        val_group = val_by_tx.get(tx_key)
        if val_group is None:
            continue

        dists = np.sum(
            (val_group["rx_pos"][:, None, :] - train_group["rx_pos"][None, :, :]) ** 2,
            axis=2,
        )
        nearest_train_idx = np.argmin(dists, axis=1)
        assigned_cluster_ids = train_cluster_ids[nearest_train_idx]
        pred_targets = cluster_centers[assigned_cluster_ids]

        preds.append(pred_targets)
        gts.append(val_group["targets"])

    for tx_key, val_group in val_by_tx.items():
        if tx_key not in train_by_tx:
            skipped_no_tx_match += int(val_group["targets"].shape[0])

    if not preds:
        raise ValueError("No evaluable validation samples with a first timestep were found.")

    pred_arr = np.concatenate(preds, axis=0)
    gt_arr = np.concatenate(gts, axis=0)
    errors = pred_arr - gt_arr

    delay_rmse = float(np.sqrt(np.mean(errors[:, 0] ** 2)))
    power_rmse = float(np.sqrt(np.mean((errors[:, 1] / 0.01) ** 2)))
    joint_rmse = float(np.sqrt(np.mean(np.sum(errors ** 2, axis=1))))

    return {
        "scenario": scenario,
        "n_clusters": int(n_clusters),
        "n_train_first_timestep": int(sum(group["targets"].shape[0] for group in train_by_tx.values())),
        "n_val_first_timestep": int(sum(group["targets"].shape[0] for group in val_by_tx.values())),
        "n_eval": int(gt_arr.shape[0]),
        "n_skipped_no_tx_match": int(skipped_no_tx_match),
        "delay_rmse": delay_rmse,
        "power_rmse": power_rmse,
        "joint_rmse": joint_rmse,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Simple KMeans baseline for first-timestep delay/power prediction."
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
        default="kmeans_first_timestep_baseline_results.csv",
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
            )
            rows.append(row)
            print(
                f"{scenario} | "
                f"delay_rmse={row['delay_rmse']:.4f}, "
                f"power_rmse={row['power_rmse']:.4f}, "
                f"joint_rmse={row['joint_rmse']:.4f}, "
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
