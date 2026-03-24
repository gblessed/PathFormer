from dataset.dataloaders import PreTrainMySeqDataLoader
import deepmimo as dm
from tqdm import tqdm
import torch
import numpy as np
import pandas as pd
import argparse
import os


DEFAULT_SCENARIOS = [
    "city_47_chicago_3p5",
    "city_23_beijing_3p5",
    "city_91_xiangyang_3p5",
    "city_17_seattle_3p5_s",
    "city_12_fortworth_3p5",
    "city_92_sãopaulo_3p5",
    "city_35_san_francisco_3p5",
    "city_10_florida_villa_7gp_1758095156175",
    "city_19_oklahoma_3p5_s",
    "city_74_chiyoda_3p5",
]


def _metrics_from_pair(pred, gt):
    delay_pred = pred[:, 0]
    delay = gt[:, 0]
    power_pred = pred[:, 1]
    power = gt[:, 1]
    phase_pred = pred[:, 2]
    phase = gt[:, 2]
    aoa_az_pred = pred[:, 3]
    aoa_az = gt[:, 3]
    aoa_el_pred = pred[:, 4]
    aoa_el = gt[:, 4]

    delay_rmse = torch.mean((delay_pred - delay) ** 2).sqrt().item()
    delay_mae = torch.mean(torch.abs(delay_pred - delay)).item()

    power_rmse = torch.mean((power_pred / 0.01 - power / 0.01) ** 2).sqrt().item()
    power_mae = torch.mean(torch.abs(power_pred / 0.01 - power / 0.01)).item()

    y_hat_angles = phase_pred / (np.pi / 180)
    y_angles = phase / (np.pi / 180)
    phase_circular_dist = (y_hat_angles - y_angles + 180) % 360 - 180
    phase_rmse = torch.mean(phase_circular_dist ** 2).sqrt().item()
    phase_mae = torch.mean(torch.abs(phase_circular_dist)).item()

    y_hat_az = aoa_az_pred / (np.pi / 180)
    y_az = aoa_az / (np.pi / 180)
    az_circular_dist = (y_hat_az - y_az + 180) % 360 - 180
    az_rmse = torch.mean(az_circular_dist ** 2).sqrt().item()
    az_mae = torch.mean(torch.abs(az_circular_dist)).item()

    y_hat_el = aoa_el_pred / (np.pi / 180)
    y_el = aoa_el / (np.pi / 180)
    el_circular_dist = (y_hat_el - y_el + 180) % 360 - 180
    el_rmse = torch.mean(el_circular_dist ** 2).sqrt().item()
    el_mae = torch.mean(torch.abs(el_circular_dist)).item()

    return {
        "delay_rmse": delay_rmse,
        "power_rmse": power_rmse,
        "phase_rmse": phase_rmse,
        "az_rmse": az_rmse,
        "el_rmse": el_rmse,
        "delay_mae": delay_mae,
        "power_mae": power_mae,
        "phase_mae": phase_mae,
        "az_mae": az_mae,
        "el_mae": el_mae,
    }


def _aggregate_metrics(metric_list):
    keys = [
        "delay_rmse",
        "power_rmse",
        "phase_rmse",
        "az_rmse",
        "el_rmse",
        "delay_mae",
        "power_mae",
        "phase_mae",
        "az_mae",
        "el_mae",
        "path_length_rmse",
        "path_length_mae",
    ]
    out = {}
    for k in keys:
        vals = [m[k] for m in metric_list if k in m]
        out[k] = float(np.mean(vals)) if len(vals) else 0.0
    return out


def _predict_from_neighbors(train_paths, train_lengths, neighbor_ids, gt_paths, knn_mode):
    """
    knn_mode:
      - 'kth': use the selected neighbor path directly
      - 'average': average the top-k neighbor paths (no weighting)
    """
    gt_no_sos = gt_paths[1:, :]
    if knn_mode == "kth":
        pred_paths = train_paths[int(neighbor_ids[0])][1:, :]
        T = min(len(pred_paths), len(gt_no_sos))
        if T == 0:
            return None, None, None
        pred = pred_paths[:T, :]
        gt = gt_no_sos[:T, :]
        pred_len_norm = float(train_lengths[int(neighbor_ids[0])] / 25.0)
        return pred, gt, pred_len_norm

    # Average mode: average across top-k neighbors.
    k_ids = [int(x) for x in neighbor_ids]
    neighbor_paths = [train_paths[j][1:, :] for j in k_ids]
    if len(neighbor_paths) == 0:
        return None, None, None
    common_len = min([len(p) for p in neighbor_paths] + [len(gt_no_sos)])
    if common_len == 0:
        return None, None, None
    stacked = torch.stack([p[:common_len, :] for p in neighbor_paths], dim=0)  # (k, T, 5)
    pred = torch.mean(stacked, dim=0)
    gt = gt_no_sos[:common_len, :]
    pred_len_norm = float(np.mean(train_lengths[k_ids]) / 25.0)
    return pred, gt, pred_len_norm


def evaluate_scenario(scenario, sort_by="power", split_by="user", train_ratio=0.8, k=1, knn_mode="kth"):
    dm.download(scenario)
    dataset = dm.load(scenario)
    train_data = PreTrainMySeqDataLoader(dataset, train=True, split_by=split_by, sort_by=sort_by, train_ratio=train_ratio)
    val_data = PreTrainMySeqDataLoader(dataset, train=False, split_by=split_by, sort_by=sort_by, train_ratio=train_ratio)

    # Build training index grouped by TX; this avoids wrong cross-TX matches.
    train_by_tx = {}
    for item in train_data:
        prompt = item[0]
        paths = item[1]
        tx_key = tuple(prompt[:3].numpy().tolist())
        user_location = prompt[3:].numpy()
        if tx_key not in train_by_tx:
            train_by_tx[tx_key] = {"locs": [], "paths": [], "lengths": []}
        train_by_tx[tx_key]["locs"].append(user_location)
        train_by_tx[tx_key]["paths"].append(paths)
        valid_len = max(paths.shape[0] - 1, 0)  # minus SOS
        train_by_tx[tx_key]["lengths"].append(valid_len)

    for tx_key in train_by_tx:
        train_by_tx[tx_key]["locs"] = np.stack(train_by_tx[tx_key]["locs"], axis=0)
        train_by_tx[tx_key]["lengths"] = np.asarray(train_by_tx[tx_key]["lengths"], dtype=np.float32)

    # Group validation samples by TX so nearest-neighbor is batched per TX.
    val_by_tx = {}
    for item in val_data:
        prompt = item[0]
        gt_paths = item[1]
        gt_norm_len = item[2].item()
        tx_key = tuple(prompt[:3].numpy().tolist())
        user_location = prompt[3:].numpy()
        if tx_key not in val_by_tx:
            val_by_tx[tx_key] = {"locs": [], "paths": [], "norm_lens": []}
        val_by_tx[tx_key]["locs"].append(user_location)
        val_by_tx[tx_key]["paths"].append(gt_paths)
        val_by_tx[tx_key]["norm_lens"].append(gt_norm_len)

    metrics = []
    skipped = 0
    for tx_key, val_group in tqdm(val_by_tx.items(), desc=f"{scenario} TX groups"):
        if tx_key not in train_by_tx:
            skipped += len(val_group["paths"])
            continue
        train_locs = train_by_tx[tx_key]["locs"]  # (Ntrain, 3)
        val_locs = np.stack(val_group["locs"], axis=0)  # (Nval, 3)

        # Fast batched NN search per TX.
        dists = np.sum((val_locs[:, None, :] - train_locs[None, :, :]) ** 2, axis=2)  # (Nval, Ntrain)
        n_train = dists.shape[1]
        k_eff = max(1, min(k, n_train))
        # Get top-k neighbor ids for each val sample (sorted by distance).
        topk_unsorted = np.argpartition(dists, kth=k_eff - 1, axis=1)[:, :k_eff]
        topk_dists = np.take_along_axis(dists, topk_unsorted, axis=1)
        order = np.argsort(topk_dists, axis=1)
        topk_idx = np.take_along_axis(topk_unsorted, order, axis=1)  # (Nval, k_eff)
        if knn_mode == "kth":
            kth_col = min(k_eff - 1, k - 1)
            selected_idx = topk_idx[:, kth_col:kth_col + 1]  # keep 2D for shared path
        else:
            selected_idx = topk_idx

        for i in range(selected_idx.shape[0]):
            pred, gt, pred_len_norm = _predict_from_neighbors(
                train_paths=train_by_tx[tx_key]["paths"],
                train_lengths=train_by_tx[tx_key]["lengths"],
                neighbor_ids=selected_idx[i],
                gt_paths=val_group["paths"][i],
                knn_mode=knn_mode,
            )
            if pred is None:
                continue
            m = _metrics_from_pair(pred, gt)
            gt_len_norm = float(val_group["norm_lens"][i])
            m["path_length_rmse"] = float(np.sqrt((pred_len_norm - gt_len_norm) ** 2))
            m["path_length_mae"] = float(np.abs(pred_len_norm - gt_len_norm))
            metrics.append(m)

    agg = _aggregate_metrics(metrics)
    agg["scenario"] = scenario
    agg["n_val"] = int(len(val_data))
    agg["n_eval"] = int(len(metrics))
    agg["n_skipped_no_tx_match"] = int(skipped)
    return agg


def parse_args():
    parser = argparse.ArgumentParser(description="KNN baseline evaluator across scenarios")
    parser.add_argument(
        "--scenarios",
        type=str,
        default=",".join(DEFAULT_SCENARIOS),
        help="Comma-separated scenario names",
    )
    parser.add_argument("--sort-by", type=str, default="power", choices=["power", "delay"])
    parser.add_argument("--split-by", type=str, default="user")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--k", type=int, default=1, help="Neighbor rank/count. kth uses rank k; average uses top-k")
    parser.add_argument("--knn-mode", type=str, default="kth", choices=["kth", "average"])
    parser.add_argument("--output-csv", type=str, default="knn_baseline_results.csv")
    return parser.parse_args()

# python knn_baseline.py --knn-mode average --k 5 --output-csv knn_baseline_results_kth_5.csv
def main():
    args = parse_args()
    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]

    results = []
    for scenario in scenarios:
        try:
            print(f"\nEvaluating scenario: {scenario}")
            row = evaluate_scenario(
                scenario=scenario,
                sort_by=args.sort_by,
                split_by=args.split_by,
                train_ratio=args.train_ratio,
                k=args.k,
                knn_mode=args.knn_mode,
            )
            results.append(row)
            row["k"] = int(args.k)
            row["knn_mode"] = args.knn_mode
            print(
                f"{scenario} | delay_rmse={row['delay_rmse']:.4f}, "
                f"power_rmse={row['power_rmse']:.4f}, phase_rmse={row['phase_rmse']:.4f}, "
                f"az_rmse={row['az_rmse']:.4f}, el_rmse={row['el_rmse']:.4f}, "
                f"mode={args.knn_mode}, k={args.k}"
            )
        except Exception as e:
            print(f"Failed scenario {scenario}: {e}")
            results.append({"scenario": scenario, "error": str(e)})

    df = pd.DataFrame(results)
    df.to_csv(args.output_csv, index=False)
    print(f"\nSaved results to {args.output_csv}")


if __name__ == "__main__":
    main()
