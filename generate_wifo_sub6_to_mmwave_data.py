import argparse
import csv
import os
from collections import defaultdict

import deepmimo as dm
import numpy as np
import torch

from utils.utils import ChannelParameters, compute_single_array_response_torch

DEFAULT_SAVE = "/home/blessedg/Pathformer/WiFo/dataset/blessed_task/"
DEFAULT_STATS_CSV = "/home/blessedg/Pathformer/WiFo/dataset/blessed_task/neighborhood_stats_sub6_to_mmwave.csv"
DURATION = 23
DEFAULT_SCENARIOS = [
    "city_0_newyork_3p5_s",
    "city_1_losangeles_3p5",
    "city_2_chicago_3p5",
    "city_3_houston_3p5",
    "city_4_phoenix_3p5",
    "city_5_philadelphia_3p5",
    "city_6_miami_3p5",
    "city_7_sandiego_3p5",
    "city_8_dallas_3p5",
    "city_9_sanfrancisco_3p5",
    "city_10_austin_3p5",
    "city_11_santaclara_3p5",
    "city_12_fortworth_3p5",
    "city_13_columbus_3p5",
    "city_17_seattle_3p5_s",
    "city_18_denver_3p5",
    "city_19_oklahoma_3p5_s",
    "city_16_sanfrancisco_3p5_lwm",
    "city_23_beijing_3p5",
    "city_31_barcelona_3p5",
    "city_35_san_francisco_3p5",
    "city_47_chicago_3p5",
    "city_89_nairobi_3p5",
    "city_91_xiangyang_3p5",
    "city_92_sãopaulo_3p5",
    "boston5g_3p5",
    "city_86_ankara_3p5",
    "city_72_capetown_3p5",
    "city_84_baoding_3p5",
    "city_95_delhi_3p5",
    "city_96_osaka_3p5",
    "city_88_tongshan_3p5",
]


def compute_beam_label_from_channel(H, S):
    if not torch.is_tensor(H):
        H = torch.from_numpy(H)
    if not torch.is_tensor(S):
        S = torch.from_numpy(S)
    H = H.to(torch.complex64)
    S = S.to(torch.complex64)
    Y = S.conj().T @ H
    prx = torch.sum(torch.abs(Y) ** 2, dim=2)
    best = torch.argmax(prx, dim=1)
    return best, prx


def make_dft_codebook(B=32, ant_params=None):
    if ant_params is None:
        ant_params = ChannelParameters().bs_antenna

    n_beams = int(B)
    u = torch.linspace(-0.95, 0.95, steps=n_beams, dtype=torch.float32).unsqueeze(0)
    azimuth = torch.asin(u)
    elevation = torch.full_like(azimuth, np.pi / 2)
    codebook = compute_single_array_response_torch(ant_params, elevation, azimuth)
    return codebook.squeeze(0)


def normalize_dataset_layout(dataset, channels):
    if hasattr(dataset, "n_ue") and isinstance(dataset.n_ue, int):
        return [dataset], [channels]
    return list(dataset), list(channels)


def infer_mmwave_scenario(sub6_scenario):
    if "_3p5_s" in sub6_scenario:
        return sub6_scenario.replace("_3p5_s", "_28", 1)
    if "_3p5_lwm" in sub6_scenario:
        return sub6_scenario.replace("_3p5_lwm", "_28", 1)
    if "3p5" in sub6_scenario:
        return sub6_scenario.replace("3p5", "28", 1)
    raise ValueError(f"Expected a sub-6 scenario containing '3p5', got: {sub6_scenario}")


def rx_pos_key(rx_pos, decimals):
    rounded = np.round(np.asarray(rx_pos, dtype=np.float64), decimals=decimals)
    return tuple(rounded.tolist())


def match_users_by_rx_pos(sub6_data, mmwave_data, decimals):
    sub6_valid = np.where(sub6_data.los != -1)[0]
    mmwave_valid = np.where(mmwave_data.los != -1)[0]

    sub6_pos_to_indices = defaultdict(list)
    for idx in sub6_valid:
        sub6_pos_to_indices[rx_pos_key(sub6_data.rx_pos[idx], decimals)].append(int(idx))

    mmwave_pos_to_indices = defaultdict(list)
    for idx in mmwave_valid:
        mmwave_pos_to_indices[rx_pos_key(mmwave_data.rx_pos[idx], decimals)].append(int(idx))

    common_keys = sorted(set(sub6_pos_to_indices) & set(mmwave_pos_to_indices))
    matched_sub6 = []
    matched_mmwave = []
    duplicate_matches = 0

    for key in common_keys:
        sub6_list = sorted(sub6_pos_to_indices[key])
        mmwave_list = sorted(mmwave_pos_to_indices[key])
        pair_count = min(len(sub6_list), len(mmwave_list))
        duplicate_matches += max(len(sub6_list), len(mmwave_list)) - pair_count
        matched_sub6.extend(sub6_list[:pair_count])
        matched_mmwave.extend(mmwave_list[:pair_count])

    return (
        np.asarray(matched_sub6, dtype=np.int64),
        np.asarray(matched_mmwave, dtype=np.int64),
        {
            "sub6_valid": int(len(sub6_valid)),
            "mmwave_valid": int(len(mmwave_valid)),
            "common_positions": int(len(common_keys)),
            "matched_users": int(len(matched_sub6)),
            "dropped_due_to_duplicates": int(duplicate_matches),
        },
    )


def squeeze_channel_to_matrix(channel):
    channel = np.asarray(channel).squeeze()
    if channel.ndim != 2:
        raise ValueError(f"Expected a 2D per-user channel after squeeze, got shape {channel.shape}")
    return channel.astype(np.complex64, copy=False)


def get_direction_candidates(rx_pos, user_pos, direction, duration):
    x_vals = rx_pos[:, 0]
    y_vals = rx_pos[:, 1]
    user_x = user_pos[0]
    user_y = user_pos[1]

    if direction == "left":
        mask = np.isclose(y_vals, user_y) & (x_vals < user_x)
        candidates = np.where(mask)[0]
        order = np.argsort(x_vals[candidates])
        ordered = candidates[order]
        return ordered[-duration:]
    if direction == "right":
        mask = np.isclose(y_vals, user_y) & (x_vals > user_x)
        candidates = np.where(mask)[0]
        order = np.argsort(-x_vals[candidates])
        ordered = candidates[order]
        return ordered[:duration]
    if direction == "down":
        mask = np.isclose(x_vals, user_x) & (y_vals < user_y)
        candidates = np.where(mask)[0]
        order = np.argsort(y_vals[candidates])
        ordered = candidates[order]
        return ordered[-duration:]
    if direction == "up":
        mask = np.isclose(x_vals, user_x) & (y_vals > user_y)
        candidates = np.where(mask)[0]
        order = np.argsort(-y_vals[candidates])
        ordered = candidates[order]
        return ordered[:duration]
    raise ValueError(f"Unknown direction: {direction}")


def pick_history_indices(rx_pos, user_idx, duration):
    user_pos = rx_pos[user_idx]
    directions = ["left", "right", "down", "up"]

    best_direction = None
    best_candidates = np.array([], dtype=np.int64)
    for direction in directions:
        candidates = get_direction_candidates(rx_pos, user_pos, direction, duration)
        if candidates.size >= duration:
            history = candidates[-duration:] if direction in {"left", "down"} else candidates[:duration]
            return history.astype(np.int64, copy=False), direction, True
        if candidates.size > best_candidates.size:
            best_candidates = candidates
            best_direction = direction

    if best_candidates.size == 0:
        return None, "self", False
    return None, best_direction, False


def build_movement_sequences(channel_tx, rx_pos, indices, duration):
    split_channels = np.asarray(channel_tx)[indices]
    split_rx_pos = np.asarray(rx_pos)[indices]

    sequence_channels = []
    kept_indices = []
    directions = []
    discarded = 0
    for local_idx in range(len(indices)):
        history_indices, direction, is_valid = pick_history_indices(split_rx_pos, local_idx, duration)
        if not is_valid:
            discarded += 1
            continue
        ordered_indices = list(history_indices) + [local_idx]
        matrices = [squeeze_channel_to_matrix(split_channels[idx]) for idx in ordered_indices]
        sequence = np.expand_dims(np.array(matrices), 0)
        sequence_channels.append(sequence.astype(np.complex64, copy=False))
        kept_indices.append(indices[local_idx])
        directions.append(direction)

    if sequence_channels:
        stacked = np.stack(sequence_channels, axis=0)
    else:
        sample_shape = squeeze_channel_to_matrix(split_channels[0]).shape if len(split_channels) else (8, 32)
        stacked = np.empty((0, 1, duration + 1, sample_shape[0], sample_shape[1]), dtype=np.complex64)

    return stacked, np.asarray(kept_indices, dtype=np.int64), directions, discarded


def compute_sequence_diversity(sequence_channels, codebook):
    if sequence_channels.size == 0:
        return {
            "n_sequences": 0,
            "history_len": 0,
            "avg_unique_beams": 0.0,
            "min_unique_beams": 0,
            "max_unique_beams": 0,
            "constant_histories": 0,
            "constant_history_ratio": 0.0,
            "avg_majority_share": 0.0,
        }

    history_only = sequence_channels[:, 0, :-1, :, :]
    flat_history = history_only.reshape(-1, history_only.shape[-2], history_only.shape[-1])
    history_labels, _ = compute_beam_label_from_channel(flat_history, codebook)
    history_labels = history_labels.view(history_only.shape[0], history_only.shape[1]).cpu().numpy()

    unique_counts = []
    constant_histories = 0
    majority_shares = []
    for row in history_labels:
        unique, counts = np.unique(row, return_counts=True)
        unique_counts.append(len(unique))
        majority_shares.append(float(counts.max() / counts.sum()))
        if len(unique) == 1:
            constant_histories += 1

    unique_counts = np.asarray(unique_counts, dtype=np.int64)
    majority_shares = np.asarray(majority_shares, dtype=np.float64)
    return {
        "n_sequences": int(history_labels.shape[0]),
        "history_len": int(history_labels.shape[1]),
        "avg_unique_beams": float(unique_counts.mean()),
        "min_unique_beams": int(unique_counts.min()),
        "max_unique_beams": int(unique_counts.max()),
        "constant_histories": int(constant_histories),
        "constant_history_ratio": float(constant_histories / max(len(unique_counts), 1)),
        "avg_majority_share": float(majority_shares.mean()),
    }


def append_stats_row(csv_path, row):
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    fieldnames = [
        "source_scenario",
        "label_scenario",
        "tx_idx",
        "split",
        "requested_users",
        "matched_users",
        "kept_users",
        "discarded_users",
        "discard_ratio",
        "history_len",
        "avg_unique_beams",
        "min_unique_beams",
        "max_unique_beams",
        "constant_histories",
        "constant_history_ratio",
        "avg_majority_share",
        "direction_counts",
        "sub6_valid",
        "mmwave_valid",
        "common_positions",
        "dropped_due_to_duplicates",
    ]
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fieldnames})


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=str, default="city_47_chicago_3p5")
    parser.add_argument("--mmwave-scenario", type=str, default=None)
    parser.add_argument("--scenarios", nargs="+", default=None)
    parser.add_argument("--all-scenarios", action="store_true")
    parser.add_argument("--save-dir", type=str, default=DEFAULT_SAVE)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--duration", type=int, default=DURATION)
    parser.add_argument("--stats-csv", type=str, default=DEFAULT_STATS_CSV)
    parser.add_argument("--n-beams", type=int, default=32)
    parser.add_argument("--match-decimals", type=int, default=4)
    return parser.parse_args()


def generate_for_scenario(sub6_scenario, mmwave_scenario, save_dir, train_ratio, seed, duration, stats_csv, n_beams, match_decimals):
    print(f"\nGenerating WiFo cross-band movement data for sub-6={sub6_scenario} label-band={mmwave_scenario}")
    sub6_dataset = dm.load(sub6_scenario)
    sub6_dataset.compute_channels(ChannelParameters())
    sub6_channels = sub6_dataset.channels

    mmwave_dataset = dm.load(mmwave_scenario)
    mmwave_dataset.compute_channels(ChannelParameters())
    mmwave_channels = mmwave_dataset.channels

    sub6_dataset, sub6_channels = normalize_dataset_layout(sub6_dataset, sub6_channels)
    mmwave_dataset, mmwave_channels = normalize_dataset_layout(mmwave_dataset, mmwave_channels)

    if len(sub6_dataset) != len(mmwave_dataset):
        print(
            f"Warning: TX count mismatch between bands: sub6={len(sub6_dataset)} mmwave={len(mmwave_dataset)}. "
            f"Using the first {min(len(sub6_dataset), len(mmwave_dataset))} TX entries."
        )

    n_tx = min(len(sub6_dataset), len(mmwave_dataset))
    codebook = make_dft_codebook(B=n_beams)

    train_channels = []
    train_labels = []
    val_channels = []
    val_labels = []
    stats_rows = []

    for tx in range(n_tx):
        sub6_data = sub6_dataset[tx]
        mmwave_data = mmwave_dataset[tx]
        sub6_channel_tx = np.asarray(sub6_channels[tx]) * 1e6
        mmwave_channel_tx = np.asarray(mmwave_channels[tx])

        matched_sub6, matched_mmwave, match_stats = match_users_by_rx_pos(
            sub6_data, mmwave_data, match_decimals
        )
        if matched_sub6.size == 0:
            print(f"TX {tx}: no matched users after rx_pos alignment, skipping.")
            continue

        indices = np.arange(matched_sub6.size)
        np.random.seed(seed + tx)
        np.random.shuffle(indices)
        split_idx = int(train_ratio * len(indices))
        train_pair_idx = indices[:split_idx]
        val_pair_idx = indices[split_idx:]

        train_sub6_idx = matched_sub6[train_pair_idx]
        val_sub6_idx = matched_sub6[val_pair_idx]
        train_mmwave_idx = matched_mmwave[train_pair_idx]
        val_mmwave_idx = matched_mmwave[val_pair_idx]

        mmwave_labels, _ = compute_beam_label_from_channel(mmwave_channel_tx.squeeze(1), codebook)

        train_label_map = {int(sub6_idx): int(mmwave_idx) for sub6_idx, mmwave_idx in zip(train_sub6_idx, train_mmwave_idx)}
        val_label_map = {int(sub6_idx): int(mmwave_idx) for sub6_idx, mmwave_idx in zip(val_sub6_idx, val_mmwave_idx)}

        train_channel_seq, kept_train_indices, train_dirs, train_discarded = build_movement_sequences(
            sub6_channel_tx, sub6_data.rx_pos, train_sub6_idx, duration
        )
        val_channel_seq, kept_val_indices, val_dirs, val_discarded = build_movement_sequences(
            sub6_channel_tx, sub6_data.rx_pos, val_sub6_idx, duration
        )

        if kept_train_indices.size > 0:
            train_labels.append(
                mmwave_labels[np.asarray([train_label_map[int(idx)] for idx in kept_train_indices], dtype=np.int64)].cpu().numpy()
            )
            train_channels.append(train_channel_seq)
        if kept_val_indices.size > 0:
            val_labels.append(
                mmwave_labels[np.asarray([val_label_map[int(idx)] for idx in kept_val_indices], dtype=np.int64)].cpu().numpy()
            )
            val_channels.append(val_channel_seq)

        train_diversity = compute_sequence_diversity(train_channel_seq, codebook)
        val_diversity = compute_sequence_diversity(val_channel_seq, codebook)
        train_dir_counts = {d: train_dirs.count(d) for d in sorted(set(train_dirs))}
        val_dir_counts = {d: val_dirs.count(d) for d in sorted(set(val_dirs))}

        print(
            f"TX {tx}: matched={match_stats['matched_users']} train_requested={len(train_sub6_idx)} val_requested={len(val_sub6_idx)} "
            f"kept_train={len(kept_train_indices)} kept_val={len(kept_val_indices)} "
            f"discarded_train={train_discarded} discarded_val={val_discarded}"
        )
        print(
            f"TX {tx} diversity: "
            f"train_avg_unique={train_diversity['avg_unique_beams']:.2f} "
            f"train_constant_ratio={train_diversity['constant_history_ratio']:.3f} "
            f"val_avg_unique={val_diversity['avg_unique_beams']:.2f} "
            f"val_constant_ratio={val_diversity['constant_history_ratio']:.3f}"
        )

        stats_rows.append(
            {
                "source_scenario": sub6_scenario,
                "label_scenario": mmwave_scenario,
                "tx_idx": tx,
                "split": "train",
                "requested_users": int(len(train_sub6_idx)),
                "matched_users": int(match_stats["matched_users"]),
                "kept_users": int(len(kept_train_indices)),
                "discarded_users": int(train_discarded),
                "discard_ratio": float(train_discarded / max(len(train_sub6_idx), 1)),
                "history_len": train_diversity["history_len"],
                "avg_unique_beams": train_diversity["avg_unique_beams"],
                "min_unique_beams": train_diversity["min_unique_beams"],
                "max_unique_beams": train_diversity["max_unique_beams"],
                "constant_histories": train_diversity["constant_histories"],
                "constant_history_ratio": train_diversity["constant_history_ratio"],
                "avg_majority_share": train_diversity["avg_majority_share"],
                "direction_counts": str(train_dir_counts),
                "sub6_valid": int(match_stats["sub6_valid"]),
                "mmwave_valid": int(match_stats["mmwave_valid"]),
                "common_positions": int(match_stats["common_positions"]),
                "dropped_due_to_duplicates": int(match_stats["dropped_due_to_duplicates"]),
            }
        )
        stats_rows.append(
            {
                "source_scenario": sub6_scenario,
                "label_scenario": mmwave_scenario,
                "tx_idx": tx,
                "split": "val",
                "requested_users": int(len(val_sub6_idx)),
                "matched_users": int(match_stats["matched_users"]),
                "kept_users": int(len(kept_val_indices)),
                "discarded_users": int(val_discarded),
                "discard_ratio": float(val_discarded / max(len(val_sub6_idx), 1)),
                "history_len": val_diversity["history_len"],
                "avg_unique_beams": val_diversity["avg_unique_beams"],
                "min_unique_beams": val_diversity["min_unique_beams"],
                "max_unique_beams": val_diversity["max_unique_beams"],
                "constant_histories": val_diversity["constant_histories"],
                "constant_history_ratio": val_diversity["constant_history_ratio"],
                "avg_majority_share": val_diversity["avg_majority_share"],
                "direction_counts": str(val_dir_counts),
                "sub6_valid": int(match_stats["sub6_valid"]),
                "mmwave_valid": int(match_stats["mmwave_valid"]),
                "common_positions": int(match_stats["common_positions"]),
                "dropped_due_to_duplicates": int(match_stats["dropped_due_to_duplicates"]),
            }
        )

    if not train_channels or not val_channels:
        raise RuntimeError(
            "No paired movement samples were created. Check scenario pairing or reduce --match-decimals."
        )

    pair_name = sub6_scenario
    train_dict = {
        "channels": torch.from_numpy(np.concatenate(train_channels, axis=0)),
        "labels": torch.from_numpy(np.concatenate(train_labels, axis=0)),
        "source_scenario": sub6_scenario,
        "label_scenario": mmwave_scenario,
    }
    train_path = os.path.join(save_dir, f"_{pair_name}_train_data.pt")
    torch.save(train_dict, train_path)
    print(
        f"Saved train channel data {train_dict['channels'].shape} "
        f"Max Label: {train_dict['labels'].max()} to {train_path}"
    )

    val_dict = {
        "channels": torch.from_numpy(np.concatenate(val_channels, axis=0)),
        "labels": torch.from_numpy(np.concatenate(val_labels, axis=0)),
        "source_scenario": sub6_scenario,
        "label_scenario": mmwave_scenario,
    }
    val_path = os.path.join(save_dir, f"_{pair_name}_val_data.pt")
    torch.save(val_dict, val_path)
    print(
        f"Saved validation channel data {val_dict['channels'].shape} "
        f"Max Label: {val_dict['labels'].max()} to {val_path}"
    )

    for row in stats_rows:
        append_stats_row(stats_csv, row)
    print(f"Appended neighborhood stats to {stats_csv}")


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    if args.all_scenarios:
        scenarios = DEFAULT_SCENARIOS
    elif args.scenarios:
        scenarios = args.scenarios
    else:
        scenarios = [args.scenario]

    if args.mmwave_scenario and len(scenarios) != 1:
        raise ValueError("--mmwave-scenario can only be used when exactly one sub-6 scenario is selected.")

    for scenario in scenarios:
        mmwave_scenario = args.mmwave_scenario or infer_mmwave_scenario(scenario)
        generate_for_scenario(
            sub6_scenario=scenario,
            mmwave_scenario=mmwave_scenario,
            save_dir=args.save_dir,
            train_ratio=args.train_ratio,
            seed=args.seed,
            duration=args.duration,
            stats_csv=args.stats_csv,
            n_beams=args.n_beams,
            match_decimals=args.match_decimals,
        )


if __name__ == "__main__":
    main()
