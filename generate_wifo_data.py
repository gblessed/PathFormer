import argparse
import csv

import deepmimo as dm
import numpy as np
import torch
import os

from utils.utils import ChannelParameters, compute_single_array_response_torch

DEFAULT_SAVE = "/home/blessedg/Pathformer/WiFo/dataset/blessed_task_user_loc/"
DEFAULT_STATS_CSV = "/home/blessedg/Pathformer/WiFo/dataset/blessed_task_user_loc/neighborhood_stats.csv"
DURATION = 23  # Number of historical users to prepend before the current user.
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
    # "city_72_capetown_3p5",
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


# def make_dft_codebook(B=8):
#     params = ChannelParameters()
#     az_t = np.linspace(-np.pi, np.pi, B, endpoint=False, dtype=np.float32)
#     el_t = np.linspace(-np.pi, np.pi, B, endpoint=False, dtype=np.float32)
#     az_new = []
#     el_new = []
#     for az in az_t:
#         for el in el_t:
#             az_new.append(az)
#             el_new.append(el)
#     az_new = torch.tensor(az_new).unsqueeze(1)
#     el_new = torch.tensor(el_new).unsqueeze(1)
#     array_response = compute_single_array_response_torch(params.bs_antenna, az_new, el_new)
#     return array_response.squeeze(2).T
    
def make_azimuth_codebook(B=8, ant_params=None):
    """Build a 64-beam azimuth sweep for the default 8x1 BS array."""
    n_beams= int(B)
    if ant_params is None:
        ant_params = ChannelParameters().bs_antenna

    azimuth = torch.linspace(-np.pi / 2, np.pi / 2, steps=n_beams, dtype=torch.float32).unsqueeze(0)
    elevation = torch.full_like(azimuth, np.pi / 2)
    codebook = compute_single_array_response_torch(ant_params, elevation, azimuth)
    return codebook.squeeze(0)


def make_dft_codebook(B=32, ant_params=None):
    """Build an easier azimuth codebook for the default 8x1 BS array."""
    if ant_params is None:
        ant_params = ChannelParameters().bs_antenna

    n_beams = int(B)

    # Sample uniformly in spatial frequency u = sin(phi), not in phi itself.
    u = torch.linspace(-0.95, 0.95, steps=n_beams, dtype=torch.float32).unsqueeze(0)
    azimuth = torch.asin(u)

    elevation = torch.full_like(azimuth, np.pi / 2)
    codebook = compute_single_array_response_torch(ant_params, elevation, azimuth)
    return codebook.squeeze(0)
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
    print(f"history_labels {history_labels.shape}")

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


def compute_los_summary(binary_los_labels):
    binary_los_labels = np.asarray(binary_los_labels, dtype=np.int64)
    total = int(binary_los_labels.size)
    if total == 0:
        return {
            "los_users": 0,
            "nlos_users": 0,
            "los_ratio": 0.0,
            "nlos_ratio": 0.0,
        }

    los_users = int(binary_los_labels.sum())
    nlos_users = int(total - los_users)
    return {
        "los_users": los_users,
        "nlos_users": nlos_users,
        "los_ratio": float(los_users / total),
        "nlos_ratio": float(nlos_users / total),
    }


def append_stats_row(csv_path, row):
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    fieldnames = [
        "scenario",
        "tx_idx",
        "split",
        "requested_users",
        "kept_users",
        "discarded_users",
        "discard_ratio",
        "history_len",
        "los_users",
        "nlos_users",
        "los_ratio",
        "nlos_ratio",
        "avg_unique_beams",
        "min_unique_beams",
        "max_unique_beams",
        "constant_histories",
        "constant_history_ratio",
        "avg_majority_share",
        "direction_counts",
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
    parser.add_argument("--scenarios", nargs="+", default=None)
    parser.add_argument("--all-scenarios", action="store_true")
    parser.add_argument("--save-dir", type=str, default=DEFAULT_SAVE)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--duration", type=int, default=DURATION)
    parser.add_argument("--stats-csv", type=str, default=DEFAULT_STATS_CSV)
    return parser.parse_args()


def generate_for_scenario(scenario, save_dir, train_ratio, seed, duration, stats_csv):
    print(f"\nGenerating WiFo movement data for {scenario}")
    dataset = dm.load(scenario)
    dataset.compute_channels(ChannelParameters())
    channels = dataset.channels
    # S = make_dft_codebook()
    S = make_azimuth_codebook(B=32)

    if hasattr(dataset, "n_ue") and isinstance(dataset.n_ue, int):
        dataset = [dataset]
        channels = [channels]

    train_channels = []
    train_labels = []
    train_los_labels = []
    train_user_locs = []
    test_channels = []
    test_labels = []
    test_los_labels = []
    test_user_locs = []
    stats_rows = []

    for tx in range(len(dataset)):
        channel_tx = np.asarray(channels[tx])
        n_ue = dataset[tx].n_ue
        indices = np.arange(n_ue)
        np.random.seed(seed + tx)
        np.random.shuffle(indices)
        split_idx = int(train_ratio * len(indices))
        train_indices = indices[:split_idx]
        test_indices = indices[split_idx:]

        use_indices = dataset[tx].los != -1
        los_tx = np.asarray(dataset[tx].los)
        train_indices = np.array([i for i in train_indices if use_indices[i]], dtype=np.int64)
        test_indices = np.array([i for i in test_indices if use_indices[i]], dtype=np.int64)
        labels, _ = compute_beam_label_from_channel(channel_tx.squeeze(1), S)

        train_channel_seq, kept_train_indices, train_dirs, train_discarded = build_movement_sequences(
            channel_tx * 1e6, dataset[tx].rx_pos, train_indices, duration
        )
        # if train_channel_seq.shape[0] > 10:
        #     dummy = train_channel_seq[10, 0, :-1, :, :]
        #     dummy_labels, _ = compute_beam_label_from_channel(dummy, S)
        #     print(f"dummy {dummy.shape}")
        #     print(f"dummy_labels: {dummy_labels}")
        test_channel_seq, kept_test_indices, test_dirs, test_discarded = build_movement_sequences(
            channel_tx * 1e6, dataset[tx].rx_pos, test_indices, duration
        )
        train_diversity = compute_sequence_diversity(train_channel_seq, S)
        test_diversity = compute_sequence_diversity(test_channel_seq, S)
        train_los_summary = compute_los_summary(los_tx[kept_train_indices] > 0)
        test_los_summary = compute_los_summary(los_tx[kept_test_indices] > 0)
        train_dir_counts = {d: train_dirs.count(d) for d in sorted(set(train_dirs))}
        test_dir_counts = {d: test_dirs.count(d) for d in sorted(set(test_dirs))}

        print(
            f"TX {tx}: train={len(train_indices)} val={len(test_indices)} "
            f"kept_train={len(kept_train_indices)} kept_val={len(kept_test_indices)} "
            f"discarded_train={train_discarded} discarded_val={test_discarded} "
            f"train_dirs={train_dir_counts} val_dirs={test_dir_counts}"
        )
        print( 
            f"TX {tx} diversity: "
            f"train_avg_unique={train_diversity['avg_unique_beams']:.2f} "
            f"train_constant_ratio={train_diversity['constant_history_ratio']:.3f} "
            f"val_avg_unique={test_diversity['avg_unique_beams']:.2f} "
            f"val_constant_ratio={test_diversity['constant_history_ratio']:.3f}"
        )

        train_channels.append(train_channel_seq)
        train_labels.append(labels[kept_train_indices].cpu().numpy())
        train_los_labels.append((los_tx[kept_train_indices] > 0).astype(np.int64, copy=False))
        train_user_locs.append(np.asarray(dataset[tx].rx_pos)[kept_train_indices])
        test_channels.append(test_channel_seq)
        test_labels.append(labels[kept_test_indices].cpu().numpy())
        test_los_labels.append((los_tx[kept_test_indices] > 0).astype(np.int64, copy=False))
        test_user_locs.append(np.asarray(dataset[tx].rx_pos)[kept_test_indices])

        stats_rows.append(
            {
                "scenario": scenario,
                "tx_idx": tx,
                "split": "train",
                "requested_users": int(len(train_indices)),
                "kept_users": int(len(kept_train_indices)),
                "discarded_users": int(train_discarded),
                "discard_ratio": float(train_discarded / max(len(train_indices), 1)),
                "history_len": train_diversity["history_len"],
                "los_users": train_los_summary["los_users"],
                "nlos_users": train_los_summary["nlos_users"],
                "los_ratio": train_los_summary["los_ratio"],
                "nlos_ratio": train_los_summary["nlos_ratio"],
                "avg_unique_beams": train_diversity["avg_unique_beams"],
                "min_unique_beams": train_diversity["min_unique_beams"],
                "max_unique_beams": train_diversity["max_unique_beams"],
                "constant_histories": train_diversity["constant_histories"],
                "constant_history_ratio": train_diversity["constant_history_ratio"],
                "avg_majority_share": train_diversity["avg_majority_share"],
                "direction_counts": str(train_dir_counts),
            }
        )
        stats_rows.append(
            {
                "scenario": scenario,
                "tx_idx": tx,
                "split": "val",
                "requested_users": int(len(test_indices)),
                "kept_users": int(len(kept_test_indices)),
                "discarded_users": int(test_discarded),
                "discard_ratio": float(test_discarded / max(len(test_indices), 1)),
                "history_len": test_diversity["history_len"],
                "los_users": test_los_summary["los_users"],
                "nlos_users": test_los_summary["nlos_users"],
                "los_ratio": test_los_summary["los_ratio"],
                "nlos_ratio": test_los_summary["nlos_ratio"],
                "avg_unique_beams": test_diversity["avg_unique_beams"],
                "min_unique_beams": test_diversity["min_unique_beams"],
                "max_unique_beams": test_diversity["max_unique_beams"],
                "constant_histories": test_diversity["constant_histories"],
                "constant_history_ratio": test_diversity["constant_history_ratio"],
                "avg_majority_share": test_diversity["avg_majority_share"],
                "direction_counts": str(test_dir_counts),
            }
        )

    data_dict = {}
    data_dict["channels"] = torch.from_numpy(np.concatenate(train_channels, axis=0))
    data_dict["labels"] = torch.from_numpy(np.concatenate(train_labels, axis=0))
    data_dict["los_labels"] = torch.from_numpy(np.concatenate(train_los_labels, axis=0)).to(torch.int64)
    data_dict["labels_user_loc"] = torch.from_numpy(np.concatenate(train_user_locs, axis=0)).to(torch.float32)
    train_path = os.path.join(save_dir, f"_{scenario}_train_data.pt")
    torch.save(data_dict, train_path)
    print(
        f'Saved train channel data {data_dict["channels"].shape} '
        f"Max Label: {data_dict['labels'].max()} "
        f"LOS labels: {data_dict['los_labels'].shape} to {train_path}"
    )

    data_dict = {}
    data_dict["channels"] = torch.from_numpy(np.concatenate(test_channels, axis=0))
    data_dict["labels"] = torch.from_numpy(np.concatenate(test_labels, axis=0))
    data_dict["los_labels"] = torch.from_numpy(np.concatenate(test_los_labels, axis=0)).to(torch.int64)
    data_dict["labels_user_loc"] = torch.from_numpy(np.concatenate(test_user_locs, axis=0)).to(torch.float32)
    val_path = os.path.join(save_dir, f"_{scenario}_val_data.pt")
    torch.save(data_dict, val_path)
    print(
        f'Saved validation channel data {data_dict["channels"].shape} '
        f"Max Label: {data_dict['labels'].max()} "
        f"LOS labels: {data_dict['los_labels'].shape} to {val_path}"
    )
    for row in stats_rows:
        append_stats_row(stats_csv, row)
    print(f"Appended neighborhood stats to {stats_csv}")


def main():
    args = parse_args()
    save_dir = args.save_dir
    train_ratio = args.train_ratio
    seed = args.seed
    duration = args.duration
    stats_csv = args.stats_csv
    os.makedirs(save_dir, exist_ok=True)

    if args.all_scenarios:
        scenarios = DEFAULT_SCENARIOS
    elif args.scenarios:
        scenarios = args.scenarios
    else:
        scenarios = [args.scenario]

    for scenario in scenarios:
        generate_for_scenario(
            scenario=scenario,
            save_dir=save_dir,
            train_ratio=train_ratio,
            seed=seed,
            duration=duration,
            stats_csv=stats_csv,
        )


if __name__ == "__main__":
    main()
