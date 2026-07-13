import argparse
import csv

import deepmimo as dm
import numpy as np
import torch
import os

from utils.utils import ChannelParameters, compute_single_array_response_torch

DEFAULT_SAVE = "/home/blessedg/Pathformer/WiFo/dataset/blessed_task/"
DEFAULT_STATS_CSV = "/home/blessedg/Pathformer/WiFo/dataset/blessed_task/neighborhood_stats.csv"
DURATION = 7  # Number of historical users to prepend before the current user.
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


def scalar_doppler_phase(doppler_hz, dt_seconds, step_idx):
    return np.exp(1j * 2 * np.pi * doppler_hz * dt_seconds * step_idx).astype(np.complex64)


def apply_scalar_doppler(channel_matrix, doppler_hz, dt_seconds, step_idx):
    return channel_matrix * scalar_doppler_phase(doppler_hz, dt_seconds, step_idx)


def add_complex_awgn(signal, snr_db, rng):
    signal = np.asarray(signal, dtype=np.complex64)
    signal_power = np.mean(np.abs(signal) ** 2)
    snr_linear = 10 ** (snr_db / 10.0)
    noise_power = signal_power / max(snr_linear, 1e-12)
    noise_std = np.sqrt(noise_power / 2.0)
    noise = noise_std * (
        rng.standard_normal(signal.shape) + 1j * rng.standard_normal(signal.shape)
    )
    return (signal + noise).astype(np.complex64)


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


def build_dynamic_movement_sequences(channel_tx, rx_pos, indices, duration, doppler_hz, dt_seconds, noise_snr_db, rng):
    split_channels = np.asarray(channel_tx)[indices]
    split_rx_pos = np.asarray(rx_pos)[indices]

    sequence_channels = []
    trajectory_rx_pos = []
    trajectory_indices = []
    kept_indices = []
    directions = []
    discarded = 0
    for local_idx in range(len(indices)):
        history_indices, direction, is_valid = pick_history_indices(split_rx_pos, local_idx, duration)
        if not is_valid:
            discarded += 1
            continue
        ordered_indices = list(history_indices) + [local_idx]
        static_matrices = [squeeze_channel_to_matrix(split_channels[idx]) for idx in ordered_indices]
        matrices = [
            add_complex_awgn(
                apply_scalar_doppler(static_matrix, doppler_hz, dt_seconds, step_idx),
                noise_snr_db,
                rng,
            )
            for step_idx, static_matrix in enumerate(static_matrices)
        ]
        sequence = np.expand_dims(np.array(matrices), 0)
        sequence_channels.append(sequence.astype(np.complex64, copy=False))
        trajectory_rx_pos.append(split_rx_pos[ordered_indices].astype(np.float32, copy=False))
        trajectory_indices.append(indices[ordered_indices].astype(np.int64, copy=False))
        kept_indices.append(indices[local_idx])
        directions.append(direction)

    if sequence_channels:
        stacked = np.stack(sequence_channels, axis=0)
        stacked_rx_pos = np.stack(trajectory_rx_pos, axis=0)
        stacked_user_indices = np.stack(trajectory_indices, axis=0)
    else:
        sample_shape = squeeze_channel_to_matrix(split_channels[0]).shape if len(split_channels) else (8, 32)
        stacked = np.empty((0, 1, duration + 1, sample_shape[0], sample_shape[1]), dtype=np.complex64)
        stacked_rx_pos = np.empty((0, duration + 1, 3), dtype=np.float32)
        stacked_user_indices = np.empty((0, duration + 1), dtype=np.int64)

    return (
        stacked,
        stacked_rx_pos,
        stacked_user_indices,
        np.asarray(kept_indices, dtype=np.int64),
        directions,
        discarded,
    )


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
    parser.add_argument("--speed", type=float, default=3000.0)
    parser.add_argument("--dt-seconds", type=float, default=0.5*1e-6)
    parser.add_argument("--noise-snr-db", type=float, default=20.0)
    parser.add_argument("--normalize", action="store_true", default=20.0)
    return parser.parse_args()


def generate_for_scenario(scenario, save_dir, train_ratio, seed, duration, stats_csv, doppler_hz, dt_seconds, noise_snr_db, normalize):
    print(f"\nGenerating dynamic noisy movement data for {scenario}")
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
    train_tx_locs = []
    train_rx_trajectory_positions = []
    train_trajectory_user_indices = []
    test_channels = []
    test_labels = []
    test_los_labels = []
    test_user_locs = []
    test_tx_locs = []
    test_rx_trajectory_positions = []
    test_trajectory_user_indices = []
    stats_rows = []

    for tx in range(len(dataset)):
        channel_tx = np.asarray(channels[tx])
        n_ue = dataset[tx].n_ue
        tx_pos_arr = np.asarray(dataset[tx].tx_pos, dtype=np.float32)
        if tx_pos_arr.ndim == 1:
            tx_pos = tx_pos_arr
        else:
            tx_pos = tx_pos_arr[0]
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
        train_rng = np.random.default_rng(seed + tx)
        test_rng = np.random.default_rng(seed + 10000 + tx)

        (
            train_channel_seq,
            train_rx_traj,
            train_traj_indices,
            kept_train_indices,
            train_dirs,
            train_discarded,
        ) = build_dynamic_movement_sequences(
            channel_tx * 1e6,
            dataset[tx].rx_pos,
            train_indices,
            duration,
            doppler_hz,
            dt_seconds,
            noise_snr_db,
            train_rng,
        )
        (
            test_channel_seq,
            test_rx_traj,
            test_traj_indices,
            kept_test_indices,
            test_dirs,
            test_discarded,
        ) = build_dynamic_movement_sequences(
            channel_tx * 1e6,
            dataset[tx].rx_pos,
            test_indices,
            duration,
            doppler_hz,
            dt_seconds,
            noise_snr_db,
            test_rng,
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
        train_tx_locs.append(np.repeat(tx_pos[None, :], len(kept_train_indices), axis=0).astype(np.float32, copy=False))
        train_rx_trajectory_positions.append(train_rx_traj)
        train_trajectory_user_indices.append(train_traj_indices)
        test_channels.append(test_channel_seq)
        test_labels.append(labels[kept_test_indices].cpu().numpy())
        test_los_labels.append((los_tx[kept_test_indices] > 0).astype(np.int64, copy=False))
        test_user_locs.append(np.asarray(dataset[tx].rx_pos)[kept_test_indices])
        test_tx_locs.append(np.repeat(tx_pos[None, :], len(kept_test_indices), axis=0).astype(np.float32, copy=False))
        test_rx_trajectory_positions.append(test_rx_traj)
        test_trajectory_user_indices.append(test_traj_indices)

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
    train_mean, train_std = data_dict["channels"].mean(), data_dict["channels"].std()
    if normalize:
        data_dict["channels"]  = (data_dict["channels"] - train_mean)/train_std
    data_dict["labels"] = torch.from_numpy(np.concatenate(train_labels, axis=0))
    data_dict["los_labels"] = torch.from_numpy(np.concatenate(train_los_labels, axis=0)).to(torch.int64)
    data_dict["labels_user_loc"] = torch.from_numpy(np.concatenate(train_user_locs, axis=0)).to(torch.float32)
    data_dict["tx_positions"] = torch.from_numpy(np.concatenate(train_tx_locs, axis=0)).to(torch.float32)
    data_dict["rx_trajectory_positions"] = torch.from_numpy(np.concatenate(train_rx_trajectory_positions, axis=0)).to(torch.float32)
    data_dict["trajectory_user_indices"] = torch.from_numpy(np.concatenate(train_trajectory_user_indices, axis=0)).to(torch.int64)
    data_dict["doppler_hz"] = float(doppler_hz)
    data_dict["dt_seconds"] = float(dt_seconds)
    data_dict["noise_snr_db"] = float(noise_snr_db)
    train_path = os.path.join(save_dir, f"_{scenario}_train_data.pt")
    torch.save(data_dict, train_path)
    print(
        f'Saved train channel data {data_dict["channels"].shape} '
        f"Max Label: {data_dict['labels'].max()} "
        f"LOS labels: {data_dict['los_labels'].shape} "
        f"RX traj: {data_dict['rx_trajectory_positions'].shape} to {train_path}"
    )

    data_dict = {}

    data_dict["channels"] = torch.from_numpy(np.concatenate(test_channels, axis=0))
    if normalize:
        data_dict["channels"]  = (data_dict["channels"] - train_mean)/train_std
    data_dict["labels"] = torch.from_numpy(np.concatenate(test_labels, axis=0))
    data_dict["los_labels"] = torch.from_numpy(np.concatenate(test_los_labels, axis=0)).to(torch.int64)
    data_dict["labels_user_loc"] = torch.from_numpy(np.concatenate(test_user_locs, axis=0)).to(torch.float32)
    data_dict["tx_positions"] = torch.from_numpy(np.concatenate(test_tx_locs, axis=0)).to(torch.float32)
    data_dict["rx_trajectory_positions"] = torch.from_numpy(np.concatenate(test_rx_trajectory_positions, axis=0)).to(torch.float32)
    data_dict["trajectory_user_indices"] = torch.from_numpy(np.concatenate(test_trajectory_user_indices, axis=0)).to(torch.int64)
    data_dict["doppler_hz"] = float(doppler_hz)
    data_dict["dt_seconds"] = float(dt_seconds)
    data_dict["noise_snr_db"] = float(noise_snr_db)
    val_path = os.path.join(save_dir, f"_{scenario}_val_data.pt")
    torch.save(data_dict, val_path)
    print(
        f'Saved validation channel data {data_dict["channels"].shape} '
        f"Max Label: {data_dict['labels'].max()} "
        f"LOS labels: {data_dict['los_labels'].shape} "
        f"RX traj: {data_dict['rx_trajectory_positions'].shape} to {val_path}"
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
    doppler_hz = (args.speed/(3*10e8)) * (3.5e9)
    dt_seconds = args.dt_seconds
    noise_snr_db = args.noise_snr_db
    os.makedirs(save_dir, exist_ok=True)

    if args.all_scenarios:
        scenarios = DEFAULT_SCENARIOS
    elif args.scenarios:
        scenarios = args.scenarios
    else:
        scenarios = [args.scenario]
    normalize = args.normalize
    for scenario in scenarios:
        generate_for_scenario(
            scenario=scenario,
            save_dir=save_dir,
            train_ratio=train_ratio,
            seed=seed,
            duration=duration,
            stats_csv=stats_csv,
            doppler_hz=doppler_hz,
            dt_seconds=dt_seconds,
            noise_snr_db=noise_snr_db,
            normalize = normalize
        )


if __name__ == "__main__":
    main()
