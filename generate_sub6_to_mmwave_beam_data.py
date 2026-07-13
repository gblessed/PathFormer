import argparse
from collections import defaultdict

import deepmimo as dm
import numpy as np
import torch

from utils.utils import ChannelParameters, compute_single_array_response_torch

DEFAULT_SAVE = "/home/blessedg/Pathformer/WiFo/dataset/blessed_task/"


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
    """Build an easier azimuth codebook for the default 8x1 BS array."""
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
    if "3p5" not in sub6_scenario:
        raise ValueError(
            f"Expected a sub-6 scenario containing '3p5', got: {sub6_scenario}"
        )
    return sub6_scenario.replace("3p5", "28", 1)


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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=str, default="city_0_newyork_3p5")
    parser.add_argument("--mmwave-scenario", type=str, default=None)
    parser.add_argument("--save-dir", type=str, default=DEFAULT_SAVE)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-beams", type=int, default=32)
    parser.add_argument("--match-decimals", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    sub6_scenario = args.scenario
    mmwave_scenario = args.mmwave_scenario or infer_mmwave_scenario(sub6_scenario)
    save_dir = args.save_dir
    train_ratio = args.train_ratio
    seed = args.seed

    print(f"Loading sub-6 scenario: {sub6_scenario}")
    sub6_dataset = dm.load(sub6_scenario)
    sub6_dataset.compute_channels(ChannelParameters())
    sub6_channels = sub6_dataset.channels

    print(f"Loading mmWave scenario: {mmwave_scenario}")
    mmwave_dataset = dm.load(mmwave_scenario)
    mmwave_dataset.compute_channels(ChannelParameters())
    mmwave_channels = mmwave_dataset.channels

    sub6_dataset, sub6_channels = normalize_dataset_layout(sub6_dataset, sub6_channels)
    mmwave_dataset, mmwave_channels = normalize_dataset_layout(mmwave_dataset, mmwave_channels)

    if len(sub6_dataset) != len(mmwave_dataset):
        print(
            f"Warning: TX count mismatch between bands: "
            f"sub6={len(sub6_dataset)} mmwave={len(mmwave_dataset)}. "
            f"Using the first {min(len(sub6_dataset), len(mmwave_dataset))} TX entries."
        )

    n_tx = min(len(sub6_dataset), len(mmwave_dataset))
    codebook = make_dft_codebook(B=args.n_beams)

    train_channels = []
    train_labels = []
    val_channels = []
    val_labels = []

    for tx in range(n_tx):
        sub6_data = sub6_dataset[tx]
        mmwave_data = mmwave_dataset[tx]
        sub6_channel_tx = np.asarray(sub6_channels[tx])
        mmwave_channel_tx = np.asarray(mmwave_channels[tx])

        matched_sub6, matched_mmwave, stats = match_users_by_rx_pos(
            sub6_data, mmwave_data, args.match_decimals
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

        mmwave_labels, _ = compute_beam_label_from_channel(mmwave_channel_tx.squeeze(1), codebook)
        scaled_sub6 = sub6_channel_tx * 1e6

        train_sub6_idx = matched_sub6[train_pair_idx]
        val_sub6_idx = matched_sub6[val_pair_idx]
        train_mmwave_idx = matched_mmwave[train_pair_idx]
        val_mmwave_idx = matched_mmwave[val_pair_idx]

        train_channels.append(scaled_sub6[train_sub6_idx])
        val_channels.append(scaled_sub6[val_sub6_idx])
        train_labels.append(mmwave_labels[train_mmwave_idx].cpu().numpy())
        val_labels.append(mmwave_labels[val_mmwave_idx].cpu().numpy())

        print(
            f"TX {tx}: matched={stats['matched_users']} "
            f"sub6_valid={stats['sub6_valid']} mmwave_valid={stats['mmwave_valid']} "
            f"common_positions={stats['common_positions']} "
            f"dropped_duplicates={stats['dropped_due_to_duplicates']} "
            f"train={len(train_pair_idx)} val={len(val_pair_idx)}"
        )

    if not train_channels or not val_channels:
        raise RuntimeError(
            "No paired samples were created. Check the scenario names or reduce --match-decimals."
        )

    pair_name = f"{sub6_scenario}_to_{mmwave_scenario}"
    pair_name = f"{sub6_scenario}"


    train_dict = {
        "channels": torch.from_numpy(np.concatenate(train_channels)).squeeze(-1),
        "labels": torch.from_numpy(np.concatenate(train_labels)),
        "source_scenario": sub6_scenario,
        "label_scenario": mmwave_scenario,
    }
    train_path = save_dir + f"_{pair_name}_train_data.pt"
    torch.save(train_dict, train_path)
    print(
        f"Saved train data {train_dict['channels'].shape} "
        f"max_label={train_dict['labels'].max()} to {train_path}"
    )

    val_dict = {
        "channels": torch.from_numpy(np.concatenate(val_channels)).squeeze(-1),
        "labels": torch.from_numpy(np.concatenate(val_labels)),
        "source_scenario": sub6_scenario,
        "label_scenario": mmwave_scenario,
    }
    val_path = save_dir + f"_{pair_name}_val_data.pt"
    torch.save(val_dict, val_path)
    print(
        f"Saved val data {val_dict['channels'].shape} "
        f"max_label={val_dict['labels'].max()} to {val_path}"
    )


if __name__ == "__main__":
    main()
