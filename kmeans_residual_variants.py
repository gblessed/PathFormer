import argparse
from pathlib import Path

import deepmimo as dm
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from dataset.dataloaders import PreTrainMySeqDataLoader
from kmeans_first_timestep_residual import (
    DEFAULT_SCENARIOS,
    MAX_GENERATE_STEPS,
    PAD_VALUE,
    _apply_standardizer,
    _build_grouped_samples,
    _build_prompt_features,
    _compute_cluster_stats,
    _compute_first_step_metrics,
    _compute_sequence_metrics,
    _compute_cluster_timestep_means,
    _fit_standardizer,
    _pad_sequence_batch,
)
from models import PathDecoder
from utils.utils import generate_paths_no_env_batch, masked_loss


class GatedResidualPathDecoder(nn.Module):
    def __init__(self, prompt_dim, hidden_dim, n_layers, n_heads, max_T, prefix_len=4, pad_value=PAD_VALUE):
        super().__init__()
        self.backbone = PathDecoder(
            prompt_dim=prompt_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            max_T=max_T,
            prefix_len=prefix_len,
            pad_value=pad_value,
        )
        self.delay_residual_head = nn.Linear(hidden_dim, 1)
        self.power_residual_head = nn.Linear(hidden_dim, 1)
        self.gate_head = nn.Linear(hidden_dim, 2)

    def forward(self, prompts, paths, interactions, baseline_seq):
        h_paths, prefix_flat = self.backbone.forward_hidden(prompts, paths, interactions)
        out = self.backbone.out(h_paths)

        delay_residual = self.delay_residual_head(h_paths).squeeze(-1)
        power_residual = self.power_residual_head(h_paths).squeeze(-1)
        gates = torch.sigmoid(self.gate_head(h_paths))

        delay_pred = baseline_seq[:, :, 0] + gates[:, :, 0] * delay_residual
        power_pred = baseline_seq[:, :, 1] + gates[:, :, 1] * power_residual

        phase_sin_pred = out[:, :, 0]
        phase_cos_pred = out[:, :, 1]
        az_sin_pred = out[:, :, 2]
        az_cos_pred = out[:, :, 3]
        el_sin_pred = out[:, :, 4]
        el_cos_pred = out[:, :, 5]

        phase_pred = torch.atan2(phase_sin_pred, phase_cos_pred)
        az_pred = torch.atan2(az_sin_pred, az_cos_pred)
        el_pred = torch.atan2(el_sin_pred, el_cos_pred)
        interaction_logits = self.backbone.interaction_head(h_paths)
        pathcounts = self.backbone.pathcount_head(prefix_flat)

        return (
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
            pathcounts,
            interaction_logits,
        )


class PrefixGatedResidualPathDecoder(nn.Module):
    def __init__(self, prompt_dim, hidden_dim, n_layers, n_heads, max_T, prefix_k, prefix_len=4, pad_value=PAD_VALUE):
        super().__init__()
        self.prefix_k = prefix_k
        self.backbone = PathDecoder(
            prompt_dim=prompt_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            max_T=max_T,
            prefix_len=prefix_len,
            pad_value=pad_value,
        )
        self.delay_residual_head = nn.Linear(hidden_dim, 1)
        self.power_residual_head = nn.Linear(hidden_dim, 1)
        self.gate_head = nn.Linear(hidden_dim, 2)

    def forward(self, prompts, paths, interactions, baseline_seq):
        h_paths, prefix_flat = self.backbone.forward_hidden(prompts, paths, interactions)
        out = self.backbone.out(h_paths)

        direct_delay = self.backbone.out_delay(h_paths).squeeze(-1)
        direct_power = self.backbone.out_power(h_paths).squeeze(-1)
        delay_residual = self.delay_residual_head(h_paths).squeeze(-1)
        power_residual = self.power_residual_head(h_paths).squeeze(-1)
        gates = torch.sigmoid(self.gate_head(h_paths))

        baseline_delay = baseline_seq[:, :, 0] + gates[:, :, 0] * delay_residual
        baseline_power = baseline_seq[:, :, 1] + gates[:, :, 1] * power_residual

        T = h_paths.shape[1]
        prefix_mask = (torch.arange(T, device=h_paths.device).unsqueeze(0) < self.prefix_k).float()
        delay_pred = prefix_mask * baseline_delay + (1.0 - prefix_mask) * direct_delay
        power_pred = prefix_mask * baseline_power + (1.0 - prefix_mask) * direct_power

        phase_sin_pred = out[:, :, 0]
        phase_cos_pred = out[:, :, 1]
        az_sin_pred = out[:, :, 2]
        az_cos_pred = out[:, :, 3]
        el_sin_pred = out[:, :, 4]
        el_cos_pred = out[:, :, 5]

        phase_pred = torch.atan2(phase_sin_pred, phase_cos_pred)
        az_pred = torch.atan2(az_sin_pred, az_cos_pred)
        el_pred = torch.atan2(el_sin_pred, el_cos_pred)
        interaction_logits = self.backbone.interaction_head(h_paths)
        pathcounts = self.backbone.pathcount_head(prefix_flat)

        return (
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
            pathcounts,
            interaction_logits,
        )


def _apply_prefix_residual_transform(paths, timestep_means, path_padding_mask, prefix_k):
    paths_residual = paths.copy()
    effective_k = min(prefix_k, paths.shape[1] - 1)
    if effective_k > 0:
        paths_residual[:, 1:1 + effective_k, :2] = (
            paths_residual[:, 1:1 + effective_k, :2] - timestep_means[:, 1:1 + effective_k, :]
        )

    invalid_mask = ~path_padding_mask
    paths_residual[invalid_mask] = PAD_VALUE
    paths_residual[:, 0, :] = 0.0
    return paths_residual


def _build_train_examples(train_group, n_clusters, random_state, prefix_k):
    first_targets = train_group["first_target"]
    n_train = first_targets.shape[0]
    k_eff = min(n_clusters, n_train)
    if k_eff <= 0:
        raise ValueError("No train samples available for this TX group.")

    kmeans = __import__("sklearn.cluster", fromlist=["KMeans"]).KMeans(
        n_clusters=k_eff,
        random_state=random_state,
        n_init=10,
    )
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
    residual_paths = _apply_prefix_residual_transform(
        padded_paths,
        sample_timestep_means,
        path_padding_mask,
        prefix_k=prefix_k,
    )

    return {
        "prompts": prompts,
        "paths": residual_paths,
        "raw_paths": padded_paths,
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


def _train_prefix_model(
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
    pbar = tqdm(range(epochs), desc="Prefix residual train", leave=False)
    for epoch in pbar:
        epoch_loss = 0.0
        n_batches = 0
        for prompts_b, paths_b, path_lengths_b, interactions_b, path_padding_mask_b in loader:
            prompts_b = prompts_b.to(device)
            paths_b = paths_b.to(device)
            path_lengths_b = path_lengths_b.to(device)
            interactions_b = interactions_b.to(device)
            path_padding_mask_b = path_padding_mask_b.to(device)

            outputs = model(prompts_b, paths_b[:, :-1, :], interactions_b[:, :-1, :])
            total_loss, *_ = masked_loss(
                *outputs,
                paths_b[:, 1:, :],
                path_lengths_b,
                interactions_b[:, 1:, :],
                pad_value=PAD_VALUE,
                interaction_weight=0.1,
                path_padding_mask=path_padding_mask_b,
            )

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_loss += float(total_loss.item())
            n_batches += 1

        pbar.set_postfix(epoch=epoch + 1, loss=f"{(epoch_loss / max(n_batches, 1)):.6f}")
    return model


def _train_gated_model(
    train_prompts,
    train_raw_paths,
    train_path_lengths,
    train_interactions,
    train_path_padding_mask,
    train_timestep_means,
    hidden_dim,
    n_layers,
    n_heads,
    lr,
    weight_decay,
    batch_size,
    epochs,
    device,
):
    model = GatedResidualPathDecoder(
        prompt_dim=train_prompts.shape[1],
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        n_heads=n_heads,
        max_T=max(train_raw_paths.shape[1] - 1, 1),
        prefix_len=4,
        pad_value=PAD_VALUE,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    ds = TensorDataset(
        torch.from_numpy(train_prompts),
        torch.from_numpy(train_raw_paths),
        torch.from_numpy(train_path_lengths),
        torch.from_numpy(train_interactions),
        torch.from_numpy(train_path_padding_mask),
        torch.from_numpy(train_timestep_means[:, 1:, :]),
    )
    loader = DataLoader(ds, batch_size=min(batch_size, len(ds)), shuffle=True)

    model.train()
    pbar = tqdm(range(epochs), desc="Gated residual train", leave=False)
    for epoch in pbar:
        epoch_loss = 0.0
        n_batches = 0
        for prompts_b, paths_b, path_lengths_b, interactions_b, path_padding_mask_b, baseline_b in loader:
            prompts_b = prompts_b.to(device)
            paths_b = paths_b.to(device)
            path_lengths_b = path_lengths_b.to(device)
            interactions_b = interactions_b.to(device)
            path_padding_mask_b = path_padding_mask_b.to(device)
            baseline_b = baseline_b.to(device)

            outputs = model(
                prompts_b,
                paths_b[:, :-1, :],
                interactions_b[:, :-1, :],
                baseline_b,
            )
            total_loss, *_ = masked_loss(
                *outputs,
                paths_b[:, 1:, :],
                path_lengths_b,
                interactions_b[:, 1:, :],
                pad_value=PAD_VALUE,
                interaction_weight=0.1,
                path_padding_mask=path_padding_mask_b,
            )

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_loss += float(total_loss.item())
            n_batches += 1

        pbar.set_postfix(epoch=epoch + 1, loss=f"{(epoch_loss / max(n_batches, 1)):.6f}")
    return model


def _train_prefix_gated_model(
    train_prompts,
    train_raw_paths,
    train_path_lengths,
    train_interactions,
    train_path_padding_mask,
    train_timestep_means,
    prefix_k,
    hidden_dim,
    n_layers,
    n_heads,
    lr,
    weight_decay,
    batch_size,
    epochs,
    device,
):
    model = PrefixGatedResidualPathDecoder(
        prompt_dim=train_prompts.shape[1],
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        n_heads=n_heads,
        max_T=max(train_raw_paths.shape[1] - 1, 1),
        prefix_k=prefix_k,
        prefix_len=4,
        pad_value=PAD_VALUE,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    ds = TensorDataset(
        torch.from_numpy(train_prompts),
        torch.from_numpy(train_raw_paths),
        torch.from_numpy(train_path_lengths),
        torch.from_numpy(train_interactions),
        torch.from_numpy(train_path_padding_mask),
        torch.from_numpy(train_timestep_means[:, 1:, :]),
    )
    loader = DataLoader(ds, batch_size=min(batch_size, len(ds)), shuffle=True)

    model.train()
    pbar = tqdm(range(epochs), desc="Prefix-gated train", leave=False)
    for epoch in pbar:
        epoch_loss = 0.0
        n_batches = 0
        for prompts_b, paths_b, path_lengths_b, interactions_b, path_padding_mask_b, baseline_b in loader:
            prompts_b = prompts_b.to(device)
            paths_b = paths_b.to(device)
            path_lengths_b = path_lengths_b.to(device)
            interactions_b = interactions_b.to(device)
            path_padding_mask_b = path_padding_mask_b.to(device)
            baseline_b = baseline_b.to(device)

            outputs = model(
                prompts_b,
                paths_b[:, :-1, :],
                interactions_b[:, :-1, :],
                baseline_b,
            )
            total_loss, *_ = masked_loss(
                *outputs,
                paths_b[:, 1:, :],
                path_lengths_b,
                interactions_b[:, 1:, :],
                pad_value=PAD_VALUE,
                interaction_weight=0.1,
                path_padding_mask=path_padding_mask_b,
            )

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_loss += float(total_loss.item())
            n_batches += 1

        pbar.set_postfix(epoch=epoch + 1, loss=f"{(epoch_loss / max(n_batches, 1)):.6f}")
    return model


@torch.no_grad()
def _generate_prefix_batched(model, prompts, timestep_means, prefix_k, batch_size, device):
    model.eval()
    generated_list = []
    pathcount_list = []

    for start in range(0, prompts.shape[0], batch_size):
        end = min(start + batch_size, prompts.shape[0])
        prompts_b = torch.from_numpy(prompts[start:end]).to(device)
        generated_b, pathcount_b, _ = generate_paths_no_env_batch(model, prompts_b, max_steps=MAX_GENERATE_STEPS)
        generated_b = generated_b.numpy().astype(np.float32)
        k_eff = min(prefix_k, MAX_GENERATE_STEPS)
        generated_b[:, :k_eff, :2] += timestep_means[start:end, 1:1 + k_eff, :]
        generated_list.append(generated_b)
        pathcount_list.append(pathcount_b.detach().cpu().numpy().astype(np.float32))

    return np.concatenate(generated_list, axis=0), np.concatenate(pathcount_list, axis=0)


@torch.no_grad()
def _generate_gated_batched(model, prompts, timestep_means, batch_size, device):
    model.eval()
    generated_list = []
    pathcount_list = []

    for start in range(0, prompts.shape[0], batch_size):
        end = min(start + batch_size, prompts.shape[0])
        prompts_b = torch.from_numpy(prompts[start:end]).to(device)
        timestep_means_b = torch.from_numpy(timestep_means[start:end]).to(device)

        cur = torch.zeros(prompts_b.shape[0], 1, 5, device=device)
        inter_str = -1 * torch.ones(prompts_b.shape[0], 1, 4, device=device)
        outputs = []
        pathcounts = None

        for _ in range(MAX_GENERATE_STEPS):
            baseline_seq = timestep_means_b[:, 1:1 + cur.shape[1], :]
            out = model(prompts_b, cur, inter_str, baseline_seq)
            d, p, _, _, ph, _, _, az, _, _, el, pathcounts, inter_logits = out
            d_t = d[:, -1]
            p_t = p[:, -1]
            ph_t = ph[:, -1]
            az_t = az[:, -1]
            el_t = el[:, -1]
            inter_pred_t = (torch.sigmoid(inter_logits[:, -1]) > 0.5).float()

            next_path = torch.stack([d_t, p_t, ph_t, az_t, el_t], dim=-1)
            outputs.append(next_path)
            cur = torch.cat([cur, next_path.unsqueeze(1)], dim=1)
            inter_str = torch.cat([inter_str, inter_pred_t.unsqueeze(1)], dim=1)

        generated_list.append(torch.stack(outputs, dim=1).detach().cpu().numpy().astype(np.float32))
        pathcount_list.append(pathcounts.detach().cpu().numpy().astype(np.float32))

    return np.concatenate(generated_list, axis=0), np.concatenate(pathcount_list, axis=0)


@torch.no_grad()
def _generate_prefix_gated_batched(model, prompts, timestep_means, batch_size, device):
    model.eval()
    generated_list = []
    pathcount_list = []

    for start in range(0, prompts.shape[0], batch_size):
        end = min(start + batch_size, prompts.shape[0])
        prompts_b = torch.from_numpy(prompts[start:end]).to(device)
        timestep_means_b = torch.from_numpy(timestep_means[start:end]).to(device)

        cur = torch.zeros(prompts_b.shape[0], 1, 5, device=device)
        inter_str = -1 * torch.ones(prompts_b.shape[0], 1, 4, device=device)
        outputs = []
        pathcounts = None

        for _ in range(MAX_GENERATE_STEPS):
            baseline_seq = timestep_means_b[:, 1:1 + cur.shape[1], :]
            out = model(prompts_b, cur, inter_str, baseline_seq)
            d, p, _, _, ph, _, _, az, _, _, el, pathcounts, inter_logits = out
            d_t = d[:, -1]
            p_t = p[:, -1]
            ph_t = ph[:, -1]
            az_t = az[:, -1]
            el_t = el[:, -1]
            inter_pred_t = (torch.sigmoid(inter_logits[:, -1]) > 0.5).float()

            next_path = torch.stack([d_t, p_t, ph_t, az_t, el_t], dim=-1)
            outputs.append(next_path)
            cur = torch.cat([cur, next_path.unsqueeze(1)], dim=1)
            inter_str = torch.cat([inter_str, inter_pred_t.unsqueeze(1)], dim=1)

        generated_list.append(torch.stack(outputs, dim=1).detach().cpu().numpy().astype(np.float32))
        pathcount_list.append(pathcounts.detach().cpu().numpy().astype(np.float32))

    return np.concatenate(generated_list, axis=0), np.concatenate(pathcount_list, axis=0)


def evaluate_scenario(
    scenario,
    mode,
    prefix_k=3,
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
    batch_size=64,
    epochs=20,
):
    dm.download(scenario)
    dataset = dm.load(scenario)

    train_data = PreTrainMySeqDataLoader(dataset, train=True, split_by=split_by, sort_by=sort_by, train_ratio=train_ratio)
    val_data = PreTrainMySeqDataLoader(dataset, train=False, split_by=split_by, sort_by=sort_by, train_ratio=train_ratio)

    train_by_tx = _build_grouped_samples(train_data)
    val_by_tx = _build_grouped_samples(val_data)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_prompt_list = []
    train_path_list = []
    train_raw_path_list = []
    train_path_length_list = []
    train_interaction_list = []
    train_path_padding_mask_list = []
    train_timestep_mean_list = []

    val_prompt_list = []
    val_timestep_mean_list = []
    val_first_target_list = []
    val_center_list = []
    val_path_list = []
    val_path_length_list = []

    skipped_no_tx_match = 0
    for tx_key, train_group in train_by_tx.items():
        train_kmeans_data = _build_train_examples(train_group, n_clusters=n_clusters, random_state=random_state, prefix_k=prefix_k)
        train_prompt_list.append(train_kmeans_data["prompts"])
        train_path_list.append(train_kmeans_data["paths"])
        train_raw_path_list.append(train_kmeans_data["raw_paths"])
        train_path_length_list.append(train_kmeans_data["path_length"])
        train_interaction_list.append(train_kmeans_data["interactions"])
        train_path_padding_mask_list.append(train_kmeans_data["path_padding_mask"])
        train_timestep_mean_list.append(train_kmeans_data["cluster_timestep_means"][train_kmeans_data["cluster_ids"]])

        val_group = val_by_tx.get(tx_key)
        if val_group is None:
            continue
        val_examples = _build_val_examples(train_kmeans_data, val_group)
        if val_examples is None:
            continue
        val_prompt_list.append(val_examples["prompts"])
        val_timestep_mean_list.append(val_examples["timestep_means"])
        val_first_target_list.append(val_examples["first_targets"])
        val_center_list.append(val_examples["centers"])
        val_path_list.append(val_examples["paths"])
        val_path_length_list.append(val_examples["path_length"])

    for tx_key, val_group in val_by_tx.items():
        if tx_key not in train_by_tx:
            skipped_no_tx_match += int(val_group["first_target"].shape[0])

    train_prompts = np.concatenate(train_prompt_list, axis=0)
    train_paths = np.concatenate(train_path_list, axis=0)
    train_raw_paths = np.concatenate(train_raw_path_list, axis=0)
    train_path_lengths = np.concatenate(train_path_length_list, axis=0)
    train_interactions = np.concatenate(train_interaction_list, axis=0)
    train_path_padding_mask = np.concatenate(train_path_padding_mask_list, axis=0)
    train_timestep_means = np.concatenate(train_timestep_mean_list, axis=0)

    val_prompts = np.concatenate(val_prompt_list, axis=0)
    val_timestep_means = np.concatenate(val_timestep_mean_list, axis=0)
    val_first_targets = np.concatenate(val_first_target_list, axis=0)
    val_centers = np.concatenate(val_center_list, axis=0)
    val_paths = np.concatenate(val_path_list, axis=0)
    val_path_lengths = np.concatenate(val_path_length_list, axis=0)

    prompt_mean, prompt_std = _fit_standardizer(train_prompts)
    train_prompts_norm = _apply_standardizer(train_prompts, prompt_mean, prompt_std)
    val_prompts_norm = _apply_standardizer(val_prompts, prompt_mean, prompt_std)

    if mode == "prefix":
        model = _train_prefix_model(
            train_prompts_norm,
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
        )
        pred_paths, pred_path_lengths = _generate_prefix_batched(
            model,
            val_prompts_norm,
            val_timestep_means,
            prefix_k=prefix_k,
            batch_size=batch_size,
            device=device,
        )
    elif mode == "gated":
        model = _train_gated_model(
            train_prompts_norm,
            train_raw_paths,
            train_path_lengths,
            train_interactions,
            train_path_padding_mask,
            train_timestep_means,
            hidden_dim,
            n_layers,
            n_heads,
            lr,
            weight_decay,
            batch_size,
            epochs,
            device,
        )
        pred_paths, pred_path_lengths = _generate_gated_batched(
            model,
            val_prompts_norm,
            val_timestep_means,
            batch_size=batch_size,
            device=device,
        )
    elif mode == "prefix_gated":
        model = _train_prefix_gated_model(
            train_prompts_norm,
            train_raw_paths,
            train_path_lengths,
            train_interactions,
            train_path_padding_mask,
            train_timestep_means,
            prefix_k,
            hidden_dim,
            n_layers,
            n_heads,
            lr,
            weight_decay,
            batch_size,
            epochs,
            device,
        )
        pred_paths, pred_path_lengths = _generate_prefix_gated_batched(
            model,
            val_prompts_norm,
            val_timestep_means,
            batch_size=batch_size,
            device=device,
        )
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    delay_rmse, power_rmse, joint_rmse, path_length_rmse, n_eval_steps = _compute_sequence_metrics(
        pred_paths,
        pred_path_lengths,
        val_paths,
        val_path_lengths,
    )
    base_delay_rmse, base_power_rmse, base_joint_rmse = _compute_first_step_metrics(val_centers, val_first_targets)

    return {
        "scenario": scenario,
        "mode": mode,
        "prefix_k": int(prefix_k),
        "n_clusters": int(n_clusters),
        "epochs": int(epochs),
        "hidden_dim": int(hidden_dim),
        "n_layers": int(n_layers),
        "n_heads": int(n_heads),
        "prompt_dim": int(train_prompts_norm.shape[1]),
        "n_train_samples": int(train_prompts.shape[0]),
        "n_val_samples": int(val_paths.shape[0]),
        "n_eval": int(val_paths.shape[0]),
        "n_eval_steps": int(n_eval_steps),
        "n_skipped_no_tx_match": int(skipped_no_tx_match),
        "kmeans_delay_rmse": base_delay_rmse,
        "kmeans_power_rmse": base_power_rmse,
        "kmeans_joint_rmse": base_joint_rmse,
        "delay_rmse": delay_rmse,
        "power_rmse": power_rmse,
        "joint_rmse": joint_rmse,
        "path_length_rmse": path_length_rmse,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Residual variants for first-timestep KMeans conditioning.")
    parser.add_argument("--scenarios", type=str, default=",".join(DEFAULT_SCENARIOS))
    parser.add_argument("--mode", type=str, required=True, choices=["prefix", "gated", "prefix_gated"])
    parser.add_argument("--prefix-k", type=int, default=3)
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
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--output-csv", type=str, default="kmeans_residual_variants_results.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    rows = []
    for scenario in scenarios:
        print(f"\nEvaluating scenario: {scenario}")
        row = evaluate_scenario(
            scenario=scenario,
            mode=args.mode,
            prefix_k=args.prefix_k,
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
        )
        rows.append(row)
        print(
            f"{scenario} | mode={row['mode']} | "
            f"delay_rmse={row['delay_rmse']:.4f}, "
            f"power_rmse={row['power_rmse']:.4f}, "
            f"joint_rmse={row['joint_rmse']:.4f}, "
            f"n_eval_steps={row['n_eval_steps']}"
        )
    output_path = Path(args.output_csv)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    main()
