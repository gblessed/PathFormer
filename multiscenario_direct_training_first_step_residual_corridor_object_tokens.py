import argparse
import os
import warnings

import deepmimo as dm
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from dataset.dataloaders import PreTrainMySeqDataLoader
from multiscenario_direct_training_first_step_residual import (
    all_scenarios,
    get_resume_checkpoint_path,
    load_best_checkpoint,
    resolve_scenarios,
)
from scene_object_token_utils import SceneObjectTokenBank
from utils.utils import add_noise_to_paths, masked_loss

warnings.filterwarnings("ignore", category=UserWarning)

csv_log_file = "muldims_weighted_first_step_residual_corridor_object_tokens_results.csv"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ObjectTokenPathDecoder(nn.Module):
    def __init__(
        self,
        prompt_dim=10,
        object_dim=22,
        hidden_dim=512,
        n_layers=8,
        n_heads=8,
        max_T=26,
        prefix_len=4,
        pad_value=0,
        include_aod=True,
    ):
        super().__init__()
        self.pad_value = pad_value
        self.hidden_dim = hidden_dim
        self.prefix_len = prefix_len
        self.max_T = max_T
        self.include_aod = include_aod

        self.prompt_to_prefix = nn.Linear(prompt_dim, prefix_len * hidden_dim)
        self.object_embed = nn.Sequential(
            nn.Linear(object_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.path_in = nn.Linear(16 if include_aod else 12, hidden_dim)
        self.pos_emb = nn.Embedding(max_T, hidden_dim)
        self.interaction_head = nn.Linear(hidden_dim, 4)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=4 * hidden_dim,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        self.out_delay = nn.Sequential(nn.Linear(hidden_dim, 1))
        self.out_power = nn.Sequential(nn.Linear(hidden_dim, 1))
        self.out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 10 if include_aod else 6),
        )
        self.pathcount_head = nn.Sequential(
            nn.Linear(prefix_len * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward_hidden(self, prompts, paths, interactions, object_tokens, object_padding_mask):
        bsz, steps, _ = paths.shape
        prefix_raw = self.prompt_to_prefix(prompts)
        prefix = prefix_raw.view(bsz, self.prefix_len, self.hidden_dim)

        phase = paths[:, :, 2]
        sinp = torch.sin(phase)
        cosp = torch.cos(phase)
        aoa_az = paths[:, :, 3]
        sin_az = torch.sin(aoa_az)
        cos_az = torch.cos(aoa_az)
        aoa_el = paths[:, :, 4]
        sin_el = torch.sin(aoa_el)
        cos_el = torch.cos(aoa_el)

        x_parts = [paths[:, :, 0], paths[:, :, 1], sinp, cosp, sin_az, cos_az, sin_el, cos_el]
        if self.include_aod:
            aod_az = paths[:, :, 5]
            sin_aod_az = torch.sin(aod_az)
            cos_aod_az = torch.cos(aod_az)
            aod_el = paths[:, :, 6]
            sin_aod_el = torch.sin(aod_el)
            cos_aod_el = torch.cos(aod_el)
            x_parts.extend([sin_aod_az, cos_aod_az, sin_aod_el, cos_aod_el])

        x = torch.stack(x_parts, dim=-1)
        interactions_clean = interactions.clone()
        interactions_clean[interactions_clean == -1] = 0
        x = torch.cat([x, interactions_clean], dim=-1)
        x = self.path_in(x)
        pos = self.pos_emb(torch.arange(steps, device=x.device))
        x = x + pos

        object_emb = self.object_embed(object_tokens)
        memory = torch.cat([prefix, object_emb], dim=1)
        prefix_mask = torch.zeros((bsz, self.prefix_len), dtype=torch.bool, device=x.device)
        memory_mask = torch.cat([prefix_mask, object_padding_mask], dim=1)
        causal_mask = torch.triu(torch.ones(steps, steps, device=x.device), diagonal=1).bool()

        h = self.decoder(
            tgt=x,
            memory=memory,
            tgt_mask=causal_mask,
            memory_key_padding_mask=memory_mask,
        )
        prefix_flat = prefix.reshape(bsz, -1)
        return h, prefix_flat


class FirstStepResidualObjectTokenDecoder(nn.Module):
    def __init__(
        self,
        prompt_dim=10,
        object_dim=22,
        hidden_dim=512,
        n_layers=8,
        n_heads=8,
        max_T=26,
        prefix_len=4,
        pad_value=0,
        include_aod=True,
    ):
        super().__init__()
        self.backbone = ObjectTokenPathDecoder(
            prompt_dim=prompt_dim,
            object_dim=object_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            max_T=max_T,
            prefix_len=prefix_len,
            pad_value=pad_value,
            include_aod=include_aod,
        )
        self.first_delay_residual_head = nn.Linear(hidden_dim, 1)
        self.first_power_residual_head = nn.Linear(hidden_dim, 1)

    def forward(self, prompts, paths, interactions, first_step_baseline, object_tokens, object_padding_mask):
        h_paths, prefix_flat = self.backbone.forward_hidden(
            prompts, paths, interactions, object_tokens, object_padding_mask
        )
        out = self.backbone.out(h_paths)

        delay_pred = self.backbone.out_delay(h_paths).squeeze(-1)
        power_pred = self.backbone.out_power(h_paths).squeeze(-1)
        delay_pred = delay_pred.clone()
        power_pred = power_pred.clone()
        delay_pred[:, 0] = first_step_baseline[:, 0] + self.first_delay_residual_head(h_paths[:, 0, :]).squeeze(-1)
        power_pred[:, 0] = first_step_baseline[:, 1] + self.first_power_residual_head(h_paths[:, 0, :]).squeeze(-1)

        phase_sin_pred = out[:, :, 0]
        phase_cos_pred = out[:, :, 1]
        az_sin_pred = out[:, :, 2]
        az_cos_pred = out[:, :, 3]
        el_sin_pred = out[:, :, 4]
        el_cos_pred = out[:, :, 5]
        aod_az_sin_pred = out[:, :, 6]
        aod_az_cos_pred = out[:, :, 7]
        aod_el_sin_pred = out[:, :, 8]
        aod_el_cos_pred = out[:, :, 9]

        phase_pred = torch.atan2(phase_sin_pred, phase_cos_pred)
        az_pred = torch.atan2(az_sin_pred, az_cos_pred)
        el_pred = torch.atan2(el_sin_pred, el_cos_pred)
        aod_az_pred = torch.atan2(aod_az_sin_pred, aod_az_cos_pred)
        aod_el_pred = torch.atan2(aod_el_sin_pred, aod_el_cos_pred)
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
            aod_az_sin_pred,
            aod_az_cos_pred,
            aod_az_pred,
            aod_el_sin_pred,
            aod_el_cos_pred,
            aod_el_pred,
            pathcounts,
            interaction_logits,
        )


class ObjectTokenResidualDataset(Dataset):
    def __init__(self, base_dataset, augmented_prompts, first_step_baselines, object_tokens, object_padding_masks):
        self.base_dataset = base_dataset
        self.augmented_prompts = augmented_prompts
        self.first_step_baselines = first_step_baselines
        self.object_tokens = object_tokens
        self.object_padding_masks = object_padding_masks
        self.pad_value = getattr(base_dataset, "pad_value", 0)

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        return (
            self.base_dataset[idx],
            self.augmented_prompts[idx],
            self.first_step_baselines[idx],
            self.object_tokens[idx],
            self.object_padding_masks[idx],
        )

    def collate_fn(self, batch):
        base_items = [item[0] for item in batch]
        aug_prompts = torch.stack([item[1] for item in batch], dim=0)
        first_step_baselines = torch.stack([item[2] for item in batch], dim=0)
        object_tokens = torch.stack([item[3] for item in batch], dim=0)
        object_padding_masks = torch.stack([item[4] for item in batch], dim=0)
        _, paths, path_lengths, interactions, env, env_prop, path_padding_mask = self.base_dataset.collate_fn(base_items)
        return (
            aug_prompts,
            paths,
            path_lengths,
            interactions,
            env,
            env_prop,
            path_padding_mask,
            first_step_baselines,
            object_tokens,
            object_padding_masks,
        )


def compute_stop_metrics(path_count, targets):
    return np.sqrt(mean_squared_error(path_count.cpu().numpy(), targets.squeeze().cpu().numpy()))


def _extract_first_step_metadata_with_objects(
    seq_dataset,
    object_bank,
    max_objects,
    nearest_rx_k,
    nearest_tx_k,
    corridor_object_k,
):
    samples = []
    for idx in range(len(seq_dataset)):
        prompt, paths, *_ = seq_dataset[idx]
        tx_pos = np.asarray(seq_dataset.dataset_filtered["tx_pos"][idx], dtype=np.float32)
        rx_pos = np.asarray(seq_dataset.dataset_filtered["rx_pos"][idx], dtype=np.float32)
        tx_key = tuple(tx_pos.tolist())
        first_target = paths[1, :2].numpy().astype(np.float32) if paths.shape[0] > 1 else np.zeros(2, dtype=np.float32)
        object_tokens, object_padding_mask = object_bank.build_object_tokens(
            tx_pos=tx_pos,
            rx_pos=rx_pos,
            max_objects=max_objects,
            nearest_rx_k=nearest_rx_k,
            nearest_tx_k=nearest_tx_k,
            corridor_k=corridor_object_k,
        )
        samples.append(
            {
                "tx_key": tx_key,
                "rx_pos": prompt[3:].numpy().astype(np.float32),
                "prompt": prompt.numpy().astype(np.float32),
                "first_target": first_target,
                "object_tokens": object_tokens.astype(np.float32),
                "object_padding_mask": object_padding_mask.astype(bool),
            }
        )
    return samples


def _compute_cluster_stats(targets, labels, centers):
    stds = np.zeros_like(centers, dtype=np.float32)
    for k in range(centers.shape[0]):
        members = targets[labels == k]
        if len(members) > 0:
            stds[k] = members.std(axis=0).astype(np.float32)
    return stds


def _fit_object_standardizer(object_token_list, object_mask_list):
    valid_rows = []
    for tokens, mask in zip(object_token_list, object_mask_list):
        valid = tokens[~mask]
        if len(valid) > 0:
            valid_rows.append(valid)
    if not valid_rows:
        feat_dim = object_token_list[0].shape[-1]
        return np.zeros((feat_dim,), dtype=np.float32), np.ones((feat_dim,), dtype=np.float32)
    arr = np.concatenate(valid_rows, axis=0).astype(np.float32)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def _apply_object_standardizer(object_token_list, object_mask_list, mean, std):
    out = []
    for tokens, mask in zip(object_token_list, object_mask_list):
        arr = ((tokens.astype(np.float32) - mean) / std).astype(np.float32)
        arr[mask] = 0.0
        out.append(torch.from_numpy(arr))
    return out


def build_first_step_assignments_with_object_tokens(
    train_data,
    val_data,
    object_bank,
    n_clusters,
    max_objects=24,
    nearest_rx_k=8,
    nearest_tx_k=4,
    corridor_object_k=12,
):
    train_meta = _extract_first_step_metadata_with_objects(
        train_data, object_bank, max_objects, nearest_rx_k, nearest_tx_k, corridor_object_k
    )
    val_meta = _extract_first_step_metadata_with_objects(
        val_data, object_bank, max_objects, nearest_rx_k, nearest_tx_k, corridor_object_k
    )

    train_aug_prompts = [None] * len(train_meta)
    train_baselines = [None] * len(train_meta)
    val_aug_prompts = [None] * len(val_meta)
    val_baselines = [None] * len(val_meta)

    train_groups = {}
    for idx, sample in enumerate(train_meta):
        train_groups.setdefault(sample["tx_key"], {"indices": [], "rx_pos": [], "targets": []})
        train_groups[sample["tx_key"]]["indices"].append(idx)
        train_groups[sample["tx_key"]]["rx_pos"].append(sample["rx_pos"])
        train_groups[sample["tx_key"]]["targets"].append(sample["first_target"])

    val_groups = {}
    for idx, sample in enumerate(val_meta):
        val_groups.setdefault(sample["tx_key"], {"indices": [], "rx_pos": []})
        val_groups[sample["tx_key"]]["indices"].append(idx)
        val_groups[sample["tx_key"]]["rx_pos"].append(sample["rx_pos"])

    for tx_key, group in train_groups.items():
        rx_pos = np.stack(group["rx_pos"], axis=0).astype(np.float32)
        targets = np.stack(group["targets"], axis=0).astype(np.float32)
        k_eff = min(n_clusters, len(targets))
        kmeans = KMeans(n_clusters=k_eff, random_state=42, n_init=10)
        labels = kmeans.fit_predict(targets)
        centers = kmeans.cluster_centers_.astype(np.float32)
        stds = _compute_cluster_stats(targets, labels, centers)

        for local_idx, dataset_idx in enumerate(group["indices"]):
            sample = train_meta[dataset_idx]
            baseline = centers[labels[local_idx]]
            std = stds[labels[local_idx]]
            prompt_aug = np.concatenate([sample["prompt"], baseline, std], axis=0).astype(np.float32)
            train_aug_prompts[dataset_idx] = torch.from_numpy(prompt_aug)
            train_baselines[dataset_idx] = torch.from_numpy(baseline.astype(np.float32))

        if tx_key not in val_groups:
            continue

        val_rx = np.stack(val_groups[tx_key]["rx_pos"], axis=0).astype(np.float32)
        dists = np.sum((val_rx[:, None, :] - rx_pos[None, :, :]) ** 2, axis=2)
        nearest_train_idx = np.argmin(dists, axis=1)
        assigned_labels = labels[nearest_train_idx]

        for local_idx, dataset_idx in enumerate(val_groups[tx_key]["indices"]):
            sample = val_meta[dataset_idx]
            baseline = centers[assigned_labels[local_idx]]
            std = stds[assigned_labels[local_idx]]
            prompt_aug = np.concatenate([sample["prompt"], baseline, std], axis=0).astype(np.float32)
            val_aug_prompts[dataset_idx] = torch.from_numpy(prompt_aug)
            val_baselines[dataset_idx] = torch.from_numpy(baseline.astype(np.float32))

    for idx in range(len(train_aug_prompts)):
        if train_aug_prompts[idx] is None:
            sample = train_meta[idx]
            zeros = np.zeros(4, dtype=np.float32)
            train_aug_prompts[idx] = torch.from_numpy(np.concatenate([sample["prompt"], zeros], axis=0).astype(np.float32))
            train_baselines[idx] = torch.zeros(2, dtype=torch.float32)
    for idx in range(len(val_aug_prompts)):
        if val_aug_prompts[idx] is None:
            sample = val_meta[idx]
            zeros = np.zeros(4, dtype=np.float32)
            val_aug_prompts[idx] = torch.from_numpy(np.concatenate([sample["prompt"], zeros], axis=0).astype(np.float32))
            val_baselines[idx] = torch.zeros(2, dtype=torch.float32)

    train_object_tokens = [sample["object_tokens"] for sample in train_meta]
    train_object_masks = [sample["object_padding_mask"] for sample in train_meta]
    val_object_tokens = [sample["object_tokens"] for sample in val_meta]
    val_object_masks = [sample["object_padding_mask"] for sample in val_meta]

    object_mean, object_std = _fit_object_standardizer(train_object_tokens, train_object_masks)
    train_object_tokens = _apply_object_standardizer(train_object_tokens, train_object_masks, object_mean, object_std)
    val_object_tokens = _apply_object_standardizer(val_object_tokens, val_object_masks, object_mean, object_std)
    train_object_masks = [torch.from_numpy(mask.astype(bool)) for mask in train_object_masks]
    val_object_masks = [torch.from_numpy(mask.astype(bool)) for mask in val_object_masks]

    return (
        train_aug_prompts,
        train_baselines,
        val_aug_prompts,
        val_baselines,
        train_object_tokens,
        train_object_masks,
        val_object_tokens,
        val_object_masks,
    )


@torch.no_grad()
def generate_paths_first_step_residual_object_tokens_batch(
    model, prompts, first_step_baselines, object_tokens, object_padding_masks, max_steps=25
):
    model.eval()
    prompts = prompts.to(device)
    first_step_baselines = first_step_baselines.to(device)
    object_tokens = object_tokens.to(device)
    object_padding_masks = object_padding_masks.to(device)

    cur = torch.zeros(prompts.shape[0], 1, 7, device=device)
    inter_str = -1 * torch.ones(prompts.shape[0], 1, 4, device=device)
    outputs = []
    outputs_inter = []
    pathcounts = None

    for _ in range(max_steps):
        (
            d,
            p,
            _,
            _,
            ph,
            _,
            _,
            az,
            _,
            _,
            el,
            _,
            _,
            aod_az,
            _,
            _,
            aod_el,
            pathcounts,
            inter_logits,
        ) = model(prompts, cur, inter_str, first_step_baselines, object_tokens, object_padding_masks)
        d_t = d[:, -1]
        p_t = p[:, -1]
        ph_t = ph[:, -1]
        az_t = az[:, -1]
        el_t = el[:, -1]
        aod_az_t = aod_az[:, -1]
        aod_el_t = aod_el[:, -1]
        inter_pred_t = (torch.sigmoid(inter_logits[:, -1]) > 0.5).float()

        next_path = torch.stack([d_t, p_t, ph_t, az_t, el_t, aod_az_t, aod_el_t], dim=-1)
        outputs.append(next_path)
        outputs_inter.append(inter_pred_t)
        cur = torch.cat([cur, next_path.unsqueeze(1)], dim=1)
        inter_str = torch.cat([inter_str, inter_pred_t.unsqueeze(1)], dim=1)

    return (
        torch.stack(outputs, dim=1).detach().cpu(),
        pathcounts.detach().cpu(),
        torch.stack(outputs_inter, dim=1).detach().cpu(),
    )


def evaluate_model(model, val_loader, max_generate=25):
    model.eval()
    delay_errors, power_errors, phase_errors = [], [], []
    az_errors, el_errors, path_length_rmses = [], [], []
    aod_az_errors, aod_el_errors = [], []
    delay_maes, power_maes, phase_maes = [], [], []
    az_maes, el_maes, path_length_maes = [], [], []
    aod_az_maes, aod_el_maes = [], []
    interaction_targets_all, interaction_preds_all = [], []

    def mean_std(values):
        if len(values) == 0:
            return 0.0, 0.0
        arr = np.asarray(values, dtype=np.float64)
        return float(np.mean(arr)), float(np.std(arr))

    with torch.no_grad():
        for (
            prompts,
            paths,
            path_lengths,
            interactions,
            env,
            env_prop,
            path_padding_mask,
            first_step_baselines,
            object_tokens,
            object_padding_masks,
        ) in tqdm(val_loader, desc="Evaluating", leave=True):
            prompts = prompts.cuda()
            paths = paths.cuda()
            path_lengths = path_lengths.cuda()
            interactions = interactions.cuda()
            first_step_baselines = first_step_baselines.cuda()
            object_tokens = object_tokens.cuda()
            object_padding_masks = object_padding_masks.cuda()

            generated, path_lengths_pred, inter_str_pred = generate_paths_first_step_residual_object_tokens_batch(
                model,
                prompts,
                first_step_baselines,
                object_tokens,
                object_padding_masks,
                max_steps=max_generate,
            )
            generated = generated.cuda()
            if path_lengths_pred.dim() > 1:
                path_lengths_pred = path_lengths_pred.squeeze(-1)

            for b in range(prompts.size(0)):
                n_valid = int(round(path_lengths[b].item() * 25))
                gt = paths[b][1 : 1 + n_valid, :7]
                gt_inter = interactions[b][1 : 1 + n_valid, :]
                T = min(len(gt), generated.size(1))
                pred = generated[b, :T]
                gt = gt[:T]
                pred_inter = inter_str_pred[b, :T]
                gt_inter = gt_inter[:T].detach().cpu()

                valid_interaction_mask = gt_inter[:, 0] != -1
                if valid_interaction_mask.any():
                    interaction_targets_all.append(gt_inter[valid_interaction_mask].numpy().astype(np.int32))
                    interaction_preds_all.append(pred_inter[valid_interaction_mask].numpy().astype(np.int32))

                delay_rmse = torch.mean((pred[:, 0] - gt[:, 0]) ** 2).sqrt().item()
                delay_mae = torch.mean(torch.abs(pred[:, 0] - gt[:, 0])).item()
                power_rmse = torch.mean(((pred[:, 1] / 0.01) - (gt[:, 1] / 0.01)) ** 2).sqrt().item()
                power_mae = torch.mean(torch.abs((pred[:, 1] / 0.01) - (gt[:, 1] / 0.01))).item()

                phase_dist = ((pred[:, 2] / (np.pi / 180)) - (gt[:, 2] / (np.pi / 180)) + 180) % 360 - 180
                phase_rmse = torch.mean(phase_dist ** 2).sqrt().item()
                phase_mae = torch.mean(torch.abs(phase_dist)).item()

                az_dist = ((pred[:, 3] / (np.pi / 180)) - (gt[:, 3] / (np.pi / 180)) + 180) % 360 - 180
                el_dist = ((pred[:, 4] / (np.pi / 180)) - (gt[:, 4] / (np.pi / 180)) + 180) % 360 - 180
                aod_az_dist = ((pred[:, 5] / (np.pi / 180)) - (gt[:, 5] / (np.pi / 180)) + 180) % 360 - 180
                aod_el_dist = ((pred[:, 6] / (np.pi / 180)) - (gt[:, 6] / (np.pi / 180)) + 180) % 360 - 180
                az_rmse = torch.mean(az_dist ** 2).sqrt().item()
                el_rmse = torch.mean(el_dist ** 2).sqrt().item()
                aod_az_rmse = torch.mean(aod_az_dist ** 2).sqrt().item()
                aod_el_rmse = torch.mean(aod_el_dist ** 2).sqrt().item()
                az_mae = torch.mean(torch.abs(az_dist)).item()
                el_mae = torch.mean(torch.abs(el_dist)).item()
                aod_az_mae = torch.mean(torch.abs(aod_az_dist)).item()
                aod_el_mae = torch.mean(torch.abs(aod_el_dist)).item()

                path_len_pred_b = path_lengths_pred[b]
                length_rmse = torch.mean((path_len_pred_b - path_lengths[b]) ** 2).sqrt().item()
                length_mae = torch.mean(torch.abs(path_len_pred_b - path_lengths[b])).item()

                delay_errors.append(delay_rmse)
                power_errors.append(power_rmse)
                phase_errors.append(phase_rmse)
                az_errors.append(az_rmse)
                el_errors.append(el_rmse)
                path_length_rmses.append(length_rmse)
                aod_az_errors.append(aod_az_rmse)
                aod_el_errors.append(aod_el_rmse)
                delay_maes.append(delay_mae)
                power_maes.append(power_mae)
                phase_maes.append(phase_mae)
                az_maes.append(az_mae)
                el_maes.append(el_mae)
                path_length_maes.append(length_mae)
                aod_az_maes.append(aod_az_mae)
                aod_el_maes.append(aod_el_mae)

    if interaction_targets_all:
        targets = np.concatenate(interaction_targets_all, axis=0)
        preds = np.concatenate(interaction_preds_all, axis=0)
        avg_interaction_accuracy = accuracy_score(targets.reshape(-1), preds.reshape(-1))
        avg_interaction_f1 = f1_score(targets.reshape(-1), preds.reshape(-1), zero_division=0)
        interaction_accuracy_per_sample = [
            accuracy_score(target.reshape(-1), pred.reshape(-1))
            for target, pred in zip(interaction_targets_all, interaction_preds_all)
        ]
        interaction_f1_per_sample = [
            f1_score(target.reshape(-1), pred.reshape(-1), zero_division=0)
            for target, pred in zip(interaction_targets_all, interaction_preds_all)
        ]
        _, std_interaction_accuracy = mean_std(interaction_accuracy_per_sample)
        _, std_interaction_f1 = mean_std(interaction_f1_per_sample)
    else:
        avg_interaction_accuracy = 0.0
        avg_interaction_f1 = 0.0
        std_interaction_accuracy = 0.0
        std_interaction_f1 = 0.0

    avg_delay, std_delay = mean_std(delay_errors)
    avg_power, std_power = mean_std(power_errors)
    avg_phase, std_phase = mean_std(phase_errors)
    avg_az, std_az = mean_std(az_errors)
    avg_el, std_el = mean_std(el_errors)
    avg_aod_az, std_aod_az = mean_std(aod_az_errors)
    avg_aod_el, std_aod_el = mean_std(aod_el_errors)
    avg_path_length_rmse, std_path_length_rmse = mean_std(path_length_rmses)
    avg_delay_mae, std_delay_mae = mean_std(delay_maes)
    avg_power_mae, std_power_mae = mean_std(power_maes)
    avg_phase_mae, std_phase_mae = mean_std(phase_maes)
    avg_az_mae, std_az_mae = mean_std(az_maes)
    avg_el_mae, std_el_mae = mean_std(el_maes)
    avg_aod_az_mae, std_aod_az_mae = mean_std(aod_az_maes)
    avg_aod_el_mae, std_aod_el_mae = mean_std(aod_el_maes)
    avg_path_length_mae, std_path_length_mae = mean_std(path_length_maes)

    return (
        avg_delay,
        std_delay,
        avg_power,
        std_power,
        avg_phase,
        std_phase,
        avg_az,
        std_az,
        avg_el,
        std_el,
        avg_aod_az,
        std_aod_az,
        avg_aod_el,
        std_aod_el,
        avg_path_length_rmse,
        std_path_length_rmse,
        avg_interaction_accuracy,
        std_interaction_accuracy,
        avg_interaction_f1,
        std_interaction_f1,
        avg_delay_mae,
        std_delay_mae,
        avg_power_mae,
        std_power_mae,
        avg_phase_mae,
        std_phase_mae,
        avg_az_mae,
        std_az_mae,
        avg_el_mae,
        std_el_mae,
        avg_aod_az_mae,
        std_aod_az_mae,
        avg_aod_el_mae,
        std_aod_el_mae,
        avg_path_length_mae,
        std_path_length_mae,
    )


def load_training_checkpoint(model, optimizer, scheduler, checkpoint_path, resume_checkpoint_path=None):
    candidate_paths = []
    if resume_checkpoint_path is not None:
        candidate_paths.append(resume_checkpoint_path)
    candidate_paths.append(checkpoint_path)
    for path in candidate_paths:
        if path and os.path.exists(path):
            checkpoint = torch.load(path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            if optimizer is not None and "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if scheduler is not None and "scheduler_state_dict" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            start_epoch = int(checkpoint.get("epoch", -1)) + 1
            best_val_loss = checkpoint.get("best_val_loss", float("inf"))
            if hasattr(best_val_loss, "item"):
                best_val_loss = float(best_val_loss.item())
            else:
                best_val_loss = float(best_val_loss)
            return start_epoch, best_val_loss, path
    return 0, float("inf"), None


def train_with_interactions(
    model, train_loader, val_loader, config, train_data, optimizer, scheduler, checkpoint_path, resume_checkpoint_path=None
):
    start_epoch, best_val_loss, resumed_from = load_training_checkpoint(
        model, optimizer, scheduler, checkpoint_path, resume_checkpoint_path=resume_checkpoint_path
    )
    if resumed_from is not None:
        print(f"Resuming training from {resumed_from} at epoch {start_epoch}")
    if start_epoch >= config["epochs"]:
        print(f"Checkpoint already reached epoch {start_epoch}; skipping training.")
        return

    for epoch in range(start_epoch, config["epochs"]):
        model.train()
        train_losses = []
        for (
            prompts,
            paths,
            path_lengths,
            interactions,
            env,
            env_prop,
            path_padding_mask,
            first_step_baselines,
            object_tokens,
            object_padding_masks,
        ) in tqdm(train_loader, desc=f"Epoch {epoch} [Train]", leave=False):
            prompts = prompts.cuda()
            paths = paths.cuda()
            path_lengths = path_lengths.cuda()
            interactions = interactions.cuda()
            path_padding_mask = path_padding_mask.cuda()
            first_step_baselines = first_step_baselines.cuda()
            object_tokens = object_tokens.cuda()
            object_padding_masks = object_padding_masks.cuda()

            paths_in = paths[:, :-1, :]
            p_noise = config.get("TARGET_NOISE_PROB", 0.0)
            if p_noise > 0:
                noise_valid_mask = path_padding_mask[:, :-1].clone()
                keep_prefix = min(2, noise_valid_mask.size(1))
                noise_valid_mask[:, :keep_prefix] = False
                paths_in = add_noise_to_paths(
                    paths_in,
                    noise_valid_mask,
                    p_noise=p_noise,
                    noise_params=config.get("TARGET_NOISE_PARAMS"),
                )
            interactions_in = interactions[:, :-1, :]
            paths_out = paths[:, 1:, :]
            interactions_out = interactions[:, 1:, :]

            outputs = model(
                prompts,
                paths_in,
                interactions_in,
                first_step_baselines,
                object_tokens,
                object_padding_masks,
            )
            total_loss, *_ = masked_loss(
                *outputs,
                paths_out,
                path_lengths,
                interactions_out,
                pad_value=train_data.pad_value,
                interaction_weight=config.get("interaction_weight", 0.1),
                path_padding_mask=path_padding_mask,
                time_step_weighted=config.get("time_step_weighted", False),
            )
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            train_losses.append(total_loss.item())

        scheduler.step()
        best_val_loss = 0
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_loss": torch.tensor(best_val_loss),
            },
            checkpoint_path,
        )
        if resume_checkpoint_path is not None:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_val_loss": torch.tensor(best_val_loss),
                },
                resume_checkpoint_path,
            )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train/evaluate corridor-aware first-step residual PathDecoder with per-object building tokens."
    )
    parser.add_argument("scenarios", nargs="*", help="Optional scenario names to run.")
    parser.add_argument("--scenario", dest="scenario_flag", action="append")
    parser.add_argument("--scenario-file", type=str)
    parser.add_argument("--shard-index", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=None)
    parser.add_argument("--csv-log-file", type=str, default=csv_log_file)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints_first_step_residual_corridor_object_tokens")
    parser.add_argument("--noise-prob", type=float, default=0.0)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--n-clusters", type=int, default=25)
    parser.add_argument("--use-material-features", action="store_true")
    parser.add_argument("--no-material-features", dest="use_material_features", action="store_false")
    parser.set_defaults(use_material_features=True)
    parser.add_argument("--max-objects", type=int, default=24)
    parser.add_argument("--nearest-rx-k", type=int, default=8)
    parser.add_argument("--nearest-tx-k", type=int, default=4)
    parser.add_argument("--corridor-object-k", type=int, default=12)
    return parser.parse_args()


def run_scenario(scenario, args):
    dataset = dm.load(scenario)
    config = {
        "BATCH_SIZE": 128,
        "PAD_VALUE": 0,
        "LR": 2e-5,
        "epochs": 300,
        "interaction_weight": 0.01,
        "experiment": f"first_step_residual_corridor_object_tokens_{scenario}",
        "hidden_dim": 512,
        "n_layers": 8,
        "n_heads": 8,
        "time_step_weighted": False,
        "TARGET_NOISE_PROB": args.noise_prob,
        "TARGET_NOISE_PARAMS": None,
    }

    base_train = PreTrainMySeqDataLoader(
        dataset, train=True, split_by="user", sort_by="power", normalizers=None, apply_normalizers=[], pad_value=0, include_aod=True
    )
    base_val = PreTrainMySeqDataLoader(
        dataset, train=False, split_by="user", sort_by="power", normalizers=None, apply_normalizers=[], pad_value=0, include_aod=True
    )

    object_bank = SceneObjectTokenBank.from_dataset(dataset, use_material_features=args.use_material_features)
    (
        train_aug_prompts,
        train_baselines,
        val_aug_prompts,
        val_baselines,
        train_object_tokens,
        train_object_masks,
        val_object_tokens,
        val_object_masks,
    ) = build_first_step_assignments_with_object_tokens(
        base_train,
        base_val,
        object_bank,
        n_clusters=args.n_clusters,
        max_objects=args.max_objects,
        nearest_rx_k=args.nearest_rx_k,
        nearest_tx_k=args.nearest_tx_k,
        corridor_object_k=args.corridor_object_k,
    )

    train_data = ObjectTokenResidualDataset(
        base_train, train_aug_prompts, train_baselines, train_object_tokens, train_object_masks
    )
    val_data = ObjectTokenResidualDataset(
        base_val, val_aug_prompts, val_baselines, val_object_tokens, val_object_masks
    )
    train_loader = DataLoader(train_data, batch_size=config["BATCH_SIZE"], shuffle=True, collate_fn=train_data.collate_fn)
    val_loader = DataLoader(val_data, batch_size=config["BATCH_SIZE"], shuffle=False, collate_fn=val_data.collate_fn)

    prompt_dim = int(train_aug_prompts[0].numel())
    object_dim = int(train_object_tokens[0].shape[-1])
    model = FirstStepResidualObjectTokenDecoder(
        prompt_dim=prompt_dim,
        object_dim=object_dim,
        hidden_dim=config["hidden_dim"],
        n_layers=config["n_layers"],
        n_heads=config["n_heads"],
        include_aod=True,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["LR"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=25, T_mult=1, eta_min=1e-8)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.checkpoint_dir, f"{config['experiment']}_best_model_checkpoint.pth")
    resume_checkpoint_path = get_resume_checkpoint_path(checkpoint_path)

    if not args.skip_train:
        train_with_interactions(
            model,
            train_loader,
            val_loader,
            config,
            train_data,
            optimizer,
            scheduler,
            checkpoint_path,
            resume_checkpoint_path=resume_checkpoint_path,
        )

    _, best_loss = load_best_checkpoint(model, checkpoint_path)
    results = evaluate_model(model, val_loader)
    (
        avg_delay,
        std_delay,
        avg_power,
        std_power,
        avg_phase,
        std_phase,
        avg_az,
        std_az,
        avg_el,
        std_el,
        avg_aod_az,
        std_aod_az,
        avg_aod_el,
        std_aod_el,
        avg_path_length_rmse,
        std_path_length_rmse,
        avg_interaction_accuracy,
        std_interaction_accuracy,
        avg_interaction_f1,
        std_interaction_f1,
        avg_delay_mae,
        std_delay_mae,
        avg_power_mae,
        std_power_mae,
        avg_phase_mae,
        std_phase_mae,
        avg_az_mae,
        std_az_mae,
        avg_el_mae,
        std_el_mae,
        avg_aod_az_mae,
        std_aod_az_mae,
        avg_aod_el_mae,
        std_aod_el_mae,
        avg_path_length_mae,
        std_path_length_mae,
    ) = results

    row = {
        "scenario": scenario,
        "n_clusters": args.n_clusters,
        "max_objects": args.max_objects,
        "nearest_rx_k": args.nearest_rx_k,
        "nearest_tx_k": args.nearest_tx_k,
        "corridor_object_k": args.corridor_object_k,
        "object_dim": object_dim,
        "prompt_dim": prompt_dim,
        "use_material_features": args.use_material_features,
        "noise_prob": args.noise_prob,
        "delay_rmse": avg_delay,
        "delay_rmse_std": std_delay,
        "power_rmse": avg_power,
        "power_rmse_std": std_power,
        "phase_rmse": avg_phase,
        "phase_rmse_std": std_phase,
        "az_rmse": avg_az,
        "az_rmse_std": std_az,
        "el_rmse": avg_el,
        "el_rmse_std": std_el,
        "aod_az_rmse": avg_aod_az,
        "aod_az_rmse_std": std_aod_az,
        "aod_el_rmse": avg_aod_el,
        "aod_el_rmse_std": std_aod_el,
        "path_length_rmse": avg_path_length_rmse,
        "path_length_rmse_std": std_path_length_rmse,
        "interaction_accuracy": avg_interaction_accuracy,
        "interaction_accuracy_std": std_interaction_accuracy,
        "interaction_f1": avg_interaction_f1,
        "interaction_f1_std": std_interaction_f1,
        "delay_mae": avg_delay_mae,
        "delay_mae_std": std_delay_mae,
        "power_mae": avg_power_mae,
        "power_mae_std": std_power_mae,
        "phase_mae": avg_phase_mae,
        "phase_mae_std": std_phase_mae,
        "avg_az_mae": avg_az_mae,
        "avg_az_mae_std": std_az_mae,
        "avg_el_mae": avg_el_mae,
        "avg_el_mae_std": std_el_mae,
        "avg_aod_az_mae": avg_aod_az_mae,
        "avg_aod_az_mae_std": std_aod_az_mae,
        "avg_aod_el_mae": avg_aod_el_mae,
        "avg_aod_el_mae_std": std_aod_el_mae,
        "path_length_mae": avg_path_length_mae,
        "path_length_mae_std": std_path_length_mae,
        "best_val_loss": best_loss.item() if hasattr(best_loss, "item") else best_loss,
    }
    pd.DataFrame([row]).to_csv(args.csv_log_file, mode="a", index=False, header=not os.path.exists(args.csv_log_file))
    print(f"✓ Results for {scenario} saved to {args.csv_log_file}")


def main():
    args = parse_args()
    scenarios = resolve_scenarios(args)
    print(f"Running {len(scenarios)} scenario(s): {scenarios}")
    for scenario in scenarios:
        run_scenario(scenario, args)


if __name__ == "__main__":
    main()
