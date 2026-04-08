import argparse
import os

import deepmimo as dm
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm

from dataset.dataloaders import PreTrainMySeqDataLoader
from k_means_utils import *
from utils.utils import add_noise_to_paths


csv_log_file = "playground_std_fix1_cluster_routed_heads_results.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test shared-decoder, cluster-routed output heads using 1-step delay/power KMeans."
    )
    parser.add_argument("scenario", nargs="?", default="city_47_chicago_3p5")
    parser.add_argument("--csv-log-file", default=csv_log_file)
    parser.add_argument("--checkpoint-dir", default="checkpoints2")
    parser.add_argument("--n-clusters", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-5)
    return parser.parse_args()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_stop_metrics(path_count, targets):
    return np.sqrt(mean_squared_error(path_count.cpu().numpy(), targets.squeeze().cpu().numpy()))


def _convert_feature_value(key, value):
    if key == "delay":
        return float(value) * 1e6
    if key == "power":
        return float(value) * 0.01
    if key in ["phase", "aoa_az", "aoa_el"]:
        return float(value) * (np.pi / 180.0)
    return float(value)


def _normalize_feature_value(dataloader_obj, key, value):
    if getattr(dataloader_obj, "normalizers", None) and getattr(dataloader_obj, "apply_normalizers", None):
        if key in dataloader_obj.apply_normalizers:
            stat = dataloader_obj.normalizers.get(key, None)
            if stat is not None:
                return (value - stat["mean"]) / (stat["std"] + 1e-8)
    return value


def _sorted_path_indices(df, idx, sort_by):
    delays_raw = np.array(df["delay"][idx])
    powers_raw = np.array(df["power"][idx])
    if sort_by == "power":
        return np.argsort(-powers_raw)
    return np.argsort(delays_raw)


def _collect_step_vectors_from_base_dataset(base_dataset, feature_keys, max_path_len):
    df = base_dataset.dataset_filtered
    n_samples = len(df["delay"])
    step_vectors = [[] for _ in range(max_path_len)]
    rx_xy_list = []
    seq_vectors = []
    valid_lens = []
    mins = np.array(base_dataset.mins)
    maxs = np.array(base_dataset.maxs)
    norm = getattr(base_dataset, "normalizers", None)
    apply_norm = getattr(base_dataset, "apply_normalizers", None) or []

    for idx in range(n_samples):
        rx_raw = np.array(df["rx_pos"][idx], dtype=np.float32)
        rx_xy = rx_raw[:2]
        if norm and "pos" in apply_norm:
            rx_xy = (rx_xy - norm["rx_pos"]["mean"][:2]) / (norm["rx_pos"]["std"][:2] + 1e-8)
        else:
            rx_xy = (rx_xy - mins[:2]) / (maxs[:2] - mins[:2] + 1e-8)
        rx_xy_list.append(rx_xy)

        seq = np.zeros((max_path_len, len(feature_keys)), dtype=np.float32)
        indices = _sorted_path_indices(df, idx, base_dataset.sort_by)
        step = 0
        for path_idx in indices:
            if step >= max_path_len:
                break
            vec = []
            broken = False
            for key in feature_keys:
                value = np.array(df[key][idx])[path_idx]
                if np.isnan(value):
                    broken = True
                    break
                value = _convert_feature_value(key, value)
                value = _normalize_feature_value(base_dataset, key, value)
                vec.append(value)
            if broken:
                break
            vec = np.array(vec, dtype=np.float32)
            seq[step] = vec
            step_vectors[step].append(vec)
            step += 1
        seq_vectors.append(seq)
        valid_lens.append(step)

    return (
        np.array(rx_xy_list, dtype=np.float32),
        np.array(seq_vectors, dtype=np.float32),
        np.array(valid_lens, dtype=np.int64),
        step_vectors,
    )


def compute_first_step_kmeans_cluster_stats(base_dataset, feature_keys, n_clusters=4, random_state=42):
    n_features = len(feature_keys)
    _, _, _, step_vectors = _collect_step_vectors_from_base_dataset(base_dataset, feature_keys, max_path_len=1)
    arr = np.array(step_vectors[0], dtype=np.float32)
    centers = np.zeros((n_clusters, n_features), dtype=np.float32)

    if len(arr) == 0:
        return centers
    if len(arr) < n_clusters:
        reps = [arr[min(i, len(arr) - 1)] for i in range(n_clusters)]
        return np.stack(reps, axis=0)

    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    km.fit(arr)
    return km.cluster_centers_.astype(np.float32)


def precompute_train_cluster_ids(base_dataset, cluster_centers, feature_keys):
    rx_xy, seq_vectors, valid_lens, _ = _collect_step_vectors_from_base_dataset(base_dataset, feature_keys, max_path_len=1)
    n_samples = seq_vectors.shape[0]
    cluster_ids = np.zeros((n_samples,), dtype=np.int64)
    for i in range(n_samples):
        if int(valid_lens[i]) < 1:
            cluster_ids[i] = 0
            continue
        vec = seq_vectors[i, 0]
        dist = np.linalg.norm(cluster_centers - vec[None, :], axis=1)
        cluster_ids[i] = int(np.argmin(dist))
    return rx_xy, cluster_ids, valid_lens


def lookup_cluster_ids_by_position(prompts, train_rx_pos, train_cluster_ids, device, prompt_rx_slice=(3, 5)):
    batch_size = prompts.shape[0]
    rx_query = prompts[:, prompt_rx_slice[0]:prompt_rx_slice[1]].detach().cpu().numpy()
    batch_ids = np.zeros((batch_size,), dtype=np.int64)
    if train_rx_pos.shape[0] == 0:
        return torch.tensor(batch_ids, dtype=torch.long, device=device)

    for b in range(batch_size):
        dist = np.sqrt(np.sum((train_rx_pos - rx_query[b]) ** 2, axis=1))
        nn_idx = int(np.argmin(dist))
        batch_ids[b] = int(train_cluster_ids[nn_idx])
    return torch.tensor(batch_ids, dtype=torch.long, device=device)


class SharedPathDecoderBackbone(nn.Module):
    def __init__(
        self,
        prompt_dim=6,
        hidden_dim=512,
        n_layers=8,
        n_heads=8,
        max_T=35,
        prefix_len=4,
        pad_value=0,
        zero_prompt_positions_for_pretrain=True,
    ):
        super().__init__()
        self.pad_value = pad_value
        self.hidden_dim = hidden_dim
        self.prompt_dim = prompt_dim
        self.prefix_len = prefix_len
        self.max_T = max_T
        self.zero_prompt_positions_for_pretrain = zero_prompt_positions_for_pretrain

        self.prompt_to_prefix = nn.Linear(prompt_dim, prefix_len * hidden_dim)
        self.path_in = nn.Linear(12, hidden_dim)
        self.pos_emb = nn.Embedding(max_T, hidden_dim)
        self.memory_pos_emb = nn.Embedding(prefix_len, hidden_dim)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=4 * hidden_dim,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

    def _prepare_prompts(self, prompts, pre_train=False):
        if pre_train and self.zero_prompt_positions_for_pretrain:
            prompts = prompts.clone()
            prompts[:, :self.prompt_dim] = 0.0
        return prompts

    def forward_hidden(self, prompts, paths, interactions, pre_train=False):
        batch_size, total_len, _ = paths.shape
        prompts = self._prepare_prompts(prompts, pre_train=pre_train)
        prefix = self.prompt_to_prefix(prompts).view(batch_size, self.prefix_len, self.hidden_dim)
        memory_pos = self.memory_pos_emb(torch.arange(self.prefix_len, device=prompts.device)).unsqueeze(0)
        memory = prefix + memory_pos

        phase = paths[:, :, 2]
        sinp = torch.sin(phase)
        cosp = torch.cos(phase)
        aoa_az = paths[:, :, 3]
        sin_az = torch.sin(aoa_az)
        cos_az = torch.cos(aoa_az)
        aoa_el = paths[:, :, 4]
        sin_el = torch.sin(aoa_el)
        cos_el = torch.cos(aoa_el)

        x = torch.stack(
            [paths[:, :, 0], paths[:, :, 1], sinp, cosp, sin_az, cos_az, sin_el, cos_el],
            dim=-1,
        )
        interactions_clean = interactions.clone()
        interactions_clean[interactions_clean == -1] = 0
        x = torch.cat([x, interactions_clean], dim=-1)
        x = self.path_in(x)

        pos = self.pos_emb(torch.arange(total_len, device=x.device)).unsqueeze(0)
        x = x + pos
        causal_mask = torch.triu(torch.ones(total_len, total_len, device=x.device), diagonal=1).bool()

        h_paths = self.decoder(tgt=x, memory=memory, tgt_mask=causal_mask)
        prefix_flat = prefix.reshape(batch_size, -1)
        return h_paths, prefix_flat


class PathDecoderClusterRoutedHeads(nn.Module):
    def __init__(self, backbone, hidden_dim, n_clusters):
        super().__init__()
        self.backbone = backbone
        self.hidden_dim = hidden_dim
        self.n_clusters = n_clusters
        self.cluster_head_router = True

        self.delay_heads = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(n_clusters)])
        self.power_heads = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(n_clusters)])
        self.angle_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, 6),
                )
                for _ in range(n_clusters)
            ]
        )
        self.interaction_heads = nn.ModuleList([nn.Linear(hidden_dim, 4) for _ in range(n_clusters)])
        self.pathcount_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(backbone.prefix_len * hidden_dim, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, 1),
                )
                for _ in range(n_clusters)
            ]
        )

    def forward(self, prompts, paths, interactions, cluster_ids=None, pre_train=False):
        h_paths, prefix_flat = self.backbone.forward_hidden(prompts, paths, interactions, pre_train=pre_train)
        batch_size, total_len, _ = h_paths.shape
        if cluster_ids is None:
            cluster_ids = torch.zeros(batch_size, dtype=torch.long, device=h_paths.device)
        cluster_ids = cluster_ids.to(device=h_paths.device, dtype=torch.long).clamp_(0, self.n_clusters - 1)

        delay_pred = torch.zeros(batch_size, total_len, device=h_paths.device, dtype=h_paths.dtype)
        power_pred = torch.zeros(batch_size, total_len, device=h_paths.device, dtype=h_paths.dtype)
        out = torch.zeros(batch_size, total_len, 6, device=h_paths.device, dtype=h_paths.dtype)
        interaction_logits = torch.zeros(batch_size, total_len, 4, device=h_paths.device, dtype=h_paths.dtype)
        pathcounts = torch.zeros(batch_size, 1, device=h_paths.device, dtype=h_paths.dtype)

        for cluster_idx in range(self.n_clusters):
            mask = cluster_ids == cluster_idx
            if not mask.any():
                continue
            h_sel = h_paths[mask]
            prefix_sel = prefix_flat[mask]
            delay_pred[mask] = self.delay_heads[cluster_idx](h_sel).squeeze(-1)
            power_pred[mask] = self.power_heads[cluster_idx](h_sel).squeeze(-1)
            out[mask] = self.angle_heads[cluster_idx](h_sel)
            interaction_logits[mask] = self.interaction_heads[cluster_idx](h_sel)
            pathcounts[mask] = self.pathcount_heads[cluster_idx](prefix_sel)

        phase_sin_pred = out[:, :, 0]
        phase_cos_pred = out[:, :, 1]
        az_sin_pred = out[:, :, 2]
        az_cos_pred = out[:, :, 3]
        el_sin_pred = out[:, :, 4]
        el_cos_pred = out[:, :, 5]

        phase_pred = torch.atan2(phase_sin_pred, phase_cos_pred)
        az_pred = torch.atan2(az_sin_pred, az_cos_pred)
        el_pred = torch.atan2(el_sin_pred, el_cos_pred)

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


def generate_paths_no_env_batch(model, prompts, max_steps=25, cluster_ids=None):
    model.eval()
    device_local = next(model.parameters()).device
    prompts = prompts.to(device_local)
    batch_size = prompts.size(0)
    cur = torch.zeros(batch_size, 1, 5, device=device_local)
    inter_str = -1 * torch.ones(batch_size, 1, 4, device=device_local)
    outputs = []
    outputs_inter_str = []

    for _ in range(max_steps):
        d, p, s, c, ph, az_s, az_c, az, el_s, el_c, el, pathcounts, inter_str_logits = model(
            prompts,
            cur,
            inter_str,
            cluster_ids=cluster_ids,
        )

        d_t = d[:, -1]
        p_t = p[:, -1]
        ph_t = ph[:, -1]
        az_t = az[:, -1]
        el_t = el[:, -1]
        inter_logits_t = inter_str_logits[:, -1]
        inter_pred_t = (torch.sigmoid(inter_logits_t) > 0.5).float()

        outputs.append(torch.stack([d_t, p_t, ph_t, az_t, el_t], dim=-1))
        outputs_inter_str.append(inter_pred_t)

        next_path = torch.stack([d_t, p_t, ph_t, az_t, el_t], dim=-1).unsqueeze(1)
        cur = torch.cat([cur, next_path], dim=1)
        inter_str = torch.cat([inter_str, inter_pred_t.unsqueeze(1)], dim=1)

    return (
        torch.stack(outputs, dim=1).detach().cpu(),
        pathcounts,
        torch.stack(outputs_inter_str, dim=1).detach().cpu(),
    )


def evaluate_model(model, val_loader, max_generate=26, cluster_lookup_data=None):
    model.eval()
    delay_errors = []
    power_errors = []
    phase_errors = []
    path_length_rmses = []
    delay_maes = []
    power_maes = []
    phase_maes = []
    path_length_maes = []
    az_errors = []
    el_errors = []
    az_maes = []
    el_maes = []
    interaction_targets_all = []
    interaction_preds_all = []

    with torch.no_grad():
        outer_bar = tqdm(val_loader, desc="Evaluating (batches)", leave=True)
        for prompts, paths, path_lengths, interactions, env, env_prop, path_padding_mask in outer_bar:
            prompts = prompts.cuda()
            paths = paths.cuda()
            path_lengths = path_lengths.cuda()
            interactions = interactions.cuda()
            cluster_ids = None
            if cluster_lookup_data is not None:
                train_rx_pos, train_cluster_ids, _ = cluster_lookup_data
                cluster_ids = lookup_cluster_ids_by_position(prompts, train_rx_pos, train_cluster_ids, prompts.device)

            generated, path_lengths_pred, inter_str_pred = generate_paths_no_env_batch(
                model,
                prompts,
                max_steps=max_generate,
                cluster_ids=cluster_ids,
            )

            generated = generated.cuda()
            inter_str_pred = inter_str_pred.cpu()
            if path_lengths_pred.dim() > 1:
                path_lengths_pred = path_lengths_pred.squeeze(-1)

            batch_delay_rmses = []
            batch_power_rmses = []
            batch_phase_rmses = []
            batch_length_rmses = []
            batch_az_rmses = []
            batch_el_rmses = []

            for b in range(prompts.size(0)):
                n_valid = int(round(path_lengths[b].item() * 25))
                gt = paths[b][1:1 + n_valid, :5]
                gt_interactions = interactions[b][1:1 + n_valid, :]

                T = min(len(gt), generated.size(1))
                pred = generated[b, :T]
                gt = gt[:T]
                pred_interactions = inter_str_pred[b, :T]
                gt_interactions = gt_interactions[:T].detach().cpu()

                valid_interaction_mask = gt_interactions[:, 0] != -1
                if valid_interaction_mask.any():
                    interaction_targets_all.append(gt_interactions[valid_interaction_mask].numpy().astype(np.int32))
                    interaction_preds_all.append(pred_interactions[valid_interaction_mask].numpy().astype(np.int32))

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

                pl_pred = path_lengths_pred[b].squeeze()
                pl_gt = path_lengths[b].squeeze()
                length_rmse = torch.mean((pl_pred - pl_gt) ** 2).sqrt().item()
                length_mae = torch.mean(torch.abs(pl_pred - pl_gt)).item()

                delay_errors.append(delay_rmse)
                power_errors.append(power_rmse)
                phase_errors.append(phase_rmse)
                path_length_rmses.append(length_rmse)
                az_errors.append(az_rmse)
                el_errors.append(el_rmse)
                delay_maes.append(delay_mae)
                power_maes.append(power_mae)
                phase_maes.append(phase_mae)
                path_length_maes.append(length_mae)
                az_maes.append(az_mae)
                el_maes.append(el_mae)
                batch_delay_rmses.append(delay_rmse)
                batch_power_rmses.append(power_rmse)
                batch_phase_rmses.append(phase_rmse)
                batch_length_rmses.append(length_rmse)
                batch_az_rmses.append(az_rmse)
                batch_el_rmses.append(el_rmse)

            if batch_delay_rmses:
                outer_bar.set_postfix(
                    {
                        "delay_rmse": f"{np.mean(batch_delay_rmses):.3f}",
                        "power_rmse": f"{np.mean(batch_power_rmses):.3f}",
                        "phase_rmse": f"{np.mean(batch_phase_rmses):.3f}",
                        "az_rmse": f"{np.mean(batch_az_rmses):.3f}",
                        "el_rmse": f"{np.mean(batch_el_rmses):.3f}",
                        "length_rmse": f"{np.mean(batch_length_rmses):.3f}",
                    }
                )

    avg_delay = np.mean(delay_errors)
    avg_power = np.mean(power_errors)
    avg_phase = np.mean(phase_errors)
    avg_az = np.mean(az_errors) if az_errors else 0.0
    avg_el = np.mean(el_errors) if el_errors else 0.0
    avg_path_length_rmse = np.mean(path_length_rmses)
    avg_delay_mae = np.mean(delay_maes)
    avg_power_mae = np.mean(power_maes)
    avg_phase_mae = np.mean(phase_maes)
    avg_az_mae = np.mean(az_maes)
    avg_el_mae = np.mean(el_maes)
    avg_path_length_mae = np.mean(path_length_maes)

    if interaction_targets_all:
        interaction_targets_np = np.concatenate(interaction_targets_all, axis=0)
        interaction_preds_np = np.concatenate(interaction_preds_all, axis=0)
        avg_interaction_accuracy = accuracy_score(interaction_targets_np.reshape(-1), interaction_preds_np.reshape(-1))
        avg_interaction_f1 = f1_score(interaction_targets_np.reshape(-1), interaction_preds_np.reshape(-1), zero_division=0)
    else:
        avg_interaction_accuracy = 0.0
        avg_interaction_f1 = 0.0

    print("\n=================  Final EVALUATION RESULTS =================")
    print(f"Delay RMSE           : {avg_delay:.4f} us")
    print(f"Power RMSE           : {avg_power:.4f} dB")
    print(f"Phase RMSE           : {avg_phase:.4f} degrees")
    print(f"AoA Azimuth RMSE     : {avg_az:.4f} degrees")
    print(f"AoA Elevation RMSE   : {avg_el:.4f} degrees")
    print(f"Path Length RMSE     : {avg_path_length_rmse:.4f}")
    print(f"Interaction Accuracy : {avg_interaction_accuracy:.4f}")
    print(f"Interaction F1       : {avg_interaction_f1:.4f}")
    print(f"Delay MAE            : {avg_delay_mae:.4f} us")
    print(f"Power MAE            : {avg_power_mae:.4f} dB")
    print(f"Phase MAE            : {avg_phase_mae:.4f} degrees")
    print(f"AoA Azimuth MAE      : {avg_az_mae:.4f} degrees")
    print(f"AoA Elevation MAE    : {avg_el_mae:.4f} degrees")
    print(f"Path Length MAE      : {avg_path_length_mae:.4f}")
    print("=====================================================\n")

    return (
        avg_delay,
        avg_power,
        avg_phase,
        avg_az,
        avg_el,
        avg_path_length_rmse,
        avg_interaction_accuracy,
        avg_interaction_f1,
        avg_delay_mae,
        avg_power_mae,
        avg_phase_mae,
        avg_az_mae,
        avg_el_mae,
        avg_path_length_mae,
    )


def train_with_interactions(model, train_loader, val_loader, config, train_data, optimizer, scheduler, checkpoint_path, cluster_lookup_data=None):
    best_val_loss = float("inf")

    for epoch in range(config["epochs"]):
        model.train()
        train_losses = []
        train_loss_delay = []
        train_loss_power = []
        train_loss_phase = []
        train_loss_path_length = []
        train_loss_interaction = []
        train_path_length_rmse = []
        train_loss_az = []
        train_loss_el = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]", leave=False)
        for prompts, paths, path_lengths, interactions, env, env_prop, path_padding_mask in pbar:
            prompts = prompts.cuda()
            paths = paths.cuda()
            path_lengths = path_lengths.cuda()
            interactions = interactions.cuda()
            path_padding_mask = path_padding_mask.cuda()

            cluster_ids = None
            if cluster_lookup_data is not None:
                train_rx_pos, train_cluster_ids, _ = cluster_lookup_data
                cluster_ids = lookup_cluster_ids_by_position(prompts, train_rx_pos, train_cluster_ids, prompts.device)

            paths_in = paths[:, :-1, :]
            p_noise = config.get("TARGET_NOISE_PROB", 0.0)
            if p_noise > 0:
                paths_in = add_noise_to_paths(
                    paths_in,
                    path_padding_mask[:, :-1],
                    p_noise=p_noise,
                    noise_params=config.get("TARGET_NOISE_PARAMS"),
                )
            interactions_in = interactions[:, :-1, :]

            paths_out = paths[:, 1:, :]
            interactions_out = interactions[:, 1:, :]

            (delay_pred, power_pred, phase_sin_pred, phase_cos_pred, phase_pred,
             az_sin_pred, az_cos_pred, az_pred, el_sin_pred, el_cos_pred, el_pred,
             path_length_pred, interaction_logits) = model(
                prompts, paths_in, interactions_in, cluster_ids=cluster_ids, pre_train=False
            )

            (total_loss, loss_delay, loss_power, loss_phase,
             loss_az, loss_el, loss_path_length, loss_interaction, loss_channel) = masked_loss(
                delay_pred, power_pred, phase_sin_pred, phase_cos_pred, phase_pred,
                az_sin_pred, az_cos_pred, az_pred, el_sin_pred, el_cos_pred, el_pred,
                path_length_pred, interaction_logits, paths_out, path_lengths,
                interactions_out, pad_value=train_data.pad_value,
                interaction_weight=config.get("interaction_weight", 0.1),
                delay_only=config.get("delay_only_loss", False),
                path_padding_mask=path_padding_mask,
            )

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            path_length_rmse = compute_stop_metrics(path_length_pred.detach().squeeze(-1), path_lengths)
            train_losses.append(total_loss.item())
            train_loss_delay.append(loss_delay.item())
            train_loss_power.append(loss_power.item())
            train_loss_phase.append(loss_phase.item())
            train_loss_path_length.append(loss_path_length.item())
            train_loss_az.append(loss_az.item())
            train_loss_el.append(loss_el.item())
            train_loss_interaction.append(loss_interaction.item())
            train_path_length_rmse.append(path_length_rmse)

            pbar.set_postfix(
                {
                    "loss": f"{total_loss.item():.4f}",
                    "delay": f"{loss_delay.item():.4f}",
                    "power": f"{loss_power.item():.4f}",
                    "phase": f"{loss_phase.item():.4f}",
                    "az": f"{loss_az.item():.4f}",
                    "el": f"{loss_el.item():.4f}",
                    "inter": f"{loss_interaction.item():.4f}",
                    "path_rmse": f"{path_length_rmse:.4f}",
                    "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                }
            )

        scheduler.step()
        avg_train_loss = np.mean(train_losses)
        avg_train_delay = np.mean(train_loss_delay)
        avg_train_power = np.mean(train_loss_power)
        avg_train_phase = np.mean(train_loss_phase)
        avg_train_az = np.mean(train_loss_az)
        avg_train_el = np.mean(train_loss_el)
        avg_train_path_length = np.mean(train_loss_path_length)
        avg_train_interaction = np.mean(train_loss_interaction)
        avg_train_path_length_rmse = np.mean(train_path_length_rmse)

        model.eval()
        val_losses = []
        val_loss_delay = []
        val_loss_power = []
        val_loss_phase = []
        val_loss_path_length = []
        val_loss_interaction = []
        val_path_length_rmse = []
        val_loss_az = []
        val_loss_el = []

        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]", leave=False)
            for prompts, paths, path_lengths, interactions, env, env_prop, path_padding_mask in pbar:
                prompts = prompts.cuda()
                paths = paths.cuda()
                path_lengths = path_lengths.cuda()
                interactions = interactions.cuda()
                path_padding_mask = path_padding_mask.cuda()

                cluster_ids = None
                if cluster_lookup_data is not None:
                    train_rx_pos, train_cluster_ids, _ = cluster_lookup_data
                    cluster_ids = lookup_cluster_ids_by_position(prompts, train_rx_pos, train_cluster_ids, prompts.device)

                paths_in = paths[:, :-1, :]
                interactions_in = interactions[:, :-1, :]
                paths_out = paths[:, 1:, :]
                interactions_out = interactions[:, 1:, :]

                (delay_pred, power_pred, phase_sin_pred, phase_cos_pred, phase_pred,
                 az_sin_pred, az_cos_pred, az_pred, el_sin_pred, el_cos_pred, el_pred,
                 path_length_pred, interaction_logits) = model(
                    prompts, paths_in, interactions_in, cluster_ids=cluster_ids, pre_train=False
                )

                (total_loss, loss_delay, loss_power, loss_phase,
                 loss_az, loss_el, loss_path_length, loss_interaction, loss_channel) = masked_loss(
                    delay_pred, power_pred, phase_sin_pred, phase_cos_pred, phase_pred,
                    az_sin_pred, az_cos_pred, az_pred, el_sin_pred, el_cos_pred, el_pred,
                    path_length_pred, interaction_logits, paths_out, path_lengths,
                    interactions_out, pad_value=train_data.pad_value,
                    interaction_weight=config.get("interaction_weight", 0.1),
                    delay_only=config.get("delay_only_loss", False),
                    path_padding_mask=path_padding_mask,
                )

                path_length_rmse = compute_stop_metrics(path_length_pred.detach().squeeze(-1), path_lengths)
                val_losses.append(total_loss.item())
                val_loss_delay.append(loss_delay.item())
                val_loss_power.append(loss_power.item())
                val_loss_phase.append(loss_phase.item())
                val_loss_az.append(loss_az.item())
                val_loss_el.append(loss_el.item())
                val_loss_path_length.append(loss_path_length.item())
                val_loss_interaction.append(loss_interaction.item())
                val_path_length_rmse.append(path_length_rmse)
                pbar.set_postfix({"val_loss": f"{total_loss.item():.4f}", "inter": f"{loss_interaction.item():.4f}"})

        avg_val_loss = np.mean(val_losses)
        avg_val_delay = np.mean(val_loss_delay)
        avg_val_power = np.mean(val_loss_power)
        avg_val_phase = np.mean(val_loss_phase)
        avg_val_az = np.mean(val_loss_az)
        avg_val_el = np.mean(val_loss_el)
        avg_val_path_length = np.mean(val_loss_path_length)
        avg_val_interaction = np.mean(val_loss_interaction)
        avg_val_path_length_rmse = np.mean(val_path_length_rmse)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
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
            print(f"  Best checkpoint saved (val_loss: {best_val_loss:.4f})")

        print(f"\nEpoch {epoch:02d}")
        print(f"  Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"    Delay: {avg_train_delay:.4f} (val: {avg_val_delay:.4f})")
        print(f"    Power: {avg_train_power:.4f} (val: {avg_val_power:.4f})")
        print(f"    Phase: {avg_train_phase:.4f} (val: {avg_val_phase:.4f})")
        print(f"    Az: {avg_train_az:.4f} (val: {avg_val_az:.4f})")
        print(f"    El: {avg_train_el:.4f} (val: {avg_val_el:.4f})")
        print(f"    Interaction: {avg_train_interaction:.4f} (val: {avg_val_interaction:.4f})")
        print(f"    PathLength: {avg_train_path_length:.4f} (val: {avg_val_path_length:.4f})")
        print(f"    PathLength RMSE: {avg_train_path_length_rmse:.4f} (val: {avg_val_path_length_rmse:.4f})")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.3e}")


def load_best_checkpoint(model, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        print(f"Warning: Checkpoint not found at {checkpoint_path}")
        return None, None
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    epoch = checkpoint["epoch"]
    best_avg_val_loss = checkpoint["best_val_loss"]
    print(f"Loaded best checkpoint from epoch {epoch} (val_loss: {best_avg_val_loss:.4f})")
    return epoch, best_avg_val_loss


def main():
    args = parse_args()
    scenario = args.scenario
    dm.download(scenario)
    dataset = dm.load(scenario)

    config = {
        "BATCH_SIZE": args.batch_size,
        "PAD_VALUE": 0,
        "USE_WANDB": False,
        "LR": args.lr,
        "epochs": args.epochs,
        "interaction_weight": 0.01,
        "experiment": f"noise_std_fix1_cluster_routed_heads_128_4_{scenario}",
        "hidden_dim": 128,
        "n_layers": 4,
        "n_heads": 4,
        "n_clusters": args.n_clusters,
        "max_path_len_clusters": 1,
        "cluster_features": ["delay", "power"],
        "delay_only_loss": False,
        "TARGET_NOISE_PROB": 0.0,
        "TARGET_NOISE_PARAMS": None,
    }

    train_data = PreTrainMySeqDataLoader(
        dataset,
        train=True,
        split_by="user",
        sort_by="power",
        pad_value=config["PAD_VALUE"],
        normalizers=None,
        apply_normalizers=[],
    )
    val_data = PreTrainMySeqDataLoader(
        dataset,
        train=False,
        split_by="user",
        sort_by="power",
        pad_value=config["PAD_VALUE"],
        normalizers=None,
        apply_normalizers=[],
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_data,
        batch_size=config["BATCH_SIZE"],
        shuffle=True,
        collate_fn=train_data.collate_fn,
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=val_data,
        batch_size=config["BATCH_SIZE"],
        shuffle=False,
        collate_fn=val_data.collate_fn,
    )

    centers = compute_first_step_kmeans_cluster_stats(
        train_data,
        feature_keys=config["cluster_features"],
        n_clusters=config["n_clusters"],
    )
    train_rx_pos, train_cluster_ids, train_valid_len = precompute_train_cluster_ids(
        train_data,
        cluster_centers=centers,
        feature_keys=config["cluster_features"],
    )
    cluster_lookup_data = (train_rx_pos, train_cluster_ids, train_valid_len)
    print(
        f"Prepared first-step cluster lookup using features={config['cluster_features']} and "
        f"n_clusters={config['n_clusters']}"
    )

    backbone = SharedPathDecoderBackbone(
        prompt_dim=6,
        hidden_dim=config["hidden_dim"],
        n_layers=config["n_layers"],
        n_heads=config["n_heads"],
        pad_value=config["PAD_VALUE"],
    ).to(device)
    model = PathDecoderClusterRoutedHeads(
        backbone,
        hidden_dim=config["hidden_dim"],
        n_clusters=config["n_clusters"],
    ).to(device)
    print("Total trainable parameters:", count_parameters(model))

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["LR"])
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=25, T_mult=1, eta_min=1e-8)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.checkpoint_dir, f"{config['experiment']}_best_model_checkpoint.pth")

    train_with_interactions(
        model,
        train_loader,
        val_loader,
        config,
        train_data,
        optimizer,
        scheduler,
        checkpoint_path,
        cluster_lookup_data=cluster_lookup_data,
    )

    load_best_checkpoint(model, checkpoint_path)
    results = evaluate_model(model, val_loader, cluster_lookup_data=cluster_lookup_data)
    (
        avg_delay,
        avg_power,
        avg_phase,
        avg_az,
        avg_el,
        avg_path_length_rmse,
        avg_interaction_accuracy,
        avg_interaction_f1,
        avg_delay_mae,
        avg_power_mae,
        avg_phase_mae,
        avg_az_mae,
        avg_el_mae,
        avg_path_length_mae,
    ) = results

    scenario_row = {
        "scenario": scenario,
        "delay_rmse": avg_delay,
        "power_rmse": avg_power,
        "phase_rmse": avg_phase,
        "az_rmse": avg_az,
        "el_rmse": avg_el,
        "path_length_rmse": avg_path_length_rmse,
        "interaction_accuracy": avg_interaction_accuracy,
        "interaction_f1": avg_interaction_f1,
        "delay_mae": avg_delay_mae,
        "power_mae": avg_power_mae,
        "phase_mae": avg_phase_mae,
        "avg_az_mae": avg_az_mae,
        "avg_el_mae": avg_el_mae,
        "path_length_mae": avg_path_length_mae,
        "n_clusters": config["n_clusters"],
        "cluster_mode": "first_step_delay_power_routed_heads",
    }

    csv_path = args.csv_log_file
    if os.path.exists(csv_path):
        pd.concat([pd.read_csv(csv_path), pd.DataFrame([scenario_row])], ignore_index=True).to_csv(csv_path, index=False)
    else:
        pd.DataFrame([scenario_row]).to_csv(csv_path, index=False)
    print(f"Saved results to {csv_path}")


if __name__ == "__main__":
    main()
