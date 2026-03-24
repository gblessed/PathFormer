import argparse
import csv
import os

import deepmimo as dm
import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from tqdm import tqdm

from dataset.dataloaders_play import PreTrainMySeqDataLoader
from utils.utils import add_noise_to_paths, masked_loss


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_SCENARIOS = [
    "city_47_chicago_3p5",
    "city_0_newyork_3p5",
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


def count_parameters(model):
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


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


def compute_feature_kmeans_cluster_stats_for_dataset(base_dataset, feature_keys, max_path_len=25, n_clusters=5, random_state=42):
    n_features = len(feature_keys)
    _, _, _, step_vectors = _collect_step_vectors_from_base_dataset(base_dataset, feature_keys, max_path_len)
    centers = np.zeros((max_path_len, n_clusters, n_features), dtype=np.float32)
    stds = np.zeros((max_path_len, n_clusters, n_features), dtype=np.float32)

    for step in range(max_path_len):
        arr = np.array(step_vectors[step], dtype=np.float32)
        if len(arr) == 0:
            continue
        if len(arr) < n_clusters:
            reps = [arr[min(i, len(arr) - 1)] for i in range(n_clusters)]
            centers[step] = np.stack(reps, axis=0)
            stds[step] = np.stack([np.std(arr, axis=0)] * n_clusters, axis=0)
        else:
            kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
            labels = kmeans.fit_predict(arr)
            centers[step] = kmeans.cluster_centers_.astype(np.float32)
            for cluster_idx in range(n_clusters):
                pts = arr[labels == cluster_idx]
                if len(pts) > 0:
                    stds[step, cluster_idx] = np.std(pts, axis=0)
    return centers, stds


def precompute_train_cluster_center_std_sequences(base_dataset, cluster_centers, cluster_stds, feature_keys, max_path_len=25):
    rx_xy, seq_vectors, valid_lens, _ = _collect_step_vectors_from_base_dataset(base_dataset, feature_keys, max_path_len)
    n_samples = seq_vectors.shape[0]
    n_features = len(feature_keys)
    out = np.zeros((n_samples, max_path_len, 2 * n_features), dtype=np.float32)

    for sample_idx in range(n_samples):
        valid = int(valid_lens[sample_idx])
        for step in range(valid):
            vec = seq_vectors[sample_idx, step]
            dist = np.linalg.norm(cluster_centers[step] - vec[None, :], axis=1)
            nn_idx = int(np.argmin(dist))
            out[sample_idx, step, :n_features] = cluster_centers[step, nn_idx]
            out[sample_idx, step, n_features:] = cluster_stds[step, nn_idx]

    return rx_xy, out, valid_lens


def lookup_cluster_center_std_by_position(prompts, train_rx_pos, train_cluster_center_std, train_valid_len, device, prompt_rx_slice=(3, 5)):
    batch_size = prompts.shape[0]
    rx_query = prompts[:, prompt_rx_slice[0]:prompt_rx_slice[1]].detach().cpu().numpy()
    max_t = train_cluster_center_std.shape[1]
    feature_dim = train_cluster_center_std.shape[2]
    batch = np.zeros((batch_size, max_t, feature_dim), dtype=np.float32)
    pad_mask = np.ones((batch_size, max_t), dtype=bool)

    for batch_idx in range(batch_size):
        if train_rx_pos.shape[0] == 0:
            continue
        pos = rx_query[batch_idx]
        dist = np.sqrt(np.sum((train_rx_pos - pos) ** 2, axis=1))
        nn_idx = int(np.argmin(dist))
        batch[batch_idx] = train_cluster_center_std[nn_idx]
        valid = int(train_valid_len[nn_idx])
        pad_mask[batch_idx, :valid] = False
        pad_mask[batch_idx, valid:] = True

    return (
        torch.tensor(batch, dtype=torch.float32, device=device),
        torch.tensor(pad_mask, dtype=torch.bool, device=device),
    )


class PathDecoderClusterEncoderAttentionFix1(nn.Module):
    def __init__(
        self,
        prompt_dim=6,
        hidden_dim=512,
        n_layers=8,
        n_heads=8,
        cluster_feature_dim=None,
        max_T=35,
        prefix_len=4,
        cluster_encoder_layers=2,
        zero_prompt_positions_for_pretrain=True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.prompt_dim = prompt_dim
        self.prefix_len = prefix_len
        self.max_T = max_T
        self.zero_prompt_positions_for_pretrain = zero_prompt_positions_for_pretrain

        self.cluster_embed = nn.Sequential(
            nn.Linear(cluster_feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.prompt_to_prefix = nn.Linear(prompt_dim, prefix_len * hidden_dim)
        self.path_in = nn.Linear(12, hidden_dim)
        self.pos_emb = nn.Embedding(max_T + 64, hidden_dim)
        self.cluster_pos_emb = nn.Embedding(max_T, hidden_dim)
        self.memory_pos_emb = nn.Embedding(prefix_len + max_T, hidden_dim)
        self.environment_embed = nn.Linear(4, hidden_dim)
        self.environment_prop_embed = nn.Linear(6, hidden_dim)
        self.interaction_head = nn.Linear(hidden_dim, 4)

        cluster_encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=4 * hidden_dim,
            batch_first=True,
        )
        self.cluster_encoder = nn.TransformerEncoder(cluster_encoder_layer, num_layers=cluster_encoder_layers)
        self.cluster_norm = nn.LayerNorm(hidden_dim)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=4 * hidden_dim,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        self.out_delay = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())
        self.out_power = nn.Sequential(nn.Linear(hidden_dim, 1))
        self.out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 6),
            nn.Tanh(),
        )
        self.pathcount_head = nn.Sequential(
            nn.Linear(prefix_len * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def _prepare_prompts(self, prompts, pre_train=False):
        if pre_train and self.zero_prompt_positions_for_pretrain:
            prompts = prompts.clone()
            prompts[:, :self.prompt_dim] = 0.0
        return prompts

    def _build_memory(self, prompts, cluster_center_std=None, cluster_pad_mask=None, pre_train=False):
        batch_size = prompts.size(0)
        device = prompts.device
        prompts = self._prepare_prompts(prompts, pre_train=pre_train)
        prefix = self.prompt_to_prefix(prompts).view(batch_size, self.prefix_len, self.hidden_dim)
        prefix_mask = torch.zeros(batch_size, self.prefix_len, dtype=torch.bool, device=device)

        if cluster_center_std is None:
            cluster_tokens = prefix.new_zeros(batch_size, 0, self.hidden_dim)
            cluster_pad_mask = torch.zeros(batch_size, 0, dtype=torch.bool, device=device)
        else:
            cluster_tokens = self.cluster_embed(cluster_center_std)
            if cluster_pad_mask is None:
                cluster_pad_mask = torch.zeros(batch_size, cluster_tokens.size(1), dtype=torch.bool, device=device)
            else:
                cluster_pad_mask = cluster_pad_mask.to(device=device, dtype=torch.bool)

            cluster_pos = self.cluster_pos_emb(torch.arange(cluster_tokens.size(1), device=device)).unsqueeze(0)
            cluster_tokens = cluster_tokens + cluster_pos
            safe_mask = cluster_pad_mask.clone()
            if safe_mask.size(1) > 0:
                all_padded = safe_mask.all(dim=1)
                safe_mask[all_padded, 0] = False
                cluster_tokens[all_padded] = 0.0
            cluster_tokens = self.cluster_encoder(cluster_tokens, src_key_padding_mask=safe_mask)
            cluster_tokens = self.cluster_norm(cluster_tokens)

        memory = torch.cat([prefix, cluster_tokens], dim=1)
        memory_pos = self.memory_pos_emb(torch.arange(memory.size(1), device=device)).unsqueeze(0)
        memory = memory + memory_pos
        memory_mask = torch.cat([prefix_mask, cluster_pad_mask], dim=1)
        return prefix, memory, memory_mask

    def forward(self, prompts, paths, interactions, environment_properties, environment, pre_train, cluster_center_std=None, cluster_pad_mask=None):
        batch_size, path_len, _ = paths.shape
        prefix, memory, memory_mask = self._build_memory(
            prompts,
            cluster_center_std=cluster_center_std,
            cluster_pad_mask=cluster_pad_mask,
            pre_train=pre_train,
        )

        env_embedding = self.environment_embed(environment).unsqueeze(1)
        env_prop_embedding = self.environment_prop_embed(environment_properties)

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
        x = torch.cat([env_embedding, env_prop_embedding, x], dim=1)

        total_len = x.size(1)
        pos = self.pos_emb(torch.arange(total_len, device=x.device)).unsqueeze(0)
        x = x + pos
        causal_mask = torch.triu(torch.ones(total_len, total_len, device=x.device), diagonal=1).bool()

        h = self.decoder(
            tgt=x,
            memory=memory,
            tgt_mask=causal_mask,
            memory_key_padding_mask=memory_mask,
        )
        h_paths = h[:, -path_len:, :]

        out = self.out(h_paths)
        delay_pred = self.out_delay(h_paths).squeeze(-1)
        power_pred = self.out_power(h_paths).squeeze(-1)
        phase_sin_pred = out[:, :, 0]
        phase_cos_pred = out[:, :, 1]
        az_sin_pred = out[:, :, 2]
        az_cos_pred = out[:, :, 3]
        el_sin_pred = out[:, :, 4]
        el_cos_pred = out[:, :, 5]

        phase_pred = torch.atan2(phase_sin_pred, phase_cos_pred)
        az_pred = torch.atan2(az_sin_pred, az_cos_pred)
        el_pred = torch.atan2(el_sin_pred, el_cos_pred)
        interaction_logits = self.interaction_head(h_paths)
        pathcounts = self.pathcount_head(prefix.reshape(batch_size, -1))

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


def generate_paths_cluster(model, prompt, env, env_prop, cluster_center_std=None, cluster_pad_mask=None, max_steps=25):
    model.eval()
    device = next(model.parameters()).device
    prompt = prompt.unsqueeze(0).to(device)
    env = env.unsqueeze(0).to(device)
    env_prop = env_prop.unsqueeze(0).to(device)
    cur = torch.zeros(1, 1, 5, device=device)
    inter_str = -1 * torch.ones(1, 1, 4, device=device)
    outputs = []
    outputs_inter = []

    for _ in range(max_steps):
        d, p, s, c, ph, az_s, az_c, az, el_s, el_c, el, pathcounts, inter_logits = model(
            prompt,
            cur,
            inter_str,
            env_prop,
            env,
            False,
            cluster_center_std=cluster_center_std,
            cluster_pad_mask=cluster_pad_mask,
        )
        d_t = d[:, -1]
        p_t = p[:, -1]
        ph_t = ph[:, -1]
        az_t = az[:, -1]
        el_t = el[:, -1]
        inter_pred_t = (torch.sigmoid(inter_logits[:, -1]) > 0.5).float()
        next_path = torch.stack([d_t, p_t, ph_t, az_t, el_t], dim=-1)
        outputs.append(next_path)
        outputs_inter.append(inter_pred_t)
        cur = torch.cat([cur, next_path.unsqueeze(1)], dim=1)
        inter_str = torch.cat([inter_str, inter_pred_t.unsqueeze(1)], dim=1)

    return (
        torch.stack(outputs, dim=1).squeeze(0).detach().cpu(),
        pathcounts.detach().cpu(),
        torch.stack(outputs_inter, dim=1).squeeze(0).detach().cpu(),
    )


def evaluate_model(model, val_loader, train_rx_pos, train_cluster_center_std, train_valid_len, max_generate=25):
    model.eval()
    delay_errors, power_errors, phase_errors = [], [], []
    az_errors, el_errors, path_length_rmses = [], [], []
    delay_maes, power_maes, phase_maes = [], [], []
    az_maes, el_maes, path_length_maes = [], [], []

    with torch.no_grad():
        for prompts, paths, path_lengths, interactions, env, env_prop, path_padding_mask in tqdm(val_loader, desc="Evaluating"):
            prompts = prompts.to(DEVICE)
            paths = paths.to(DEVICE)
            path_lengths = path_lengths.to(DEVICE)
            env = env.to(DEVICE)
            env_prop = env_prop.to(DEVICE)

            cluster_center_std, cluster_pad_mask = lookup_cluster_center_std_by_position(
                prompts,
                train_rx_pos,
                train_cluster_center_std,
                train_valid_len,
                DEVICE,
            )

            for batch_idx in range(prompts.size(0)):
                pred, path_lengths_pred, _ = generate_paths_cluster(
                    model,
                    prompts[batch_idx],
                    env[batch_idx],
                    env_prop[batch_idx],
                    cluster_center_std=cluster_center_std[batch_idx:batch_idx + 1],
                    cluster_pad_mask=cluster_pad_mask[batch_idx:batch_idx + 1],
                    max_steps=max_generate,
                )
                pred = pred.to(DEVICE)
                n_valid = int(round(path_lengths[batch_idx].item() * 25))
                gt = paths[batch_idx][1:1 + n_valid, :5]
                t = min(len(gt), len(pred))
                gt = gt[:t]
                pred = pred[:t]

                delay_rmse = torch.mean((pred[:, 0] - gt[:, 0]) ** 2).sqrt().item()
                delay_mae = torch.mean(torch.abs(pred[:, 0] - gt[:, 0])).item()
                power_rmse = torch.mean((pred[:, 1] / 0.01 - gt[:, 1] / 0.01) ** 2).sqrt().item()
                power_mae = torch.mean(torch.abs(pred[:, 1] / 0.01 - gt[:, 1] / 0.01)).item()

                phase_diff = (pred[:, 2] / (np.pi / 180) - gt[:, 2] / (np.pi / 180) + 180) % 360 - 180
                az_diff = (pred[:, 3] / (np.pi / 180) - gt[:, 3] / (np.pi / 180) + 180) % 360 - 180
                el_diff = (pred[:, 4] / (np.pi / 180) - gt[:, 4] / (np.pi / 180) + 180) % 360 - 180

                phase_rmse = torch.mean(phase_diff ** 2).sqrt().item()
                phase_mae = torch.mean(torch.abs(phase_diff)).item()
                az_rmse = torch.mean(az_diff ** 2).sqrt().item()
                az_mae = torch.mean(torch.abs(az_diff)).item()
                el_rmse = torch.mean(el_diff ** 2).sqrt().item()
                el_mae = torch.mean(torch.abs(el_diff)).item()

                length_rmse = torch.mean((path_lengths_pred.squeeze() - path_lengths[batch_idx].squeeze()) ** 2).sqrt().item()
                length_mae = torch.mean(torch.abs(path_lengths_pred.squeeze() - path_lengths[batch_idx].squeeze())).item()

                delay_errors.append(delay_rmse)
                power_errors.append(power_rmse)
                phase_errors.append(phase_rmse)
                az_errors.append(az_rmse)
                el_errors.append(el_rmse)
                path_length_rmses.append(length_rmse)
                delay_maes.append(delay_mae)
                power_maes.append(power_mae)
                phase_maes.append(phase_mae)
                az_maes.append(az_mae)
                el_maes.append(el_mae)
                path_length_maes.append(length_mae)

    return {
        "delay_rmse": float(np.mean(delay_errors)),
        "power_rmse": float(np.mean(power_errors)),
        "phase_rmse": float(np.mean(phase_errors)),
        "az_rmse": float(np.mean(az_errors)),
        "el_rmse": float(np.mean(el_errors)),
        "path_length_rmse": float(np.mean(path_length_rmses)),
        "delay_mae": float(np.mean(delay_maes)),
        "power_mae": float(np.mean(power_maes)),
        "phase_mae": float(np.mean(phase_maes)),
        "az_mae": float(np.mean(az_maes)),
        "el_mae": float(np.mean(el_maes)),
        "path_length_mae": float(np.mean(path_length_maes)),
    }


def load_checkpoint(model, checkpoint_path, strict=True):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state, strict=strict)
    epoch = checkpoint.get("epoch", -1)
    best_val_loss = checkpoint.get("best_val_loss", None)
    print(f"Loaded checkpoint from {checkpoint_path} (epoch {epoch})")
    return epoch, best_val_loss


def freeze_for_prompt_cross_attn_adaptation(model):
    # Start from the pretrained masked-user-position decoder and only adapt
    # the geometry-to-prefix bridge plus decoder cross-attention.
    for param in model.parameters():
        param.requires_grad = False

    model.prompt_to_prefix.requires_grad_(True)

    for layer in model.decoder.layers:
        for param in layer.multihead_attn.parameters():
            param.requires_grad = True


def unfreeze_all(model):
    for param in model.parameters():
        param.requires_grad = True


def train_with_interactions(model, train_loader, val_loader, train_data, config, checkpoint_path):
    warmup_epochs = max(0, min(config.get("cross_attn_warmup_epochs", 0), config["epochs"]))
    if warmup_epochs > 0:
        freeze_for_prompt_cross_attn_adaptation(model)
        print(
            "Stage 1: finetuning only prompt_to_prefix and decoder cross-attention "
            f"for {warmup_epochs} epoch(s). Trainable parameters: {count_parameters(model)}"
        )
    else:
        unfreeze_all(model)
        print(f"Stage 1 skipped: training all parameters immediately. Trainable parameters: {count_parameters(model)}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["LR"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=config["scheduler_t0"], T_mult=1, eta_min=1e-8
    )

    cluster_centers, cluster_stds = compute_feature_kmeans_cluster_stats_for_dataset(
        train_data,
        feature_keys=config["cluster_features"],
        max_path_len=config["max_path_len_clusters"],
        n_clusters=config["n_clusters"],
    )
    train_rx_pos, train_cluster_center_std, train_valid_len = precompute_train_cluster_center_std_sequences(
        train_data,
        cluster_centers=cluster_centers,
        cluster_stds=cluster_stds,
        feature_keys=config["cluster_features"],
        max_path_len=config["max_path_len_clusters"],
    )
    print(f"Prepared local cluster lookup with features={config['cluster_features']}")

    best_val_loss = float("inf")
    for epoch in range(config["epochs"]):
        if epoch == warmup_epochs and warmup_epochs > 0:
            unfreeze_all(model)
            print(
                "Stage 2: unfreezing the full model after prompt/cross-attention adaptation. "
                f"Trainable parameters: {count_parameters(model)}"
            )

        model.train()
        train_losses = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]", leave=False)
        for prompts, paths, path_lengths, interactions, env, env_prop, path_padding_mask in pbar:
            prompts = prompts.to(DEVICE)
            paths = paths.to(DEVICE)
            path_lengths = path_lengths.to(DEVICE)
            interactions = interactions.to(DEVICE)
            env = env.to(DEVICE)
            env_prop = env_prop.to(DEVICE)
            path_padding_mask = path_padding_mask.to(DEVICE)

            cluster_center_std, cluster_pad_mask = lookup_cluster_center_std_by_position(
                prompts, train_rx_pos, train_cluster_center_std, train_valid_len, DEVICE
            )

            paths_in = paths[:, :-1, :]
            if config["TARGET_NOISE_PROB"] > 0:
                paths_in = add_noise_to_paths(
                    paths_in,
                    path_padding_mask[:, :-1],
                    p_noise=config["TARGET_NOISE_PROB"],
                    noise_params=config.get("TARGET_NOISE_PARAMS"),
                )
            interactions_in = interactions[:, :-1, :]
            paths_out = paths[:, 1:, :]
            interactions_out = interactions[:, 1:, :]

            outputs = model(
                prompts,
                paths_in,
                interactions_in,
                env_prop,
                env,
                False,
                cluster_center_std=cluster_center_std,
                cluster_pad_mask=cluster_pad_mask,
            )
            total_loss, loss_delay, loss_power, loss_phase, loss_az, loss_el, _, loss_interaction, _ = masked_loss(
                *outputs,
                paths_out,
                path_lengths,
                interactions_out,
                pad_value=config["PAD_VALUE"],
                interaction_weight=config["interaction_weight"],
                path_padding_mask=path_padding_mask,
            )

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(total_loss.item())
            pbar.set_postfix(loss=f"{total_loss.item():.4f}", delay=f"{loss_delay.item():.4f}", power=f"{loss_power.item():.4f}", inter=f"{loss_interaction.item():.4f}")

        model.eval()
        val_losses = []
        with torch.no_grad():
            for prompts, paths, path_lengths, interactions, env, env_prop, path_padding_mask in tqdm(val_loader, desc=f"Epoch {epoch} [Val]", leave=False):
                prompts = prompts.to(DEVICE)
                paths = paths.to(DEVICE)
                path_lengths = path_lengths.to(DEVICE)
                interactions = interactions.to(DEVICE)
                env = env.to(DEVICE)
                env_prop = env_prop.to(DEVICE)
                path_padding_mask = path_padding_mask.to(DEVICE)

                cluster_center_std, cluster_pad_mask = lookup_cluster_center_std_by_position(
                    prompts, train_rx_pos, train_cluster_center_std, train_valid_len, DEVICE
                )
                paths_in = paths[:, :-1, :]
                interactions_in = interactions[:, :-1, :]
                paths_out = paths[:, 1:, :]
                interactions_out = interactions[:, 1:, :]

                outputs = model(
                    prompts,
                    paths_in,
                    interactions_in,
                    env_prop,
                    env,
                    False,
                    cluster_center_std=cluster_center_std,
                    cluster_pad_mask=cluster_pad_mask,
                )
                total_loss, *_ = masked_loss(
                    *outputs,
                    paths_out,
                    path_lengths,
                    interactions_out,
                    pad_value=config["PAD_VALUE"],
                    interaction_weight=config["interaction_weight"],
                    path_padding_mask=path_padding_mask,
                )
                val_losses.append(total_loss.item())

        avg_train_loss = float(np.mean(train_losses))
        avg_val_loss = float(np.mean(val_losses))
        scheduler.step()
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
            print(f"  Saved best checkpoint at val_loss={best_val_loss:.4f}")

        print(f"Epoch {epoch:02d} | train={avg_train_loss:.4f} | val={avg_val_loss:.4f}")

    load_checkpoint(model, checkpoint_path)
    return evaluate_model(model, val_loader, train_rx_pos, train_cluster_center_std, train_valid_len, max_generate=config["max_generate"])


def append_result(path, row):
    file_exists = os.path.exists(path)
    fieldnames = [
        "scenario",
        "pretrain_checkpoint",
        "delay_rmse",
        "power_rmse",
        "phase_rmse",
        "az_rmse",
        "el_rmse",
        "path_length_rmse",
        "delay_mae",
        "power_mae",
        "phase_mae",
        "az_mae",
        "el_mae",
        "path_length_mae",
    ]
    with open(path, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fieldnames})


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune scenario models from cluster-pretrained checkpoints.")
    parser.add_argument("--scenarios", type=str, default=",".join(DEFAULT_SCENARIOS))
    parser.add_argument("--base-experiment", type=str, default="cluster_noisy_pretrain")
    parser.add_argument("--pretrain-checkpoint-dir", type=str, default="checkpoints20M")
    parser.add_argument("--finetune-checkpoint-dir", type=str, default="checkpoints_finetune_clusters")
    parser.add_argument("--results-csv", type=str, default="pretrain_finetune_clusters_results.csv")
    parser.add_argument("--n-clusters", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--n-layers", type=int, default=8)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--scheduler-t0", type=int, default=20)
    parser.add_argument("--cross-attn-warmup-epochs", type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_args()
    scenarios = [scenario.strip() for scenario in args.scenarios.split(",") if scenario.strip()]
    os.makedirs(args.finetune_checkpoint_dir, exist_ok=True)

    base_checkpoint_name = f"{args.base_experiment}_clusters_{args.n_clusters}_best_model_checkpoint.pth"
    base_checkpoint_path = os.path.join(args.pretrain_checkpoint_dir, base_checkpoint_name)

    for scenario in scenarios:
        print(f"\n######### Finetuning cluster-pretrained model on {scenario} #########")
        dataset = dm.load(scenario)
        config = {
            "BATCH_SIZE": args.batch_size,
            "PAD_VALUE": 0,
            "LR": args.lr,
            "epochs": args.epochs,
            "interaction_weight": 0.01,
            "hidden_dim": args.hidden_dim,
            "n_layers": args.n_layers,
            "n_heads": args.n_heads,
            "n_clusters": args.n_clusters,
            "max_path_len_clusters": 25,
            "cluster_features": ["delay", "power"],
            "TARGET_NOISE_PROB": 0.2,
            "TARGET_NOISE_PARAMS": None,
            "scheduler_t0": args.scheduler_t0,
            "cross_attn_warmup_epochs": args.cross_attn_warmup_epochs,
            "max_generate": 25,
        }

        model = PathDecoderClusterEncoderAttentionFix1(
            hidden_dim=config["hidden_dim"],
            n_layers=config["n_layers"],
            n_heads=config["n_heads"],
            cluster_feature_dim=2 * len(config["cluster_features"]),
            prompt_dim=6,
        ).to(DEVICE)
        print(f"Trainable parameters: {count_parameters(model)}")

        load_checkpoint(model, base_checkpoint_path)

        train_data = PreTrainMySeqDataLoader(dataset, train=True, split_by="user", sort_by="power", pad_value=config["PAD_VALUE"])
        val_data = PreTrainMySeqDataLoader(dataset, train=False, split_by="user", sort_by="power", pad_value=config["PAD_VALUE"])
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=config["BATCH_SIZE"], shuffle=True, collate_fn=train_data.collate_fn)
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=config["BATCH_SIZE"], shuffle=False, collate_fn=val_data.collate_fn)

        finetune_checkpoint = os.path.join(
            args.finetune_checkpoint_dir,
            f"finetune_clusters_{scenario}_{args.base_experiment}_clusters_{args.n_clusters}.pth",
        )
        metrics = train_with_interactions(model, train_loader, val_loader, train_data, config, finetune_checkpoint)
        metrics["scenario"] = scenario
        metrics["pretrain_checkpoint"] = base_checkpoint_path
        append_result(args.results_csv, metrics)
        print(f"{scenario} results: delay_rmse={metrics['delay_rmse']:.4f}, power_rmse={metrics['power_rmse']:.4f}, phase_rmse={metrics['phase_rmse']:.4f}")


if __name__ == "__main__":
    main()
