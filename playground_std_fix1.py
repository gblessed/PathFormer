# %%
# %%
# !pip install DeepMIMO==4.0.0b10

# %%
# %%
# =============================================================================
# 1. IMPORTS AND WARNINGS SETUP
#    - Load necessary PyTorch modules, utilities, and suppress UserWarnings
# =============================================================================
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import torch
from tqdm import tqdm
import math
# from utils import (generate_channels_and_labels, tokenizer_train, tokenizer, make_sample, nmse_loss,
                #    create_train_dataloader, patch_maker, count_parameters, train_lwm)
from collections import defaultdict
import numpy as np
# import pretrained_model  # Assuming this contains the LWM model definition
import matplotlib.pyplot as plt
import warnings
import os
import bisect
# from collections import defaultdict
from tqdm import tqdm
warnings.filterwarnings("ignore", category=UserWarning)
# from utils import *
import deepmimo as dm
from sklearn.metrics import mean_squared_error
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.cluster import KMeans
from models import PathDecoder, GPTPathDecoder
from dataset.dataloaders import PreTrainMySeqDataLoader
from k_means_utils import *
from utils.utils import add_noise_to_paths

from tqdm import tqdm
import torch
import numpy as np
import pandas as pd
import os

csv_log_file = "playground_std_fix1_final_scenario_results.csv"


def parse_args():
    parser = argparse.ArgumentParser(description="Run cluster-conditioned playground training/eval for a single scenario.")
    parser.add_argument("scenario", nargs="?", default="city_47_chicago_3p5")
    parser.add_argument("--csv-log-file", default=csv_log_file)
    parser.add_argument("--checkpoint-dir", default="checkpoints2")
    parser.add_argumet("--n_clusters", type = int, default=5)
    return parser.parse_args()


args = parse_args()
csv_log_file = args.csv_log_file

# %%
# scenario = 'city_89_nairobi_3p5'
scenario = args.scenario

dm.download(scenario)
dataset = dm.load(scenario, )

# %%
dataset.scene.plot()


# %%
dm.info()


# %%
config = {
    "BATCH_SIZE":128,
    "PAD_VALUE": 0,
    "USE_WANDB": False,
    "LR":2e-5,
    "epochs" : 100,
    "interaction_weight": 0.01,
    "experiment": f"noise_std_fix1_enc_direct_{scenario}_interacaction_all_inter_str_dec_all_repeat",
    "hidden_dim": 512,
    "n_layers": 8,
    "n_heads": 8,
    "use_delay_kmeans": True,
    "n_clusters": args.n_clusters,
    "max_path_len_clusters": 10,
    "cluster_features": ["delay", "power"],
    "delay_only_loss": False,
    "TARGET_NOISE_PROB": 0.2,
    "TARGET_NOISE_PARAMS": None,
    "use_cluster_conditioning": True,
    "use_cluster_mlp_head": False,
    "pretrained_checkpoint": "checkpoints2/noise_enc_direct_city_47_chicago_3p5_interacaction_all_inter_str_dec_all_repeat_best_model_checkpoint.pth",
}




# %%



# %%


# %%
train_data  = PreTrainMySeqDataLoader(dataset, train=True, split_by="user", sort_by="power", pad_value=config["PAD_VALUE"])

train_loader = torch.utils.data.DataLoader(
    dataset     = train_data,
    batch_size  = config['BATCH_SIZE'],
    shuffle     = True,
    collate_fn= train_data.collate_fn
    )
val_data  = PreTrainMySeqDataLoader(dataset, train=False, split_by="user", sort_by="power", pad_value=config["PAD_VALUE"])
val_loader = torch.utils.data.DataLoader(
    dataset     = val_data,
    batch_size  = config['BATCH_SIZE'],
    shuffle     = False,
    collate_fn= val_data.collate_fn
    )

for item in train_loader:
    print(f"Prompt shape: {item[0].shape}, Paths shape: {item[1].shape}, Num paths shape: {item[2].shape}")
    
    break


# %%
print("No. of Train Points   : ", train_data.__len__())
print("Batch Size           : ", config["BATCH_SIZE"])
print("Train Batches        : ", train_loader.__len__())
print("No. of Train Points   : ", val_data.__len__())
print("Val Batches          : ", val_loader.__len__())

def compute_stop_metrics(path_count, targets, pad_value=0):
    """

    Args:

    """

    rmse = np.sqrt(mean_squared_error(path_count.cpu().numpy(), targets.squeeze().cpu().numpy()))
    
    return rmse 


def compute_interaction_metrics_from_binary_predictions(interaction_preds, interaction_targets):
    interaction_mask = (interaction_targets[:, :, 0] != -1)

    if not interaction_mask.any():
        return 0.0, 0.0

    valid_preds = interaction_preds[interaction_mask].int().cpu().numpy()
    valid_targets = interaction_targets[interaction_mask].int().cpu().numpy()

    accuracy = accuracy_score(valid_targets.reshape(-1), valid_preds.reshape(-1))
    f1 = f1_score(valid_targets.reshape(-1), valid_preds.reshape(-1), zero_division=0)
    return accuracy, f1


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


def compute_feature_kmeans_cluster_stats(base_dataset, feature_keys, max_path_len=26, n_clusters=4, random_state=42):
    n_features = len(feature_keys)
    _, _, _, step_vectors = _collect_step_vectors_from_base_dataset(base_dataset, feature_keys, max_path_len)
    centers = np.zeros((max_path_len, n_clusters, n_features), dtype=np.float32)
    stds = np.zeros((max_path_len, n_clusters, n_features), dtype=np.float32)
    for t in range(max_path_len):
        arr = np.array(step_vectors[t], dtype=np.float32)
        if len(arr) == 0:
            continue
        if len(arr) < n_clusters:
            reps = [arr[min(i, len(arr) - 1)] for i in range(n_clusters)]
            c = np.stack(reps, axis=0)
            s = np.stack([np.std(arr, axis=0)] * n_clusters, axis=0)
        else:
            km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
            labels = km.fit_predict(arr)
            c = km.cluster_centers_.astype(np.float32)
            s = np.zeros_like(c)
            for k in range(n_clusters):
                pts = arr[labels == k]
                if len(pts) > 0:
                    s[k] = np.std(pts, axis=0)
        centers[t] = c
        stds[t] = s
    return centers, stds


def precompute_train_cluster_center_std_sequences(base_dataset, cluster_centers, cluster_stds, feature_keys, max_path_len=26):
    n_features = len(feature_keys)
    rx_xy, seq_vectors, valid_lens, _ = _collect_step_vectors_from_base_dataset(base_dataset, feature_keys, max_path_len)
    n_samples = seq_vectors.shape[0]
    out = np.zeros((n_samples, max_path_len, 2 * n_features), dtype=np.float32)
    for i in range(n_samples):
        valid = int(valid_lens[i])
        for t in range(valid):
            vec = seq_vectors[i, t]
            cc = cluster_centers[t]
            dist = np.linalg.norm(cc - vec[None, :], axis=1)
            nn_idx = int(np.argmin(dist))
            out[i, t, :n_features] = cluster_centers[t, nn_idx]
            out[i, t, n_features:] = cluster_stds[t, nn_idx]
    return rx_xy, out, valid_lens


def lookup_cluster_center_std_by_position(prompts, train_rx_pos, train_cluster_center_std, train_valid_len, device, prompt_rx_slice=(3, 5)):
    batch_size = prompts.shape[0]
    rx_query = prompts[:, prompt_rx_slice[0]:prompt_rx_slice[1]].detach().cpu().numpy()
    max_t = train_cluster_center_std.shape[1]
    feature_dim = train_cluster_center_std.shape[2]
    batch = np.zeros((batch_size, max_t, feature_dim), dtype=np.float32)
    pad_mask = np.ones((batch_size, max_t), dtype=bool)
    if train_rx_pos.shape[0] == 0:
        return (
            torch.tensor(batch, dtype=torch.float32, device=device),
            torch.tensor(pad_mask, dtype=torch.bool, device=device),
        )
    for b in range(batch_size):
        dist = np.sqrt(np.sum((train_rx_pos - rx_query[b]) ** 2, axis=1))
        nn_idx = int(np.argmin(dist))
        batch[b] = train_cluster_center_std[nn_idx]
        valid = int(train_valid_len[nn_idx])
        pad_mask[b, :valid] = False
        pad_mask[b, valid:] = True
    return (
        torch.tensor(batch, dtype=torch.float32, device=device),
        torch.tensor(pad_mask, dtype=torch.bool, device=device),
    )


class PathDecoderClusterCenterStdAdapter(nn.Module):
    def __init__(self, backbone, hidden_dim, cluster_feature_dim, prompt_dim=6):
        super().__init__()
        self.backbone = backbone
        self.cluster_embed = nn.Sequential(
            nn.Linear(cluster_feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.cluster_to_prompt = nn.Linear(hidden_dim, prompt_dim)

    def forward(self, prompts, paths, interactions, cluster_center_std=None, cluster_pad_mask=None):
        batch_size = prompts.shape[0]
        if cluster_center_std is None:
            cluster_summary = torch.zeros(
                batch_size,
                self.cluster_to_prompt.in_features,
                device=prompts.device,
                dtype=prompts.dtype,
            )
        else:
            emb = self.cluster_embed(cluster_center_std)
            if cluster_pad_mask is not None:
                valid = (~cluster_pad_mask).float().unsqueeze(-1)
                denom = valid.sum(dim=1).clamp_min(1.0)
                cluster_summary = (emb * valid).sum(dim=1) / denom
            else:
                cluster_summary = emb.mean(dim=1)
        prompts_aug = prompts + self.cluster_to_prompt(cluster_summary)
        return self.backbone(prompts_aug, paths, interactions)


class PathDecoderClusterEncoderAttentionFix1(nn.Module):
    """Cluster-aware decoder with boolean memory masking and ordered memory tokens."""

    def __init__(
        self,
        prompt_dim=6,
        hidden_dim=512,
        n_layers=6,
        n_heads=4,
        cluster_feature_dim=None,
        max_T=35,
        prefix_len=4,
        pad_value=500,
        cluster_encoder_layers=2,
        zero_prompt_positions_for_pretrain=True,
    ):
        super().__init__()
        self.pad_value = pad_value
        self.hidden_dim = hidden_dim
        self.prompt_dim = prompt_dim
        self.cluster_feature_dim = cluster_feature_dim
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
        self.pos_emb = nn.Embedding(max_T, hidden_dim)
        self.cluster_pos_emb = nn.Embedding(max_T, hidden_dim)
        self.memory_pos_emb = nn.Embedding(prefix_len + max_T, hidden_dim)

        cluster_encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=4 * hidden_dim,
            batch_first=True,
        )
        self.cluster_encoder = nn.TransformerEncoder(cluster_encoder_layer, num_layers=cluster_encoder_layers)
        self.cluster_norm = nn.LayerNorm(hidden_dim)

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
            nn.Linear(hidden_dim, 6),
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
                cluster_pad_mask = torch.zeros(
                    batch_size,
                    cluster_tokens.size(1),
                    dtype=torch.bool,
                    device=device,
                )
            else:
                cluster_pad_mask = cluster_pad_mask.to(device=device, dtype=torch.bool)

            cluster_pos = self.cluster_pos_emb(torch.arange(cluster_tokens.size(1), device=device)).unsqueeze(0)
            cluster_tokens = cluster_tokens + cluster_pos
            safe_cluster_pad_mask = cluster_pad_mask.clone()
            if safe_cluster_pad_mask.size(1) > 0:
                all_padded = safe_cluster_pad_mask.all(dim=1)
                safe_cluster_pad_mask[all_padded, 0] = False
                cluster_tokens[all_padded] = 0.0
            cluster_tokens = self.cluster_encoder(
                cluster_tokens,
                src_key_padding_mask=safe_cluster_pad_mask,
            )
            cluster_tokens = self.cluster_norm(cluster_tokens)

        memory = torch.cat([prefix, cluster_tokens], dim=1)
        memory_pos = self.memory_pos_emb(torch.arange(memory.size(1), device=device)).unsqueeze(0)
        memory = memory + memory_pos
        memory_mask = torch.cat([prefix_mask, cluster_pad_mask], dim=1)
        return prefix, memory, memory_mask

    def forward(self, prompts, paths, interactions, cluster_center_std=None, cluster_pad_mask=None, pre_train=False):
        batch_size, total_len, _ = paths.shape
        prefix, memory, memory_mask = self._build_memory(
            prompts,
            cluster_center_std=cluster_center_std,
            cluster_pad_mask=cluster_pad_mask,
            pre_train=pre_train,
        )

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

        h_paths = self.decoder(
            tgt=x,
            memory=memory,
            tgt_mask=causal_mask,
            memory_key_padding_mask=memory_mask,
        )

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

        prefix_flat = prefix.reshape(batch_size, -1)
        pathcounts = self.pathcount_head(prefix_flat)

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

    def forward_hidden(self, prompts, paths, interactions, cluster_center_std=None, cluster_pad_mask=None, pre_train=False):
        batch_size, total_len, _ = paths.shape
        prefix, memory, memory_mask = self._build_memory(
            prompts,
            cluster_center_std=cluster_center_std,
            cluster_pad_mask=cluster_pad_mask,
            pre_train=pre_train,
        )

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

        h_paths = self.decoder(
            tgt=x,
            memory=memory,
            tgt_mask=causal_mask,
            memory_key_padding_mask=memory_mask,
        )
        prefix_flat = prefix.reshape(batch_size, -1)
        return h_paths, prefix_flat


def generate_paths_no_env_batch(model, prompts, max_steps=25, stop_threshold=0.5, cluster_center_std=None, cluster_pad_mask=None):
    model.eval()
    device = next(model.parameters()).device
    prompts = prompts.to(device)
    batch_size = prompts.size(0)
    cur = torch.zeros(batch_size, 1, 5, device=device)
    inter_str = -1 * torch.ones(batch_size, 1, 4, device=device)
    outputs = []
    outputs_inter_str = []

    for _ in range(max_steps):
        d, p, s, c, ph, az_s, az_c, az, el_s, el_c, el, pathcounts, inter_str_logits = model(
            prompts,
            cur,
            inter_str,
            cluster_center_std=cluster_center_std,
            cluster_pad_mask=cluster_pad_mask,
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



def evaluate_model(model, val_loader, max_generate=26, log_to_wandb=False, pad_value=0, data_stats=None,
                   cluster_lookup_data=None):
    """
    cluster_lookup_data: (train_rx_pos, train_cluster_center_std, train_valid_len) for NN lookup by (x,y)
    """
    model.eval()
    device = next(model.parameters()).device

    delay_errors = []
    power_errors = []
    phase_errors = []
    path_length_rmses = []



    delay_maes = []
    power_maes = []
    phase_maes = []
    path_length_maes = []
    # AoA metrics
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
            path_padding_mask = path_padding_mask.cuda()
            cluster_center_std, cluster_pad_mask = None, None
            if cluster_lookup_data is not None:
                train_rx_pos, train_cluster_center_std, train_valid_len = cluster_lookup_data
                cluster_center_std, cluster_pad_mask = lookup_cluster_center_std_by_position(
                    prompts, train_rx_pos, train_cluster_center_std, train_valid_len, device
                )

            generated, path_lengths_pred, inter_str_pred = generate_paths_no_env_batch(
                model,
                prompts,
                max_steps=max_generate,
                cluster_center_std=cluster_center_std,
                cluster_pad_mask=cluster_pad_mask,
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
                # Use path length for valid positions (mask-based; pad value is 0)
                n_valid = int(round(path_lengths[b].item() * 25))
                gt = paths[b][1:1 + n_valid, :5]
                gt_interactions = interactions[b][1:1 + n_valid, :]

                T = min(len(gt), generated.size(1))
                pred = generated[b, :T]
                gt = gt[:T]
                pred_interactions = inter_str_pred[b, :T]
                gt_interactions = gt_interactions[:T].detach().cpu()

                valid_interaction_mask = (gt_interactions[:, 0] != -1)
                if valid_interaction_mask.any():
                    interaction_targets_all.append(gt_interactions[valid_interaction_mask].numpy().astype(np.int32))
                    interaction_preds_all.append(pred_interactions[valid_interaction_mask].numpy().astype(np.int32))

                delay_pred = pred[:,0]
                delay = gt[:,0]

                power_pred = pred[:,1]
                power = gt[:,1]

                phase_pred = pred[:,2]
                phase = gt[:,2]

                aoa_az_pred = pred[:,3]
                aoa_az = gt[:,3]
              
                aoa_el_pred = pred[:,4]
                aoa_el = gt[:,4]

                # Denormalize when data_stats provided (model was trained on normalized data)
                if data_stats is not None:
                    def denorm(x, key):
                        if key not in data_stats:
                            return x
                        s, m = data_stats[key]["std"], data_stats[key]["mean"]
                        return x * s + m
                    delay_pred = denorm(pred[:,0], "delay")
                    delay = denorm(gt[:,0], "delay")
                    power_pred = denorm(pred[:,1], "power")
                    power = denorm(gt[:,1], "power")
                    phase_pred = denorm(pred[:,2], "phase")
                    phase = denorm(gt[:,2], "phase")
                    aoa_az_pred = denorm(pred[:,3], "aoa_az")
                    aoa_az = denorm(gt[:,3], "aoa_az")
                    aoa_el_pred = denorm(pred[:,4], "aoa_el")
                    aoa_el = denorm(gt[:,4], "aoa_el")


                # ---- Compute Metrics ----
                delay_rmse = torch.mean((delay_pred - delay)**2).sqrt().item()
                delay_mae = torch.mean(torch.abs(delay_pred - delay)).item()

                power_rmse = torch.mean((power_pred/0.01 -power/0.01)**2).sqrt().item()
                power_mae = torch.mean((torch.abs(power_pred/0.01 -power/0.01))).item()


                # Phase errors
                y_hat_angles = (phase_pred / (np.pi/180))
                y_angles = (phase / (np.pi/180))
                phase_circular_dist = (y_hat_angles - y_angles + 180) % 360 - 180
                phase_rmse = torch.mean(phase_circular_dist**2).sqrt().item()
                phase_mae = torch.mean(torch.abs(phase_circular_dist)).item()

                # AoA azimuth errors
                y_hat_az = (aoa_az_pred / (np.pi/180))
                y_az = (aoa_az / (np.pi/180))

                az_circular_dist = (y_hat_az - y_az + 180) % 360 - 180
                az_rmse = torch.mean(az_circular_dist**2).sqrt().item()
                az_mae = torch.mean(torch.abs(az_circular_dist)).item()

                # AoA elevation errors
                y_hat_el = (aoa_el_pred / (np.pi/180))
                y_el = (aoa_el/ (np.pi/180))
                el_circular_dist = (y_hat_el - y_el + 180) % 360 - 180
                el_rmse = torch.mean(el_circular_dist**2).sqrt().item()
                el_mae = torch.mean(torch.abs(el_circular_dist)).item()

                # Path length RMSE (path_lengths in [0,1]; pathcounts raw; use same scale)
                pl_pred = path_lengths_pred[b].squeeze()
                pl_gt = path_lengths[b].squeeze()
                length_rmse = (torch.mean((pl_pred - pl_gt)**2)).sqrt().item()
                length_mae = (torch.mean(torch.abs(pl_pred - pl_gt))).item()


                # Save metrics
                delay_errors.append(delay_rmse)
                power_errors.append(power_rmse)
                phase_errors.append(phase_rmse)
                path_length_rmses.append(length_rmse)
                # AoA
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

                # wandb logging
                if log_to_wandb:
                    wandb.log({
                        "test_delay_rmse": delay_rmse,
                        "test_power_rmse": power_rmse,
                        "test_phase_circ_err": phase_rmse,
                        "test_stop_length_rmse": length_rmse,
                        "test_az_rmse": az_rmse,
                        "test_el_rmse": el_rmse,
                        "test_delay_mae": delay_mae,
                        "test_power_mae": power_mae,
                        "test_phase_circ_err_mae": phase_mae,
                        "test_az_mae": az_mae,
                        "test_el_mae": el_mae,
                        "test_stop_length_mae": length_mae,
                    })

            if batch_delay_rmses:
                outer_bar.set_postfix({
                    "delay_rmse": f"{np.mean(batch_delay_rmses):.3f}",
                    "power_rmse": f"{np.mean(batch_power_rmses):.3f}",
                    "phase_rmse": f"{np.mean(batch_phase_rmses):.3f}",
                    "az_rmse": f"{np.mean(batch_az_rmses):.3f}",
                    "el_rmse": f"{np.mean(batch_el_rmses):.3f}",
                    "length_rmse": f"{np.mean(batch_length_rmses):.3f}",
                })
            

    # ---- Final Aggregated Results ----
    avg_delay = np.mean(delay_errors)
    avg_power = np.mean(power_errors)
    avg_phase = np.mean(phase_errors)
    avg_az = np.mean(az_errors) if len(az_errors) > 0 else 0.0
    avg_el = np.mean(el_errors) if len(el_errors) > 0 else 0.0
    avg_path_length_rmse = np.mean(path_length_rmses)
   
    avg_delay_mae = np.mean(delay_maes)
    avg_power_mae = np.mean(power_maes)
    avg_phase_mae = np.mean(phase_maes)
    avg_az_mae = np.mean(az_maes)
    avg_el_mae = np.mean(el_maes)
    avg_path_length_mae= np.mean(path_length_maes)
    if interaction_targets_all:
        interaction_targets_np = np.concatenate(interaction_targets_all, axis=0)
        interaction_preds_np = np.concatenate(interaction_preds_all, axis=0)
        avg_interaction_accuracy = accuracy_score(
            interaction_targets_np.reshape(-1),
            interaction_preds_np.reshape(-1),
        )
        avg_interaction_f1 = f1_score(
            interaction_targets_np.reshape(-1),
            interaction_preds_np.reshape(-1),
            zero_division=0,
        )
    else:
        avg_interaction_accuracy = 0.0
        avg_interaction_f1 = 0.0

    print("\n=================  Final EVALUATION RESULTS =================")
    print(f"Delay RMSE           : {avg_delay:.4f} µs")
    print(f"Power RMSE           : {avg_power:.4f} dB")
    print(f"Phase RMSE           : {avg_phase:.4f} degrees")
    print(f"AoA Azimuth RMSE     : {avg_az:.4f} degrees")
    print(f"AoA Elevation RMSE   : {avg_el:.4f} degrees")
    print(f"Path Length RMSE     : {avg_path_length_rmse:.4f}")
    print(f"Interaction Accuracy : {avg_interaction_accuracy:.4f}")
    print(f"Interaction F1       : {avg_interaction_f1:.4f}")
        
    print(f"Delay MAE           : {avg_delay_mae:.4f} µs")
    print(f"Power MAE           : {avg_power_mae:.4f} dB")
    print(f"Phase MAE           : {avg_phase_mae:.4f} degrees")
    print(f"AoA Azimuth MAE     : {avg_az_mae:.4f} degrees")
    print(f"AoA Elevation MAE   : {avg_el_mae:.4f} degrees")
    print(f"Path Length MAE     : {avg_path_length_mae:.4f}")
    print("=====================================================\n")

    if log_to_wandb:
        wandb.run.summary["test_delay_rmse"] = avg_delay
        wandb.run.summary["test_power_rmse"] = avg_power
        wandb.run.summary["test_phase_circ_err"] = avg_phase
        wandb.run.summary["test_path_length_rmse"] = avg_path_length_rmse
        wandb.run.summary["test_interaction_accuracy"] = avg_interaction_accuracy
        wandb.run.summary["test_interaction_f1"] = avg_interaction_f1
        
        wandb.run.summary["test_delay_mae"] = avg_delay_mae
        wandb.run.summary["test_power_mae"] = avg_power_mae
        wandb.run.summary["test_phase_circ_err_mae"] = avg_phase_mae
        wandb.run.summary["test_path_length_mae"] = avg_path_length_mae

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




def show_example(model, val_loader, sample_index=0, k=25, plot=True, cluster_lookup_data=None):
    model.eval()
    batch = next(iter(val_loader))
    prompts, paths, path_lengths, interactions = batch[0], batch[1], batch[2], batch[3]
    path_padding_mask = batch[6] if len(batch) > 6 else None
    
    prompts = prompts.cuda()
    paths = paths.cuda()
    path_lengths = path_lengths.cuda()
    device = next(model.parameters()).device
    
    cluster_center_std_b, cluster_pad_b = None, None
    if cluster_lookup_data is not None:
        prom_b = prompts[sample_index:sample_index+1]
        train_rx_pos, train_cluster_center_std, train_valid_len = cluster_lookup_data
        cluster_center_std_b, cluster_pad_b = lookup_cluster_center_std_by_position(
            prom_b, train_rx_pos, train_cluster_center_std, train_valid_len, device
        )
    pred_paths, path_lengths_pred, inter_str_pred = generate_paths_no_env_batch(
        model,
        prompts[sample_index:sample_index+1],
        max_steps=25,
        cluster_center_std=cluster_center_std_b,
        cluster_pad_mask=cluster_pad_b,
    )
    
    pred = pred_paths[0]  # (T,3)
    n_valid = int(round(path_lengths[sample_index].item() * 25))
    gt = paths[sample_index][1:1 + n_valid, :3]  # Extract only 3D components (T,3)

    print("\n--- Ground Truth Length {} ".format(len(gt)))

    print("\n--- Model Predict Length (first {} paths) ---".format(path_lengths_pred.item()))
    
    
    print(gt[:k])
    print(pred[:k])

    if plot:
        T = min(len(gt), len(pred))
        # print("len_path", len(pred), "actual = ", T)
        pred = pred[:T]
        gt = gt[:T]

        fig, axs = plt.subplots(3,1, figsize=(10,12))

        axs[0].plot(gt[:,0].cpu(), label="GT Delay", marker='o')
        axs[0].plot(pred[:,0].cpu(), label="Pred Delay", marker='x')
        axs[0].set_title("Path Delay (µs)")
        axs[0].legend()

        axs[1].plot(gt[:,1].cpu()*0.01, label="GT Power", marker='o')
        axs[1].plot(pred[:,1].cpu()*0.01, label="Pred Power", marker='x')
        axs[1].set_title("Path Power dB")
        axs[1].legend()

        axs[2].plot(gt[:,2].cpu()/(np.pi/180), label="GT Phase", marker='o')
        axs[2].plot(pred[:,2].cpu()/(np.pi/180), label="Pred Phase", marker='x')
        axs[2].set_title("Path Phase (degrees)")

        axs[2].legend()

        plt.tight_layout()
        plt.show()


# def evaluate_generation(val_loader, n_samples=3):
#     model.eval()
#     for i, (prompts, paths) in enumerate(val_loader):
#         if i >= n_samples:
#             break
#         pred, path_lengths_pred = generate_paths_no_env(model, prompts[0])  # autoregressive generation
#         print(f"path lengths pred: {path_lengths_pred[0]}")
#         print(f"\nSample {i}")
#         print("GT paths (first 5):")
#         print(paths[0][:5])
#         print("Predicted paths (first 5):")
#         print(pred[0][:5])

# # %%
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PathDecoderClusterMLPHead(nn.Module):
    """
    Frozen PathDecoder backbone + MLP head that takes (hidden_state, cluster_embedding) per step.
    Load a checkpoint from multiscenario_direct_training (no clusters), freeze backbone, train only this head.
    """
    def __init__(self, backbone, hidden_dim, n_clusters=4, max_path_len_clusters=26):
        super().__init__()
        self.backbone = backbone
        self.hidden_dim = hidden_dim
        self.n_clusters = n_clusters
        self.max_path_len_clusters = max_path_len_clusters
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.cluster_embed = nn.Linear(1, hidden_dim)
        head_in = hidden_dim * 2
        self.out_delay = nn.Linear(head_in, 1)
        self.out_power = nn.Linear(head_in, 1)
        self.out = nn.Sequential(
            nn.Linear(head_in, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 6),
        )
        self.interaction_head = nn.Linear(head_in, 4)
        self.cluster_mlp_head = True

    def forward(self, prompts, paths, interactions, cluster_emb=None, cluster_pad_mask=None):
        B, T, _ = paths.shape
        h_paths, prefix_flat = self.backbone.forward_hidden(prompts, paths, interactions)
        if cluster_emb is None:
            cluster_emb = torch.zeros(B, T, self.hidden_dim, device=h_paths.device, dtype=h_paths.dtype)
        else:
            if cluster_emb.shape[-1] == 1:
                cluster_emb = self.cluster_embed(cluster_emb)
            if cluster_emb.shape[1] != T:
                if cluster_emb.shape[1] < T:
                    pad = torch.zeros(B, T - cluster_emb.shape[1], self.hidden_dim, device=cluster_emb.device, dtype=cluster_emb.dtype)
                    cluster_emb = torch.cat([cluster_emb, pad], dim=1)
                else:
                    cluster_emb = cluster_emb[:, :T]
        concat = torch.cat([h_paths, cluster_emb], dim=-1)
        delay_pred = self.out_delay(concat).squeeze(-1)
        power_pred = self.out_power(concat).squeeze(-1)
        out = self.out(concat)
        phase_sin_pred = out[:, :, 0]
        phase_cos_pred = out[:, :, 1]
        phase_pred = torch.atan2(phase_sin_pred, phase_cos_pred)
        az_sin_pred = out[:, :, 2]
        az_cos_pred = out[:, :, 3]
        el_sin_pred = out[:, :, 4]
        el_cos_pred = out[:, :, 5]
        az_pred = torch.atan2(az_sin_pred, az_cos_pred)
        el_pred = torch.atan2(el_sin_pred, el_cos_pred)
        interaction_logits = self.interaction_head(concat)
        path_length_pred = self.backbone.pathcount_head(prefix_flat)
        return (delay_pred, power_pred, phase_sin_pred, phase_cos_pred, phase_pred,
                az_sin_pred, az_cos_pred, az_pred, el_sin_pred, el_cos_pred, el_pred,
                path_length_pred, interaction_logits)


def train_with_interactions(model, train_loader, val_loader, config, train_data, task=None, cluster_lookup_data=None):
    """
    cluster_lookup_data: (train_rx_pos, train_cluster_center_std, train_valid_len) for NN lookup by (x,y)
    """
    device = next(model.parameters()).device
    best_val_loss = float('inf')


    for epoch in range(config["epochs"]):
        # -------------------- TRAINING --------------------
        model.train()
        train_losses = []
        train_loss_delay = []
        train_loss_power = []
        train_loss_phase = []
        train_loss_path_length = []
        train_loss_interaction = []  # NEW
        train_path_length_rmse = []
        train_loss_az = []
        train_loss_el = []
        train_ch_nmse = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]", leave=False)
        for prompts, paths, path_lengths, interactions, env, env_prop, path_padding_mask in pbar:
            prompts = prompts.cuda()
            paths = paths.cuda()
            path_lengths = path_lengths.cuda()
            interactions = interactions.cuda()
            path_padding_mask = path_padding_mask.cuda()
            cluster_center_std, cluster_pad_mask = None, None
            if cluster_lookup_data is not None:
                train_rx_pos, train_cluster_center_std, train_valid_len = cluster_lookup_data
                cluster_center_std, cluster_pad_mask = lookup_cluster_center_std_by_position(
                    prompts, train_rx_pos, train_cluster_center_std, train_valid_len, device
                )

            paths_in = paths[:, :-1, :]
            p_noise = config.get("TARGET_NOISE_PROB", 0.0)
            if p_noise > 0:
                paths_in = add_noise_to_paths(paths_in, path_padding_mask[:, :-1], p_noise=p_noise,
                                              noise_params=config.get("TARGET_NOISE_PARAMS"))
            interactions_in = interactions[:, :-1, :]

            paths_out = paths[:, 1:, :]
            interactions_out = interactions[:, 1:, :]

            if hasattr(model, "cluster_mlp_head"):
                (delay_pred, power_pred, phase_sin_pred, phase_cos_pred, phase_pred,
                 az_sin_pred, az_cos_pred, az_pred, el_sin_pred, el_cos_pred, el_pred,
                 path_length_pred, interaction_logits) = model(prompts, paths_in, interactions_in,
                                                              cluster_emb=cluster_center_std, cluster_pad_mask=cluster_pad_mask)
            elif hasattr(model, "cluster_to_prompt"):
                (delay_pred, power_pred, phase_sin_pred, phase_cos_pred, phase_pred,
                 az_sin_pred, az_cos_pred, az_pred, el_sin_pred, el_cos_pred, el_pred,
                 path_length_pred, interaction_logits) = model(prompts, paths_in, interactions_in,
                                                              cluster_center_std=cluster_center_std, cluster_pad_mask=cluster_pad_mask)
            else:
                (delay_pred, power_pred, phase_sin_pred, phase_cos_pred, phase_pred,
                 az_sin_pred, az_cos_pred, az_pred, el_sin_pred, el_cos_pred, el_pred,
                 path_length_pred, interaction_logits) = model(prompts, paths_in, interactions_in)

            (total_loss, loss_delay, loss_power, loss_phase,
             loss_az, loss_el, loss_path_length, loss_interaction,loss_channel) = masked_loss(
                delay_pred, power_pred, phase_sin_pred, phase_cos_pred,phase_pred,
                az_sin_pred, az_cos_pred, az_pred, el_sin_pred, el_cos_pred,el_pred,
                path_length_pred, interaction_logits, paths_out, path_lengths,
                interactions_out, finetune=task, pad_value=train_data.pad_value,
                interaction_weight=config.get("interaction_weight", 0.1),
                delay_only=config.get("delay_only_loss", False),
                path_padding_mask=path_padding_mask
            )
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            path_length_rmse = compute_stop_metrics(path_length_pred.detach().squeeze(-1), 
                                                    path_lengths)
            ch_nmse = 0
            if epoch >= 0:
                pass
                # pred_power_linear = 10**( ((power_pred.cpu().detach().numpy())/0.01)/10)
                # pred_delay_secs = delay_pred.cpu().detach().numpy()/ 1e6


                # delay_t = paths_out[:, :, 0].cpu().detach().numpy()
                # power_t = paths_out[:, :, 1].cpu().detach().numpy()
                # phase = paths_out[:, :, 2].cpu().detach().numpy()
                # az = paths_out[:, :, 3].cpu().detach().numpy()
                # el = paths_out[:, :, 4].cpu().detach().numpy()
                # power_linear = 10**( (power_t/0.01)/10)
                # delay_secs = delay_t/ 1e6

                # predicted_channels = mycomputer.compute_channels(pred_power_linear,pred_delay_secs, phase_pred.cpu().detach().numpy(), az_pred.cpu().detach().numpy(), el_pred.cpu().detach().numpy(),kwargs=None  )
                # gt_channels = mycomputer.compute_channels(power_linear,delay_secs, phase, az, el ,kwargs=None )
   


                # ch_nmse = compute_channel_nmse(predicted_channels, gt_channels)
            train_ch_nmse.append(ch_nmse)
            train_losses.append(total_loss.item())
            train_loss_delay.append(loss_delay.item())
            train_loss_power.append(loss_power.item())
            train_loss_phase.append(loss_phase.item())
            train_loss_path_length.append(loss_path_length.item())
            # track aoa losses
            # if 'train_loss_az' not in locals():
            #     train_loss_az = []
            #     train_loss_el = []
            train_loss_az.append(loss_az.item())
            train_loss_el.append(loss_el.item())
            train_loss_interaction.append(loss_interaction.item())  # NEW
            train_path_length_rmse.append(path_length_rmse)
            current_lr = optimizer.param_groups[0]["lr"]
            pbar.set_postfix({
                "loss": f"{total_loss.item():.4f}",
                "delay": f"{loss_delay.item():.4f}",
                
                "power": f"{loss_power.item():.4f}",
                "phase": f"{loss_phase.item():.4f}",
                "az": f"{loss_az.item():.4f}",
                "el": f"{loss_el.item():.4f}",
                "inter": f"{loss_interaction.item():.4f}",  # NEW
                "path_rmse": f"{path_length_rmse:.4f}",
                "ch_nmse":f"{ch_nmse:.4f}",
                "lr": f"{current_lr:.2e}"
            })
        scheduler.step()
        try:
            attn = getattr(model.decoder.layers[-1], "cross_attn_weights", None) if hasattr(model, "decoder") else None
        except Exception:
            attn = None
        # print("Train delay_pred->",delay_pred[0])
        # print("Train actual->",paths_out[0, :, 0])
        avg_train_loss = np.mean(train_losses)
        avg_train_delay = np.mean(train_loss_delay)
        avg_train_power = np.mean(train_loss_power)
        avg_train_phase = np.mean(train_loss_phase)
        avg_train_az = np.mean(train_loss_az) 
        avg_train_el = np.mean(train_loss_el)
        avg_train_path_length = np.mean(train_loss_path_length)
        avg_train_interaction = np.mean(train_loss_interaction)  # NEW
        avg_train_path_length_rmse = np.mean(train_path_length_rmse)

        # -------------------- VALIDATION --------------------
        model.eval()
        val_losses = []
        val_loss_delay = []
        val_loss_power = []
        val_loss_phase = []
        val_loss_path_length = []
        val_loss_interaction = []  # NEW
        val_path_length_rmse = []
        val_loss_az = []
        val_loss_el = []

        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]", leave=False)
            # prepare val aoa loss lists

            for prompts, paths, path_lengths, interactions, env, env_prop, path_padding_mask in pbar:
                prompts = prompts.cuda()
                paths = paths.cuda()
                path_lengths = path_lengths.cuda()
                interactions = interactions.cuda()
                path_padding_mask = path_padding_mask.cuda()
                cluster_center_std, cluster_pad_mask = None, None
                if cluster_lookup_data is not None:
                    train_rx_pos, train_cluster_center_std, train_valid_len = cluster_lookup_data
                    cluster_center_std, cluster_pad_mask = lookup_cluster_center_std_by_position(
                        prompts, train_rx_pos, train_cluster_center_std, train_valid_len, device
                    )

                paths_in = paths[:, :-1, :]
                interactions_in = interactions[:, :-1, :]

                paths_out = paths[:, 1:, :]
                interactions_out = interactions[:, 1:, :]

                if hasattr(model, "cluster_mlp_head"):
                    (delay_pred, power_pred, phase_sin_pred, phase_cos_pred, phase_pred,
                     az_sin_pred, az_cos_pred, az_pred, el_sin_pred, el_cos_pred, el_pred,
                     path_length_pred, interaction_logits) = model(prompts, paths_in, interactions_in,
                                                                  cluster_emb=cluster_center_std, cluster_pad_mask=cluster_pad_mask)
                elif hasattr(model, "cluster_to_prompt"):
                    (delay_pred, power_pred, phase_sin_pred, phase_cos_pred, phase_pred,
                     az_sin_pred, az_cos_pred, az_pred, el_sin_pred, el_cos_pred, el_pred,
                     path_length_pred, interaction_logits) = model(prompts, paths_in, interactions_in,
                                                                  cluster_center_std=cluster_center_std, cluster_pad_mask=cluster_pad_mask)
                else:
                    (delay_pred, power_pred, phase_sin_pred, phase_cos_pred, phase_pred,
                     az_sin_pred, az_cos_pred, az_pred, el_sin_pred, el_cos_pred, el_pred,
                     path_length_pred, interaction_logits) = model(prompts, paths_in, interactions_in)
                
                try:
                    if hasattr(model, "decoder") and hasattr(model.decoder, "layers"):
                        attn = getattr(model.decoder.layers[-1], "cross_attn_weights", None)
                    elif hasattr(model, "backbone") and hasattr(model.backbone, "decoder"):
                        attn = getattr(model.backbone.decoder.layers[-1], "cross_attn_weights", None)
                    else:
                        attn = None
                except Exception:
                    attn = None

                (total_loss, loss_delay, loss_power, loss_phase,
                loss_az, loss_el, loss_path_length, loss_interaction,loss_channel) = masked_loss(
                    delay_pred, power_pred, phase_sin_pred, phase_cos_pred,phase_pred,
                    az_sin_pred, az_cos_pred, az_pred, el_sin_pred, el_cos_pred,el_pred,
                    path_length_pred, interaction_logits, paths_out, path_lengths,
                    interactions_out, finetune=task, pad_value=train_data.pad_value,
                    interaction_weight=config.get("interaction_weight", 0.1),
                    delay_only=config.get("delay_only_loss", False),
                    path_padding_mask=path_padding_mask
                )

                path_length_rmse = compute_stop_metrics(path_length_pred.detach().squeeze(-1), 
                                                       path_lengths)

                val_losses.append(total_loss.item())
                val_loss_delay.append(loss_delay.item())
                val_loss_power.append(loss_power.item())
                val_loss_phase.append(loss_phase.item())
                val_loss_az.append(loss_az.item())
                val_loss_el.append(loss_el.item())
                val_loss_path_length.append(loss_path_length.item())
                val_loss_interaction.append(loss_interaction.item())  # NEW
                val_path_length_rmse.append(path_length_rmse)

                pbar.set_postfix({
                    "val_loss": f"{total_loss.item():.4f}",
                    "inter": f"{loss_interaction.item():.4f}",  # NEW
                })
        # print("Val delay_pred->",delay_pred[0])
        # print("Val actual->",paths_out[0, :, 0])

        
        # print("val attn->",attn[0, 1,])
        avg_val_loss = np.mean(val_losses)
        avg_val_delay = np.mean(val_loss_delay)
        avg_val_power = np.mean(val_loss_power)
        avg_val_phase = np.mean(val_loss_phase)
        avg_val_az = np.mean(val_loss_az) 
        avg_val_el = np.mean(val_loss_el)
        avg_val_path_length = np.mean(val_loss_path_length)
        avg_val_interaction = np.mean(val_loss_interaction)  # NEW
        avg_val_path_length_rmse = np.mean(val_path_length_rmse)

        # scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        # -------------------- CHECKPOINT SAVING --------------------
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': torch.tensor(best_val_loss),
            }, checkpoint_path)
            print(f"  ✓ Best checkpoint saved (val_loss: {best_val_loss:.4f})")

        if config.get("USE_WANDB", False):
            import wandb
            wandb.log({
                "train_loss": avg_train_loss,
                "train_loss_delay": avg_train_delay,
                "train_loss_power": avg_train_power,
                "train_loss_phase": avg_train_phase,
                "train_loss_az": avg_train_az,
                "train_loss_el": avg_train_el,
                "train_loss_path_length": avg_train_path_length,
                "train_loss_interaction": avg_train_interaction,  # NEW
                "train_path_length_rmse": avg_train_path_length_rmse,

                "val_loss": avg_val_loss,
                "val_loss_delay": avg_val_delay,
                "val_loss_power": avg_val_power,
                "val_loss_phase": avg_val_phase,
                "val_loss_az": avg_val_az,
                "val_loss_el": avg_val_el,
                "val_loss_path_length": avg_val_path_length,
                "val_loss_interaction": avg_val_interaction,  # NEW
                "val_path_length_rmse": avg_val_path_length_rmse,
                "epoch": epoch,
                "lr": current_lr,
            })

        print(f"\nEpoch {epoch:02d}")
        print(f"  Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"    Delay: {avg_train_delay:.4f} (val: {avg_val_delay:.4f})")
        print(f"    Power: {avg_train_power:.4f} (val: {avg_val_power:.4f})")
        print(f"    Phase: {avg_train_phase:.4f} (val: {avg_val_phase:.4f})")
        print(f"    Az: {avg_train_az:.4f} (val: {avg_val_az:.4f})")
        print(f"    El: {avg_train_el:.4f} (val: {avg_val_el:.4f})")

        print(f"    Interaction: {avg_train_interaction:.4f} (val: {avg_val_interaction:.4f})")  # NEW
        print(f"    PathLength: {avg_train_path_length:.4f} (val: {avg_val_path_length:.4f})")  # NEW

        print(f"  LR: {current_lr:.3e}")

# model = PathDecoder().to(device)




# %%
if config["USE_WANDB"]:
    import wandb

    wandb.init(
        project="deepmimo-path-decoder",
        config = config
        # config={
        #     "batch_size": train_loader.batch_size,
        #     "split_type": train_data.split_by,
        # }
    )





# %%


# %%
all_scenarios = [scenario]

for scenario in all_scenarios:
# %%
    dataset = dm.load(scenario, )
    print(f"######### Training on Scenario {scenario}  #########")
    config = {
        "BATCH_SIZE": 128,
        "PAD_VALUE": 0,
        "USE_WANDB": False,
        "LR": 2e-5,
        "epochs": 300,
        "interaction_weight": 0.01,
        "experiment": f"noise_std_fix1_enc_direct_{scenario}_interacaction_all_inter_str_dec_all_repeat",
        "hidden_dim": 512,
        "n_layers": 8,
        "n_heads": 8,
        "use_cluster_conditioning": True,
        "n_clusters": 25,
        "max_path_len_clusters": 1,
        "cluster_features": ["delay", "power"],
        "delay_only_loss": False,
        "TARGET_NOISE_PROB": 0.2,
        "TARGET_NOISE_PARAMS": None,
        "use_cluster_mlp_head": False,
        "pretrained_checkpoint": "checkpoints2/noise_enc_direct_city_47_chicago_3p5_interacaction_all_inter_str_dec_all_repeat_best_model_checkpoint.pth",
    }

    train_data  = PreTrainMySeqDataLoader(dataset, train=True, split_by="user", sort_by="power", pad_value=config["PAD_VALUE"], normalizers=None, apply_normalizers=[])
    val_data  = PreTrainMySeqDataLoader(dataset, train=False, split_by="user", sort_by="power", pad_value=config["PAD_VALUE"], normalizers=None, apply_normalizers=[])

    n_clusters = config.get("n_clusters", 4)
    max_path_len_clusters = config.get("max_path_len_clusters", 26)
    cluster_lookup_data = None

    base_model_kwargs = dict(
        prompt_dim=6,
        hidden_dim=config["hidden_dim"],
        n_layers=config["n_layers"],
        n_heads=config["n_heads"],
        pad_value=config["PAD_VALUE"],
    )
    # ckpt_path = config.get("pretrained_checkpoint", "checkpoints2/noise_enc_direct_city_47_chicago_3p5_interacaction_all_inter_str_dec_all_repeat_best_model_checkpoint.pth")

    if config.get("use_cluster_conditioning", True):
        model = PathDecoderClusterEncoderAttentionFix1(
            cluster_feature_dim=2 * len(config.get("cluster_features", ["delay", "power"])),
            **base_model_kwargs,
        ).to(device)
        # if os.path.exists(ckpt_path):
        #     ckpt = torch.load(ckpt_path, map_location=device)
        #     model.load_state_dict(ckpt["model_state_dict"], strict=False)
        #     print(f"Loaded shared decoder weights from {ckpt_path}")
        centers, stds = compute_feature_kmeans_cluster_stats(
            train_data,
            feature_keys=config.get("cluster_features", ["delay", "power"]),
            max_path_len=max_path_len_clusters,
            n_clusters=n_clusters,
        )
        train_rx_pos, train_cluster_center_std, train_valid_len = precompute_train_cluster_center_std_sequences(
            train_data,
            cluster_centers=centers,
            cluster_stds=stds,
            feature_keys=config.get("cluster_features", ["delay", "power"]),
            max_path_len=max_path_len_clusters,
        )
        cluster_lookup_data = (train_rx_pos, train_cluster_center_std, train_valid_len)
        print(f"Prepared cluster center+std lookup for features={config.get('cluster_features', ['delay', 'power'])}")
    elif config.get("use_cluster_mlp_head", False):
        backbone = PathDecoder(**base_model_kwargs).to(device)
        # if os.path.exists(ckpt_path):
        #     ckpt = torch.load(ckpt_path, map_location=device)
        #     backbone.load_state_dict(ckpt["model_state_dict"], strict=True)
        #     print(f"Loaded backbone from {ckpt_path}")
        model = PathDecoderClusterMLPHead(backbone, config["hidden_dim"], n_clusters=n_clusters, max_path_len_clusters=max_path_len_clusters).to(device)
    else:
        model = PathDecoder(**base_model_kwargs).to(device)
        # if os.path.exists(ckpt_path):
        #     ckpt = torch.load(ckpt_path, map_location=device)
        #     model.load_state_dict(ckpt["model_state_dict"], strict=True)
        #     print(f"Loaded backbone from {ckpt_path}")
    print("Total trainable parameters:", count_parameters(model))
    # print(f"train_tx_idx: {train_tx_idx.shape} {train_tx_idx[:6]}")
    # print(f"train_rx_pos: {train_rx_pos.shape} {train_rx_pos[:6]}")
    # print(f"train_valid_len: {train_valid_len.shape} {train_valid_len.max()}")
    # print(f"train_step_means: {train_step_means.shape} {train_step_means[0].T}")



    optimizer = torch.optim.AdamW(model.parameters(), lr=config["LR"])
    # scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=2, mode="min")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=25,      # Restart every 10 epochs
        T_mult=1,    # Double the period after each restart
        eta_min=1e-8 # Minimum LR
    )

    # Initialize best checkpoint tracking (based on path_length loss)

    # scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=2, mode="min")

    checkpoint_path = f"{config['experiment']}_best_model_checkpoint.pth"
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.checkpoint_dir, checkpoint_path)

    train_loader = torch.utils.data.DataLoader(
        dataset     = train_data,
        batch_size  = config['BATCH_SIZE'],
        shuffle     = True,
        collate_fn= train_data.collate_fn
        )
    val_loader = torch.utils.data.DataLoader(
        dataset     = val_data,
        batch_size  = config['BATCH_SIZE'],
        shuffle     = False,
        collate_fn= val_data.collate_fn
        )

    # Train
    train_with_interactions(model, train_loader, val_loader, config, train_data, cluster_lookup_data=cluster_lookup_data)
    
    # %% [markdown]
    # 
    # evaluate_generation(train_loader)
    # 

    # %%
    def load_best_checkpoint(model, checkpoint_path="checkpoints2/best_model_checkpoint.pth"):
        """
        Load the best model checkpoint saved during training.
        
        Args:
            model: The model instance to load the checkpoint into
            checkpoint_path: Path to the checkpoint file
        
        Returns:
            epoch: Epoch at which best checkpoint was saved
            best_val_loss: Best validation loss achieved
        """
        if not os.path.exists(checkpoint_path):
            print(f"Warning: Checkpoint not found at {checkpoint_path}")
            return None, None
        
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint['epoch']
        best_avg_val_loss = checkpoint['best_val_loss']
        
        print(f"✓ Loaded best checkpoint from epoch {epoch} (val_loss: {best_avg_val_loss:.4f})")
        return epoch, best_avg_val_loss
    # torch.serialization.add_safe_globals([np._core.multiarray.scalar])
    # torch.serialization.add_safe_globals([np.dtype])

    # Load best checkpoint for inference/evaluation
    best_epoch, best_loss = load_best_checkpoint(model, checkpoint_path=checkpoint_path)

    # %%
    checkpoint_path

    # %%
    results = evaluate_model(model, val_loader, pad_value=config["PAD_VALUE"], data_stats=None,
                             cluster_lookup_data=cluster_lookup_data)
    # print(results)
    avg_delay, avg_power, avg_phase, avg_az, avg_el, avg_path_length_rmse, avg_interaction_accuracy, avg_interaction_f1, avg_delay_mae, avg_power_mae, avg_phase_mae, avg_az_mae, avg_el_mae, avg_path_length_mae  = results
    # (avg_delay, avg_power, avg_phase, avg_path_length_rmse, 
    #  avg_delay_mae, avg_power_mae, avg_phase_mae, avg_path_length_mae) = results
    scenario_row = {
            "scenario": scenario,
            "delay_rmse": avg_delay,
            "power_rmse": avg_power,
            "phase_rmse": avg_phase,
            "az_rmse": avg_az,
            "el_rmse": avg_el,
            "phase_rmse": avg_phase,
            "path_length_rmse": avg_path_length_rmse,
            "interaction_accuracy": avg_interaction_accuracy,
            "interaction_f1": avg_interaction_f1,
            "delay_mae": avg_delay_mae,
            "power_mae": avg_power_mae,
            "phase_mae": avg_phase_mae,
            "avg_az_mae": avg_az_mae,
            "avg_el_mae": avg_el_mae,
            "path_length_mae": avg_path_length_mae,
            "best_val_loss": best_loss.item() if hasattr(best_loss, 'item') else best_loss
        }

        # 4. Append to CSV
    df = pd.DataFrame([scenario_row])
    # header=not os.path.exists(...) ensures the header is only written once
    df.to_csv(csv_log_file, mode='a', index=False, header=not os.path.exists(csv_log_file))

    print(f"✓ Results for {scenario} saved to {csv_log_file}")
    del dataset, train_loader, val_loader, model


    # %%
    # show_example(model, val_loader, sample_index=24)





# %%
best_epoch, best_loss = load_best_checkpoint(model, checkpoint_path=checkpoint_path)

# %%
checkpoint_path

# %%
results = evaluate_model(model, val_loader, pad_value=config["PAD_VALUE"], data_stats=None,
                            cluster_lookup_data=cluster_lookup_data)

# %%


# %%



# %%


# %%
