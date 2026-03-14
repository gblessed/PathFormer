import os
import csv
from collections import defaultdict

import deepmimo as dm
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

from dataset.dataloaders import PreTrainMySeqDataLoader
from utils.utils import add_noise_to_paths, masked_loss


MAX_PATHS = 25
CSV_LOG_FILE = "neighborhood_final_scenario_results.csv"


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_stop_metrics(path_count, targets):
    return np.sqrt(mean_squared_error(path_count.detach().cpu().numpy(), targets.squeeze().detach().cpu().numpy()))


def _path_token_features(paths, interactions):
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
    inter = interactions.clone()
    inter[inter == -1] = 0
    return torch.cat([x, inter], dim=-1)


class NeighborhoodSmoothPathDecoder(nn.Module):
    """
    Direct decoder with explicit neighborhood priors.
    The prior is a compact spatially smoothed atlas over training users, not raw retrieval.
    """

    def __init__(self, prompt_dim=6, hidden_dim=512, n_layers=6, n_heads=4, max_T=35, prefix_len=4, pad_value=500):
        super().__init__()
        self.pad_value = pad_value
        self.hidden_dim = hidden_dim
        self.prefix_len = prefix_len
        self.max_T = max_T
        self.prompt_feature_dim = 17
        self.neighborhood_feature_dim = 15  # mean(5) + std(5) + interaction_mean(4) + valid_flag(1)

        self.prompt_to_prefix = nn.Linear(self.prompt_feature_dim, prefix_len * hidden_dim)
        self.path_in = nn.Linear(12, hidden_dim)
        self.neighborhood_in = nn.Linear(self.neighborhood_feature_dim, hidden_dim)
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.pos_emb = nn.Embedding(max_T, hidden_dim)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=4 * hidden_dim,
            batch_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        self.interaction_head = nn.Linear(hidden_dim, 4)
        self.out_delay = nn.Linear(hidden_dim, 1)
        self.out_power = nn.Linear(hidden_dim, 1)
        self.out = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 6),
        )
        self.pathcount_head = nn.Sequential(
            nn.Linear(prefix_len * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def _make_prompt_features(self, prompts, smooth_centroid=None):
        tx = prompts[:, :3]
        rx = prompts[:, 3:6]
        delta = rx - tx
        dist = torch.norm(delta, dim=-1, keepdim=True)
        unit_dir = delta / dist.clamp_min(1e-6)
        if smooth_centroid is None:
            smooth_centroid = torch.zeros_like(rx)
        local_offset = rx - smooth_centroid
        return torch.cat([tx, rx, delta, dist, unit_dir, local_offset, torch.log1p(dist)], dim=-1)

    def forward(self, prompts, paths, interactions, smooth_features=None, smooth_mask=None, smooth_centroid=None, **kwargs):
        bsz, tgt_len, _ = paths.shape
        prompt_features = self._make_prompt_features(prompts, smooth_centroid=smooth_centroid)
        prefix = self.prompt_to_prefix(prompt_features).view(bsz, self.prefix_len, self.hidden_dim)

        tgt = self.path_in(_path_token_features(paths, interactions))
        tgt = tgt + self.pos_emb(torch.arange(tgt_len, device=paths.device)).unsqueeze(0)

        if smooth_features is None:
            smooth_features = torch.zeros(
                bsz, self.max_T, self.neighborhood_feature_dim, device=paths.device, dtype=paths.dtype
            )
        if smooth_mask is None:
            smooth_mask = torch.ones(bsz, self.max_T, device=paths.device, dtype=torch.bool)

        smooth_tokens = self.neighborhood_in(smooth_features[:, :self.max_T, :])
        smooth_tokens = smooth_tokens + self.pos_emb(torch.arange(smooth_tokens.size(1), device=paths.device)).unsqueeze(0)

        aligned_smooth = smooth_tokens[:, :tgt_len, :]
        gate = self.fusion_gate(torch.cat([tgt, aligned_smooth], dim=-1))
        tgt = tgt + gate * aligned_smooth

        memory = torch.cat([prefix, smooth_tokens], dim=1)
        prefix_mask = torch.zeros(bsz, self.prefix_len, device=paths.device, dtype=torch.bool)
        memory_mask = torch.cat([prefix_mask, smooth_mask[:, :smooth_tokens.size(1)]], dim=1)
        causal_mask = torch.triu(torch.ones(tgt_len, tgt_len, device=paths.device), diagonal=1).bool()

        h = self.decoder(tgt=tgt, memory=memory, tgt_mask=causal_mask, memory_key_padding_mask=memory_mask)

        out = self.out(h)
        delay_pred = self.out_delay(h).squeeze(-1)
        power_pred = self.out_power(h).squeeze(-1)
        phase_sin_pred = out[:, :, 0]
        phase_cos_pred = out[:, :, 1]
        az_sin_pred = out[:, :, 2]
        az_cos_pred = out[:, :, 3]
        el_sin_pred = out[:, :, 4]
        el_cos_pred = out[:, :, 5]
        phase_pred = torch.atan2(phase_sin_pred, phase_cos_pred)
        az_pred = torch.atan2(az_sin_pred, az_cos_pred)
        el_pred = torch.atan2(el_sin_pred, el_cos_pred)
        interaction_logits = self.interaction_head(h)
        pathcounts = self.pathcount_head(prefix.reshape(bsz, -1))
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


def _to_key(arr):
    return tuple(np.asarray(arr, dtype=np.float32).round(4).tolist())


def _sample_to_fixed(paths, interactions, max_paths=MAX_PATHS):
    path_arr = np.asarray(paths, dtype=np.float32)
    inter_arr = np.asarray(interactions, dtype=np.float32)
    out_paths = np.zeros((max_paths, 5), dtype=np.float32)
    out_inters = np.zeros((max_paths, 4), dtype=np.float32)
    out_mask = np.zeros(max_paths, dtype=bool)
    valid = max(min(path_arr.shape[0] - 1, max_paths), 0)
    if valid > 0:
        out_paths[:valid] = path_arr[1:1 + valid]
        clean_inter = inter_arr[1:1 + valid]
        clean_inter[clean_inter < 0] = 0
        out_inters[:valid] = clean_inter
        out_mask[:valid] = True
    return out_paths, out_inters, out_mask


def build_neighborhood_atlas(train_data, cell_size_xyz=(1.5, 1.5, 1.0), max_paths=MAX_PATHS):
    cell_size_xyz = np.asarray(cell_size_xyz, dtype=np.float32)
    atlas = defaultdict(lambda: defaultdict(lambda: {
        "sum_rx": np.zeros(3, dtype=np.float64),
        "samples": 0,
        "sum_paths": np.zeros((max_paths, 5), dtype=np.float64),
        "sum_sq_paths": np.zeros((max_paths, 5), dtype=np.float64),
        "sum_inter": np.zeros((max_paths, 4), dtype=np.float64),
        "count": np.zeros((max_paths, 1), dtype=np.float64),
    }))

    for idx in tqdm(range(len(train_data)), desc="Building neighborhood atlas", leave=False):
        prompt, paths, _, interactions, *_ = train_data[idx]
        prompt_np = prompt.numpy()
        tx_key = _to_key(prompt_np[:3])
        rx = np.asarray(prompt_np[3:6], dtype=np.float32)
        cell_idx = tuple(np.floor(rx / cell_size_xyz).astype(np.int32).tolist())
        fixed_paths, fixed_inter, fixed_mask = _sample_to_fixed(paths.numpy(), interactions.numpy(), max_paths=max_paths)

        bucket = atlas[tx_key][cell_idx]
        bucket["sum_rx"] += rx
        bucket["samples"] += 1
        mask_float = fixed_mask.astype(np.float64)[:, None]
        bucket["sum_paths"] += fixed_paths * mask_float
        bucket["sum_sq_paths"] += (fixed_paths ** 2) * mask_float
        bucket["sum_inter"] += fixed_inter * mask_float
        bucket["count"] += mask_float

    finalized = {}
    for tx_key, cells in atlas.items():
        centroids = []
        mean_paths = []
        std_paths = []
        mean_inter = []
        valid_masks = []
        support = []
        for _, bucket in cells.items():
            samples = max(bucket["samples"], 1)
            counts = np.clip(bucket["count"], 1.0, None)
            mean_p = bucket["sum_paths"] / counts
            var_p = np.maximum(bucket["sum_sq_paths"] / counts - mean_p ** 2, 0.0)
            std_p = np.sqrt(var_p + 1e-8)
            mean_i = bucket["sum_inter"] / counts
            valid = (bucket["count"][:, 0] > 0).astype(np.float32)

            centroids.append(bucket["sum_rx"] / samples)
            mean_paths.append(mean_p.astype(np.float32))
            std_paths.append(std_p.astype(np.float32))
            mean_inter.append(mean_i.astype(np.float32))
            valid_masks.append(valid.astype(np.float32))
            support.append(float(samples))

        finalized[tx_key] = {
            "centroids": np.stack(centroids, axis=0).astype(np.float32),
            "mean_paths": np.stack(mean_paths, axis=0).astype(np.float32),
            "std_paths": np.stack(std_paths, axis=0).astype(np.float32),
            "mean_inter": np.stack(mean_inter, axis=0).astype(np.float32),
            "valid_masks": np.stack(valid_masks, axis=0).astype(np.float32),
            "support": np.asarray(support, dtype=np.float32),
        }
    return finalized


def query_neighborhood_atlas(prompts, atlas, top_k=4, sigma=1.5, max_paths=MAX_PATHS):
    device = prompts.device
    prompts_np = prompts.detach().cpu().numpy()
    bsz = prompts_np.shape[0]
    smooth_features = np.zeros((bsz, max_paths, 15), dtype=np.float32)
    smooth_mask = np.ones((bsz, max_paths), dtype=bool)
    smooth_centroid = np.zeros((bsz, 3), dtype=np.float32)

    for b in range(bsz):
        tx_key = _to_key(prompts_np[b, :3])
        if tx_key not in atlas:
            continue
        entry = atlas[tx_key]
        rx = prompts_np[b, 3:6]
        centroids = entry["centroids"]
        d2 = np.sum((centroids - rx[None, :]) ** 2, axis=1)
        k_eff = min(top_k, len(d2))
        topk = np.argpartition(d2, kth=k_eff - 1)[:k_eff]
        local_d2 = d2[topk]
        weights = np.exp(-local_d2 / max(2.0 * sigma * sigma, 1e-6))
        weights = weights * np.clip(entry["support"][topk], 1.0, None)
        weights = weights / np.clip(weights.sum(), 1e-8, None)

        centroid = (entry["centroids"][topk] * weights[:, None]).sum(axis=0)
        valid_mass = (entry["valid_masks"][topk] * weights[:, None]).sum(axis=0)
        mean_paths = (entry["mean_paths"][topk] * weights[:, None, None]).sum(axis=0)
        std_paths = (entry["std_paths"][topk] * weights[:, None, None]).sum(axis=0)
        mean_inter = (entry["mean_inter"][topk] * weights[:, None, None]).sum(axis=0)

        valid_flag = (valid_mass > 1e-3).astype(np.float32)[:, None]
        smooth_features[b] = np.concatenate([mean_paths, std_paths, mean_inter, valid_flag], axis=-1)
        smooth_mask[b] = valid_flag[:, 0] < 0.5
        smooth_centroid[b] = centroid.astype(np.float32)

    return (
        torch.tensor(smooth_features, dtype=torch.float32, device=device),
        torch.tensor(smooth_mask, dtype=torch.bool, device=device),
        torch.tensor(smooth_centroid, dtype=torch.float32, device=device),
    )


def generate_paths_with_neighborhood(model, prompt, atlas, max_steps=25, top_k=4, sigma=1.5):
    model.eval()
    device = next(model.parameters()).device
    prompt = prompt.unsqueeze(0).to(device)
    smooth_features, smooth_mask, smooth_centroid = query_neighborhood_atlas(
        prompt, atlas, top_k=top_k, sigma=sigma, max_paths=max_steps
    )

    cur = torch.zeros(1, 1, 5, device=device)
    inter_str = -1 * torch.ones(1, 1, 4, device=device)
    outputs = []
    outputs_inter = []

    for _ in range(max_steps):
        d, p, s, c, ph, az_s, az_c, az, el_s, el_c, el, pathcounts, inter_logits = model(
            prompt,
            cur,
            inter_str,
            smooth_features=smooth_features,
            smooth_mask=smooth_mask,
            smooth_centroid=smooth_centroid,
        )
        d_t = d[:, -1]
        p_t = p[:, -1]
        ph_t = ph[:, -1]
        az_t = az[:, -1]
        el_t = el[:, -1]
        inter_t = (torch.sigmoid(inter_logits[:, -1]) > 0.5).float()

        next_path = torch.stack([d_t, p_t, ph_t, az_t, el_t], dim=-1)
        outputs.append(next_path)
        outputs_inter.append(inter_t)
        cur = torch.cat([cur, next_path.unsqueeze(1)], dim=1)
        inter_str = torch.cat([inter_str, inter_t.unsqueeze(1)], dim=1)

    return (
        torch.stack(outputs, dim=1).squeeze(0).detach().cpu(),
        pathcounts.detach().cpu(),
        torch.stack(outputs_inter, dim=1).squeeze(0).detach().cpu(),
    )


def evaluate_model(model, val_loader, atlas, max_generate=26, top_k=4, sigma=1.5):
    model.eval()
    delay_errors, power_errors, phase_errors = [], [], []
    az_errors, el_errors, path_length_rmses = [], [], []
    delay_maes, power_maes, phase_maes = [], [], []
    az_maes, el_maes, path_length_maes = [], [], []

    with torch.no_grad():
        for prompts, paths, path_lengths, interactions, env, env_prop, path_padding_mask in tqdm(val_loader, desc="Evaluating"):
            prompts = prompts.cuda()
            paths = paths.cuda()
            path_lengths = path_lengths.cuda()

            for b in range(prompts.size(0)):
                generated, path_lengths_pred, _ = generate_paths_with_neighborhood(
                    model, prompts[b], atlas, max_steps=max_generate, top_k=top_k, sigma=sigma
                )
                generated = generated.cuda()
                n_valid = int(round(path_lengths[b].item() * 25))
                gt = paths[b][1:1 + n_valid, :5]
                t = min(len(gt), len(generated))
                pred = generated[:t]
                gt = gt[:t]

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
                length_rmse = torch.mean((path_lengths_pred.squeeze() - path_lengths[b].squeeze()) ** 2).sqrt().item()
                length_mae = torch.mean(torch.abs(path_lengths_pred.squeeze() - path_lengths[b].squeeze())).item()

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

    return (
        np.mean(delay_errors),
        np.mean(power_errors),
        np.mean(phase_errors),
        np.mean(az_errors),
        np.mean(el_errors),
        np.mean(path_length_rmses),
        np.mean(delay_maes),
        np.mean(power_maes),
        np.mean(phase_maes),
        np.mean(az_maes),
        np.mean(el_maes),
        np.mean(path_length_maes),
    )


def train_with_interactions(model, train_loader, val_loader, config, train_data, atlas):
    device = next(model.parameters()).device
    best_val_loss = float("inf")

    for epoch in range(config["epochs"]):
        model.train()
        train_losses = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]", leave=False)
        for prompts, paths, path_lengths, interactions, env, env_prop, path_padding_mask in pbar:
            prompts = prompts.to(device)
            paths = paths.to(device)
            path_lengths = path_lengths.to(device)
            interactions = interactions.to(device)
            path_padding_mask = path_padding_mask.to(device)

            smooth_features, smooth_mask, smooth_centroid = query_neighborhood_atlas(
                prompts,
                atlas,
                top_k=config["atlas_top_k"],
                sigma=config["atlas_sigma"],
                max_paths=config["max_generate"],
            )

            paths_in = paths[:, :-1, :]
            if config.get("TARGET_NOISE_PROB", 0.0) > 0:
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
                smooth_features=smooth_features,
                smooth_mask=smooth_mask,
                smooth_centroid=smooth_centroid,
            )
            total_loss, loss_delay, loss_power, loss_phase, loss_az, loss_el, loss_path_length, loss_interaction, _ = masked_loss(
                *outputs,
                paths_out,
                path_lengths,
                interactions_out,
                pad_value=train_data.pad_value,
                interaction_weight=config["interaction_weight"],
                path_padding_mask=path_padding_mask,
                time_step_weighted=config.get("time_step_weighted", False),
            )

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            train_losses.append(total_loss.item())
            pbar.set_postfix(loss=f"{total_loss.item():.4f}", delay=f"{loss_delay.item():.4f}", power=f"{loss_power.item():.4f}")

        scheduler.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for prompts, paths, path_lengths, interactions, env, env_prop, path_padding_mask in tqdm(val_loader, desc=f"Epoch {epoch} [Val]", leave=False):
                prompts = prompts.to(device)
                paths = paths.to(device)
                path_lengths = path_lengths.to(device)
                interactions = interactions.to(device)
                path_padding_mask = path_padding_mask.to(device)

                smooth_features, smooth_mask, smooth_centroid = query_neighborhood_atlas(
                    prompts,
                    atlas,
                    top_k=config["atlas_top_k"],
                    sigma=config["atlas_sigma"],
                    max_paths=config["max_generate"],
                )

                paths_in = paths[:, :-1, :]
                interactions_in = interactions[:, :-1, :]
                paths_out = paths[:, 1:, :]
                interactions_out = interactions[:, 1:, :]

                outputs = model(
                    prompts,
                    paths_in,
                    interactions_in,
                    smooth_features=smooth_features,
                    smooth_mask=smooth_mask,
                    smooth_centroid=smooth_centroid,
                )
                total_loss, *_ = masked_loss(
                    *outputs,
                    paths_out,
                    path_lengths,
                    interactions_out,
                    pad_value=train_data.pad_value,
                    interaction_weight=config["interaction_weight"],
                    path_padding_mask=path_padding_mask,
                    time_step_weighted=config.get("time_step_weighted", False),
                )
                val_losses.append(total_loss.item())

        avg_train = float(np.mean(train_losses))
        avg_val = float(np.mean(val_losses))
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_val_loss": best_val_loss,
                },
                checkpoint_path,
            )
        print(f"Epoch {epoch:03d} | train={avg_train:.4f} | val={avg_val:.4f}")


def load_best_checkpoint(model, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        return None, None
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint["epoch"], checkpoint["best_val_loss"]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
all_scenarios = [
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

for scenario in all_scenarios[:1]:
    dataset = dm.load(scenario)
    print(f"######### Training on Scenario {scenario} #########")

    config = {
        "BATCH_SIZE": 128,
        "PAD_VALUE": 0,
        "USE_WANDB": False,
        "LR": 2e-5,
        "epochs": 100,
        "interaction_weight": 0.01,
        "experiment": f"neighborhood_smooth_{scenario}_direct",
        "hidden_dim": 512,
        "n_layers": 8,
        "n_heads": 8,
        "time_step_weighted": False,
        "TARGET_NOISE_PROB": 0.2,
        "TARGET_NOISE_PARAMS": None,
        "atlas_cell_size_xyz": (1.5, 1.5, 1.0),
        "atlas_top_k": 4,
        "atlas_sigma": 1.5,
        "max_generate": 25,
    }

    train_data = PreTrainMySeqDataLoader(dataset, train=True, split_by="user", sort_by="power", normalizers=None, apply_normalizers=[], pad_value=config["PAD_VALUE"])
    val_data = PreTrainMySeqDataLoader(dataset, train=False, split_by="user", sort_by="power", normalizers=None, apply_normalizers=[], pad_value=config["PAD_VALUE"])
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=config["BATCH_SIZE"], shuffle=True, collate_fn=train_data.collate_fn)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=config["BATCH_SIZE"], shuffle=False, collate_fn=val_data.collate_fn)

    atlas = build_neighborhood_atlas(train_data, cell_size_xyz=config["atlas_cell_size_xyz"], max_paths=config["max_generate"])
    model = NeighborhoodSmoothPathDecoder(
        hidden_dim=config["hidden_dim"],
        n_layers=config["n_layers"],
        n_heads=config["n_heads"],
        max_T=config["max_generate"],
        pad_value=config["PAD_VALUE"],
    ).to(device)
    print("Total trainable parameters:", count_parameters(model))

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["LR"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=25, T_mult=1, eta_min=1e-8)

    checkpoint_path = os.path.join("checkpoints2", f"{config['experiment']}_best_model_checkpoint.pth")
    os.makedirs("checkpoints2", exist_ok=True)

    train_with_interactions(model, train_loader, val_loader, config, train_data, atlas)
    best_epoch, best_loss = load_best_checkpoint(model, checkpoint_path)

    results = evaluate_model(
        model,
        val_loader,
        atlas,
        max_generate=config["max_generate"],
        top_k=config["atlas_top_k"],
        sigma=config["atlas_sigma"],
    )
    avg_delay, avg_power, avg_phase, avg_az, avg_el, avg_path_length_rmse, avg_delay_mae, avg_power_mae, avg_phase_mae, avg_az_mae, avg_el_mae, avg_path_length_mae = results

    scenario_row = {
        "scenario": scenario,
        "delay_rmse": avg_delay,
        "power_rmse": avg_power,
        "phase_rmse": avg_phase,
        "az_rmse": avg_az,
        "el_rmse": avg_el,
        "path_length_rmse": avg_path_length_rmse,
        "delay_mae": avg_delay_mae,
        "power_mae": avg_power_mae,
        "phase_mae": avg_phase_mae,
        "avg_az_mae": avg_az_mae,
        "avg_el_mae": avg_el_mae,
        "path_length_mae": avg_path_length_mae,
        "best_val_loss": best_loss,
    }
    df = pd.DataFrame([scenario_row])
    df.to_csv(CSV_LOG_FILE, mode="a", index=False, header=not os.path.exists(CSV_LOG_FILE))
    print(f"Saved results for {scenario} to {CSV_LOG_FILE}")
