# =============================================================================
# Channel finetuning: use MULTIPATH EMBEDDINGS to predict the channel (frozen
# backbone + channel head). Pool: first step, last step, or mean of valid steps.
# =============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import warnings
from typing import Literal

import deepmimo as dm
from sklearn.metrics import mean_squared_error

from models import PathDecoder
from dataset.dataloaders import PreTrainMySeqDataLoader
from utils.utils import (
    generate_paths_no_env,
    generate_paths_no_env_batch,
    masked_loss,
    add_noise_to_paths,
    ChannelParameters,
    compute_single_array_response_torch,
    generate_MIMO_channel_torch,
)
import generator_things.consts as c

warnings.filterwarnings("ignore", category=UserWarning)

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

# -----------------------------------------------------------------------------
# Config: path checkpoint (trained without channel loss) and channel-finetune
# -----------------------------------------------------------------------------
def get_config(scenario, path_checkpoint_path):
    return {
        "BATCH_SIZE": 128,
        "PAD_VALUE": 0,
        "USE_WANDB": False,
        "LR": 1e-5,
        "epochs": 5,
        "interaction_weight": 0.01,
        "experiment": f"channel_finetune_embed_{scenario}",
        "hidden_dim": 512,
        "n_layers": 8,
        "n_heads": 8,
        "TARGET_NOISE_PROB": 0.0,
        "TARGET_NOISE_PARAMS": None,
        "GRAD_CLIP_NORM": 1.0,
        "PATH_CHECKPOINT": path_checkpoint_path,
        "channel_loss_weight": 1.0,
        "embed_pool_mode": "mean",
        "max_generated_steps": 25,
    }


def get_channel_numel():
    """Number of real numbers to represent the channel (2 * num_complex_elements)."""
    params = ChannelParameters()
    sc = np.array(params.ofdm[c.PARAMSET_OFDM_SC_SAMP])
    n_sc = sc.size
    M_tx = int(np.prod(params.bs_antenna[c.PARAMSET_ANT_SHAPE]))
    M_rx = int(np.prod(params.ue_antenna[c.PARAMSET_ANT_SHAPE]))
    return 2 * M_rx * M_tx * n_sc


def build_gt_channel_batch(paths_out, path_padding_mask, pad_value, device):
    """
    paths_out: (B, T, 5) [delay, power, phase, aoa_az, aoa_el]
    path_padding_mask: (B, T) True = padding
    Returns: (B, 1, M_rx, M_tx, n_sc) complex on device
    """
    params = ChannelParameters()
    B, T, _ = paths_out.shape
    mask = ~path_padding_mask  # True = valid
    delay_secs = paths_out[:, :, 0] / 1e6
    power_t = paths_out[:, :, 1].masked_fill(path_padding_mask, 0)
    power_linear = 10 ** ((power_t / 0.01) / 10)
    power_linear = power_linear.masked_fill(path_padding_mask, float("nan"))
    phase_degs = torch.rad2deg(paths_out[:, :, 2])
    az_t = paths_out[:, :, 3]
    el_t = paths_out[:, :, 4]
    default_dopplers = torch.zeros(B, T, device=device, dtype=paths_out.dtype)
    array_response = compute_single_array_response_torch(
        params.bs_antenna, az_t, el_t
    )
    # generate_MIMO_channel_torch expects 3D (n_ues, n_ant, n_paths); it unsqueeze(1) then unpacks 4 dims.
    # array_response is (B, n_ant, T) — pass as-is.
    gt_channel = generate_MIMO_channel_torch(
        array_response,
        power_linear,
        delay_secs,
        phase_degs,
        default_dopplers,
        ofdm_params=params.ofdm,
        freq_domain=params.freq_domain,
    )
    # Generator returns (B, 1, M_tx, n_sc); match pred_ch shape (B, 1, M_rx, M_tx, n_sc) with M_rx=1
    if gt_channel.dim() == 4:
        gt_channel = gt_channel.unsqueeze(2)
    return gt_channel


class PathDecoderChannelFromEmbedding(nn.Module):
    """
    Frozen PathDecoder backbone + channel head. Uses hidden states from the
    backbone's own autoregressive rollout to predict the channel.
    """

    def __init__(
        self,
        backbone: PathDecoder,
        hidden_dim: int,
        channel_numel: int,
        pool_mode: Literal["first", "last", "mean"] = "mean",
        max_generated_steps: int = 25,
    ):
        super().__init__()
        self.backbone = backbone
        self.hidden_dim = hidden_dim
        self.channel_numel = channel_numel
        self.pool_mode = pool_mode
        self.max_generated_steps = max_generated_steps
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.channel_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, channel_numel),
        )

    def _pool_embeddings(self, h_paths):
        if self.pool_mode == "first":
            return h_paths[:, 0]
        if self.pool_mode == "last":
            return h_paths[:, -1]
        return h_paths.mean(dim=1)

    def _rollout_generated_hidden(self, prompts, max_steps):
        device = prompts.device
        B = prompts.size(0)
        cur = torch.zeros(B, 1, 5, device=device, dtype=prompts.dtype)
        inter_str = -1 * torch.ones(B, 1, 4, device=device, dtype=prompts.dtype)
        hidden_steps = []

        for _ in range(max_steps):
            h_paths, _ = self.backbone.forward_hidden(prompts, cur, inter_str)
            last_h = h_paths[:, -1, :]

            out = self.backbone.out(last_h)
            d_t = self.backbone.out_delay(last_h).squeeze(-1)
            p_t = self.backbone.out_power(last_h).squeeze(-1)
            ph_t = torch.atan2(out[:, 0], out[:, 1])
            az_t = torch.atan2(out[:, 2], out[:, 3])
            el_t = torch.atan2(out[:, 4], out[:, 5])
            inter_probs_t = torch.sigmoid(self.backbone.interaction_head(last_h))

            next_path = torch.stack([d_t, p_t, ph_t, az_t, el_t], dim=-1)
            hidden_steps.append(last_h)
            cur = torch.cat([cur, next_path.unsqueeze(1)], dim=1)
            inter_str = torch.cat([inter_str, inter_probs_t.unsqueeze(1)], dim=1)

        return torch.stack(hidden_steps, dim=1)

    def forward(self, prompts, paths_in, interactions_in, path_padding_mask=None):
        """
        The channel head uses hidden states from the backbone's generated rollout,
        not teacher-forced ground-truth paths. `paths_in` is used only to infer
        rollout length when provided.

        Returns pred_channel: (B, 1, M_rx, M_tx, n_sc) complex.
        """
        B = prompts.size(0)
        rollout_steps = paths_in.size(1) if paths_in is not None else self.max_generated_steps
        h_paths = self._rollout_generated_hidden(prompts, rollout_steps)
        agg = self._pool_embeddings(h_paths)
        ch_flat = self.channel_head(agg)
        half = self.channel_numel // 2
        real = ch_flat[:, :half]
        imag = ch_flat[:, half : half * 2]
        params = ChannelParameters()
        sc = np.array(params.ofdm[c.PARAMSET_OFDM_SC_SAMP])
        n_sc = sc.size
        M_tx = int(np.prod(params.bs_antenna[c.PARAMSET_ANT_SHAPE]))
        M_rx = int(np.prod(params.ue_antenna[c.PARAMSET_ANT_SHAPE]))
        real = real.view(B, M_rx, M_tx, n_sc)
        imag = imag.view(B, M_rx, M_tx, n_sc)
        pred_ch = torch.complex(real, imag).unsqueeze(1)
        return pred_ch


def compute_stop_metrics(path_count, targets, pad_value=0):
    y_pred = path_count.cpu().numpy().flatten()
    y_true = targets.squeeze().cpu().numpy().flatten()
    y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=0.0, neginf=0.0)
    y_true = np.nan_to_num(y_true, nan=0.0, posinf=0.0, neginf=0.0)
    return np.sqrt(mean_squared_error(y_true, y_pred))


# Evaluation: same as multiscenario (autoregressive + torch channel) but also
# report avg_ch_score and avg_ch_nmse (dB) like channel_playground.ipynb
def evaluate_model(model, val_loader, train_data_pad_value=0, max_generate=26, log_to_wandb=False):
    model.eval()
    CH_SCORE_DB_MIN, CH_SCORE_DB_MAX = -20.0, 0.0

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
    ch_nmse_log_list = []
    raw_ch_nmses_db = []
    ch_scores = []

    with torch.no_grad():
        outer_bar = tqdm(val_loader, desc="Evaluating (batches)", leave=True)
        for prompts, paths, path_lengths, interactions, env, env_prop, path_padding_mask in outer_bar:
            prompts = prompts.cuda()
            paths = paths.cuda()
            path_lengths = path_lengths.cuda()
            inner_bar = tqdm(range(prompts.size(0)), desc="   Processing samples", leave=False)

            for b in inner_bar:
                generated, path_lengths_pred, inter_str_pred = generate_paths_no_env(
                    model, prompts[b], max_steps=max_generate
                )
                generated = generated.cuda()
                n_valid = int(round(path_lengths[b].item() * 25))
                gt = paths[b][1 : 1 + n_valid, :5]
                T = min(len(gt), len(generated))
                pred = generated[:T]
                gt = gt[:T]

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

                length_rmse = (torch.mean((path_lengths_pred - path_lengths[b]) ** 2)).sqrt().item()
                length_mae = (torch.mean(torch.abs(path_lengths_pred - path_lengths[b]))).item()

                ch_nmse_dB = 0.0
                ch_score = 0.0
                if T > 0:
                    params = ChannelParameters()
                    delay_secs_gt = gt[:, 0] / 1e6
                    power_linear_gt = 10 ** ((gt[:, 1] / 0.01) / 10)
                    phase_degs_gt = torch.rad2deg(gt[:, 2])
                    array_resp_gt = compute_single_array_response_torch(
                        params.bs_antenna, gt[:, 3].unsqueeze(0), gt[:, 4].unsqueeze(0)
                    )
                    dopplers = torch.zeros(1, T, device=gt.device)
                    gt_ch = generate_MIMO_channel_torch(
                        array_resp_gt,
                        power_linear_gt.unsqueeze(0),
                        delay_secs_gt.unsqueeze(0),
                        phase_degs_gt.unsqueeze(0),
                        dopplers,
                        ofdm_params=params.ofdm,
                        freq_domain=params.freq_domain,
                    )
                    power_pred_clamped = pred[:, 1].clamp(-15000.0, 500.0)
                    power_linear_pred = 10 ** ((power_pred_clamped / 0.01) / 10)
                    phase_degs_pred = torch.rad2deg(pred[:, 2])
                    delay_secs_pred = pred[:, 0] / 1e6
                    array_resp_pred = compute_single_array_response_torch(
                        params.bs_antenna, pred[:, 3].unsqueeze(0), pred[:, 4].unsqueeze(0)
                    )
                    pred_ch = generate_MIMO_channel_torch(
                        array_resp_pred,
                        power_linear_pred.unsqueeze(0),
                        delay_secs_pred.unsqueeze(0),
                        phase_degs_pred.unsqueeze(0),
                        dopplers,
                        ofdm_params=params.ofdm,
                        freq_domain=params.freq_domain,
                    )
                    scale = 1e6
                    gt_s = gt_ch * scale
                    pred_s = pred_ch * scale
                    mse = (
                        (gt_s.real - pred_s.real) ** 2 + (gt_s.imag - pred_s.imag) ** 2
                    ).mean().item()
                    gt_norm_sq = (gt_s.real ** 2 + gt_s.imag ** 2).mean().item()
                    nmse = mse / (max(gt_norm_sq, 1e-6) + 1e-10)
                    ch_nmse_log = np.log10(nmse + 1e-10)
                    ch_nmse_log_list.append(ch_nmse_log)
                    ch_nmse_dB = 10.0 * ch_nmse_log
                    raw_ch_nmses_db.append(ch_nmse_dB)
                    normalized = (ch_nmse_dB - CH_SCORE_DB_MIN) / (
                        CH_SCORE_DB_MAX - CH_SCORE_DB_MIN
                    )
                    score = 1.0 - normalized
                    ch_score = max(0.0, min(1.0, score))
                    ch_scores.append(ch_score)

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
                postfix_dict = {
                    "delay_rmse": f"{delay_rmse:.3f}",
                    "ch_nmse": f"{ch_nmse_dB:.3f}" if T > 0 else "n/a",
                    "ch_score": f"{ch_score:.3f}" if T > 0 else "n/a",
                }
                inner_bar.set_postfix(postfix_dict)

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
    avg_ch_nmse_log = np.mean(ch_nmse_log_list) if ch_nmse_log_list else 0.0
    avg_ch_nmse = np.mean(raw_ch_nmses_db) if raw_ch_nmses_db else 0.0
    avg_ch_score = np.mean(ch_scores) if ch_scores else 0.0

    print("\n=================  Final EVALUATION RESULTS =================")
    print(f"Delay RMSE           : {avg_delay:.4f} µs")
    print(f"Power RMSE           : {avg_power:.4f} dB")
    print(f"Phase RMSE           : {avg_phase:.4f} degrees")
    print(f"AoA Azimuth RMSE     : {avg_az:.4f} degrees")
    print(f"AoA Elevation RMSE   : {avg_el:.4f} degrees")
    print(f"Path Length RMSE     : {avg_path_length_rmse:.4f}")
    print(f"Delay MAE           : {avg_delay_mae:.4f} µs")
    print(f"Power MAE           : {avg_power_mae:.4f} dB")
    print(f"Phase MAE           : {avg_phase_mae:.4f} degrees")
    print(f"AoA Azimuth MAE     : {avg_az_mae:.4f} degrees")
    print(f"AoA Elevation MAE   : {avg_el_mae:.4f} degrees")
    print(f"Path Length MAE     : {avg_path_length_mae:.4f}")
    print(f"Channel log10(NMSE) : {avg_ch_nmse_log:.4f}")
    print(f"avg_ch_score        :       {avg_ch_score:.4f}")
    print(f"avg_ch_nmse         :       {avg_ch_nmse:.4f} (dB)")
    print("=====================================================\n")

    if log_to_wandb:
        import wandb
        wandb.run.summary["test_ch_nmse_log"] = avg_ch_nmse_log
        wandb.run.summary["test_avg_ch_score"] = avg_ch_score
        wandb.run.summary["test_avg_ch_nmse_dB"] = avg_ch_nmse

    return (
        avg_delay,
        avg_power,
        avg_phase,
        avg_az,
        avg_el,
        avg_path_length_rmse,
        avg_delay_mae,
        avg_power_mae,
        avg_phase_mae,
        avg_az_mae,
        avg_el_mae,
        avg_path_length_mae,
        avg_ch_nmse_log,
        avg_ch_score,
        avg_ch_nmse,
    )


def evaluate_model_batch(model, val_loader, train_data_pad_value=0, max_generate=26, log_to_wandb=False):
    """Batch inference: one forward per batch. Same metrics as evaluate_model including avg_ch_score and avg_ch_nmse."""
    model.eval()
    CH_SCORE_DB_MIN, CH_SCORE_DB_MAX = -20.0, 0.0
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
    ch_nmse_log_list = []
    raw_ch_nmses_db = []
    ch_scores = []
    with torch.no_grad():
        outer_bar = tqdm(val_loader, desc="Evaluating (batches)", leave=True)
        for prompts, paths, path_lengths, interactions, env, env_prop, path_padding_mask in outer_bar:
            prompts = prompts.cuda()
            paths = paths.cuda()
            path_lengths = path_lengths.cuda()
            B = prompts.size(0)
            generated, pathcounts, _ = generate_paths_no_env_batch(model, prompts, max_steps=max_generate)
            generated = generated.cuda()
            if pathcounts.dim() > 1:
                pathcounts = pathcounts.squeeze(-1)
            for b in range(B):
                n_valid = int(round(path_lengths[b].item() * 25))
                gt = paths[b][1 : 1 + n_valid, :5]
                T = min(len(gt), generated.size(1))
                pred = generated[b, :T]
                gt = gt[:T]
                path_lengths_pred_b = pathcounts[b].squeeze()
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
                length_rmse = (torch.mean((path_lengths_pred_b - path_lengths[b]) ** 2)).sqrt().item()
                length_mae = (torch.mean(torch.abs(path_lengths_pred_b - path_lengths[b]))).item()
                ch_nmse_dB = 0.0
                ch_score = 0.0
                if T > 0:
                    params = ChannelParameters()
                    delay_secs_gt = gt[:, 0] / 1e6
                    power_linear_gt = 10 ** ((gt[:, 1] / 0.01) / 10)
                    phase_degs_gt = torch.rad2deg(gt[:, 2])
                    array_resp_gt = compute_single_array_response_torch(
                        params.bs_antenna, gt[:, 3].unsqueeze(0), gt[:, 4].unsqueeze(0)
                    )
                    dopplers = torch.zeros(1, T, device=gt.device)
                    gt_ch = generate_MIMO_channel_torch(
                        array_resp_gt,
                        power_linear_gt.unsqueeze(0),
                        delay_secs_gt.unsqueeze(0),
                        phase_degs_gt.unsqueeze(0),
                        dopplers,
                        ofdm_params=params.ofdm,
                        freq_domain=params.freq_domain,
                    )
                    power_pred_clamped = pred[:, 1].clamp(-15000.0, 500.0)
                    power_linear_pred = 10 ** ((power_pred_clamped / 0.01) / 10)
                    phase_degs_pred = torch.rad2deg(pred[:, 2])
                    delay_secs_pred = pred[:, 0] / 1e6
                    array_resp_pred = compute_single_array_response_torch(
                        params.bs_antenna, pred[:, 3].unsqueeze(0), pred[:, 4].unsqueeze(0)
                    )
                    pred_ch = generate_MIMO_channel_torch(
                        array_resp_pred,
                        power_linear_pred.unsqueeze(0),
                        delay_secs_pred.unsqueeze(0),
                        phase_degs_pred.unsqueeze(0),
                        dopplers,
                        ofdm_params=params.ofdm,
                        freq_domain=params.freq_domain,
                    )
                    scale = 1e6
                    gt_s = gt_ch * scale
                    pred_s = pred_ch * scale
                    mse = (
                        (gt_s.real - pred_s.real) ** 2 + (gt_s.imag - pred_s.imag) ** 2
                    ).mean().item()
                    gt_norm_sq = (gt_s.real ** 2 + gt_s.imag ** 2).mean().item()
                    nmse = mse / (max(gt_norm_sq, 1e-6) + 1e-10)
                    ch_nmse_log = np.log10(nmse + 1e-10)
                    ch_nmse_log_list.append(ch_nmse_log)
                    ch_nmse_dB = 10.0 * ch_nmse_log
                    raw_ch_nmses_db.append(ch_nmse_dB)
                    normalized = (ch_nmse_dB - CH_SCORE_DB_MIN) / (
                        CH_SCORE_DB_MAX - CH_SCORE_DB_MIN
                    )
                    score = 1.0 - normalized
                    ch_score = max(0.0, min(1.0, score))
                    ch_scores.append(ch_score)
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
    avg_ch_nmse_log = np.mean(ch_nmse_log_list) if ch_nmse_log_list else 0.0
    avg_ch_nmse = np.mean(raw_ch_nmses_db) if raw_ch_nmses_db else 0.0
    avg_ch_score = np.mean(ch_scores) if ch_scores else 0.0
    print("\n=================  Final EVALUATION RESULTS (batch) =================")
    print(f"Delay RMSE           : {avg_delay:.4f} µs")
    print(f"Power RMSE           : {avg_power:.4f} dB")
    print(f"Phase RMSE           : {avg_phase:.4f} degrees")
    print(f"AoA Azimuth RMSE     : {avg_az:.4f} degrees")
    print(f"AoA Elevation RMSE   : {avg_el:.4f} degrees")
    print(f"Path Length RMSE     : {avg_path_length_rmse:.4f}")
    print(f"Delay MAE           : {avg_delay_mae:.4f} µs")
    print(f"Power MAE           : {avg_power_mae:.4f} dB")
    print(f"Phase MAE           : {avg_phase_mae:.4f} degrees")
    print(f"AoA Azimuth MAE     : {avg_az_mae:.4f} degrees")
    print(f"AoA Elevation MAE   : {avg_el_mae:.4f} degrees")
    print(f"Path Length MAE     : {avg_path_length_mae:.4f}")
    print(f"Channel log10(NMSE) : {avg_ch_nmse_log:.4f}")
    print(f"avg_ch_score        :       {avg_ch_score:.4f}")
    print(f"avg_ch_nmse         :       {avg_ch_nmse:.4f} (dB)")
    print("=====================================================\n")
    if log_to_wandb:
        import wandb
        wandb.run.summary["test_ch_nmse_log"] = avg_ch_nmse_log
        wandb.run.summary["test_avg_ch_score"] = avg_ch_score
        wandb.run.summary["test_avg_ch_nmse_dB"] = avg_ch_nmse
    return (
        avg_delay,
        avg_power,
        avg_phase,
        avg_az,
        avg_el,
        avg_path_length_rmse,
        avg_delay_mae,
        avg_power_mae,
        avg_phase_mae,
        avg_az_mae,
        avg_el_mae,
        avg_path_length_mae,
        avg_ch_nmse_log,
        avg_ch_score,
        avg_ch_nmse,
    )


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def evaluate_channel_from_embed(model, val_loader, train_data_pad_value=0):
    """Evaluate channel NMSE/score when model predicts channel from embeddings (no path generation)."""
    model.eval()
    CH_SCORE_DB_MIN, CH_SCORE_DB_MAX = -20.0, 0.0
    scale = 1e6
    ch_nmse_log_list = []
    raw_ch_nmses_db = []
    ch_scores = []
    with torch.no_grad():
        for prompts, paths, path_lengths, interactions, env, env_prop, path_padding_mask in tqdm(
            val_loader, desc="Eval channel from embed"
        ):
            prompts = prompts.cuda()
            paths = paths.cuda()
            path_padding_mask = path_padding_mask.cuda()
            interactions = interactions.cuda()
            paths_in = paths[:, :-1, :]
            interactions_in = interactions[:, :-1, :]
            paths_out = paths[:, 1:, :]
            pred_ch = model(prompts, paths_in, interactions_in, path_padding_mask)
            gt_ch = build_gt_channel_batch(
                paths_out, path_padding_mask[:, 1:], train_data_pad_value, pred_ch.device
            )
            gt_s = gt_ch * scale
            pred_s = pred_ch * scale
            mse = ((gt_s.real - pred_s.real) ** 2 + (gt_s.imag - pred_s.imag) ** 2).mean().item()
            gt_norm_sq = (gt_s.real ** 2 + gt_s.imag ** 2).mean().item()
            nmse = mse / (max(gt_norm_sq, 1e-6) + 1e-10)
            ch_nmse_log = np.log10(nmse + 1e-10)
            ch_nmse_dB = 10.0 * ch_nmse_log
            ch_nmse_log_list.append(ch_nmse_log)
            raw_ch_nmses_db.append(ch_nmse_dB)
            normalized = (ch_nmse_dB - CH_SCORE_DB_MIN) / (CH_SCORE_DB_MAX - CH_SCORE_DB_MIN)
            score = 1.0 - normalized
            ch_scores.append(max(0.0, min(1.0, score)))
    avg_ch_nmse_log = np.mean(ch_nmse_log_list)
    avg_ch_nmse = np.mean(raw_ch_nmses_db)
    avg_ch_score = np.mean(ch_scores)
    print("\n=================  Channel-from-embed EVALUATION =================")
    print(f"Channel log10(NMSE) : {avg_ch_nmse_log:.4f}")
    print(f"avg_ch_score        :       {avg_ch_score:.4f}")
    print(f"avg_ch_nmse         :       {avg_ch_nmse:.4f} (dB)")
    print("=====================================================\n")
    return avg_ch_nmse_log, avg_ch_score, avg_ch_nmse


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_path_checkpoint(model, checkpoint_path):
    """Load only model state from a path-prediction checkpoint (no channel loss)."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Path checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state, strict=False)
    epoch = checkpoint.get("epoch", -1)
    print(f"✓ Loaded path-prediction checkpoint from {checkpoint_path} (epoch {epoch})")
    return epoch


def load_channel_checkpoint(model, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Channel checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    epoch = checkpoint.get("epoch", -1)
    best_val_ch_loss = checkpoint.get("best_val_ch_loss", None)
    if hasattr(best_val_ch_loss, "item"):
        best_val_ch_loss = best_val_ch_loss.item()
    print(
        f"✓ Loaded channel checkpoint from {checkpoint_path} "
        f"(epoch {epoch}, best_val_ch_loss={best_val_ch_loss})"
    )
    return epoch, best_val_ch_loss


def train_channel_finetune(
    model,
    train_loader,
    val_loader,
    config,
    train_data,
    optimizer,
    scheduler,
    checkpoint_path,
):
    """Train only on channel loss. For PathDecoderChannelFromEmbedding: pred channel from embeddings vs GT."""
    best_val_ch_loss = float("inf")
    channel_numel = get_channel_numel()
    scale = 1e6

    for epoch in range(config["epochs"]):
        model.train()
        train_ch_losses = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]", leave=False)
        for (
            prompts,
            paths,
            path_lengths,
            interactions,
            env,
            env_prop,
            path_padding_mask,
        ) in pbar:
            prompts = prompts.cuda()
            paths = paths.cuda()
            path_lengths = path_lengths.cuda()
            interactions = interactions.cuda()
            path_padding_mask = path_padding_mask.cuda()

            paths_in = paths[:, :-1, :]
            interactions_in = interactions[:, :-1, :]
            paths_out = paths[:, 1:, :]

            pred_ch = model(prompts, paths_in, interactions_in, path_padding_mask)
            gt_ch = build_gt_channel_batch(
                paths_out, path_padding_mask[:, 1:], train_data.pad_value, pred_ch.device
            )
            gt_s = gt_ch * scale
            pred_s = pred_ch * scale
            mse = ((gt_s.real - pred_s.real) ** 2 + (gt_s.imag - pred_s.imag) ** 2).mean()
            gt_norm_sq = (gt_s.real ** 2 + gt_s.imag ** 2).mean().clamp(min=1e-6)
            channel_loss = mse / (gt_norm_sq + 1e-10)
            channel_loss = channel_loss.clamp(max=100.0)

            train_ch_losses.append(channel_loss.item())
            optimizer.zero_grad()
            channel_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.get("GRAD_CLIP_NORM", 1.0))
            optimizer.step()
            pbar.set_postfix({"ch_loss": f"{channel_loss.item():.4f}"})

        scheduler.step()
        avg_train_ch = np.mean(train_ch_losses)

        model.eval()
        val_ch_losses = []
        with torch.no_grad():
            for (
                prompts,
                paths,
                path_lengths,
                interactions,
                env,
                env_prop,
                path_padding_mask,
            ) in tqdm(val_loader, desc=f"Epoch {epoch} [Val]", leave=False):
                prompts = prompts.cuda()
                paths = paths.cuda()
                path_lengths = path_lengths.cuda()
                interactions = interactions.cuda()
                path_padding_mask = path_padding_mask.cuda()
                paths_in = paths[:, :-1, :]
                interactions_in = interactions[:, :-1, :]
                paths_out = paths[:, 1:, :]
                pred_ch = model(prompts, paths_in, interactions_in, path_padding_mask)
                gt_ch = build_gt_channel_batch(
                    paths_out, path_padding_mask[:, 1:], train_data.pad_value, pred_ch.device
                )
                gt_s = gt_ch * scale
                pred_s = pred_ch * scale
                mse = ((gt_s.real - pred_s.real) ** 2 + (gt_s.imag - pred_s.imag) ** 2).mean().item()
                gt_norm_sq = (gt_s.real ** 2 + gt_s.imag ** 2).mean().item()
                v = mse / (max(gt_norm_sq, 1e-6) + 1e-10)
                val_ch_losses.append(min(v, 100.0))

        avg_val_ch = np.mean(val_ch_losses)
        current_lr = optimizer.param_groups[0]["lr"]

        if avg_val_ch < best_val_ch_loss:
            best_val_ch_loss = avg_val_ch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_val_ch_loss": torch.tensor(best_val_ch_loss),
                },
                checkpoint_path,
            )
            print(f"  ✓ Best channel checkpoint saved (val_ch_loss: {best_val_ch_loss:.4f})")

        if config.get("USE_WANDB", False):
            import wandb
            wandb.log(
                {
                    "train_ch_loss": avg_train_ch,
                    "val_ch_loss": avg_val_ch,
                    "epoch": epoch,
                    "lr": current_lr,
                }
            )

        print(
            f"\nEpoch {epoch:02d}  train_ch_loss: {avg_train_ch:.4f}  val_ch_loss: {avg_val_ch:.4f}  lr: {current_lr:.2e}"
        )


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Channel finetuning/evaluation from generated PathDecoder embeddings"
    )
    parser.add_argument("scenarios", nargs="*", help="Explicit scenario names to run.")
    parser.add_argument("--scenario", dest="scenario_flag", action="append", help="Scenario name to run. Can be repeated.")
    parser.add_argument("--scenario-file", type=str, help="Optional file with one scenario per line.")
    parser.add_argument("--path_checkpoint", type=str, default=None, help="Optional explicit path checkpoint. Use only with a single scenario.")
    parser.add_argument("--path-checkpoint-dir", type=str, default="checkpoints2", help="Directory containing path checkpoints from multiscenario_direct_training.py.")
    parser.add_argument("--channel-checkpoint-dir", type=str, default="checkpoints_channel_finetune", help="Directory for saving/loading channel finetune checkpoints.")
    parser.add_argument("--csv-log-file", type=str, default="channel_finetune_embed_results.csv", help="CSV file for per-scenario results.")
    parser.add_argument("--epochs", type=int, default=50, help="Channel finetune epochs")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate for channel finetune")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--eval_only", action="store_true", help="Only load the saved channel checkpoint and run evaluation")
    parser.add_argument(
        "--embed_pool",
        type=str,
        choices=["first", "last", "mean"],
        default="mean",
        help="How to pool generated path embeddings for channel prediction: first step, last step, or mean over generated steps",
    )
    parser.add_argument(
        "--max_generated_steps",
        type=int,
        default=25,
        help="Number of autoregressive rollout steps used to build generated embeddings.",
    )
    return parser.parse_args()


def resolve_scenarios(args):
    scenarios = []
    if args.scenarios:
        scenarios.extend(args.scenarios)
    if args.scenario_flag:
        scenarios.extend(args.scenario_flag)
    if args.scenario_file:
        with open(args.scenario_file, "r", encoding="utf-8") as handle:
            scenarios.extend([line.strip() for line in handle if line.strip()])
    if not scenarios:
        scenarios = list(all_scenarios[2:])
    return scenarios


def default_path_checkpoint(path_checkpoint_dir, scenario):
    experiment = f"snoise_enc_direct_{scenario}_interacaction_all_inter_str_dec_all_repeat"
    return os.path.join(path_checkpoint_dir, f"{experiment}_best_model_checkpoint.pth")


def run_scenario(scenario, args):
    if args.path_checkpoint is not None and len(resolve_scenarios(args)) > 1:
        raise ValueError("--path_checkpoint can only be used when evaluating a single scenario.")

    path_checkpoint_path = args.path_checkpoint or default_path_checkpoint(
        args.path_checkpoint_dir,
        scenario,
    )
    dm.download(scenario)
    dataset = dm.load(scenario)

    config = get_config(scenario, path_checkpoint_path)
    config["epochs"] = args.epochs
    config["LR"] = args.lr
    config["BATCH_SIZE"] = args.batch_size
    config["embed_pool_mode"] = args.embed_pool
    config["max_generated_steps"] = args.max_generated_steps

    channel_numel = get_channel_numel()
    backbone = PathDecoder(
        hidden_dim=config["hidden_dim"],
        n_layers=config["n_layers"],
        n_heads=config["n_heads"],
    ).to(device)
    load_path_checkpoint(backbone, path_checkpoint_path)
    model = PathDecoderChannelFromEmbedding(
        backbone,
        hidden_dim=config["hidden_dim"],
        channel_numel=channel_numel,
        pool_mode=config["embed_pool_mode"],
        max_generated_steps=config["max_generated_steps"],
    ).to(device)
    print(f"######### Channel embedding eval on Scenario {scenario} #########")
    print("Total trainable parameters (channel head only):", count_parameters(model))

    train_data = PreTrainMySeqDataLoader(
        dataset,
        train=True,
        split_by="user",
        sort_by="power",
        normalizers=None,
        apply_normalizers=[],
        pad_value=config["PAD_VALUE"],
    )
    train_loader = torch.utils.data.DataLoader(
        dataset=train_data,
        batch_size=config["BATCH_SIZE"],
        shuffle=True,
        collate_fn=train_data.collate_fn,
    )
    val_data = PreTrainMySeqDataLoader(
        dataset,
        train=False,
        split_by="user",
        sort_by="power",
        normalizers=None,
        apply_normalizers=[],
        pad_value=config["PAD_VALUE"],
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=val_data,
        batch_size=config["BATCH_SIZE"],
        shuffle=False,
        collate_fn=val_data.collate_fn,
    )

    os.makedirs(args.channel_checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(
        args.channel_checkpoint_dir,
        f"{config['experiment']}_best_channel_checkpoint.pth",
    )

    if not args.eval_only:
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()), lr=config["LR"]
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=25, T_mult=1, eta_min=1e-8
        )
        train_channel_finetune(
            model,
            train_loader,
            val_loader,
            config,
            train_data,
            optimizer,
            scheduler,
            checkpoint_path,
        )
        if os.path.exists(checkpoint_path):
            load_channel_checkpoint(model, checkpoint_path)
    else:
        load_channel_checkpoint(model, checkpoint_path)

    avg_ch_nmse_log, avg_ch_score, avg_ch_nmse = evaluate_channel_from_embed(
        model, val_loader, train_data_pad_value=config["PAD_VALUE"]
    )

    row = {
        "scenario": scenario,
        "path_checkpoint": path_checkpoint_path,
        "channel_checkpoint": checkpoint_path,
        "embed_pool": config["embed_pool_mode"],
        "max_generated_steps": config["max_generated_steps"],
        "eval_only": args.eval_only,
        "ch_nmse_log": avg_ch_nmse_log,
        "avg_ch_score": avg_ch_score,
        "avg_ch_nmse_dB": avg_ch_nmse,
    }
    df = pd.DataFrame([row])
    df.to_csv(
        args.csv_log_file,
        mode="a",
        index=False,
        header=not os.path.exists(args.csv_log_file),
    )
    print(f"✓ Results for {scenario} saved to {args.csv_log_file}")
    return row


def main():
    args = parse_args()
    scenarios = resolve_scenarios(args)
    if not scenarios:
        print("No scenarios selected for this run.")
        return

    print(f"Running {len(scenarios)} scenario(s): {scenarios}")
    for scenario in scenarios:
        try:
            run_scenario(scenario, args)
        except Exception as exc:
            print(f"✗ Failed scenario {scenario}: {exc}")
            error_row = pd.DataFrame([{"scenario": scenario, "error": str(exc)}])
            error_row.to_csv(
                args.csv_log_file,
                mode="a",
                index=False,
                header=not os.path.exists(args.csv_log_file),
            )


if __name__ == "__main__":
    main()
