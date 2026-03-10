# =============================================================================
# Channel-only finetuning: load a path-prediction checkpoint (no channel loss)
# and finetune only on channel loss. Uses latest model/dataloader from
# multiscenario_direct_training_channel.py. Reports avg_ch_score and nmse
# like channel_playground.ipynb.
# =============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import warnings
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

warnings.filterwarnings("ignore", category=UserWarning)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------------------------------------------------------
# Config: path checkpoint (trained without channel loss) and channel-finetune
# -----------------------------------------------------------------------------
def get_config(scenario, path_checkpoint_path):
    return {
        "BATCH_SIZE": 128,
        "PAD_VALUE": 0,
        "USE_WANDB": False,
        "LR": 1e-5,
        "epochs": 50,
        "interaction_weight": 0.01,
        "experiment": f"channel_finetune_{scenario}",
        "hidden_dim": 512,
        "n_layers": 8,
        "n_heads": 8,
        "TARGET_NOISE_PROB": 0.0,
        "TARGET_NOISE_PARAMS": None,
        "GRAD_CLIP_NORM": 1.0,
        "PATH_CHECKPOINT": path_checkpoint_path,
        "channel_loss_weight": 1.0,
    }


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


def train_channel_finetune(model, train_loader, val_loader, config, train_data):
    """Train only on channel loss (backward on channel_loss only)."""
    best_val_ch_loss = float("inf")
    channel_loss_weight = config.get("channel_loss_weight", 1.0)

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

            (
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
                path_length_pred,
                interaction_logits,
            ) = model(prompts, paths_in, interactions_in)

            (
                total_loss,
                loss_delay,
                loss_power,
                loss_phase,
                loss_az,
                loss_el,
                loss_path_length,
                loss_interaction,
                loss_channel,
            ) = masked_loss(
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
                path_length_pred,
                interaction_logits,
                paths_out,
                path_lengths,
                interactions_out,
                finetune="channel_estimation",
                pad_value=train_data.pad_value,
                interaction_weight=config.get("interaction_weight", 0.1),
                path_padding_mask=path_padding_mask,
            )

            loss_channel_val = (
                loss_channel.item() if torch.is_tensor(loss_channel) else float(loss_channel)
            )
            train_ch_losses.append(loss_channel_val)
            if torch.is_tensor(loss_channel) and loss_channel.requires_grad:
                total_loss_channel_only = channel_loss_weight * loss_channel
                optimizer.zero_grad()
                total_loss_channel_only.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.get("GRAD_CLIP_NORM", 1.0)
                )
                optimizer.step()

            pbar.set_postfix({"ch_loss": f"{loss_channel_val:.4f}"})

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
                interactions_out = interactions[:, 1:, :]
                (
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
                    path_length_pred,
                    interaction_logits,
                ) = model(prompts, paths_in, interactions_in)
                (
                    _,
                    _,
                    _,
                    _,
                    _,
                    _,
                    _,
                    _,
                    loss_channel,
                ) = masked_loss(
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
                    path_length_pred,
                    interaction_logits,
                    paths_out,
                    path_lengths,
                    interactions_out,
                    finetune="channel_estimation",
                    pad_value=train_data.pad_value,
                    interaction_weight=config.get("interaction_weight", 0.1),
                    path_padding_mask=path_padding_mask,
                )
                v = loss_channel.item() if torch.is_tensor(loss_channel) else float(loss_channel)
                val_ch_losses.append(v)

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


# -----------------------------------------------------------------------------
# Main: one scenario, load path checkpoint, finetune on channel, then evaluate
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Channel-only finetuning from path checkpoint")
    parser.add_argument("--scenario", type=str, default="city_47_chicago_3p5", help="DeepMIMO scenario")
    # parser.add_argument(
    #     "--path_checkpoint",
    #     type=str,
    #     required=True,
    #     help="Path to .pth checkpoint trained on path prediction (no channel loss)",
    # )
    parser.add_argument("--epochs", type=int, default=50, help="Channel finetune epochs")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate for channel finetune")
    parser.add_argument("--eval_only", action="store_true", help="Only load path checkpoint and run evaluation")
    args = parser.parse_args()

    scenario = args.scenario
    # path_checkpoint_path = args.path_checkpoint
    # /home/blessedg/Pathformer/checkpoints2/noise_enc_direct_city_35_san_francisco_3p5_interacaction_all_inter_str_dec_all_repeat_best_model_checkpoint.pth
    # path_checkpoint_path = f"/home/blessedg/Pathformer/checkpoints2/noise_enc_direct_{scenario}_interacaction_all_inter_str_dec_all_repeat_best_model_checkpoint.pth"
    path_checkpoint_path = f"/home/blessedg/Pathformer/checkpoints2/delay_only_enc_direct_{scenario}_interacaction_all_inter_str_dec_all_repeat_best_model_checkpoint.pth"
    

    dm.download(scenario)
    dataset = dm.load(scenario)

    config = get_config(scenario, path_checkpoint_path)
    config["epochs"] = args.epochs
    config["LR"] = args.lr

    model = PathDecoder(
        hidden_dim=config["hidden_dim"],
        n_layers=config["n_layers"],
        n_heads=config["n_heads"],
    ).to(device)
    print("Total trainable parameters:", count_parameters(model))

    load_path_checkpoint(model, path_checkpoint_path)

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

    os.makedirs("checkpoints_channel_finetune", exist_ok=True)
    checkpoint_path = os.path.join(
        "checkpoints_channel_finetune",
        f"{config['experiment']}_best_channel_checkpoint.pth",
    )

    if not args.eval_only:
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["LR"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=25, T_mult=1, eta_min=1e-8
        )
        train_channel_finetune(model, train_loader, val_loader, config, train_data)
        if os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            print(f"✓ Loaded best channel checkpoint for final evaluation")
    else:
        print("eval_only: skipping channel finetune, evaluating path checkpoint as-is.")

    # results = evaluate_model(model, val_loader, train_data_pad_value=config["PAD_VALUE"])
    results = evaluate_model_batch(model, val_loader, train_data_pad_value=config["PAD_VALUE"])
    (
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
    ) = results

    csv_log_file = "channel_finetune_results.csv"
    row = {
        "scenario": scenario,
        "path_checkpoint": path_checkpoint_path,
        "delay_rmse": avg_delay,
        "power_rmse": avg_power,
        "phase_rmse": avg_phase,
        "az_rmse": avg_az,
        "el_rmse": avg_el,
        "path_length_rmse": avg_path_length_rmse,
        "ch_nmse_log": avg_ch_nmse_log,
        "avg_ch_score": avg_ch_score,
        "avg_ch_nmse_dB": avg_ch_nmse,
    }
    df = pd.DataFrame([row])
    df.to_csv(
        csv_log_file,
        mode="a",
        index=False,
        header=not os.path.exists(csv_log_file),
    )
    print(f"✓ Results saved to {csv_log_file}")
