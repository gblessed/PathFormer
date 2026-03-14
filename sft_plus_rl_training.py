# =============================================================================
# SFT + RL training: improve SFT checkpoint with reinforcement learning.
# - Loads a trained SFT checkpoint (from multiscenario_direct_training.py).
# - Uses reward-weighted loss + SFT regularization so eval metrics improve.
# - Uses batch eval from channel_finetune_direct_training.py; saves best by eval.
# Does NOT modify multiscenario_direct_training.py.
# =============================================================================
import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import deepmimo as dm

from models import PathDecoder
from dataset.dataloaders import PreTrainMySeqDataLoader
from utils.utils import (
    generate_paths_no_env_batch,
    masked_loss,
    add_noise_to_paths,
)
# Batch evaluation (same as channel_finetune_direct_training.py)
from channel_finetune_direct_training import evaluate_model_batch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_config(scenario, sft_checkpoint_path):
    """Config aligned with multiscenario_direct_training.py + RL hyperparams."""
    return {
        "BATCH_SIZE": 128,
        "PAD_VALUE": 0,
        "USE_WANDB": False,
        "LR": 1e-5,  # Lower LR for RL stability
        "epochs": 50,
        "interaction_weight": 0.01,
        "experiment": f"rl_after_sft_{scenario}",
        "hidden_dim": 512,
        "n_layers": 8,
        "n_heads": 8,
        "TARGET_NOISE_PROB": 0.0,
        "TARGET_NOISE_PARAMS": None,
        "SFT_CHECKPOINT": sft_checkpoint_path,
        # RL: reward = -composite_error; we maximize reward
        "rl_reward_weights": {
            "delay_rmse": 1.0,
            "power_rmse": 1.0,
            "phase_rmse": 0.0,
            "az_rmse": 0.5,
            "el_rmse": 0.5,
            "path_length_rmse": 0.0,  # Path length is critical for eval
        },
        "sft_coef": 0.0,   # Keep 20% SFT loss so we don't forget
        "rl_baseline": "batch",  # "batch" or "running"
        "eval_every_epochs": 2,
        "max_generate": 26,
        "GRAD_CLIP_NORM": 1.0,
    }


def compute_per_sample_composite_error(
    delay_pred, power_pred, phase_pred, az_pred, el_pred, path_length_pred,
    paths_out, path_lengths, path_padding_mask, reward_weights, device
):
    """
    Compute per-sample composite error (same scale as eval metrics).
    paths_out: (B, T, 5), path_lengths: (B,), path_padding_mask: (B, T_path) with True = valid.
    Returns: (B,) tensor of composite errors (lower = better).
    """
    B = paths_out.size(0)
    mask = path_padding_mask[:, 1:]  # (B, T) for paths_out
    T = paths_out.size(1)
    if mask.size(1) != T:
        mask = mask[:, :T]

    delay_gt = paths_out[:, :, 0]
    power_gt = paths_out[:, :, 1]
    phase_gt = paths_out[:, :, 2]
    az_gt = paths_out[:, :, 3]
    el_gt = paths_out[:, :, 4]

    # Clamp preds to same ranges as in eval
    power_pred_safe = power_pred.clamp(-15000.0, 500.0)

    errors = []
    for b in range(B):
        m = mask[b]
        if m.sum() == 0:
            errors.append(torch.tensor(0.0, device=device))
            continue
        # RMSE per sample (over valid steps)
        delay_rmse = ((delay_pred[b] - delay_gt[b])[m].pow(2).mean().clamp(min=1e-12).sqrt())
        power_rmse = ((power_pred_safe[b] / 0.01 - power_gt[b] / 0.01)[m].pow(2).mean().clamp(min=1e-12).sqrt())
        # Circular phase/angle errors
        phase_diff = torch.atan2(torch.sin(phase_pred[b] - phase_gt[b]), torch.cos(phase_pred[b] - phase_gt[b]))[m]
        phase_rmse = (phase_diff.pow(2).mean().clamp(min=1e-12).sqrt())
        az_diff = torch.atan2(torch.sin(az_pred[b] - az_gt[b]), torch.cos(az_pred[b] - az_gt[b]))[m]
        az_rmse = (az_diff.pow(2).mean().clamp(min=1e-12).sqrt())
        el_diff = torch.atan2(torch.sin(el_pred[b] - el_gt[b]), torch.cos(el_pred[b] - el_gt[b]))[m]
        el_rmse = (el_diff.pow(2).mean().clamp(min=1e-12).sqrt())
        pl = path_length_pred[b].squeeze()
        if pl.dim() > 0:
            pl = pl.squeeze(0)
        path_length_rmse = (pl - path_lengths[b]).abs()
        comp = (
            reward_weights["delay_rmse"] * delay_rmse
            + reward_weights["power_rmse"] * power_rmse
            + reward_weights["phase_rmse"] * phase_rmse
            + reward_weights["az_rmse"] * az_rmse
            + reward_weights["el_rmse"] * el_rmse
            + reward_weights["path_length_rmse"] * path_length_rmse
        )
        errors.append(comp)
    return torch.stack(errors)  # (B,)


def compute_per_sample_mse(
    delay_pred, power_pred, phase_sin_pred, phase_cos_pred, phase_pred,
    az_sin_pred, az_cos_pred, az_pred, el_sin_pred, el_cos_pred, el_pred,
    paths_out, path_padding_mask, device
):
    """Per-sample MSE (over valid steps) for REINFORCE weight. Returns (B,) tensor."""
    B = paths_out.size(0)
    mask = path_padding_mask[:, 1:]
    T = paths_out.size(1)
    if mask.size(1) != T:
        mask = mask[:, :T]
    delay_gt = paths_out[:, :, 0]
    power_gt = paths_out[:, :, 1]
    phase_gt = paths_out[:, :, 2]
    sinp_gt = torch.sin(phase_gt)
    cosp_gt = torch.cos(phase_gt)
    az_gt = paths_out[:, :, 3]
    sin_az_gt = torch.sin(az_gt)
    cos_az_gt = torch.cos(az_gt)
    el_gt = paths_out[:, :, 4]
    sin_el_gt = torch.sin(el_gt)
    cos_el_gt = torch.cos(el_gt)

    per_sample = []
    for b in range(B):
        m = mask[b]
        if m.sum() == 0:
            per_sample.append(torch.tensor(0.0, device=device))

            continue
        
        s = (
            (delay_pred[b] - delay_gt[b])[m].pow(2).mean()
            + (power_pred[b] - power_gt[b])[m].pow(2).mean()
            + (phase_sin_pred[b] - sinp_gt[b])[m].pow(2).mean() + (phase_cos_pred[b] - cosp_gt[b])[m].pow(2).mean()
            + (az_sin_pred[b] - sin_az_gt[b])[m].pow(2).mean() + (az_cos_pred[b] - cos_az_gt[b])[m].pow(2).mean()
            + (el_sin_pred[b] - sin_el_gt[b])[m].pow(2).mean() + (el_cos_pred[b] - cos_el_gt[b])[m].pow(2).mean()
        )
        per_sample.append(s)
    return torch.stack(per_sample)


def load_sft_checkpoint(model, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"SFT checkpoint not found: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=True)
    epoch = ckpt.get("epoch", -1)
    print(f"✓ Loaded SFT checkpoint from {checkpoint_path} (epoch {epoch})")
    return epoch


def train_rl_after_sft(model, train_loader, val_loader, config, train_data, checkpoint_path, best_eval_metric_cb):
    """
    RL loop: reward-weighted loss + SFT regularization.
    best_eval_metric_cb: callable that runs batch eval and returns the metric we want to minimize (e.g. composite RMSE).
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["LR"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=15, T_mult=1, eta_min=1e-8
    )
    reward_weights = config["rl_reward_weights"]
    sft_coef = config["sft_coef"]
    running_reward = None
    alpha = 0.99  # for running baseline

    best_eval = float("inf")

    for epoch in range(config["epochs"]):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [RL]", leave=False)
        for prompts, paths, path_lengths, interactions, env, env_prop, path_padding_mask in pbar:
            prompts = prompts.to(device)
            paths = paths.to(device)
            path_lengths = path_lengths.to(device)
            interactions = interactions.to(device)
            path_padding_mask = path_padding_mask.to(device)

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

            # Per-sample composite error (same as eval) -> reward = -error
            composite_error = compute_per_sample_composite_error(
                delay_pred,
                power_pred,
                phase_pred,
                az_pred,
                el_pred,
                path_length_pred,
                paths_out,
                path_lengths,
                path_padding_mask,
                reward_weights,
                device,
            )
            reward = -composite_error  # (B,)

            # Baseline for variance reduction
            baseline = reward.mean().detach()
            if config["rl_baseline"] == "running":
                if running_reward is None:
                    running_reward = baseline.item()
                running_reward = alpha * running_reward + (1 - alpha) * baseline.item()
                baseline = torch.tensor(running_reward, device=device, dtype=reward.dtype)

            # Per-sample MSE for REINFORCE weight
            mse_per_sample = compute_per_sample_mse(
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
                paths_out,
                path_padding_mask,
                device,
            )
            # RL loss: minimize (baseline - reward) * mse -> push high-reward samples to lower MSE
            baseline = 0
            # print(f"reward: {reward.shape}, mse_per_sample: {mse_per_sample.shape} ")
            advantage = (reward - baseline).detach()
            #normalize advantage to be between 0 and 1
            # advantage = (advantage - advantage.min()) / (advantage.max() - advantage.min())
     
            
            # print(f"advantage: {advantage.shape}, ")
            loss_rl = (advantage * mse_per_sample).mean()
            # loss_rl =  (mse_per_sample).mean()

            # SFT loss (standard masked loss) so we don't forget
            (total_sft, *_) = masked_loss(
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
                finetune=None,
                pad_value=train_data.pad_value,
                interaction_weight=config.get("interaction_weight", 0.1),
                path_padding_mask=path_padding_mask,
            )

            loss = (1.0 - sft_coef) * loss_rl + sft_coef * total_sft

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.get("GRAD_CLIP_NORM", 1.0))
            optimizer.step()

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "rl": f"{loss_rl.item():.4f}",
                "sft": f"{total_sft.item():.4f}",
                "reward": f"{reward.mean().item():.4f}",
            })

        scheduler.step()

        # Periodic eval and save best by eval metric
        if (epoch + 1) % config["eval_every_epochs"] == 0 or epoch == 0:
            eval_metric = best_eval_metric_cb(model, val_loader)
            if eval_metric < best_eval:
                best_eval = eval_metric
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_eval_metric": best_eval,
                }, checkpoint_path)
                print(f"  ✓ Best checkpoint saved (eval metric: {best_eval:.4f})")
    return best_eval


def make_eval_metric_cb(config):
    """Returns a callable that runs batch eval and returns a scalar to minimize (composite of RMSEs)."""
    weights = config["rl_reward_weights"]

    def run_eval_and_return_metric(model, val_loader):
        results = evaluate_model_batch(
            model,
            val_loader,
            train_data_pad_value=config["PAD_VALUE"],
            max_generate=config["max_generate"],
            log_to_wandb=False,
        )
        (
            avg_delay,
            avg_power,
            avg_phase,
            avg_az,
            avg_el,
            avg_path_length_rmse,
            *_,
        ) = results
        composite = (
            weights["delay_rmse"] * avg_delay
            + weights["power_rmse"] * avg_power
            + weights["phase_rmse"] * avg_phase
            + weights["az_rmse"] * avg_az
            + weights["el_rmse"] * avg_el
            + weights["path_length_rmse"] * avg_path_length_rmse
        )
        return composite

    return run_eval_and_return_metric


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    parser = argparse.ArgumentParser(description="RL after SFT for path prediction")
    parser.add_argument("--scenario", type=str, default="city_47_chicago_3p5", help="DeepMIMO scenario")
    parser.add_argument("--sft_checkpoint", type=str, default=None,
                        help="Path to SFT .pth checkpoint. Default: checkpoints2/noise_enc_direct_<scenario>_...")
    parser.add_argument("--epochs", type=int, default=50, help="RL epochs")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--sft_coef", type=float, default=0.2, help="Weight of SFT loss (rest is RL)")
    parser.add_argument("--eval_every_epochs", type=int, default=None, help="Eval every N epochs (default from config)")
    parser.add_argument("--trial_suffix", type=str, default="", help="Suffix for checkpoint file (for multi-trial runs)")
    parser.add_argument("--eval_only", action="store_true", help="Only load SFT and run batch eval")
    args = parser.parse_args()

    scenario = args.scenario
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if args.sft_checkpoint is None:
        sft_checkpoint_path = os.path.join(
            base_dir,
            "checkpoints2",
            f"snoise_enc_direct_{scenario}_interacaction_all_inter_str_dec_all_repeat_best_model_checkpoint.pth",
        )
    else:
        sft_checkpoint_path = args.sft_checkpoint

    dm.download(scenario)
    dataset = dm.load(scenario)

    config = get_config(scenario, sft_checkpoint_path)
    config["epochs"] = args.epochs
    config["LR"] = args.lr
    config["sft_coef"] = args.sft_coef
    if args.eval_every_epochs is not None:
        config["eval_every_epochs"] = args.eval_every_epochs

    model = PathDecoder(
        hidden_dim=config["hidden_dim"],
        n_layers=config["n_layers"],
        n_heads=config["n_heads"],
    ).to(device)
    print("Total trainable parameters:", count_parameters(model))

    load_sft_checkpoint(model, sft_checkpoint_path)

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

    os.makedirs("checkpoints_rl_after_sft", exist_ok=True)
    trial_suffix = getattr(args, "trial_suffix", "") or ""
    ckpt_name = f"{config['experiment']}_best_eval.pth"
    if trial_suffix:
        ckpt_name = ckpt_name.replace(".pth", f"_{trial_suffix}.pth")
    checkpoint_path = os.path.join("checkpoints_rl_after_sft", ckpt_name)

    if args.eval_only:
        print("eval_only: running batch eval on SFT checkpoint.")
        run_eval = make_eval_metric_cb(config)
        composite = run_eval(model, val_loader)
        print(f"\n[SFT baseline] composite eval metric (lower=better): {composite:.4f}")
        return

    best_eval_metric_cb = make_eval_metric_cb(config)
    train_rl_after_sft(model, train_loader, val_loader, config, train_data, checkpoint_path, best_eval_metric_cb)

    # Load best for final eval
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print("✓ Loaded best RL checkpoint for final evaluation")
    print("\n========== Final batch evaluation ==========")
    results = evaluate_model_batch(
        model,
        val_loader,
        train_data_pad_value=config["PAD_VALUE"],
        max_generate=config["max_generate"],
    )
    print("Done.")


if __name__ == "__main__":
    main()
