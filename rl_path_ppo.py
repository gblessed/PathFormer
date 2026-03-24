import argparse
import copy
import os
import random
from dataclasses import dataclass

import deepmimo as dm
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Bernoulli, Normal
from tqdm import tqdm

from dataset.dataloaders import PreTrainMySeqDataLoader
from models import PathDecoder
from utils.utils import (
    masked_loss,
    ChannelParameters,
    compute_single_array_response_torch,
    generate_MIMO_channel_torch,
)


def safe_angle_from_components(y: torch.Tensor, x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Bound atan2 gradients by normalizing near-zero component pairs."""
    comps = torch.stack([y, x], dim=-1)
    comps = torch.nn.functional.normalize(comps, dim=-1, eps=eps)
    return torch.atan2(comps[..., 0], comps[..., 1])


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class PPOPathPolicy(nn.Module):
    def __init__(self, path_decoder: PathDecoder, init_log_std: float = -2.5):
        super().__init__()
        self.path_decoder = path_decoder
        self.value_head = nn.Linear(path_decoder.hidden_dim, 1)
        self.cont_log_std = nn.Parameter(torch.full((4,), init_log_std))

    def forward_step(self, prompts, paths, interactions):
        h_paths, prefix_flat = self.path_decoder.forward_hidden(prompts, paths, interactions)
        h_last = h_paths[:, -1, :]

        delay = self.path_decoder.out_delay(h_last).squeeze(-1)
        power = self.path_decoder.out_power(h_last).squeeze(-1)
        out = self.path_decoder.out(h_last)
        phase_sin = out[:, 0]
        phase_cos = out[:, 1]
        phase = safe_angle_from_components(phase_sin, phase_cos)
        az = safe_angle_from_components(out[:, 2], out[:, 3])
        el = safe_angle_from_components(out[:, 4], out[:, 5])
        interaction_logits = self.path_decoder.interaction_head(h_last)
        pathcount = self.path_decoder.pathcount_head(prefix_flat).squeeze(-1)
        value = self.value_head(h_last).squeeze(-1)

        mean_full = torch.stack([delay, power, phase, az, el], dim=-1)
        return mean_full, interaction_logits, pathcount, value

    def continuous_dist(self, mean_full):
        mean_cont = torch.stack(
            [mean_full[:, 0], mean_full[:, 1], mean_full[:, 3], mean_full[:, 4]],
            dim=-1,
        )
        std = torch.exp(self.cont_log_std).clamp(min=1e-5)
        std = std.unsqueeze(0).expand_as(mean_cont)
        return Normal(mean_cont, std)


def build_action_from_sample(mean_full, sampled_cont):
    return torch.stack(
        [
            sampled_cont[:, 0],
            sampled_cont[:, 1],
            mean_full[:, 2],
            sampled_cont[:, 2],
            sampled_cont[:, 3],
        ],
        dim=-1,
    )


def clamp_action_full(action_full: torch.Tensor) -> torch.Tensor:
    """Keep sampled actions within broad data-like ranges before autoregressive feedback."""
    delay = action_full[:, 0].clamp(0.0, 2.5)
    power = action_full[:, 1].clamp(-2.0, 0.5)
    phase = action_full[:, 2].clamp(-np.pi, np.pi)
    az = action_full[:, 3].clamp(-np.pi, np.pi)
    el = action_full[:, 4].clamp(0.0, np.pi)
    return torch.stack([delay, power, phase, az, el], dim=-1)


def masked_mean(values: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mask = mask.to(values.dtype)
    return (values * mask).sum() / mask.sum().clamp_min(eps)


def binary_f1_score(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    preds = preds.float()
    targets = targets.float()
    tp = ((preds == 1) & (targets == 1)).sum().float()
    fp = ((preds == 1) & (targets == 0)).sum().float()
    fn = ((preds == 0) & (targets == 1)).sum().float()
    denom = 2 * tp + fp + fn
    if denom.item() == 0:
        return preds.new_tensor(0.0)
    return 2 * tp / denom


def build_simple_channel_from_paths(paths: torch.Tensor, phase_source_paths: torch.Tensor) -> torch.Tensor:
    """
    Build a channel tensor from path parameters.

    `paths` provides delay, power, and AoA values.
    `phase_source_paths` provides phase values. This lets us use GT phase for both
    the predicted and GT channel construction.
    """
    params = ChannelParameters()
    B, T, _ = paths.shape
    delay_secs = paths[:, :, 0] / 1e6
    power_linear = 10 ** ((paths[:, :, 1] / 0.01) / 10)
    phase_degs = torch.rad2deg(phase_source_paths[:, :, 2])
    az_t = paths[:, :, 3]
    el_t = paths[:, :, 4]
    dopplers = torch.zeros(B, T, device=paths.device, dtype=paths.dtype)
    array_response = compute_single_array_response_torch(params.bs_antenna, az_t, el_t)
    channel = generate_MIMO_channel_torch(
        array_response,
        power_linear,
        delay_secs,
        phase_degs,
        dopplers,
        ofdm_params=params.ofdm,
        freq_domain=params.freq_domain,
    )
    if channel.dim() == 4:
        channel = channel.unsqueeze(2)
    return channel


def compute_reward_components(
    generated_paths,
    gt_paths,
    gt_path_lengths,
):
    batch_rewards = []
    metrics = {
        "channel_mse": [],
        "reward": [],
    }

    for b in range(generated_paths.size(0)):
        n_valid = int(round(gt_path_lengths[b].item() * 25))
        horizon = min(n_valid, generated_paths.size(1), gt_paths.size(1))
        gt = gt_paths[b, :horizon, :5]
        pred = generated_paths[b, :horizon]

        if horizon == 0:
            channel_mse = gt_path_lengths.new_tensor(0.0)
        else:
            gt_batch = gt.unsqueeze(0)
            pred_batch = pred.unsqueeze(0)
            gt_channel = 1e6*build_simple_channel_from_paths(gt_batch, gt_batch)
            pred_channel = 1e6*build_simple_channel_from_paths(pred_batch, gt_batch)
            # print(pred_channel[0,0,0,0,0], gt_channel[0,0,0,0,0])
            channel_mse = (
                (gt_channel.real - pred_channel.real) ** 2
                + (gt_channel.imag - pred_channel.imag) ** 2
            ).mean()

        reward = -channel_mse

        batch_rewards.append(reward)
        metrics["channel_mse"].append(channel_mse.item())
        metrics["reward"].append(reward.item())

    rewards = torch.stack(batch_rewards).view(-1)
    metric_means = {k: float(np.mean(v)) if v else 0.0 for k, v in metrics.items()}
    return rewards, metric_means


def rollout_policy(policy, prompts, max_steps, deterministic=False):
    device = prompts.device
    batch_size = prompts.size(0)
    cur_paths = torch.zeros(batch_size, 1, 5, device=device)
    cur_interactions = -1 * torch.ones(batch_size, 1, 4, device=device)

    generated_paths = []
    generated_interactions = []
    log_probs = []
    values = []
    entropies = []
    pathcount_pred = None

    for _ in range(max_steps):
        mean_full, interaction_logits, pathcount_pred, value = policy.forward_step(
            prompts, cur_paths, cur_interactions
        )
        cont_dist = policy.continuous_dist(mean_full)
        inter_dist = Bernoulli(logits=interaction_logits)

        if deterministic:
            sampled_cont = cont_dist.mean
            sampled_inter = (torch.sigmoid(interaction_logits) > 0.5).float()
        else:
            sampled_cont = cont_dist.rsample()
            sampled_inter = inter_dist.sample()

        action_full = build_action_from_sample(mean_full, sampled_cont)
        action_full = clamp_action_full(action_full)
        log_prob = cont_dist.log_prob(sampled_cont).sum(dim=-1)
        log_prob = log_prob + inter_dist.log_prob(sampled_inter).sum(dim=-1)
        entropy = cont_dist.entropy().sum(dim=-1) + inter_dist.entropy().sum(dim=-1)

        generated_paths.append(action_full)
        generated_interactions.append(sampled_inter)
        log_probs.append(log_prob)
        values.append(value)
        entropies.append(entropy)

        cur_paths = torch.cat([cur_paths, action_full.unsqueeze(1)], dim=1)
        cur_interactions = torch.cat([cur_interactions, sampled_inter.unsqueeze(1)], dim=1)

    return {
        "paths": torch.stack(generated_paths, dim=1),
        "interactions": torch.stack(generated_interactions, dim=1),
        "log_probs": torch.stack(log_probs, dim=1),
        "values": torch.stack(values, dim=1),
        "entropies": torch.stack(entropies, dim=1),
        "pathcount_pred": pathcount_pred,
    }


def recompute_policy_terms(policy, prompts, sampled_paths, sampled_interactions):
    device = prompts.device
    batch_size, max_steps, _ = sampled_paths.shape
    cur_paths = torch.zeros(batch_size, 1, 5, device=device)
    cur_interactions = -1 * torch.ones(batch_size, 1, 4, device=device)

    log_probs = []
    values = []
    entropies = []
    mean_fulls = []
    interaction_logits_all = []

    for t in range(max_steps):
        mean_full, interaction_logits, _, value = policy.forward_step(prompts, cur_paths, cur_interactions)
        cont_dist = policy.continuous_dist(mean_full)
        inter_dist = Bernoulli(logits=interaction_logits)
        sampled_cont = torch.stack(
            [
                sampled_paths[:, t, 0],
                sampled_paths[:, t, 1],
                sampled_paths[:, t, 3],
                sampled_paths[:, t, 4],
            ],
            dim=-1,
        )
        sampled_inter = sampled_interactions[:, t]

        log_prob = cont_dist.log_prob(sampled_cont).sum(dim=-1)
        log_prob = log_prob + inter_dist.log_prob(sampled_inter).sum(dim=-1)
        entropy = cont_dist.entropy().sum(dim=-1) + inter_dist.entropy().sum(dim=-1)

        log_probs.append(log_prob)
        values.append(value)
        entropies.append(entropy)
        mean_fulls.append(mean_full)
        interaction_logits_all.append(interaction_logits)

        cur_paths = torch.cat([cur_paths, sampled_paths[:, t].unsqueeze(1)], dim=1)
        cur_interactions = torch.cat([cur_interactions, sampled_inter.unsqueeze(1)], dim=1)

    return (
        torch.stack(log_probs, dim=1),
        torch.stack(values, dim=1),
        torch.stack(entropies, dim=1),
        torch.stack(mean_fulls, dim=1),
        torch.stack(interaction_logits_all, dim=1),
    )


def compute_supervised_anchor_loss(policy, batch, interaction_weight, pad_value):
    device = next(policy.parameters()).device
    prompts, paths, path_lengths, interactions, _, _, path_padding_mask = batch
    prompts = prompts.to(device)
    paths = paths.to(device)
    path_lengths = path_lengths.to(device)
    interactions = interactions.to(device)
    path_padding_mask = path_padding_mask.to(device)

    paths_in = paths[:, :-1, :]
    interactions_in = interactions[:, :-1, :]
    paths_out = paths[:, 1:, :]
    interactions_out = interactions[:, 1:, :]

    outputs = policy.path_decoder(prompts, paths_in, interactions_in)
    total_loss, *_ = masked_loss(
        *outputs,
        paths_out,
        path_lengths,
        interactions_out,
        pad_value=pad_value,
        interaction_weight=interaction_weight,
        path_padding_mask=path_padding_mask,
    )
    pathcount_pred = outputs[11].squeeze(-1)
    pathcount_loss = (pathcount_pred - path_lengths.squeeze(-1)).pow(2).mean()
    return total_loss + pathcount_loss


def evaluate_policy(policy, val_loader, max_steps):
    policy.eval()
    reward_values = []
    channel_mses = []

    with torch.no_grad():
        for prompts, paths, path_lengths, interactions, env, env_prop, path_padding_mask in tqdm(
            val_loader, desc="RL Eval", leave=False
        ):
            device = next(policy.parameters()).device
            prompts = prompts.to(device)
            paths = paths.to(device)
            path_lengths = path_lengths.to(device)
            interactions = interactions.to(device)

            rollout = rollout_policy(policy, prompts, max_steps=max_steps, deterministic=True)
            gt_paths = paths[:, 1:, :]
            rewards, metrics = compute_reward_components(
                rollout["paths"],
                gt_paths,
                path_lengths,
            )

            reward_values.append(rewards.mean().item())
            channel_mses.append(metrics["channel_mse"])

    return {
        "reward": float(np.mean(reward_values)) if reward_values else 0.0,
        "channel_mse": float(np.mean(channel_mses)) if channel_mses else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description="PPO fine-tuning for path generation with channel-MSE reward.")
    parser.add_argument("--scenario", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=26)
    parser.add_argument("--lr", type=float, default=2e-6)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.001)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--sft-anchor-weight", type=float, default=1.0)
    parser.add_argument("--interaction-weight", type=float, default=0.01)
    parser.add_argument("--pad-value", type=float, default=0.0)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--n-layers", type=int, default=8)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", type=str, default="checkpoints_rl")
    parser.add_argument("--init-log-std", type=float, default=-2.5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--kl-coef", type=float, default=0.05)
    parser.add_argument("--max-kl", type=float, default=5.0)
    parser.add_argument("--ref-anchor-weight", type=float, default=1.0)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dm.download(args.scenario)
    dataset = dm.load(args.scenario)

    train_data = PreTrainMySeqDataLoader(
        dataset,
        train=True,
        split_by="user",
        sort_by="power",
        normalizers=None,
        apply_normalizers=[],
        pad_value=args.pad_value,
    )
    val_data = PreTrainMySeqDataLoader(
        dataset,
        train=False,
        split_by="user",
        sort_by="power",
        normalizers=None,
        apply_normalizers=[],
        pad_value=args.pad_value,
    )

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=train_data.collate_fn,
    )
    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=val_data.collate_fn,
    )

    base_model = PathDecoder(
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        pad_value=args.pad_value,
    ).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    base_model.load_state_dict(state_dict)

    policy = PPOPathPolicy(base_model, init_log_std=args.init_log_std).to(device)
    policy.cont_log_std.requires_grad_(False)
    ref_policy = copy.deepcopy(policy).to(device)
    ref_policy.eval()
    for param in ref_policy.parameters():
        param.requires_grad_(False)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr)

    os.makedirs(args.save_dir, exist_ok=True)
    best_val_reward = -float("inf")
    best_path = os.path.join(
        args.save_dir,
        f"ppo_{args.scenario}_best.pt",
    )

    for epoch in range(args.epochs):
        policy.train()
        train_rewards = []
        train_policy_losses = []
        train_value_losses = []
        train_entropy_values = []
        train_kl_values = []
        train_ref_anchor_values = []
        train_anchor_losses = []
        train_channel_mses = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [RL Train]", leave=False)
        for batch in pbar:
            prompts, paths, path_lengths, interactions, env, env_prop, path_padding_mask = batch
            prompts = prompts.to(device)
            paths = paths.to(device)
            path_lengths = path_lengths.to(device)
            interactions = interactions.to(device)

            rollout = rollout_policy(policy, prompts, max_steps=args.max_steps, deterministic=False)
            rewards, reward_metrics = compute_reward_components(
                rollout["paths"],
                paths[:, 1:, :],
                path_lengths,
            )
            valid_steps = torch.round(path_lengths.squeeze(-1) * 25.0).long().clamp(min=1, max=args.max_steps)
            step_idx = torch.arange(args.max_steps, device=device).unsqueeze(0)
            valid_mask = step_idx < valid_steps.unsqueeze(1)
            time_idx = torch.arange(args.max_steps, device=device, dtype=rollout["values"].dtype)
            discounts = args.gamma ** (args.max_steps - 1 - time_idx)
            returns = rewards.unsqueeze(1) * discounts.unsqueeze(0) * valid_mask.to(rollout["values"].dtype)
            advantages = returns - rollout["values"].detach()
            adv_mean = masked_mean(advantages, valid_mask)
            adv_centered = advantages - adv_mean
            adv_var = masked_mean(adv_centered.pow(2), valid_mask)
            advantages = adv_centered / adv_var.sqrt().clamp(min=1e-6)
            advantages = advantages * valid_mask.to(advantages.dtype)

            log_probs_new, values_new, entropies, mean_full_new, interaction_logits_new = recompute_policy_terms(
                policy,
                prompts,
                rollout["paths"].detach(),
                rollout["interactions"].detach(),
            )
            with torch.no_grad():
                log_probs_ref, _, _, mean_full_ref, interaction_logits_ref = recompute_policy_terms(
                    ref_policy,
                    prompts,
                    rollout["paths"].detach(),
                    rollout["interactions"].detach(),
                )

            log_ratio = (log_probs_new - rollout["log_probs"].detach()).clamp(min=-20.0, max=20.0)
            ratio = torch.exp(log_ratio)
            clipped_ratio = torch.clamp(ratio, 1.0 - args.clip_eps, 1.0 + args.clip_eps)
            surr = torch.min(ratio * advantages, clipped_ratio * advantages)
            policy_loss = -masked_mean(surr, valid_mask)
            value_loss = 0.5 * masked_mean((returns - values_new).pow(2), valid_mask)
            entropy_bonus = masked_mean(entropies, valid_mask)
            approx_kl = torch.clamp(masked_mean(log_probs_new - log_probs_ref, valid_mask), min=0.0)
            ref_scales = mean_full_new.new_tensor([1.0, 0.5, np.pi, np.pi, np.pi / 2.0]).view(1, 1, 5)
            ref_mean_loss = ((mean_full_new - mean_full_ref) / ref_scales).pow(2).mean(dim=-1)
            ref_mean_loss = masked_mean(ref_mean_loss, valid_mask)
            ref_inter_loss = (
                torch.sigmoid(interaction_logits_new) - torch.sigmoid(interaction_logits_ref)
            ).pow(2).mean(dim=-1)
            ref_inter_loss = masked_mean(ref_inter_loss, valid_mask)
            ref_anchor_loss = ref_mean_loss + 0.25 * ref_inter_loss
            total_loss = (
                policy_loss
                + args.value_coef * value_loss
                + args.kl_coef * torch.clamp(approx_kl, max=args.max_kl)
                + args.ref_anchor_weight * ref_anchor_loss
                - args.entropy_coef * entropy_bonus
            )

            anchor_loss_value = torch.tensor(0.0, device=device)
            if args.sft_anchor_weight > 0:
                anchor_loss_value = compute_supervised_anchor_loss(
                    policy,
                    batch,
                    interaction_weight=args.interaction_weight,
                    pad_value=args.pad_value,
                )
                total_loss = total_loss + args.sft_anchor_weight * anchor_loss_value

            optimizer.zero_grad()
            if not torch.isfinite(total_loss):
                print("  Skipping batch due to non-finite total loss.")
                continue
            total_loss.backward()
            has_non_finite_grad = False
            for param in policy.parameters():
                if param.grad is not None and not torch.isfinite(param.grad).all():
                    has_non_finite_grad = True
                    break
            if has_non_finite_grad:
                optimizer.zero_grad(set_to_none=True)
                print("  Skipping batch due to non-finite gradients.")
                continue
            torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip)
            optimizer.step()

            params_finite = all(torch.isfinite(param).all() for param in policy.parameters())
            if not params_finite:
                raise RuntimeError("Encountered non-finite policy parameters after optimizer step.")

            train_rewards.append(rewards.mean().item())
            train_policy_losses.append(policy_loss.item())
            train_value_losses.append(value_loss.item())
            train_entropy_values.append(entropy_bonus.item())
            train_kl_values.append(approx_kl.item())
            train_ref_anchor_values.append(ref_anchor_loss.item())
            train_anchor_losses.append(anchor_loss_value.item())
            train_channel_mses.append(reward_metrics["channel_mse"])

            pbar.set_postfix(
                {
                    "reward": f"{train_rewards[-1]:.3f}",
                    "pi": f"{train_policy_losses[-1]:.3f}",
                    "vf": f"{train_value_losses[-1]:.3f}",
                    "ch_mse": f"{train_channel_mses[-1]:.3e}",
                }
            )

        val_metrics = evaluate_policy(
            policy,
            val_loader,
            max_steps=args.max_steps,
        )

        print(f"\nEpoch {epoch:02d}")
        print(
            f"  Train Reward: {np.mean(train_rewards):.4f} | "
            f"Val Reward: {val_metrics['reward']:.4f}"
        )
        print(
            f"  Train Policy Loss: {np.mean(train_policy_losses):.4f} | "
            f"Value Loss: {np.mean(train_value_losses):.4f} | "
            f"Entropy: {np.mean(train_entropy_values):.4f} | "
            f"KL: {np.mean(train_kl_values):.4f} | "
            f"Ref: {np.mean(train_ref_anchor_values):.4f}"
        )
        print(
            f"  Train Channel MSE: {np.mean(train_channel_mses):.6e} | "
            f"Val Channel MSE: {val_metrics['channel_mse']:.6e}"
        )

        if val_metrics["reward"] > best_val_reward:
            best_val_reward = val_metrics["reward"]
            torch.save(
                {
                    "epoch": epoch,
                    "policy_state_dict": policy.state_dict(),
                    "model_state_dict": policy.path_decoder.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_reward": best_val_reward,
                    "args": vars(args),
                },
                best_path,
            )
            print(f"  Saved best PPO checkpoint to {best_path}")


if __name__ == "__main__":
    main()


'''
python rl_path_ppo.py \
  --scenario city_47_chicago_3p5 \
  --checkpoint checkpoints2/snoise_enc_direct_city_47_chicago_3p5_interacaction_all_inter_str_dec_all_repeat_best_model_checkpoint.pth

'''
