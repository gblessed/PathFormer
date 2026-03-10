# Two-Stage RL for Channel Optimization (Stage 2: PPO with terminal channel NMSE reward)
# Assumes a pre-trained PathDecoder (SFT) is loaded from checkpoint.
# Uses stochastic policy π(a|s) = N(μ_θ(s), diag(exp(ω))²) and PPO.
# If reward drops: use --warmup_deterministic_epochs 1, smaller --init_log_std (-2.5), lower --lr (5e-6).

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import argparse
from tqdm import tqdm
from collections import defaultdict

from models import PathDecoder
from dataset.dataloaders import PreTrainMySeqDataLoader
from utils.utils import (
    ChannelParameters,
    compute_single_array_response_torch,
    generate_MIMO_channel_torch,
)

# ---------------------------------------------------------------------------
# Channel reward: terminal NMSE -> R = -log10(NMSE + eps)
# ---------------------------------------------------------------------------

def compute_channel_nmse_reward(
    paths_pred: torch.Tensor,   # (B, T, 5) delay, power, phase, az, el
    paths_gt: torch.Tensor,     # (B, T, 5)
    path_lengths: torch.Tensor,  # (B,) or (B,1) normalized count in [0,1]; n_paths = round(path_lengths * 25)
    params: ChannelParameters,
    device: torch.device,
    eps: float = 1e-10,
    scale: float = 1e6,
) -> torch.Tensor:
    """Compute per-sample reward = -log10(NMSE + eps). paths in same units as model (delay µs, power 0.01*dB, phase rad, az/el rad)."""
    B = paths_pred.size(0)
    path_lengths = path_lengths.squeeze()
    if path_lengths.dim() == 0:
        path_lengths = path_lengths.unsqueeze(0)
    rewards = torch.zeros(B, device=device, dtype=paths_pred.dtype)
    for b in range(B):
        n_paths = int(round(path_lengths[b].item() * 25))
        n_paths = max(1, min(n_paths, paths_pred.size(1), paths_gt.size(1)))
        pred = paths_pred[b, :n_paths]   # (T, 5)
        gt = paths_gt[b, :n_paths]       # (T, 5)
        if n_paths == 0:
            rewards[b] = 0.0
            continue
        # Same preprocessing as in channel training eval
        delay_secs_gt = gt[:, 0] / 1e6
        power_linear_gt = 10 ** ((gt[:, 1].clamp(-15000, 500) / 0.01) / 10)
        phase_degs_gt = torch.rad2deg(gt[:, 2])
        array_resp_gt = compute_single_array_response_torch(
            params.bs_antenna, gt[:, 3].unsqueeze(0), gt[:, 4].unsqueeze(0)
        )
        dopplers = torch.zeros(1, n_paths, device=device)
        gt_ch = generate_MIMO_channel_torch(
            array_resp_gt,
            power_linear_gt.unsqueeze(0),
            delay_secs_gt.unsqueeze(0),
            phase_degs_gt.unsqueeze(0),
            dopplers,
            ofdm_params=params.ofdm,
            freq_domain=params.freq_domain,
        )
        power_pred_c = pred[:, 1].clamp(-15000.0, 500.0)
        power_linear_pred = 10 ** ((power_pred_c / 0.01) / 10)
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
        gt_s = gt_ch * scale
        pred_s = pred_ch * scale
        mse = ((gt_s.real - pred_s.real) ** 2 + (gt_s.imag - pred_s.imag) ** 2).mean()
        gt_norm_sq = (gt_s.real ** 2 + gt_s.imag ** 2).mean()
        nmse = mse / (gt_norm_sq.clamp(min=1e-6) + 1e-10)
        rewards[b] = -torch.log10(nmse + eps)
    return rewards


# ---------------------------------------------------------------------------
# Policy wrapper: μ_θ(s) from PathDecoder + learnable log_std ω
# ---------------------------------------------------------------------------

class PathDecoderPolicyWithValue(nn.Module):
    def __init__(self, path_decoder: PathDecoder, init_log_std: float = -3.0):
        super().__init__()
        self.path_decoder = path_decoder
        hidden_dim = path_decoder.hidden_dim
        self.log_std = nn.Parameter(torch.full((5,), init_log_std))
        self.value_head = nn.Linear(hidden_dim, 1)

    def get_mean_and_hidden(self, prompts, paths, interactions):
        """Returns (mean (B,T,5), h_paths (B,T,hidden)) for last step only we use mean[:, -1], h_paths[:, -1]."""
        d, p, ps, pc, ph, az_s, az_c, az, el_s, el_c, el, _, _ = self.path_decoder(prompts, paths, interactions)
        mean = torch.stack([d, p, ph, az, el], dim=-1)
        h_paths, _ = self.path_decoder.forward_hidden(prompts, paths, interactions)
        return mean, h_paths

    def forward(self, prompts, paths, interactions, return_value=True):
        mean, h_paths = self.get_mean_and_hidden(prompts, paths, interactions)
        std = torch.exp(self.log_std).clamp(min=1e-6)
        if return_value:
            v = self.value_head(h_paths[:, -1, :]).squeeze(-1)
            return mean, std, v
        return mean, std

    def sample_actions(self, mean: torch.Tensor, std: torch.Tensor):
        """mean (B, 5), std (5,) -> actions (B, 5), log_prob (B,)."""
        B = mean.size(0)
        std_b = std.unsqueeze(0).expand(B, -1)
        dist = torch.distributions.Normal(mean, std_b)
        actions = dist.rsample()
        log_prob = dist.log_prob(actions).sum(dim=-1)
        return actions, log_prob

    def log_prob_actions(self, mean: torch.Tensor, std: torch.Tensor, actions: torch.Tensor):
        """mean (B, 5), std (5,), actions (B, 5) -> log_prob (B,)."""
        B = mean.size(0)
        std_b = std.unsqueeze(0).expand(B, -1)
        dist = torch.distributions.Normal(mean, std_b)
        return dist.log_prob(actions).sum(dim=-1)


# ---------------------------------------------------------------------------
# Rollout: autoregressive generation with sampling, store for PPO
# ---------------------------------------------------------------------------

def rollout(
    policy: PathDecoderPolicyWithValue,
    prompts: torch.Tensor,
    interactions_gt: torch.Tensor,
    path_lengths: torch.Tensor,
    max_steps: int,
    device: torch.device,
    deterministic: bool = False,
    return_value: bool = True,
):
    """
    Roll out the policy. If deterministic=True, use mean action. If return_value=False (GRPO), values are zeros.
    Returns: generated (B,T,5), log_probs_old (B,T), values (B,T).
    """
    B = prompts.size(0)
    cur = torch.zeros(B, 1, 5, device=device)
    inter_str = -1 * torch.ones(B, 1, 4, device=device)
    actions_list = []
    log_probs_list = []
    values_list = []
    policy.eval()

    for t in range(max_steps):
        out = policy(prompts, cur, inter_str, return_value=return_value)
        if return_value:
            mean, std, v = out
        else:
            mean, std = out
            v = torch.zeros(prompts.size(0), device=device, dtype=mean.dtype)
        mean_t = mean[:, -1, :]
        std_t = std
        if deterministic:
            actions_t = mean_t
            log_prob_t = policy.log_prob_actions(mean_t, std_t, actions_t)
        else:
            actions_t, log_prob_t = policy.sample_actions(mean_t, std_t)
        actions_list.append(actions_t)
        log_probs_list.append(log_prob_t)
        values_list.append(v)
        # Get interaction prediction while cur and inter_str still have same length
        with torch.no_grad():
            _, _, _, _, _, _, _, _, _, _, _, _, inter_logits = policy.path_decoder(prompts, cur, inter_str)
            inter_pred = (torch.sigmoid(inter_logits[:, -1, :]) > 0.5).float()
        inter_str = torch.cat([inter_str, inter_pred.unsqueeze(1)], dim=1)
        next_path = actions_t.unsqueeze(1)
        cur = torch.cat([cur, next_path], dim=1)

    generated = torch.stack([torch.stack([actions_list[t][b] for t in range(max_steps)], dim=0) for b in range(B)], dim=0)
    log_probs_old = torch.stack(log_probs_list, dim=1)
    values = torch.stack(values_list, dim=1)
    return generated, log_probs_old, values


# ---------------------------------------------------------------------------
# PPO training step
# ---------------------------------------------------------------------------

def ppo_loss(
    policy: PathDecoderPolicyWithValue,
    prompts: torch.Tensor,
    paths_gt: torch.Tensor,
    path_lengths: torch.Tensor,
    interactions_gt: torch.Tensor,
    generated: torch.Tensor,
    log_probs_old: torch.Tensor,
    values: torch.Tensor,
    params: ChannelParameters,
    device: torch.device,
    clip_eps: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    advantages: torch.Tensor = None,
):
    """If advantages is provided (e.g. for GRPO), use it and skip value loss. Else A = returns - values, normalized."""
    B, T = generated.size(0), generated.size(1)
    rewards = compute_channel_nmse_reward(generated, paths_gt, path_lengths, params, device)
    rewards_per_t = rewards.unsqueeze(1).expand(-1, T)
    returns = rewards_per_t
    if advantages is not None:
        pass  # use provided advantages
    else:
        advantages = returns - values.detach()
        adv_mean = advantages.mean()
        adv_std = advantages.std()
        if adv_std > 1e-8:
            advantages = (advantages - adv_mean) / (adv_std + 1e-8)

    cur = torch.zeros(B, 1, 5, device=device)
    inter_str = -1 * torch.ones(B, 1, 4, device=device)
    log_probs_new_list = []
    values_new_list = []
    entropies_list = []

    for t in range(T):
        mean, std, v = policy(prompts, cur, inter_str, return_value=True)
        mean_t = mean[:, -1, :]
        actions_t = generated[:, t, :]
        log_prob_new = policy.log_prob_actions(mean_t, std, actions_t)
        std_b = std.unsqueeze(0).expand(B, -1)
        dist = torch.distributions.Normal(mean_t, std_b)
        entropy_t = dist.entropy().sum(dim=-1)
        log_probs_new_list.append(log_prob_new)
        values_new_list.append(v)
        entropies_list.append(entropy_t)
        # Keep cur and inter_str in sync: get inter_logits before appending to cur
        with torch.no_grad():
            _, _, _, _, _, _, _, _, _, _, _, _, inter_logits = policy.path_decoder(prompts, cur, inter_str)
            inter_pred = (torch.sigmoid(inter_logits[:, -1, :]) > 0.5).float()
        inter_str = torch.cat([inter_str, inter_pred.unsqueeze(1)], dim=1)
        next_path = actions_t.unsqueeze(1)
        cur = torch.cat([cur, next_path], dim=1)

    log_probs_new = torch.stack(log_probs_new_list, dim=1)
    values_new = torch.stack(values_new_list, dim=1)
    entropies = torch.stack(entropies_list, dim=1)

    ratio = torch.exp(log_probs_new - log_probs_old)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()
    if value_coef > 0 and advantages is None:
        value_loss = F.mse_loss(values_new, returns)
    else:
        value_loss = torch.tensor(0.0, device=device)
    entropy_loss = -entropies.mean()
    loss = policy_loss + value_coef * value_loss - entropy_coef * entropy_loss
    return loss, rewards_per_t[:, 0].mean().item(), policy_loss.item(), value_loss.item()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to SFT PathDecoder checkpoint")
    parser.add_argument("--scenario", type=str, default="city_47_chicago_3p5")
    parser.add_argument("--save_dir", type=str, default="grpo")

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_steps", type=int, default=25)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--value_coef", type=float, default=0.5)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    
    parser.add_argument("--init_log_std", type=float, default=-2.0,
                        help="Initial log_std (smaller = less exploration, more stable from SFT)")
    parser.add_argument("--warmup_deterministic_epochs", type=int, default=1,
                        help="Use mean action (no sampling) for this many epochs to stabilize")
    parser.add_argument("--algorithm", type=str, default="ppo", choices=["ppo", "grpo"],
                        help="ppo: value baseline + advantage norm. grpo: group-relative advantages, no critic.")
    parser.add_argument("--group_size", type=int, default=4,
                        help="GRPO only: number of rollouts per prompt for group-relative advantage")
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--n_layers", type=int, default=8)
    parser.add_argument("--n_heads", type=int, default=8)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)
    params = ChannelParameters()

    import deepmimo as dm
    dm.download(args.scenario)
    dataset = dm.load(args.scenario)
    train_data = PreTrainMySeqDataLoader(
        dataset, train=True, split_by="user", sort_by="power",
        normalizers=None, apply_normalizers=[], pad_value=0
    )
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, collate_fn=train_data.collate_fn
    )

    path_decoder = PathDecoder(
        hidden_dim=args.hidden_dim, n_layers=args.n_layers, n_heads=args.n_heads
    ).to(device)

    base_model_checkpoint_path = f"{args.checkpoint}_best_model_checkpoint.pth"
    base_model_checkpoint_path = os.path.join("checkpoints2", base_model_checkpoint_path)
    ckpt = torch.load(base_model_checkpoint_path, map_location=device)
    if "model_state_dict" in ckpt:
        path_decoder.load_state_dict(ckpt["model_state_dict"], strict=True)
    else:
        path_decoder.load_state_dict(ckpt, strict=True)
    policy = PathDecoderPolicyWithValue(path_decoder, init_log_std=args.init_log_std).to(device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        policy.train()
        run_reward = []
        run_pl = []
        run_vl = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [{getattr(args, 'algorithm', 'ppo').upper()}]")
        for batch in pbar:
            if len(batch) == 7:
                prompts, paths, path_lengths, interactions, env, env_prop, path_padding_mask = batch
            else:
                prompts, paths, path_lengths, interactions = batch
            prompts = prompts.to(device)
            paths = paths.to(device)
            path_lengths = path_lengths.to(device)
            interactions = interactions.to(device)
            B = prompts.size(0)
            paths_gt = paths[:, 1 : 1 + args.max_steps, :5]
            if paths_gt.size(1) < args.max_steps:
                pad = torch.zeros(B, args.max_steps - paths_gt.size(1), 5, device=device)
                paths_gt = torch.cat([paths_gt, pad], dim=1)
            interactions_gt = interactions[:, 1 : 1 + args.max_steps, :]
            if interactions_gt.size(1) < args.max_steps:
                pad_i = -1 * torch.ones(B, args.max_steps - interactions_gt.size(1), 4, device=device)
                interactions_gt = torch.cat([interactions_gt, pad_i], dim=1)

            use_grpo = getattr(args, "algorithm", "ppo") == "grpo"
            if use_grpo:
                G = getattr(args, "group_size", 4)
                # Expand each prompt G times for group rollouts
                prompts_g = prompts.repeat_interleave(G, dim=0)
                paths_gt_g = paths_gt.repeat_interleave(G, dim=0)
                path_lengths_g = path_lengths.repeat_interleave(G, dim=0)
                interactions_gt_g = interactions_gt.repeat_interleave(G, dim=0)
                generated, log_probs_old, values = rollout(
                    policy, prompts_g, interactions_gt_g, path_lengths_g, args.max_steps, device,
                    deterministic=False, return_value=False,
                )
                rewards_g = compute_channel_nmse_reward(generated, paths_gt_g, path_lengths_g, params, device)
                R = rewards_g.view(B, G)
                adv_group = (R - R.mean(dim=1, keepdim=True)) / (R.std(dim=1, keepdim=True) + 1e-8)
                advantages_grpo = adv_group.view(B * G, 1).expand(-1, args.max_steps)
                values_dummy = torch.zeros(B * G, args.max_steps, device=device, dtype=generated.dtype)
                loss, rew, pl, vl = ppo_loss(
                    policy, prompts_g, paths_gt_g, path_lengths_g, interactions_gt_g,
                    generated, log_probs_old, values_dummy, params, device,
                    clip_eps=args.clip_eps, value_coef=0.0, entropy_coef=args.entropy_coef,
                    advantages=advantages_grpo,
                )
            else:
                generated, log_probs_old, values = rollout(
                    policy, prompts, interactions_gt, path_lengths, args.max_steps, device,
                    deterministic=(epoch < getattr(args, "warmup_deterministic_epochs", 0)),
                    return_value=True,
                )
                loss, rew, pl, vl = ppo_loss(
                    policy, prompts, paths_gt, path_lengths, interactions_gt,
                    generated, log_probs_old, values, params, device,
                    clip_eps=args.clip_eps, value_coef=args.value_coef, entropy_coef=args.entropy_coef,
                )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            run_reward.append(rew)
            run_pl.append(pl)
            run_vl.append(vl)
            pbar.set_postfix(reward=f"{rew:.3f}", policy_loss=f"{pl:.4f}", value_loss=f"{vl:.4f}")

        avg_reward = np.mean(run_reward)
        print(f"Epoch {epoch} avg reward (terminal -log10 NMSE): {avg_reward:.4f}")
        torch.save({
            "policy_state_dict": policy.state_dict(),
            "epoch": epoch,
            "avg_reward": avg_reward,
        }, os.path.join(args.save_dir, "rl_channel_ppo_latest.pt"))

    print("Done.")


if __name__ == "__main__":
    main()
