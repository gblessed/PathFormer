import argparse
import os
import warnings
from pathlib import Path

import deepmimo as dm
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from dataset.dataloaders import PreTrainMySeqDataLoader
from models import PathDecoder
from utils.utils import masked_loss

warnings.filterwarnings("ignore", category=UserWarning)

csv_log_file = "muldims_weighted_first_step_residual_results.csv"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FirstStepResidualPathDecoder(torch.nn.Module):
    def __init__(self, prompt_dim=10, hidden_dim=512, n_layers=8, n_heads=8, max_T=26, prefix_len=4, pad_value=0):
        super().__init__()
        self.backbone = PathDecoder(
            prompt_dim=prompt_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            max_T=max_T,
            prefix_len=prefix_len,
            pad_value=pad_value,
        )
        self.first_delay_residual_head = torch.nn.Linear(hidden_dim, 1)
        self.first_power_residual_head = torch.nn.Linear(hidden_dim, 1)

    def forward(self, prompts, paths, interactions, first_step_baseline):
        h_paths, prefix_flat = self.backbone.forward_hidden(prompts, paths, interactions)
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

        phase_pred = torch.atan2(phase_sin_pred, phase_cos_pred)
        az_pred = torch.atan2(az_sin_pred, az_cos_pred)
        el_pred = torch.atan2(el_sin_pred, el_cos_pred)
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
            pathcounts,
            interaction_logits,
        )


class FirstStepResidualDataset(Dataset):
    def __init__(self, base_dataset, augmented_prompts, first_step_baselines):
        self.base_dataset = base_dataset
        self.augmented_prompts = augmented_prompts
        self.first_step_baselines = first_step_baselines

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        return self.base_dataset[idx], self.augmented_prompts[idx], self.first_step_baselines[idx]

    def collate_fn(self, batch):
        base_items = [item[0] for item in batch]
        aug_prompts = torch.stack([item[1] for item in batch], dim=0)
        first_step_baselines = torch.stack([item[2] for item in batch], dim=0)
        _, paths, path_lengths, interactions, env, env_prop, path_padding_mask = self.base_dataset.collate_fn(base_items)
        return aug_prompts, paths, path_lengths, interactions, env, env_prop, path_padding_mask, first_step_baselines


def compute_stop_metrics(path_count, targets):
    return np.sqrt(mean_squared_error(path_count.cpu().numpy(), targets.squeeze().cpu().numpy()))


def compute_interaction_metrics_from_logits(interaction_logits, interaction_targets):
    interaction_mask = (interaction_targets[:, :, 0] != -1)
    if not interaction_mask.any():
        return 0.0, 0.0
    valid_logits = interaction_logits[interaction_mask]
    valid_targets = interaction_targets[interaction_mask]
    valid_preds = (torch.sigmoid(valid_logits) > 0.5).int().detach().cpu().numpy()
    valid_targets = valid_targets.int().detach().cpu().numpy()
    accuracy = accuracy_score(valid_targets.reshape(-1), valid_preds.reshape(-1))
    f1 = f1_score(valid_targets.reshape(-1), valid_preds.reshape(-1), zero_division=0)
    return accuracy, f1


def _extract_first_step_metadata(seq_dataset):
    samples = []
    for idx in range(len(seq_dataset)):
        prompt, paths, *_ = seq_dataset[idx]
        tx_key = tuple(prompt[:3].numpy().tolist())
        rx_pos = prompt[3:].numpy().astype(np.float32)
        first_target = paths[1, :2].numpy().astype(np.float32) if paths.shape[0] > 1 else np.zeros(2, dtype=np.float32)
        samples.append({"tx_key": tx_key, "rx_pos": rx_pos, "first_target": first_target})
    return samples


def _compute_cluster_stats(targets, labels, centers):
    stds = np.zeros_like(centers, dtype=np.float32)
    for k in range(centers.shape[0]):
        members = targets[labels == k]
        if len(members) > 0:
            stds[k] = members.std(axis=0).astype(np.float32)
    return stds


def build_first_step_assignments(train_data, val_data, n_clusters):
    train_meta = _extract_first_step_metadata(train_data)
    val_meta = _extract_first_step_metadata(val_data)

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
            prompt, *_ = train_data[dataset_idx]
            baseline = centers[labels[local_idx]]
            std = stds[labels[local_idx]]
            train_aug_prompts[dataset_idx] = torch.from_numpy(np.concatenate([prompt.numpy(), baseline, std], axis=0).astype(np.float32))
            train_baselines[dataset_idx] = torch.from_numpy(baseline.astype(np.float32))

        if tx_key not in val_groups:
            continue

        val_rx = np.stack(val_groups[tx_key]["rx_pos"], axis=0).astype(np.float32)
        dists = np.sum((val_rx[:, None, :] - rx_pos[None, :, :]) ** 2, axis=2)
        nearest_train_idx = np.argmin(dists, axis=1)
        assigned_labels = labels[nearest_train_idx]

        for local_idx, dataset_idx in enumerate(val_groups[tx_key]["indices"]):
            prompt, *_ = val_data[dataset_idx]
            baseline = centers[assigned_labels[local_idx]]
            std = stds[assigned_labels[local_idx]]
            val_aug_prompts[dataset_idx] = torch.from_numpy(np.concatenate([prompt.numpy(), baseline, std], axis=0).astype(np.float32))
            val_baselines[dataset_idx] = torch.from_numpy(baseline.astype(np.float32))

    for idx in range(len(train_aug_prompts)):
        if train_aug_prompts[idx] is None:
            prompt, *_ = train_data[idx]
            zeros = np.zeros(4, dtype=np.float32)
            train_aug_prompts[idx] = torch.from_numpy(np.concatenate([prompt.numpy(), zeros], axis=0).astype(np.float32))
            train_baselines[idx] = torch.zeros(2, dtype=torch.float32)
    for idx in range(len(val_aug_prompts)):
        if val_aug_prompts[idx] is None:
            prompt, *_ = val_data[idx]
            zeros = np.zeros(4, dtype=np.float32)
            val_aug_prompts[idx] = torch.from_numpy(np.concatenate([prompt.numpy(), zeros], axis=0).astype(np.float32))
            val_baselines[idx] = torch.zeros(2, dtype=torch.float32)

    return train_aug_prompts, train_baselines, val_aug_prompts, val_baselines


@torch.no_grad()
def generate_paths_first_step_residual_batch(model, prompts, first_step_baselines, max_steps=25):
    model.eval()
    device = next(model.parameters()).device
    prompts = prompts.to(device)
    first_step_baselines = first_step_baselines.to(device)

    cur = torch.zeros(prompts.shape[0], 1, 5, device=device)
    inter_str = -1 * torch.ones(prompts.shape[0], 1, 4, device=device)

    outputs = []
    outputs_inter = []
    pathcounts = None

    for _ in range(max_steps):
        d, p, _, _, ph, _, _, az, _, _, el, pathcounts, inter_logits = model(prompts, cur, inter_str, first_step_baselines)
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
        torch.stack(outputs, dim=1).detach().cpu(),
        pathcounts.detach().cpu(),
        torch.stack(outputs_inter, dim=1).detach().cpu(),
    )


def evaluate_model(model, val_loader, max_generate=25):
    model.eval()
    delay_errors, power_errors, phase_errors = [], [], []
    az_errors, el_errors, path_length_rmses = [], [], []
    delay_maes, power_maes, phase_maes = [], [], []
    az_maes, el_maes, path_length_maes = [], [], []
    interaction_targets_all, interaction_preds_all = [], []

    with torch.no_grad():
        for prompts, paths, path_lengths, interactions, env, env_prop, path_padding_mask, first_step_baselines in tqdm(val_loader, desc="Evaluating", leave=True):
            prompts = prompts.cuda()
            paths = paths.cuda()
            path_lengths = path_lengths.cuda()
            interactions = interactions.cuda()
            first_step_baselines = first_step_baselines.cuda()

            generated, path_lengths_pred, inter_str_pred = generate_paths_first_step_residual_batch(
                model,
                prompts,
                first_step_baselines,
                max_steps=max_generate,
            )
            generated = generated.cuda()
            if path_lengths_pred.dim() > 1:
                path_lengths_pred = path_lengths_pred.squeeze(-1)

            for b in range(prompts.size(0)):
                n_valid = int(round(path_lengths[b].item() * 25))
                gt = paths[b][1:1 + n_valid, :5]
                gt_inter = interactions[b][1:1 + n_valid, :]
                T = min(len(gt), generated.size(1))
                pred = generated[b, :T]
                gt = gt[:T]
                pred_inter = inter_str_pred[b, :T]
                gt_inter = gt_inter[:T].detach().cpu()

                valid_interaction_mask = (gt_inter[:, 0] != -1)
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
                az_rmse = torch.mean(az_dist ** 2).sqrt().item()
                el_rmse = torch.mean(el_dist ** 2).sqrt().item()
                az_mae = torch.mean(torch.abs(az_dist)).item()
                el_mae = torch.mean(torch.abs(el_dist)).item()

                path_len_pred_b = path_lengths_pred[b]
                length_rmse = torch.mean((path_len_pred_b - path_lengths[b]) ** 2).sqrt().item()
                length_mae = torch.mean(torch.abs(path_len_pred_b - path_lengths[b])).item()

                delay_errors.append(delay_rmse); power_errors.append(power_rmse); phase_errors.append(phase_rmse)
                az_errors.append(az_rmse); el_errors.append(el_rmse); path_length_rmses.append(length_rmse)
                delay_maes.append(delay_mae); power_maes.append(power_mae); phase_maes.append(phase_mae)
                az_maes.append(az_mae); el_maes.append(el_mae); path_length_maes.append(length_mae)

    if interaction_targets_all:
        targets = np.concatenate(interaction_targets_all, axis=0)
        preds = np.concatenate(interaction_preds_all, axis=0)
        avg_interaction_accuracy = accuracy_score(targets.reshape(-1), preds.reshape(-1))
        avg_interaction_f1 = f1_score(targets.reshape(-1), preds.reshape(-1), zero_division=0)
    else:
        avg_interaction_accuracy = 0.0
        avg_interaction_f1 = 0.0

    return (
        np.mean(delay_errors), np.mean(power_errors), np.mean(phase_errors),
        np.mean(az_errors), np.mean(el_errors), np.mean(path_length_rmses),
        avg_interaction_accuracy, avg_interaction_f1,
        np.mean(delay_maes), np.mean(power_maes), np.mean(phase_maes),
        np.mean(az_maes), np.mean(el_maes), np.mean(path_length_maes),
    )


def train_with_interactions(model, train_loader, val_loader, config, train_data, optimizer, scheduler, checkpoint_path):
    best_val_loss = float("inf")
    for epoch in range(config["epochs"]):
        model.train()
        train_losses = []
        for prompts, paths, path_lengths, interactions, env, env_prop, path_padding_mask, first_step_baselines in tqdm(train_loader, desc=f"Epoch {epoch} [Train]", leave=False):
            prompts = prompts.cuda()
            paths = paths.cuda()
            path_lengths = path_lengths.cuda()
            interactions = interactions.cuda()
            path_padding_mask = path_padding_mask.cuda()
            first_step_baselines = first_step_baselines.cuda()

            paths_in = paths[:, :-1, :]
            interactions_in = interactions[:, :-1, :]
            paths_out = paths[:, 1:, :]
            interactions_out = interactions[:, 1:, :]

            outputs = model(prompts, paths_in, interactions_in, first_step_baselines)
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

        model.eval()
        val_losses = []
        with torch.no_grad():
            for prompts, paths, path_lengths, interactions, env, env_prop, path_padding_mask, first_step_baselines in tqdm(val_loader, desc=f"Epoch {epoch} [Val]", leave=False):
                prompts = prompts.cuda()
                paths = paths.cuda()
                path_lengths = path_lengths.cuda()
                interactions = interactions.cuda()
                path_padding_mask = path_padding_mask.cuda()
                first_step_baselines = first_step_baselines.cuda()

                outputs = model(prompts, paths[:, :-1, :], interactions[:, :-1, :], first_step_baselines)
                total_loss, *_ = masked_loss(
                    *outputs,
                    paths[:, 1:, :],
                    path_lengths,
                    interactions[:, 1:, :],
                    pad_value=train_data.pad_value,
                    interaction_weight=config.get("interaction_weight", 0.1),
                    path_padding_mask=path_padding_mask,
                    time_step_weighted=config.get("time_step_weighted", False),
                )
                val_losses.append(total_loss.item())

        avg_val_loss = float(np.mean(val_losses))
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


all_scenarios = ['city_23_beijing_3p5', 'city_91_xiangyang_3p5', 'city_17_seattle_3p5_s', 'city_12_fortworth_3p5', 'city_92_sãopaulo_3p5', 'city_35_san_francisco_3p5', 'city_10_florida_villa_7gp_1758095156175', 'city_19_oklahoma_3p5_s', 'city_74_chiyoda_3p5'][1:]


def parse_args():
    parser = argparse.ArgumentParser(description="Train/evaluate first-step residual PathDecoder across scenarios.")
    parser.add_argument("scenarios", nargs="*", help="Optional scenario names to run.")
    parser.add_argument("--scenario", dest="scenario_flag", action="append")
    parser.add_argument("--scenario-file", type=str)
    parser.add_argument("--shard-index", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=None)
    parser.add_argument("--csv-log-file", type=str, default=csv_log_file)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints_first_step_residual")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--n-clusters", type=int, default=20)
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
        scenarios = list(all_scenarios)
    if args.num_shards is not None or args.shard_index is not None:
        if args.num_shards is None or args.shard_index is None:
            raise ValueError("Both --num-shards and --shard-index must be provided together.")
        scenarios = [s for i, s in enumerate(scenarios) if i % args.num_shards == args.shard_index]
    return scenarios


def load_best_checkpoint(model, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        return None, None
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint["epoch"], checkpoint["best_val_loss"]


def run_scenario(scenario, args):
    dataset = dm.load(scenario)
    config = {
        "BATCH_SIZE": 128,
        "PAD_VALUE": 0,
        "LR": 2e-5,
        "epochs": 300,
        "interaction_weight": 0.01,
        "experiment": f"first_step_residual_{scenario}",
        "hidden_dim": 512,
        "n_layers": 8,
        "n_heads": 8,
        "time_step_weighted": False,
    }



    base_train = PreTrainMySeqDataLoader(dataset, train=True, split_by="user", sort_by="power", normalizers=None, apply_normalizers=[], pad_value=config["PAD_VALUE"])

    normalizer = {
        "rx_pos": {
            "max": -1e9 * np.ones(3),
            "min": 1e9 * np.ones(3),
        }
    }

    for data in base_train:
        prompt = data[0]
        rx = prompt[3:6]
        if isinstance(rx, torch.Tensor):
            rx = rx.cpu().numpy()
        rx_x, rx_y, rx_z = float(rx[0]), float(rx[1]), float(rx[2])

        normalizer["rx_pos"]["max"][0] = max(normalizer["rx_pos"]["max"][0], rx_x)
        normalizer["rx_pos"]["max"][1] = max(normalizer["rx_pos"]["max"][1], rx_y)
        normalizer["rx_pos"]["max"][2] = max(normalizer["rx_pos"]["max"][2], rx_z)

        normalizer["rx_pos"]["min"][0] = min(normalizer["rx_pos"]["min"][0], rx_x)
        normalizer["rx_pos"]["min"][1] = min(normalizer["rx_pos"]["min"][1], rx_y)
        normalizer["rx_pos"]["min"][2] = min(normalizer["rx_pos"]["min"][2], rx_z)    
    # compute min/max rx_pos over training prompts'
    print(prompt)

    print(normalizer)


    base_train = PreTrainMySeqDataLoader(dataset, train=True, split_by="user", sort_by="power", normalizers=normalizer, apply_normalizers=["rx_pos"], pad_value=config["PAD_VALUE"])
    print(f"base_train: {base_train[0]}")
    base_val = PreTrainMySeqDataLoader(dataset, train=False, split_by="user", sort_by="power", normalizers=normalizer, apply_normalizers=["rx_pos"], pad_value=config["PAD_VALUE"])

    train_aug_prompts, train_baselines, val_aug_prompts, val_baselines = build_first_step_assignments(base_train, base_val, n_clusters=args.n_clusters)

    train_data = FirstStepResidualDataset(base_train, train_aug_prompts, train_baselines)
    val_data = FirstStepResidualDataset(base_val, val_aug_prompts, val_baselines)
    train_loader = DataLoader(train_data, batch_size=config["BATCH_SIZE"], shuffle=True, collate_fn=train_data.collate_fn)
    val_loader = DataLoader(val_data, batch_size=config["BATCH_SIZE"], shuffle=False, collate_fn=val_data.collate_fn)

    model = FirstStepResidualPathDecoder(prompt_dim=14, hidden_dim=config["hidden_dim"], n_layers=config["n_layers"], n_heads=config["n_heads"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["LR"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=25, T_mult=1, eta_min=1e-8)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.checkpoint_dir, f"{config['experiment']}_best_model_checkpoint.pth")

    if not args.skip_train:
        train_with_interactions(model, train_loader, val_loader, config, base_train, optimizer, scheduler, checkpoint_path)

    _, best_loss = load_best_checkpoint(model, checkpoint_path)
    results = evaluate_model(model, val_loader)
    avg_delay, avg_power, avg_phase, avg_az, avg_el, avg_path_length_rmse, avg_interaction_accuracy, avg_interaction_f1, avg_delay_mae, avg_power_mae, avg_phase_mae, avg_az_mae, avg_el_mae, avg_path_length_mae = results
    row = {
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
