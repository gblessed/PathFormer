import argparse
import math
import os
import warnings

import deepmimo as dm
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.dataloaders import PreTrainMySeqDataLoader
from models import PathDecoder
from multiscenario_direct_training_first_step_residual import (
    FirstStepResidualDataset,
    all_scenarios,
    get_resume_checkpoint_path,
    load_best_checkpoint,
    resolve_scenarios,
)
from multiscenario_direct_training_first_step_residual_corridor import (
    build_first_step_assignments_with_corridor,
)
from scene_feature_utils import SceneFeatureBank
from utils.utils import add_noise_to_paths

warnings.filterwarnings("ignore", category=UserWarning)

csv_log_file = "muldims_weighted_first_step_residual_corridor_classifier_results.csv"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FIELD_NAMES = ["delay", "power", "phase", "aoa_az", "aoa_el", "aod_az", "aod_el"]
ANGLE_FIELDS = {"phase", "aoa_az", "aoa_el", "aod_az", "aod_el"}


def wrap_angle_tensor(x):
    return torch.atan2(torch.sin(x), torch.cos(x))


class PathValueTokenizer:
    def __init__(self, num_bins=64, field_names=None):
        self.num_bins = int(num_bins)
        self.field_names = field_names or list(FIELD_NAMES)
        self.field_to_idx = {name: idx for idx, name in enumerate(self.field_names)}
        self.boundaries = {}
        self.centers = {}

    def fit(self, seq_dataset):
        collected = {name: [] for name in self.field_names}
        for idx in range(len(seq_dataset)):
            _, paths, *_ = seq_dataset[idx]
            if paths.size(0) <= 1:
                continue
            target = paths[1:, : len(self.field_names)].numpy().astype(np.float32)
            for field_idx, name in enumerate(self.field_names):
                values = target[:, field_idx]
                values = values[np.isfinite(values)]
                if values.size:
                    collected[name].append(values)

        for name in self.field_names:
            if name in ANGLE_FIELDS:
                edges = np.linspace(-np.pi, np.pi, self.num_bins + 1, dtype=np.float32)
            else:
                if not collected[name]:
                    raise ValueError(f"No training values found for field '{name}'.")
                values = np.concatenate(collected[name], axis=0)
                quantiles = np.linspace(0.0, 1.0, self.num_bins + 1, dtype=np.float64)
                edges = np.quantile(values, quantiles).astype(np.float32)
                edges = self._make_monotonic(edges)

            centers = 0.5 * (edges[:-1] + edges[1:])
            if name in ANGLE_FIELDS:
                centers = np.arctan2(np.sin(centers), np.cos(centers)).astype(np.float32)
            self.boundaries[name] = torch.tensor(edges[1:-1], dtype=torch.float32)
            self.centers[name] = torch.tensor(centers, dtype=torch.float32)

        return self

    @staticmethod
    def _make_monotonic(edges):
        out = edges.copy()
        for i in range(1, out.size):
            if out[i] <= out[i - 1]:
                out[i] = out[i - 1] + 1e-6
        return out

    def encode_values(self, values, field_name):
        boundaries = self.boundaries[field_name].to(values.device)
        if field_name in ANGLE_FIELDS:
            values = wrap_angle_tensor(values)
        return torch.bucketize(values, boundaries)

    def decode_indices(self, indices, field_name):
        centers = self.centers[field_name].to(indices.device)
        decoded = centers[indices.clamp(min=0, max=centers.numel() - 1)]
        if field_name in ANGLE_FIELDS:
            decoded = wrap_angle_tensor(decoded)
        return decoded

    def encode_paths(self, paths, valid_mask, skip_first=False):
        labels = torch.full(
            (paths.size(0), paths.size(1), len(self.field_names)),
            -100,
            dtype=torch.long,
            device=paths.device,
        )
        for field_idx, name in enumerate(self.field_names):
            field_values = paths[:, :, field_idx]
            mask = valid_mask.clone()
            if skip_first and mask.size(1) > 0:
                mask[:, 0] = False
            if not mask.any():
                continue
            field_labels = labels[:, :, field_idx]
            field_labels[mask] = self.encode_values(field_values[mask], name)
        return labels

    def dequantize_labels(self, labels, reference_paths):
        out = reference_paths.clone()
        for field_idx, name in enumerate(self.field_names):
            valid = labels[:, :, field_idx] >= 0
            if not valid.any():
                continue
            decoded = self.decode_indices(labels[:, :, field_idx].clamp_min(0), name)
            out[:, :, field_idx][valid] = decoded[valid]
        return out

    def quantize_teacher_forcing_inputs(self, paths, valid_mask):
        labels = self.encode_paths(paths, valid_mask=valid_mask, skip_first=True)
        return self.dequantize_labels(labels, paths)

    def labels_to_paths(self, label_tensor):
        out = torch.zeros(
            label_tensor.size(0),
            label_tensor.size(1),
            len(self.field_names),
            device=label_tensor.device,
            dtype=torch.float32,
        )
        for field_idx, name in enumerate(self.field_names):
            out[:, :, field_idx] = self.decode_indices(label_tensor[:, :, field_idx], name)
        return out


class FirstStepResidualCorridorClassifier(torch.nn.Module):
    def __init__(
        self,
        prompt_dim=10,
        hidden_dim=512,
        n_layers=8,
        n_heads=8,
        max_T=26,
        prefix_len=4,
        pad_value=0,
        include_aod=True,
        num_bins=64,
    ):
        super().__init__()
        self.field_names = list(FIELD_NAMES)
        self.num_bins = int(num_bins)
        self.backbone = PathDecoder(
            prompt_dim=prompt_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            max_T=max_T,
            prefix_len=prefix_len,
            pad_value=pad_value,
            include_aod=include_aod,
        )
        self.class_heads = torch.nn.ModuleDict(
            {name: torch.nn.Linear(hidden_dim, self.num_bins) for name in self.field_names}
        )

    def forward(self, prompts, paths, interactions, first_step_baseline=None):
        del first_step_baseline
        h_paths, prefix_flat = self.backbone.forward_hidden(prompts, paths, interactions)
        field_logits = {name: head(h_paths) for name, head in self.class_heads.items()}
        interaction_logits = self.backbone.interaction_head(h_paths)
        pathcounts = self.backbone.pathcount_head(prefix_flat)
        return {
            "field_logits": field_logits,
            "interaction_logits": interaction_logits,
            "pathcounts": pathcounts,
        }


def compute_interaction_metrics(interaction_logits, interaction_targets):
    interaction_mask = interaction_targets[:, :, 0] != -1
    if not interaction_mask.any():
        return 0.0, 0.0
    valid_logits = interaction_logits[interaction_mask]
    valid_targets = interaction_targets[interaction_mask]
    valid_preds = (torch.sigmoid(valid_logits) > 0.5).int().detach().cpu().numpy()
    valid_targets = valid_targets.int().detach().cpu().numpy()
    accuracy = accuracy_score(valid_targets.reshape(-1), valid_preds.reshape(-1))
    f1 = f1_score(valid_targets.reshape(-1), valid_preds.reshape(-1), zero_division=0)
    return accuracy, f1


def prepare_batch_for_classifier(paths, interactions, path_padding_mask, tokenizer):
    input_mask = path_padding_mask[:, :-1].clone()
    target_mask = path_padding_mask[:, 1:].clone()
    paths_in = tokenizer.quantize_teacher_forcing_inputs(paths[:, :-1, : len(FIELD_NAMES)], input_mask)
    paths_out = paths[:, 1:, : len(FIELD_NAMES)]
    target_labels = tokenizer.encode_paths(paths_out, valid_mask=target_mask, skip_first=False)
    interactions_in = interactions[:, :-1, :]
    interactions_out = interactions[:, 1:, :]
    return paths_in, interactions_in, paths_out, interactions_out, target_labels, target_mask


def compute_classifier_loss(outputs, target_labels, target_mask, path_lengths, interactions_out, interaction_weight=0.01, pathcount_weight=0.05):
    field_losses = []
    for field_idx, field_name in enumerate(FIELD_NAMES):
        logits = outputs["field_logits"][field_name]
        labels = target_labels[:, :, field_idx]
        valid = target_mask & (labels >= 0)
        if valid.any():
            field_losses.append(F.cross_entropy(logits[valid], labels[valid]))

    if not field_losses:
        class_loss = torch.tensor(0.0, device=path_lengths.device)
    else:
        class_loss = torch.stack(field_losses).mean()

    interaction_mask = interactions_out[:, :, 0] != -1
    if interaction_mask.any():
        interaction_loss = F.binary_cross_entropy_with_logits(
            outputs["interaction_logits"][interaction_mask],
            interactions_out[interaction_mask],
        )
    else:
        interaction_loss = torch.tensor(0.0, device=path_lengths.device)

    pathcount_pred = outputs["pathcounts"]
    if pathcount_pred.dim() > 1:
        pathcount_pred = pathcount_pred.squeeze(-1)
    pathcount_target = path_lengths.squeeze(-1)
    pathcount_loss = F.mse_loss(pathcount_pred, pathcount_target)

    total_loss = class_loss + interaction_weight * interaction_loss + pathcount_weight * pathcount_loss
    return total_loss, class_loss, interaction_loss, pathcount_loss


@torch.no_grad()
def generate_paths_classifier_batch(model, tokenizer, prompts, first_step_baselines, max_steps=25):
    model.eval()
    prompts = prompts.to(device)
    first_step_baselines = first_step_baselines.to(device)

    cur = torch.zeros(prompts.shape[0], 1, len(FIELD_NAMES), device=device)
    inter_str = -1 * torch.ones(prompts.shape[0], 1, 4, device=device)

    outputs = []
    outputs_inter = []
    pathcounts = None

    for _ in range(max_steps):
        pred = model(prompts, cur, inter_str, first_step_baselines)
        step_labels = []
        for field_name in FIELD_NAMES:
            logits = pred["field_logits"][field_name][:, -1, :]
            step_labels.append(torch.argmax(logits, dim=-1))
        step_labels = torch.stack(step_labels, dim=-1)
        next_path = tokenizer.labels_to_paths(step_labels.unsqueeze(1)).squeeze(1)
        inter_pred_t = (torch.sigmoid(pred["interaction_logits"][:, -1]) > 0.5).float()

        outputs.append(next_path)
        outputs_inter.append(inter_pred_t)
        cur = torch.cat([cur, next_path.unsqueeze(1)], dim=1)
        inter_str = torch.cat([inter_str, inter_pred_t.unsqueeze(1)], dim=1)
        pathcounts = pred["pathcounts"]

    generated = torch.stack(outputs, dim=1).detach().cpu()
    inter_str_pred = torch.stack(outputs_inter, dim=1).detach().cpu()
    pathcounts = pathcounts.detach().cpu()
    return generated, pathcounts, inter_str_pred


def _mean_std(values):
    if not values:
        return 0.0, 0.0
    arr = np.asarray(values, dtype=np.float64)
    return float(np.mean(arr)), float(np.std(arr))


@torch.no_grad()
def evaluate_classifier_model(model, val_loader, tokenizer, max_generate=25):
    model.eval()

    delay_errors, power_errors, phase_errors = [], [], []
    az_errors, el_errors, path_length_rmses = [], [], []
    aod_az_errors, aod_el_errors = [], []
    delay_maes, power_maes, phase_maes = [], [], []
    az_maes, el_maes, path_length_maes = [], [], []
    aod_az_maes, aod_el_maes = [], []
    interaction_targets_all, interaction_preds_all = [], []
    val_total_losses, val_class_losses = [], []
    nll_tokens = []
    per_sample_top1, per_sample_top3, per_sample_adj = [], [], []

    for prompts, paths, path_lengths, interactions, env, env_prop, path_padding_mask, first_step_baselines in tqdm(val_loader, desc="Evaluating", leave=True):
        del env, env_prop
        prompts = prompts.to(device)
        paths = paths.to(device)
        path_lengths = path_lengths.to(device)
        interactions = interactions.to(device)
        path_padding_mask = path_padding_mask.to(device)
        first_step_baselines = first_step_baselines.to(device)

        paths_in, interactions_in, paths_out, interactions_out, target_labels, target_mask = prepare_batch_for_classifier(
            paths, interactions, path_padding_mask, tokenizer
        )
        outputs = model(prompts, paths_in, interactions_in, first_step_baselines)
        total_loss, class_loss, _, _ = compute_classifier_loss(
            outputs,
            target_labels,
            target_mask,
            path_lengths,
            interactions_out,
        )
        val_total_losses.append(total_loss.item())
        val_class_losses.append(class_loss.item())

        batch_top1 = torch.zeros(paths.size(0), device=paths.device)
        batch_top3 = torch.zeros(paths.size(0), device=paths.device)
        batch_adj = torch.zeros(paths.size(0), device=paths.device)
        batch_counts = torch.zeros(paths.size(0), device=paths.device)

        for field_idx, field_name in enumerate(FIELD_NAMES):
            logits = outputs["field_logits"][field_name]
            labels = target_labels[:, :, field_idx]
            valid = target_mask & (labels >= 0)
            if not valid.any():
                continue

            log_probs = F.log_softmax(logits, dim=-1)
            nll = -log_probs.gather(-1, labels.unsqueeze(-1).clamp_min(0)).squeeze(-1)
            nll_tokens.extend(nll[valid].detach().cpu().tolist())

            pred_top1 = torch.argmax(logits, dim=-1)
            pred_top3 = torch.topk(logits, k=min(3, logits.size(-1)), dim=-1).indices
            correct_top1 = (pred_top1 == labels) & valid
            correct_top3 = pred_top3.eq(labels.unsqueeze(-1)).any(dim=-1) & valid
            correct_adj = ((pred_top1 - labels).abs() <= 1) & valid

            batch_top1 += correct_top1.float().sum(dim=1)
            batch_top3 += correct_top3.float().sum(dim=1)
            batch_adj += correct_adj.float().sum(dim=1)
            batch_counts += valid.float().sum(dim=1)

        valid_samples = batch_counts > 0
        if valid_samples.any():
            per_sample_top1.extend((batch_top1[valid_samples] / batch_counts[valid_samples]).detach().cpu().tolist())
            per_sample_top3.extend((batch_top3[valid_samples] / batch_counts[valid_samples]).detach().cpu().tolist())
            per_sample_adj.extend((batch_adj[valid_samples] / batch_counts[valid_samples]).detach().cpu().tolist())

        generated, path_lengths_pred, inter_str_pred = generate_paths_classifier_batch(
            model,
            tokenizer,
            prompts,
            first_step_baselines,
            max_steps=max_generate,
        )
        generated = generated.to(device)

        if path_lengths_pred.dim() > 1:
            path_lengths_pred = path_lengths_pred.squeeze(-1)

        for b in range(prompts.size(0)):
            n_valid = int(round(path_lengths[b].item() * 25))
            gt = paths[b][1:1 + n_valid, : len(FIELD_NAMES)]
            gt_inter = interactions[b][1:1 + n_valid, :]
            T = min(len(gt), generated.size(1))
            pred = generated[b, :T]
            gt = gt[:T]
            pred_inter = inter_str_pred[b, :T]
            gt_inter = gt_inter[:T].detach().cpu()

            valid_interaction_mask = gt_inter[:, 0] != -1
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
            aod_az_dist = ((pred[:, 5] / (np.pi / 180)) - (gt[:, 5] / (np.pi / 180)) + 180) % 360 - 180
            aod_el_dist = ((pred[:, 6] / (np.pi / 180)) - (gt[:, 6] / (np.pi / 180)) + 180) % 360 - 180
            az_rmse = torch.mean(az_dist ** 2).sqrt().item()
            el_rmse = torch.mean(el_dist ** 2).sqrt().item()
            aod_az_rmse = torch.mean(aod_az_dist ** 2).sqrt().item()
            aod_el_rmse = torch.mean(aod_el_dist ** 2).sqrt().item()
            az_mae = torch.mean(torch.abs(az_dist)).item()
            el_mae = torch.mean(torch.abs(el_dist)).item()
            aod_az_mae = torch.mean(torch.abs(aod_az_dist)).item()
            aod_el_mae = torch.mean(torch.abs(aod_el_dist)).item()

            path_len_pred_b = path_lengths_pred[b]
            length_rmse = torch.mean((path_len_pred_b - path_lengths[b]) ** 2).sqrt().item()
            length_mae = torch.mean(torch.abs(path_len_pred_b - path_lengths[b])).item()

            delay_errors.append(delay_rmse)
            power_errors.append(power_rmse)
            phase_errors.append(phase_rmse)
            az_errors.append(az_rmse)
            el_errors.append(el_rmse)
            aod_az_errors.append(aod_az_rmse)
            aod_el_errors.append(aod_el_rmse)
            path_length_rmses.append(length_rmse)

            delay_maes.append(delay_mae)
            power_maes.append(power_mae)
            phase_maes.append(phase_mae)
            az_maes.append(az_mae)
            el_maes.append(el_mae)
            aod_az_maes.append(aod_az_mae)
            aod_el_maes.append(aod_el_mae)
            path_length_maes.append(length_mae)

    if interaction_targets_all:
        targets = np.concatenate(interaction_targets_all, axis=0)
        preds = np.concatenate(interaction_preds_all, axis=0)
        avg_interaction_accuracy = accuracy_score(targets.reshape(-1), preds.reshape(-1))
        avg_interaction_f1 = f1_score(targets.reshape(-1), preds.reshape(-1), zero_division=0)
        interaction_accuracy_per_sample = [
            accuracy_score(target.reshape(-1), pred.reshape(-1))
            for target, pred in zip(interaction_targets_all, interaction_preds_all)
        ]
        interaction_f1_per_sample = [
            f1_score(target.reshape(-1), pred.reshape(-1), zero_division=0)
            for target, pred in zip(interaction_targets_all, interaction_preds_all)
        ]
        _, std_interaction_accuracy = _mean_std(interaction_accuracy_per_sample)
        _, std_interaction_f1 = _mean_std(interaction_f1_per_sample)
    else:
        avg_interaction_accuracy = 0.0
        avg_interaction_f1 = 0.0
        std_interaction_accuracy = 0.0
        std_interaction_f1 = 0.0

    avg_nll, std_nll = _mean_std(nll_tokens)
    perplexity = float(math.exp(avg_nll)) if nll_tokens else 0.0
    avg_top1, std_top1 = _mean_std(per_sample_top1)
    avg_top3, std_top3 = _mean_std(per_sample_top3)
    avg_adj, std_adj = _mean_std(per_sample_adj)
    avg_val_loss, std_val_loss = _mean_std(val_total_losses)
    avg_val_class_loss, std_val_class_loss = _mean_std(val_class_losses)

    avg_delay, std_delay = _mean_std(delay_errors)
    avg_power, std_power = _mean_std(power_errors)
    avg_phase, std_phase = _mean_std(phase_errors)
    avg_az, std_az = _mean_std(az_errors)
    avg_el, std_el = _mean_std(el_errors)
    avg_aod_az, std_aod_az = _mean_std(aod_az_errors)
    avg_aod_el, std_aod_el = _mean_std(aod_el_errors)
    avg_path_length_rmse, std_path_length_rmse = _mean_std(path_length_rmses)
    avg_delay_mae, std_delay_mae = _mean_std(delay_maes)
    avg_power_mae, std_power_mae = _mean_std(power_maes)
    avg_phase_mae, std_phase_mae = _mean_std(phase_maes)
    avg_az_mae, std_az_mae = _mean_std(az_maes)
    avg_el_mae, std_el_mae = _mean_std(el_maes)
    avg_aod_az_mae, std_aod_az_mae = _mean_std(aod_az_maes)
    avg_aod_el_mae, std_aod_el_mae = _mean_std(aod_el_maes)
    avg_path_length_mae, std_path_length_mae = _mean_std(path_length_maes)

    return {
        "delay_rmse": avg_delay,
        "delay_rmse_std": std_delay,
        "power_rmse": avg_power,
        "power_rmse_std": std_power,
        "phase_rmse": avg_phase,
        "phase_rmse_std": std_phase,
        "az_rmse": avg_az,
        "az_rmse_std": std_az,
        "el_rmse": avg_el,
        "el_rmse_std": std_el,
        "aod_az_rmse": avg_aod_az,
        "aod_az_rmse_std": std_aod_az,
        "aod_el_rmse": avg_aod_el,
        "aod_el_rmse_std": std_aod_el,
        "path_length_rmse": avg_path_length_rmse,
        "path_length_rmse_std": std_path_length_rmse,
        "interaction_accuracy": avg_interaction_accuracy,
        "interaction_accuracy_std": std_interaction_accuracy,
        "interaction_f1": avg_interaction_f1,
        "interaction_f1_std": std_interaction_f1,
        "delay_mae": avg_delay_mae,
        "delay_mae_std": std_delay_mae,
        "power_mae": avg_power_mae,
        "power_mae_std": std_power_mae,
        "phase_mae": avg_phase_mae,
        "phase_mae_std": std_phase_mae,
        "avg_az_mae": avg_az_mae,
        "avg_az_mae_std": std_az_mae,
        "avg_el_mae": avg_el_mae,
        "avg_el_mae_std": std_el_mae,
        "avg_aod_az_mae": avg_aod_az_mae,
        "avg_aod_az_mae_std": std_aod_az_mae,
        "avg_aod_el_mae": avg_aod_el_mae,
        "avg_aod_el_mae_std": std_aod_el_mae,
        "path_length_mae": avg_path_length_mae,
        "path_length_mae_std": std_path_length_mae,
        "nll": avg_nll,
        "nll_std": std_nll,
        "perplexity": perplexity,
        "top1_accuracy": avg_top1,
        "top1_accuracy_std": std_top1,
        "top3_accuracy": avg_top3,
        "top3_accuracy_std": std_top3,
        "adjacent_accuracy": avg_adj,
        "adjacent_accuracy_std": std_adj,
        "val_total_loss": avg_val_loss,
        "val_total_loss_std": std_val_loss,
        "val_class_loss": avg_val_class_loss,
        "val_class_loss_std": std_val_class_loss,
    }


def load_training_checkpoint(model, optimizer, scheduler, checkpoint_path, resume_checkpoint_path=None):
    candidate_paths = []
    if resume_checkpoint_path is not None:
        candidate_paths.append(resume_checkpoint_path)
    candidate_paths.append(checkpoint_path)

    for path in candidate_paths:
        if path and os.path.exists(path):
            checkpoint = torch.load(path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            if optimizer is not None and "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if scheduler is not None and "scheduler_state_dict" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            start_epoch = int(checkpoint.get("epoch", -1)) + 1
            best_val_loss = checkpoint.get("best_val_loss", float("inf"))
            if hasattr(best_val_loss, "item"):
                best_val_loss = float(best_val_loss.item())
            else:
                best_val_loss = float(best_val_loss)
            return start_epoch, best_val_loss, path
    return 0, float("inf"), None


def train_classifier(model, train_loader, val_loader, config, tokenizer, optimizer, scheduler, checkpoint_path, resume_checkpoint_path=None):
    start_epoch, best_val_loss, resumed_from = load_training_checkpoint(
        model,
        optimizer,
        scheduler,
        checkpoint_path,
        resume_checkpoint_path=resume_checkpoint_path,
    )
    if resumed_from is not None:
        print(f"Resuming training from {resumed_from} at epoch {start_epoch}")
    if start_epoch >= config["epochs"]:
        print(f"Checkpoint already reached epoch {start_epoch}; skipping training.")
        return

    for epoch in range(start_epoch, config["epochs"]):
        model.train()
        train_losses = []
        train_class_losses = []
        for prompts, paths, path_lengths, interactions, env, env_prop, path_padding_mask, first_step_baselines in tqdm(train_loader, desc=f"Epoch {epoch} [Train]", leave=False):
            del env, env_prop
            prompts = prompts.to(device)
            paths = paths.to(device)
            path_lengths = path_lengths.to(device)
            interactions = interactions.to(device)
            path_padding_mask = path_padding_mask.to(device)
            first_step_baselines = first_step_baselines.to(device)

            paths_in, interactions_in, paths_out, interactions_out, target_labels, target_mask = prepare_batch_for_classifier(
                paths, interactions, path_padding_mask, tokenizer
            )
            p_noise = config.get("TARGET_NOISE_PROB", 0.0)
            if p_noise > 0:
                noise_valid_mask = path_padding_mask[:, :-1].clone()
                keep_prefix = min(2, noise_valid_mask.size(1))
                noise_valid_mask[:, :keep_prefix] = False
                paths_in = add_noise_to_paths(
                    paths_in,
                    noise_valid_mask,
                    p_noise=p_noise,
                    noise_params=config.get("TARGET_NOISE_PARAMS"),
                )

            outputs = model(prompts, paths_in, interactions_in, first_step_baselines)
            total_loss, class_loss, _, _ = compute_classifier_loss(
                outputs,
                target_labels,
                target_mask,
                path_lengths,
                interactions_out,
                interaction_weight=config.get("interaction_weight", 0.01),
                pathcount_weight=config.get("pathcount_weight", 0.05),
            )
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            train_losses.append(total_loss.item())
            train_class_losses.append(class_loss.item())

        scheduler.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for prompts, paths, path_lengths, interactions, env, env_prop, path_padding_mask, first_step_baselines in tqdm(val_loader, desc=f"Epoch {epoch} [Val]", leave=False):
                del env, env_prop
                prompts = prompts.to(device)
                paths = paths.to(device)
                path_lengths = path_lengths.to(device)
                interactions = interactions.to(device)
                path_padding_mask = path_padding_mask.to(device)
                first_step_baselines = first_step_baselines.to(device)

                paths_in, interactions_in, paths_out, interactions_out, target_labels, target_mask = prepare_batch_for_classifier(
                    paths, interactions, path_padding_mask, tokenizer
                )
                outputs = model(prompts, paths_in, interactions_in, first_step_baselines)
                total_loss, _, _, _ = compute_classifier_loss(
                    outputs,
                    target_labels,
                    target_mask,
                    path_lengths,
                    interactions_out,
                    interaction_weight=config.get("interaction_weight", 0.01),
                    pathcount_weight=config.get("pathcount_weight", 0.05),
                )
                val_losses.append(total_loss.item())

        avg_train_loss = float(np.mean(train_losses)) if train_losses else 0.0
        avg_train_class_loss = float(np.mean(train_class_losses)) if train_class_losses else 0.0
        avg_val_loss = float(np.mean(val_losses)) if val_losses else float("inf")

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

        if resume_checkpoint_path is not None:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_val_loss": torch.tensor(best_val_loss),
                },
                resume_checkpoint_path,
            )

        print(
            f"epoch {epoch:03d} train_total_loss={avg_train_loss:.4f} "
            f"train_class_loss={avg_train_class_loss:.4f} val_total_loss={avg_val_loss:.4f}"
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Train/evaluate a corridor-aware binned-classifier PathDecoder across scenarios.")
    parser.add_argument("scenarios", nargs="*", help="Optional scenario names to run.")
    parser.add_argument("--scenario", dest="scenario_flag", action="append")
    parser.add_argument("--scenario-file", type=str)
    parser.add_argument("--shard-index", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=None)
    parser.add_argument("--csv-log-file", type=str, default=csv_log_file)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints_first_step_residual_corridor_classifier")
    parser.add_argument("--noise-prob", type=float, default=0.0)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--n-clusters", type=int, default=25)
    parser.add_argument("--nearest-k", type=int, default=5)
    parser.add_argument("--corridor-k", type=int, default=5)
    parser.add_argument("--corridor-bins", type=int, default=8)
    parser.add_argument("--num-value-bins", type=int, default=64)
    parser.add_argument("--use-material-features", action="store_true")
    parser.add_argument("--no-material-features", dest="use_material_features", action="store_false")
    parser.set_defaults(use_material_features=True)
    return parser.parse_args()


def run_scenario(scenario, args):
    dataset = dm.load(scenario)
    config = {
        "BATCH_SIZE": 128,
        "PAD_VALUE": 0,
        "LR": 2e-5,
        "epochs": 300,
        "interaction_weight": 0.01,
        "pathcount_weight": 0.05,
        "experiment": f"first_step_residual_corridor_classifier_{scenario}",
        "hidden_dim": 512,
        "n_layers": 8,
        "n_heads": 8,
        "TARGET_NOISE_PROB": args.noise_prob,
        "TARGET_NOISE_PARAMS": None,
    }

    base_train = PreTrainMySeqDataLoader(
        dataset,
        train=True,
        split_by="user",
        sort_by="power",
        normalizers=None,
        apply_normalizers=[],
        pad_value=config["PAD_VALUE"],
        include_aod=True,
    )
    base_val = PreTrainMySeqDataLoader(
        dataset,
        train=False,
        split_by="user",
        sort_by="power",
        normalizers=None,
        apply_normalizers=[],
        pad_value=config["PAD_VALUE"],
        include_aod=True,
    )

    scene_bank = SceneFeatureBank.from_dataset(dataset, use_material_features=args.use_material_features)
    train_aug_prompts, train_baselines, val_aug_prompts, val_baselines = build_first_step_assignments_with_corridor(
        base_train,
        base_val,
        scene_bank,
        n_clusters=args.n_clusters,
        nearest_k=args.nearest_k,
        corridor_k=args.corridor_k,
        corridor_bins=args.corridor_bins,
    )

    train_data = FirstStepResidualDataset(base_train, train_aug_prompts, train_baselines)
    val_data = FirstStepResidualDataset(base_val, val_aug_prompts, val_baselines)
    train_loader = DataLoader(train_data, batch_size=config["BATCH_SIZE"], shuffle=True, collate_fn=train_data.collate_fn)
    val_loader = DataLoader(val_data, batch_size=config["BATCH_SIZE"], shuffle=False, collate_fn=val_data.collate_fn)

    tokenizer = PathValueTokenizer(num_bins=args.num_value_bins).fit(base_train)

    prompt_dim = int(train_aug_prompts[0].numel())
    model = FirstStepResidualCorridorClassifier(
        prompt_dim=prompt_dim,
        hidden_dim=config["hidden_dim"],
        n_layers=config["n_layers"],
        n_heads=config["n_heads"],
        num_bins=args.num_value_bins,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["LR"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=25, T_mult=1, eta_min=1e-8)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.checkpoint_dir, f"{config['experiment']}_best_model_checkpoint.pth")
    resume_checkpoint_path = get_resume_checkpoint_path(checkpoint_path)

    if not args.skip_train:
        train_classifier(
            model,
            train_loader,
            val_loader,
            config,
            tokenizer,
            optimizer,
            scheduler,
            checkpoint_path,
            resume_checkpoint_path=resume_checkpoint_path,
        )

    best_epoch, best_loss = load_best_checkpoint(model, checkpoint_path)
    if best_epoch is not None:
        print(f"Loaded best checkpoint from epoch {best_epoch} (val_loss: {float(best_loss):.4f})")

    results = evaluate_classifier_model(model, val_loader, tokenizer)
    row = {
        "scenario": scenario,
        "n_clusters": args.n_clusters,
        "nearest_k": args.nearest_k,
        "corridor_k": args.corridor_k,
        "corridor_bins": args.corridor_bins,
        "num_value_bins": args.num_value_bins,
        "prompt_dim": prompt_dim,
        "use_material_features": args.use_material_features,
        "noise_prob": args.noise_prob,
        "best_val_loss": float(best_loss) if best_loss is not None else float("nan"),
    }
    row.update(results)
    pd.DataFrame([row]).to_csv(
        args.csv_log_file,
        mode="a",
        index=False,
        header=not os.path.exists(args.csv_log_file),
    )
    print(f"✓ Results for {scenario} saved to {args.csv_log_file}")


def main():
    args = parse_args()
    scenarios = resolve_scenarios(args)
    if not scenarios:
        scenarios = list(all_scenarios)
    print(f"Running {len(scenarios)} scenario(s): {scenarios}")
    for scenario in scenarios:
        run_scenario(scenario, args)


if __name__ == "__main__":
    main()
