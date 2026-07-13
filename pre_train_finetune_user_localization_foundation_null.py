import argparse
import hashlib
import os
import warnings

import deepmimo as dm
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from tqdm import tqdm

from dataset.dataloaders import PreTrainMySeqDataLoader
from multiscenario_direct_training_first_step_residual import (
    FirstStepResidualDataset,
    FirstStepResidualPathDecoder,
    load_best_checkpoint,
    resolve_scenarios,
)
from multiscenario_direct_training_first_step_residual_corridor import (
    build_first_step_assignments_with_corridor,
)
from scene_feature_utils import SceneFeatureBank
from utils.utils import count_parameters

warnings.filterwarnings("ignore", category=UserWarning)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_CSV_LOG = "/home/blessedg/Pathformer/logs/foundation_localization_null_corridor.csv"




def parse_args():
    parser = argparse.ArgumentParser(
        description="Downstream XY localization finetuning from the corridor-concat foundation backbone with nulled user-dependent prompt fields."
    )
    parser.add_argument("scenarios", nargs="*", help="Optional scenario names to run.")
    parser.add_argument("--scenario", dest="scenario_flag", action="append")
    parser.add_argument("--scenario-file", type=str)
    parser.add_argument("--shard-index", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=None)
    parser.add_argument("--csv-log-file", type=str, default=DEFAULT_CSV_LOG)
    parser.add_argument(
        "--pretrained-checkpoint",
        type=str,
        default="/home/blessedg/Pathformer/checkpoints_first_step_residual_corridor_concat/first_step_residual_corridor_concat_27scenarios_44710a4a_best_model_checkpoint.pth",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="/home/blessedg/Pathformer/checkpoints_foundation_localization_null_corridor",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--pool-mode", choices=["first", "last", "mean"], default="mean")
    parser.add_argument("--unfreeze-backbone", action="store_true")
    parser.add_argument("--n-clusters", type=int, default=25)
    parser.add_argument("--nearest-k", type=int, default=5)
    parser.add_argument("--corridor-k", type=int, default=5)
    parser.add_argument("--corridor-bins", type=int, default=8)
    parser.add_argument("--use-material-features", action="store_true")
    parser.add_argument("--no-material-features", dest="use_material_features", action="store_false")
    parser.add_argument("--null-rx-pos", action="store_true")
    parser.add_argument("--keep-rx-pos", dest="null_rx_pos", action="store_false")
    parser.add_argument("--null-scene-features", action="store_true")
    parser.add_argument("--keep-scene-features", dest="null_scene_features", action="store_false")
    parser.add_argument("--null-cluster-prior", action="store_true")
    parser.set_defaults(use_material_features=True, null_rx_pos=True, null_scene_features=True)
    return parser.parse_args()


def _scenario_group_name(scenarios):
    ordered = sorted(scenarios)
    digest = hashlib.md5("||".join(ordered).encode("utf-8")).hexdigest()[:8]
    if len(ordered) <= 3:
        prefix = "_".join(s.replace("/", "_") for s in ordered)
    else:
        prefix = f"{len(ordered)}scenarios"
    return f"{prefix}_{digest}"


def _null_prompt_fields(prompt_tensor, base_prompt_dim, null_rx_pos, null_scene_features, null_cluster_prior):
    prompt_tensor = prompt_tensor.clone()

    # The base prompt is [tx_x, tx_y, tx_z, rx_x, rx_y, rx_z]. After standardization,
    # zero corresponds to the train-set mean and acts as an "unknown" placeholder.
    if null_rx_pos:
        prompt_tensor[3:base_prompt_dim] = 0.0

    cluster_start = base_prompt_dim
    cluster_end = base_prompt_dim + 4
    if null_cluster_prior:
        prompt_tensor[cluster_start:cluster_end] = 0.0

    if null_scene_features:
        prompt_tensor[cluster_end:] = 0.0

    return prompt_tensor


class LocalizationResidualDataset(Dataset):
    def __init__(self, base_dataset, targets_xy):
        self.base_dataset = base_dataset
        self.targets_xy = targets_xy.float().cpu()

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        return self.base_dataset[idx], self.targets_xy[idx]

    def collate_fn(self, batch):
        base_items = [item[0] for item in batch]
        labels = torch.stack([item[1] for item in batch], dim=0)
        collated = self.base_dataset.collate_fn(base_items)
        return (*collated, labels)


def _make_concat_collate_fn(reference_dataset):
    def _collate_fn(batch):
        base_items = [item[0] for item in batch]
        labels = torch.stack([item[1] for item in batch], dim=0)
        collated = reference_dataset.base_dataset.collate_fn(base_items)
        return (*collated, labels)

    return _collate_fn


class FoundationXYLocalizer(nn.Module):
    def __init__(self, backbone, hidden_dim=1024, pool_mode="mean", train_backbone=False):
        super().__init__()
        self.backbone = backbone
        self.pool_mode = pool_mode
        self.train_backbone = train_backbone
        if not train_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        self.localization_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 2),
        )

    def extract_summary_features(self, prompts, paths, interactions, path_padding_mask):
        h_paths, _ = self.backbone.backbone.forward_hidden(prompts, paths, interactions)
        valid_mask = path_padding_mask.bool()
        if valid_mask.size(1) != h_paths.size(1):
            valid_mask = valid_mask[:, : h_paths.size(1)]
        if valid_mask.size(1) > 0:
            valid_mask = valid_mask.clone()
            valid_mask[:, 0] = False

        if self.pool_mode == "first":
            pooled = h_paths[:, 0, :]
        elif self.pool_mode == "last":
            last_idx = valid_mask.sum(dim=1).clamp(min=1) - 1
            pooled = h_paths[torch.arange(h_paths.size(0), device=h_paths.device), last_idx, :]
        else:
            valid_float = valid_mask.unsqueeze(-1).float()
            denom = valid_float.sum(dim=1).clamp(min=1.0)
            pooled = (h_paths * valid_float).sum(dim=1) / denom
        return pooled

    def forward(self, prompts, paths, interactions, path_padding_mask):
        summary = self.extract_summary_features(prompts, paths, interactions, path_padding_mask)
        return self.localization_head(summary)


def compute_xy_stats(coord_tensors):
    coords = torch.cat(coord_tensors, dim=0).float()
    mean = coords.mean(dim=0)
    std = coords.std(dim=0)
    std = torch.where(std < 1e-6, torch.ones_like(std), std)
    return mean, std


def build_concat_localization_datasets(scenarios, args, pad_value):
    train_wrappers = []
    val_wrappers = []
    train_coord_tensors = []
    prompt_dim = None

    for scenario in scenarios:
        print(f"Preparing scenario: {scenario}")
        dataset = dm.load(scenario)
        base_train = PreTrainMySeqDataLoader(
            dataset,
            train=True,
            split_by="user",
            sort_by="power",
            normalizers=None,
            apply_normalizers=[],
            pad_value=pad_value,
            include_aod=True,
        )
        base_val = PreTrainMySeqDataLoader(
            dataset,
            train=False,
            split_by="user",
            sort_by="power",
            normalizers=None,
            apply_normalizers=[],
            pad_value=pad_value,
            include_aod=True,
        )

        raw_prompt_dim = len(base_train[0][0])
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

        train_aug_prompts = [
            _null_prompt_fields(
                prompt,
                base_prompt_dim=raw_prompt_dim,
                null_rx_pos=args.null_rx_pos,
                null_scene_features=args.null_scene_features,
                null_cluster_prior=args.null_cluster_prior,
            )
            for prompt in train_aug_prompts
        ]
        val_aug_prompts = [
            _null_prompt_fields(
                prompt,
                base_prompt_dim=raw_prompt_dim,
                null_rx_pos=args.null_rx_pos,
                null_scene_features=args.null_scene_features,
                null_cluster_prior=args.null_cluster_prior,
            )
            for prompt in val_aug_prompts
        ]

        if prompt_dim is None:
            prompt_dim = int(train_aug_prompts[0].numel())
        elif prompt_dim != int(train_aug_prompts[0].numel()):
            raise ValueError(
                f"Prompt dimension mismatch across scenarios: expected {prompt_dim}, got {int(train_aug_prompts[0].numel())} for {scenario}"
            )

        train_data = FirstStepResidualDataset(base_train, train_aug_prompts, train_baselines)
        val_data = FirstStepResidualDataset(base_val, val_aug_prompts, val_baselines)

        train_targets_xy = torch.tensor(np.asarray(base_train.dataset_filtered["rx_pos"], dtype=np.float32)[:, :2])
        val_targets_xy = torch.tensor(np.asarray(base_val.dataset_filtered["rx_pos"], dtype=np.float32)[:, :2])
        train_coord_tensors.append(train_targets_xy)

        train_wrappers.append(LocalizationResidualDataset(train_data, train_targets_xy))
        val_wrappers.append(LocalizationResidualDataset(val_data, val_targets_xy))
        print(f"  train samples={len(train_data)} | val samples={len(val_data)}")

    coord_mean, coord_std = compute_xy_stats(train_coord_tensors)

    normalized_train_wrappers = []
    normalized_val_wrappers = []
    for train_wrapper, val_wrapper in zip(train_wrappers, val_wrappers):
        train_targets = (train_wrapper.targets_xy - coord_mean) / coord_std
        val_targets = (val_wrapper.targets_xy - coord_mean) / coord_std
        normalized_train_wrappers.append(LocalizationResidualDataset(train_wrapper.base_dataset, train_targets))
        normalized_val_wrappers.append(LocalizationResidualDataset(val_wrapper.base_dataset, val_targets))

    combined_train = ConcatDataset(normalized_train_wrappers)
    combined_val = ConcatDataset(normalized_val_wrappers)
    collate_fn = _make_concat_collate_fn(normalized_train_wrappers[0])
    return combined_train, combined_val, collate_fn, prompt_dim, coord_mean, coord_std


def unpack_batch(batch):
    prompts, paths, path_lengths, interactions, env, env_prop, path_padding_mask, first_step_baseline, coords_xy = batch
    return (
        prompts.to(device),
        paths.to(device),
        interactions.to(device),
        path_padding_mask.to(device),
        coords_xy.to(device),
    )


def evaluate_localizer(model, loader, coord_mean, coord_std):
    model.eval()
    losses = []
    errors = []
    criterion = nn.SmoothL1Loss()
    coord_mean = coord_mean.to(device)
    coord_std = coord_std.to(device)
    with torch.no_grad():
        for batch in tqdm(loader, desc="Eval localization", leave=False):
            prompts, paths, interactions, path_padding_mask, coords_xy = unpack_batch(batch)
            preds = model(prompts, paths, interactions, path_padding_mask)
            loss = criterion(preds, coords_xy)
            losses.append(loss.item())

            preds_meter = preds * coord_std + coord_mean
            coords_meter = coords_xy * coord_std + coord_mean
            batch_errors = torch.linalg.norm(preds_meter - coords_meter, dim=1)
            errors.extend(batch_errors.detach().cpu().numpy().tolist())

    return {
        "val_loss": float(np.mean(losses)) if losses else float("inf"),
        "mde_m": float(np.mean(errors)) if errors else float("inf"),
        "median_mde_m": float(np.median(errors)) if errors else float("inf"),
        "p90_mde_m": float(np.percentile(errors, 90)) if errors else float("inf"),
    }


def train_localizer(model, train_loader, val_loader, args, checkpoint_path, coord_mean, coord_std):
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=25,
        T_mult=1,
        eta_min=1e-6,
    )
    criterion = nn.SmoothL1Loss()
    best_val_mde = float("inf")

    for epoch in range(args.epochs):
        model.train()
        train_losses = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:03d} [Localization]", leave=False)
        for batch in pbar:
            prompts, paths, interactions, path_padding_mask, coords_xy = unpack_batch(batch)
            preds = model(prompts, paths, interactions, path_padding_mask)
            loss = criterion(preds, coords_xy)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip_norm)
            optimizer.step()

            train_losses.append(loss.item())
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        scheduler.step()
        val_metrics = evaluate_localizer(model, val_loader, coord_mean, coord_std)
        print(
            f"Epoch {epoch:03d} "
            f"train_loss={np.mean(train_losses):.4f} "
            f"val_loss={val_metrics['val_loss']:.4f} "
            f"val_mde={val_metrics['mde_m']:.2f}m "
            f"val_med={val_metrics['median_mde_m']:.2f}m "
            f"val_p90={val_metrics['p90_mde_m']:.2f}m"
        )

        if val_metrics["mde_m"] < best_val_mde:
            best_val_mde = val_metrics["mde_m"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "best_val_mde": best_val_mde,
                    "coord_mean": coord_mean.cpu(),
                    "coord_std": coord_std.cpu(),
                    "pool_mode": args.pool_mode,
                    "null_rx_pos": args.null_rx_pos,
                    "null_scene_features": args.null_scene_features,
                    "null_cluster_prior": args.null_cluster_prior,
                    "unfreeze_backbone": args.unfreeze_backbone,
                },
                checkpoint_path,
            )
            print(f"  ✓ Saved best localization checkpoint to {checkpoint_path}")


def load_localization_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


def main():
    args = parse_args()
    if not args.scenarios and not args.scenario_flag and not args.scenario_file:
        raise ValueError("Provide at least one scenario or a scenario file.")
    scenarios = resolve_scenarios(args)

    if args.shard_index is not None or args.num_shards is not None:
        if args.shard_index is None or args.num_shards is None:
            raise ValueError("Provide both --shard-index and --num-shards.")
        scenarios = [s for idx, s in enumerate(scenarios) if idx % args.num_shards == args.shard_index]

    if not scenarios:
        raise ValueError("No scenarios selected after sharding.")
    if not os.path.exists(args.pretrained_checkpoint):
        raise FileNotFoundError(f"Foundation checkpoint not found: {args.pretrained_checkpoint}")

    experiment_suffix = _scenario_group_name(scenarios)
    print(f"Running localization for {len(scenarios)} scenario(s): {scenarios}")
    print(f"Using foundation checkpoint: {args.pretrained_checkpoint}")

    train_data, val_data, collate_fn, prompt_dim, coord_mean, coord_std = build_concat_localization_datasets(
        scenarios,
        args,
        pad_value=0,
    )
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    backbone = FirstStepResidualPathDecoder(
        prompt_dim=prompt_dim,
        hidden_dim=512 * 2,
        n_layers=8 + 4,
        n_heads=8,
    ).to(device)
    _, _ = load_best_checkpoint(backbone, args.pretrained_checkpoint)
    model = FoundationXYLocalizer(
        backbone=backbone,
        hidden_dim=512 * 2,
        pool_mode=args.pool_mode,
        train_backbone=args.unfreeze_backbone,
    ).to(device)
    print(f"Trainable localization parameters: {count_parameters(model)}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.csv_log_file) or ".", exist_ok=True)
    checkpoint_path = os.path.join(
        args.checkpoint_dir,
        f"foundation_xy_localizer_{experiment_suffix}.pth",
    )

    if args.eval_only or args.skip_train:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Localization checkpoint not found: {checkpoint_path}")
        checkpoint = load_localization_checkpoint(model, checkpoint_path)
        coord_mean = checkpoint["coord_mean"]
        coord_std = checkpoint["coord_std"]
    else:
        train_localizer(model, train_loader, val_loader, args, checkpoint_path, coord_mean, coord_std)
        checkpoint = load_localization_checkpoint(model, checkpoint_path)
        coord_mean = checkpoint["coord_mean"]
        coord_std = checkpoint["coord_std"]

    metrics = evaluate_localizer(model, val_loader, coord_mean, coord_std)
    row = {
        "scenario_group": experiment_suffix,
        "num_scenarios": len(scenarios),
        "scenarios": "|".join(scenarios),
        "pool_mode": args.pool_mode,
        "unfreeze_backbone": args.unfreeze_backbone,
        "null_rx_pos": args.null_rx_pos,
        "null_scene_features": args.null_scene_features,
        "null_cluster_prior": args.null_cluster_prior,
        "n_clusters": args.n_clusters,
        "nearest_k": args.nearest_k,
        "corridor_k": args.corridor_k,
        "corridor_bins": args.corridor_bins,
        "prompt_dim": prompt_dim,
        "train_samples": len(train_data),
        "val_samples": len(val_data),
        "mde_m": metrics["mde_m"],
        "median_mde_m": metrics["median_mde_m"],
        "p90_mde_m": metrics["p90_mde_m"],
        "val_loss": metrics["val_loss"],
        "backbone_checkpoint": args.pretrained_checkpoint,
        "localization_checkpoint": checkpoint_path,
    }
    pd.DataFrame([row]).to_csv(
        args.csv_log_file,
        mode="a",
        index=False,
        header=not os.path.exists(args.csv_log_file),
    )
    print(
        f"Localization results | mean={metrics['mde_m']:.2f}m "
        f"median={metrics['median_mde_m']:.2f}m p90={metrics['p90_mde_m']:.2f}m"
    )


if __name__ == "__main__":
    main()
