import argparse
import os
import warnings

import deepmimo as dm
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from dataset.dataloaders import PreTrainMySeqDataLoader
from multiscenario_direct_training_first_step_residual import (
    FirstStepResidualDataset,
    FirstStepResidualPathDecoder,
    generate_paths_first_step_residual_batch,
    load_best_checkpoint as load_residual_checkpoint,
    resolve_scenarios,
)
from multiscenario_direct_training_first_step_residual_corridor import (
    build_first_step_assignments_with_corridor,
)
from scene_feature_utils import SceneFeatureBank
from utils.utils import count_parameters

warnings.filterwarnings("ignore", category=UserWarning)

csv_log_file = "los_prediction_finetuning_foundation.csv"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser(
        description="LOS head finetuning and zero-shot evaluation for the corridor-concat foundation model."
    )
    parser.add_argument("scenarios", nargs="*", help="Optional scenario names to run.")
    parser.add_argument("--scenario", dest="scenario_flag", action="append")
    parser.add_argument("--scenario-file", type=str)
    parser.add_argument("--shard-index", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=None)
    parser.add_argument("--csv-log-file", type=str, default=csv_log_file)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--pretrained-checkpoint",
        type=str,
        default="/home/blessedg/Pathformer/checkpoints_first_step_residual_corridor_concat/first_step_residual_corridor_concat_27scenarios_44710a4a_best_model_checkpoint.pth",
    )
    parser.add_argument(
        "--los-head-checkpoint-dir",
        type=str,
        default="/home/blessedg/Pathformer/checkpoints_los_heads_foundation",
    )
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--n-clusters", type=int, default=25)
    parser.add_argument("--nearest-k", type=int, default=5)
    parser.add_argument("--corridor-k", type=int, default=5)
    parser.add_argument("--corridor-bins", type=int, default=8)
    parser.add_argument("--train-val-ratio", type=float, default=0.125)
    parser.add_argument("--max-generate", type=int, default=25)
    parser.add_argument("--use-material-features", action="store_true")
    parser.add_argument("--no-material-features", dest="use_material_features", action="store_false")
    parser.set_defaults(use_material_features=True)
    return parser.parse_args()


def default_scenarios():
    return [
        "city_47_chicago_3p5",
        "city_0_newyork_3p5_s",
        "city_1_losangeles_3p5",
        "city_2_chicago_3p5",
        "city_3_houston_3p5",
        "city_4_phoenix_3p5",
        "city_5_philadelphia_3p5",
        "city_6_miami_3p5",
        "city_7_sandiego_3p5",
        "city_8_dallas_3p5",
        "city_9_sanfrancisco_3p5",
        "city_10_austin_3p5",
        "city_11_santaclara_3p5",
        "city_12_fortworth_3p5",
        "city_13_columbus_3p5",
        "city_17_seattle_3p5_s",
        "city_18_denver_3p5",
        "city_19_oklahoma_3p5_s",
        "city_16_sanfrancisco_3p5_lwm",
        "city_23_beijing_3p5",
        "city_31_barcelona_3p5",
        "city_35_san_francisco_3p5",
        "city_89_nairobi_3p5",
        "city_91_xiangyang_3p5",
        "city_92_sãopaulo_3p5",
        "boston5g_3p5",
        "city_86_ankara_3p5",
        "city_72_capetown_3p5",
        "city_84_baoding_3p5",
        "city_95_delhi_3p5",
        "city_96_osaka_3p5",
        "city_88_tongshan_3p5",
    ]


class CachedBinaryFeatureDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features.float().cpu()
        self.labels = labels.float().cpu()

    def __len__(self):
        return self.features.size(0)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class LOSHeadFinetuner(nn.Module):
    def __init__(self, backbone, hidden_dim=1024):
        super().__init__()
        self.backbone = backbone
        self.los_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def extract_summary_features(self, prompts, paths, interactions, path_padding_mask):
        with torch.inference_mode():
            h_paths, _ = self.backbone.backbone.forward_hidden(prompts, paths, interactions)
        valid_mask = path_padding_mask.bool()
        if valid_mask.size(1) != h_paths.size(1):
            valid_mask = valid_mask[:, : h_paths.size(1)]
        if valid_mask.size(1) > 0:
            valid_mask = valid_mask.clone()
            valid_mask[:, 0] = False
        valid_float = valid_mask.unsqueeze(-1).float()
        denom = valid_float.sum(dim=1).clamp(min=1.0)
        return (h_paths * valid_float).sum(dim=1) / denom

    def classify_summary_features(self, summary_features):
        return self.los_head(summary_features).squeeze(-1)

    def forward(self, prompts, paths, interactions, path_padding_mask):
        summary = self.extract_summary_features(prompts, paths, interactions, path_padding_mask)
        return self.classify_summary_features(summary)


def freeze_backbone(module):
    for param in module.parameters():
        param.requires_grad = False


def _make_loader(dataset, batch_size, shuffle, collate_fn=None):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)


def build_corridor_datasets(dataset, batch_size, pad_value, args):
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
    base_heldout = PreTrainMySeqDataLoader(
        dataset,
        train=False,
        split_by="user",
        sort_by="power",
        normalizers=None,
        apply_normalizers=[],
        pad_value=pad_value,
        include_aod=True,
    )
    scene_bank = SceneFeatureBank.from_dataset(dataset, use_material_features=args.use_material_features)
    train_aug_prompts, train_baselines, heldout_aug_prompts, heldout_baselines = build_first_step_assignments_with_corridor(
        base_train,
        base_heldout,
        scene_bank,
        n_clusters=args.n_clusters,
        nearest_k=args.nearest_k,
        corridor_k=args.corridor_k,
        corridor_bins=args.corridor_bins,
    )
    train_data = FirstStepResidualDataset(base_train, train_aug_prompts, train_baselines)
    heldout_data = FirstStepResidualDataset(base_heldout, heldout_aug_prompts, heldout_baselines)
    return train_data, heldout_data


def get_binary_los_labels(dataset):
    los = np.asarray(dataset.base_dataset.dataset_filtered["los"])
    return torch.from_numpy((los > 0).astype(np.float32))


def split_indices(n_items, seed, val_ratio):
    if n_items == 0:
        return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.int64)

    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_items)
    if n_items == 1:
        return indices, indices[:0]

    val_size = int(round(n_items * val_ratio))
    val_size = min(max(val_size, 1), n_items - 1)
    return indices[:val_size], indices[val_size:]


@torch.no_grad()
def precompute_backbone_features(model, dataset, batch_size, desc):
    model.eval()
    loader = _make_loader(dataset, batch_size, False, dataset.collate_fn)
    all_features = []
    for batch in tqdm(loader, desc=desc, leave=False):
        prompts, paths, path_lengths, interactions, env, env_prop, path_padding_mask, first_step_baselines = batch
        prompts = prompts.to(device)
        paths = paths.to(device)
        interactions = interactions.to(device)
        path_padding_mask = path_padding_mask.to(device)
        features = model.extract_summary_features(prompts, paths, interactions, path_padding_mask)
        all_features.append(features.cpu())
    return torch.cat(all_features, dim=0) if all_features else torch.empty((0, 0), dtype=torch.float32)


def _scaled_power_to_db(power_scaled):
    return power_scaled / 0.01


def _power_db_to_linear(power_db):
    clipped = np.clip(power_db, -200.0, 200.0)
    return np.power(10.0, clipped / 10.0)


@torch.no_grad()
def precompute_zero_shot_scores(backbone, dataset, batch_size, max_generate, desc):
    backbone.eval()
    loader = _make_loader(dataset, batch_size, False, dataset.collate_fn)
    first_power_scores = []
    dominance_scores = []

    for batch in tqdm(loader, desc=desc, leave=False):
        prompts, paths, path_lengths, interactions, env, env_prop, path_padding_mask, first_step_baselines = batch
        generated, pathcount_pred, _ = generate_paths_first_step_residual_batch(
            backbone,
            prompts.to(device),
            first_step_baselines.to(device),
            max_steps=max_generate,
        )
        if pathcount_pred.dim() > 1:
            pathcount_pred = pathcount_pred.squeeze(-1)
        valid_counts = torch.clamp(torch.round(pathcount_pred * max_generate).long(), min=1, max=max_generate)

        generated_powers = generated[:, :, 1].numpy()
        valid_counts_np = valid_counts.cpu().numpy()
        for sample_powers, count in zip(generated_powers, valid_counts_np):
            valid_powers = sample_powers[: int(count)]
            valid_power_db = _scaled_power_to_db(valid_powers)
            valid_power_linear = _power_db_to_linear(valid_power_db)
            first_power_scores.append(float(valid_power_db[0]))
            dominance_scores.append(float(valid_power_linear[0] / max(valid_power_linear.sum(), 1e-12)))

    return {
        "first_power_db": torch.tensor(first_power_scores, dtype=torch.float32),
        "dominance_ratio": torch.tensor(dominance_scores, dtype=torch.float32),
    }


def subset_tensor(tensor, indices):
    index_tensor = torch.as_tensor(indices, dtype=torch.long)
    return tensor.index_select(0, index_tensor)


def compute_binary_metrics(labels, predictions):
    labels_np = np.asarray(labels, dtype=np.int64)
    predictions_np = np.asarray(predictions, dtype=np.int64)
    return {
        "accuracy": float(accuracy_score(labels_np, predictions_np)),
        "f1": float(f1_score(labels_np, predictions_np, zero_division=0)),
        "precision": float(precision_score(labels_np, predictions_np, zero_division=0)),
        "recall": float(recall_score(labels_np, predictions_np, zero_division=0)),
    }


def find_best_threshold(scores, labels):
    scores_np = np.asarray(scores, dtype=np.float64)
    labels_np = np.asarray(labels, dtype=np.int64)
    if scores_np.size == 0:
        return {"threshold": 0.5, "accuracy": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0}

    unique_scores = np.unique(scores_np)
    if unique_scores.size <= 256:
        thresholds = unique_scores
    else:
        thresholds = np.quantile(scores_np, np.linspace(0.0, 1.0, 257))
        thresholds = np.unique(thresholds)

    best = None
    for threshold in thresholds:
        predictions = (scores_np >= threshold).astype(np.int64)
        metrics = compute_binary_metrics(labels_np, predictions)
        candidate = {"threshold": float(threshold), **metrics}
        if best is None:
            best = candidate
            continue
        if candidate["f1"] > best["f1"] + 1e-12:
            best = candidate
        elif abs(candidate["f1"] - best["f1"]) <= 1e-12 and candidate["accuracy"] > best["accuracy"]:
            best = candidate
    return best


def evaluate_scores_with_threshold(scores, labels, threshold):
    scores_np = np.asarray(scores, dtype=np.float64)
    labels_np = np.asarray(labels, dtype=np.int64)
    predictions = (scores_np >= threshold).astype(np.int64)
    return compute_binary_metrics(labels_np, predictions)


def select_best_zero_shot_rule(val_scores_by_name, val_labels):
    best = None
    val_labels_np = val_labels.cpu().numpy().astype(np.int64)
    for score_name, score_tensor in val_scores_by_name.items():
        threshold_metrics = find_best_threshold(score_tensor.cpu().numpy(), val_labels_np)
        candidate = {"score_name": score_name, **threshold_metrics}
        if best is None:
            best = candidate
            continue
        if candidate["f1"] > best["f1"] + 1e-12:
            best = candidate
        elif abs(candidate["f1"] - best["f1"]) <= 1e-12 and candidate["accuracy"] > best["accuracy"]:
            best = candidate
    return best


def collect_logits_and_labels(model, loader):
    model.eval()
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for features, labels in tqdm(loader, desc="Collect logits [foundation]", leave=False):
            features = features.to(device)
            logits = model.classify_summary_features(features)
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
    if not all_logits:
        return torch.empty(0), torch.empty(0)
    return torch.cat(all_logits, dim=0), torch.cat(all_labels, dim=0)


def evaluate_los_head(model, loader, threshold=0.5):
    logits, labels = collect_logits_and_labels(model, loader)
    probabilities = torch.sigmoid(logits).numpy()
    labels_np = labels.numpy().astype(np.int64)
    metrics = evaluate_scores_with_threshold(probabilities, labels_np, threshold)
    return metrics, probabilities, labels_np


def train_los_head(model, train_loader, val_loader, config, checkpoint_path):
    train_labels = train_loader.dataset.labels
    positive_count = float(train_labels.sum().item())
    negative_count = float(train_labels.numel() - positive_count)
    if positive_count > 0 and negative_count > 0:
        pos_weight = torch.tensor([negative_count / positive_count], dtype=torch.float32, device=device)
    else:
        pos_weight = None

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.los_head.parameters(), lr=config["LR"], weight_decay=config["WEIGHT_DECAY"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=25,
        T_mult=1,
        eta_min=1e-8,
    )
    best_val_f1 = -1.0

    for epoch in range(config["epochs"]):
        model.train()
        train_losses = []
        train_logits = []
        train_targets = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [foundation Train]", leave=False)
        for features, labels in pbar:
            features = features.to(device)
            labels = labels.to(device)
            logits = model.classify_summary_features(features)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.los_head.parameters(), max_norm=1.0)
            optimizer.step()

            train_losses.append(loss.item())
            train_logits.append(logits.detach().cpu())
            train_targets.append(labels.detach().cpu())
            train_probs = torch.sigmoid(torch.cat(train_logits, dim=0)).numpy()
            train_labels_np = torch.cat(train_targets, dim=0).numpy().astype(np.int64)
            train_predictions = (train_probs >= 0.5).astype(np.int64)
            train_metrics = compute_binary_metrics(train_labels_np, train_predictions)
            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "train_acc": f"{train_metrics['accuracy']:.4f}",
                    "train_f1": f"{train_metrics['f1']:.4f}",
                }
            )

        val_metrics_default, val_probs, val_labels_np = evaluate_los_head(model, val_loader, threshold=0.5)
        tuned_val = find_best_threshold(val_probs, val_labels_np)
        scheduler.step()

        print(
            f"Epoch {epoch:02d} [foundation] "
            f"train_loss={np.mean(train_losses):.4f} "
            f"val_acc@0.5={val_metrics_default['accuracy']:.4f} "
            f"val_f1@0.5={val_metrics_default['f1']:.4f} "
            f"val_best_f1={tuned_val['f1']:.4f} "
            f"val_best_thr={tuned_val['threshold']:.4f}"
        )

        if tuned_val["f1"] > best_val_f1:
            best_val_f1 = tuned_val["f1"]
            torch.save(
                {
                    "epoch": epoch,
                    "los_head_state_dict": model.los_head.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_val_f1": best_val_f1,
                    "best_threshold": tuned_val["threshold"],
                    "model_family": "foundation_corridor_concat",
                },
                checkpoint_path,
            )
            print(f"  ✓ Saved LOS head checkpoint to {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.los_head.load_state_dict(checkpoint["los_head_state_dict"])
    return checkpoint.get("best_threshold", 0.5)


def build_foundation_objects(dataset, args, config):
    pad_value = config["PAD_VALUE"]
    train_data, heldout_data = build_corridor_datasets(dataset, args.batch_size, pad_value, args)

    prompt_dim = int(train_data.augmented_prompts[0].numel())
    backbone = FirstStepResidualPathDecoder(
        prompt_dim=prompt_dim,
        hidden_dim=512 * 2,
        n_layers=8 + 4,
        n_heads=8,
    ).to(device)
    load_residual_checkpoint(backbone, config["backbone_checkpoint_path"])
    freeze_backbone(backbone)

    los_model = LOSHeadFinetuner(backbone=backbone, hidden_dim=512 * 2).to(device)
    train_features = precompute_backbone_features(
        model=los_model,
        dataset=train_data,
        batch_size=args.batch_size,
        desc="Cache train features [foundation]",
    )
    heldout_features = precompute_backbone_features(
        model=los_model,
        dataset=heldout_data,
        batch_size=args.batch_size,
        desc="Cache test features [foundation]",
    )
    train_labels = get_binary_los_labels(train_data)
    heldout_labels = get_binary_los_labels(heldout_data)
    train_scores = precompute_zero_shot_scores(
        backbone=backbone,
        dataset=train_data,
        batch_size=args.batch_size,
        max_generate=args.max_generate,
        desc="Zero-shot generated powers [train]",
    )
    heldout_scores = precompute_zero_shot_scores(
        backbone=backbone,
        dataset=heldout_data,
        batch_size=args.batch_size,
        max_generate=args.max_generate,
        desc="Zero-shot generated powers [test]",
    )

    train_indices, val_indices = split_indices(len(train_data), seed=42, val_ratio=args.train_val_ratio)
    cached_train = CachedBinaryFeatureDataset(subset_tensor(train_features, train_indices), subset_tensor(train_labels, train_indices))
    cached_val = CachedBinaryFeatureDataset(subset_tensor(train_features, val_indices), subset_tensor(train_labels, val_indices))
    cached_test = CachedBinaryFeatureDataset(heldout_features, heldout_labels)
    train_loader = _make_loader(cached_train, args.batch_size, True)
    val_loader = _make_loader(cached_val, args.batch_size, False)
    test_loader = _make_loader(cached_test, args.batch_size, False)

    zero_shot_split = {
        "val_labels": subset_tensor(train_labels, val_indices),
        "test_labels": heldout_labels,
        "val_scores": {name: subset_tensor(scores, val_indices) for name, scores in train_scores.items()},
        "test_scores": heldout_scores,
    }
    return los_model, backbone, train_loader, val_loader, test_loader, zero_shot_split


def main():
    args = parse_args()
    if not args.scenarios and not args.scenario_flag and not args.scenario_file:
        scenarios = default_scenarios()
    else:
        scenarios = resolve_scenarios(args)
    if not scenarios:
        print("No scenarios selected.")
        return

    if not os.path.exists(args.pretrained_checkpoint):
        raise FileNotFoundError(f"Foundation checkpoint not found: {args.pretrained_checkpoint}")

    os.makedirs(args.los_head_checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.csv_log_file) or ".", exist_ok=True)
    print(f"Running foundation LOS finetuning for {len(scenarios)} scenario(s): {scenarios}")
    print(f"Using foundation checkpoint: {args.pretrained_checkpoint}")

    for scenario in scenarios:
        print(f"\nLOS finetuning for {scenario} [foundation_corridor_concat]")
        dataset = dm.load(scenario)
        config = {
            "PAD_VALUE": 0,
            "LR": args.lr,
            "WEIGHT_DECAY": args.weight_decay,
            "epochs": args.epochs,
            "backbone_checkpoint_path": args.pretrained_checkpoint,
        }
        model, _, train_loader, val_loader, test_loader, zero_shot_split = build_foundation_objects(dataset, args, config)
        print(f"foundation_corridor_concat trainable LOS head parameters: {count_parameters(model.los_head)}")

        head_checkpoint_path = os.path.join(
            args.los_head_checkpoint_dir,
            f"los_head_foundation_corridor_concat_{scenario}.pth",
        )

        if args.skip_train and os.path.exists(head_checkpoint_path):
            checkpoint = torch.load(head_checkpoint_path, map_location=device)
            model.los_head.load_state_dict(checkpoint["los_head_state_dict"])
            finetuned_threshold = checkpoint.get("best_threshold", 0.5)
        elif args.skip_train:
            print(f"Skipping {scenario} [foundation_corridor_concat] because LOS head checkpoint is missing: {head_checkpoint_path}")
            continue
        else:
            finetuned_threshold = train_los_head(
                model,
                train_loader,
                val_loader,
                config,
                head_checkpoint_path,
            )

        zero_shot_best = select_best_zero_shot_rule(
            zero_shot_split["val_scores"],
            zero_shot_split["val_labels"],
        )
        zero_shot_test_metrics = evaluate_scores_with_threshold(
            zero_shot_split["test_scores"][zero_shot_best["score_name"]].cpu().numpy(),
            zero_shot_split["test_labels"].cpu().numpy().astype(np.int64),
            zero_shot_best["threshold"],
        )
        finetuned_test_metrics, _, _ = evaluate_los_head(
            model,
            test_loader,
            threshold=finetuned_threshold,
        )

        row = {
            "scenario": scenario,
            "model_family": "foundation_corridor_concat",
            "backbone_checkpoint_path": args.pretrained_checkpoint,
            "los_head_checkpoint_path": head_checkpoint_path,
            "use_material_features": args.use_material_features,
            "n_clusters": args.n_clusters,
            "nearest_k": args.nearest_k,
            "corridor_k": args.corridor_k,
            "corridor_bins": args.corridor_bins,
            "train_val_ratio": args.train_val_ratio,
            "zero_shot_score_name": zero_shot_best["score_name"],
            "zero_shot_threshold": zero_shot_best["threshold"],
            "zero_shot_val_acc": zero_shot_best["accuracy"],
            "zero_shot_val_f1": zero_shot_best["f1"],
            "zero_shot_test_acc": zero_shot_test_metrics["accuracy"],
            "zero_shot_test_f1": zero_shot_test_metrics["f1"],
            "zero_shot_test_precision": zero_shot_test_metrics["precision"],
            "zero_shot_test_recall": zero_shot_test_metrics["recall"],
            "finetuned_threshold": finetuned_threshold,
            "finetuned_test_acc": finetuned_test_metrics["accuracy"],
            "finetuned_test_f1": finetuned_test_metrics["f1"],
            "finetuned_test_precision": finetuned_test_metrics["precision"],
            "finetuned_test_recall": finetuned_test_metrics["recall"],
        }
        pd.DataFrame([row]).to_csv(
            args.csv_log_file,
            mode="a",
            index=False,
            header=not os.path.exists(args.csv_log_file),
        )
        print(
            f"{scenario} [foundation_corridor_concat] | "
            f"zero-shot {zero_shot_best['score_name']} test_acc={zero_shot_test_metrics['accuracy']:.4f}, "
            f"zero-shot test_f1={zero_shot_test_metrics['f1']:.4f}, "
            f"finetuned test_acc={finetuned_test_metrics['accuracy']:.4f}, "
            f"finetuned test_f1={finetuned_test_metrics['f1']:.4f}"
        )


if __name__ == "__main__":
    main()
