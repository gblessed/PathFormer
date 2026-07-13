# coding=utf-8
import argparse
import csv
import os
import random
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from model import WiFo_model


ALL_SCENARIOS = [
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
    "city_47_chicago_3p5",
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

RESULT_FIELDS = [
    "scenario",
    "status",
    "best_epoch",
    "train_samples",
    "val_samples",
    "train_loss",
    "train_mde_m",
    "val_loss",
    "val_mde_m",
    "val_median_mde_m",
    "val_p90_mde_m",
    "checkpoint",
    "error",
]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_checkpoint_path(path):
    if os.path.isfile(path):
        return path
    if os.path.isfile(f"{path}.pkl"):
        return f"{path}.pkl"
    raise FileNotFoundError(f"Could not find checkpoint at {path} or {path}.pkl")


def normalize_size(size):
    if size == "middle":
        return "base"
    return size


def split_scenarios(scenarios, num_splits, split_index):
    if num_splits < 1:
        raise ValueError(f"num_splits must be >= 1, got {num_splits}")
    if split_index < 0 or split_index >= num_splits:
        raise ValueError(f"split_index must be in [0, {num_splits - 1}], got {split_index}")
    chunk_size = (len(scenarios) + num_splits - 1) // num_splits
    start = split_index * chunk_size
    end = min(start + chunk_size, len(scenarios))
    return scenarios[start:end]


def compute_localization_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    errors = np.linalg.norm(y_true - y_pred, axis=1)
    return {
        "mde_m": float(np.mean(errors)) if errors.size else 0.0,
        "median_mde_m": float(np.median(errors)) if errors.size else 0.0,
        "p90_mde_m": float(np.percentile(errors, 90)) if errors.size else 0.0,
    }


class UserLocalizationMovementDataset(Dataset):
    def __init__(self, pt_path):
        payload = torch.load(pt_path, map_location="cpu", weights_only=False)
        channels = payload["channels"].to(torch.complex64)
        labels_user_loc = payload["labels_user_loc"].to(torch.float32)

        if channels.dim() != 5 or channels.shape[1] != 1:
            raise ValueError(
                f"Expected generated movement channels with shape (N, 1, T, H, W), got {tuple(channels.shape)}"
            )
        if labels_user_loc.dim() != 2 or labels_user_loc.shape[1] < 2:
            raise ValueError(
                f"Expected labels_user_loc with shape (N, >=2), got {tuple(labels_user_loc.shape)}"
            )

        self.inputs = torch.cat((channels.real, channels.imag), dim=1).float()
        self.targets_xy = labels_user_loc[:, :2]

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets_xy[idx]


class LocalizationMLPHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2),
        )

    def forward(self, x):
        return self.net(x)


class WiFoLocalizer(nn.Module):
    def __init__(self, backbone, hidden_dim, dropout=0.1, freeze_backbone=False):
        super().__init__()
        self.backbone = backbone
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            self.backbone.eval()
            for param in self.backbone.parameters():
                param.requires_grad = False
        self.head = LocalizationMLPHead(backbone.embed_dim, hidden_dim, dropout=dropout)

    def encode_no_mask(self, x):
        batch_size, _, T, H, W = x.shape
        tokens = self.backbone.Embedding(x)
        Tp = T // self.backbone.t_patch_size
        Hp = H // self.backbone.patch_size
        Wp = W // self.backbone.patch_size
        input_size = (Tp, Hp, Wp)
        ids_keep = torch.arange(tokens.shape[1], device=x.device).unsqueeze(0).expand(batch_size, -1)

        if self.backbone.pos_emb in ("SinCos", "trivial"):
            tokens = tokens + self.backbone.pos_embed_enc(ids_keep, batch_size, input_size)
        elif self.backbone.pos_emb == "SinCos_3D":
            tokens = tokens + self.backbone.pos_embed_enc_3d(ids_keep, batch_size, input_size, scale=[1, 1, 1])

        for block in self.backbone.blocks:
            tokens = block(tokens)
        tokens = self.backbone.norm(tokens)
        return tokens[:, -1, :]

    def forward(self, x):
        if self.freeze_backbone:
            self.backbone.eval()
            with torch.no_grad():
                embedding = self.encode_no_mask(x)
        else:
            embedding = self.encode_no_mask(x)
        return self.head(embedding)


def create_backbone(args):
    model_args = SimpleNamespace(
        size=normalize_size(args.size),
        t_patch_size=args.t_patch_size,
        patch_size=args.patch_size,
        pos_emb=args.pos_emb,
        no_qkv_bias=args.no_qkv_bias,
    )
    return WiFo_model(model_args)


def run_epoch(model, loader, criterion, optimizer, device, train):
    model.train(train)
    if model.freeze_backbone:
        model.backbone.eval()

    total_loss = 0.0
    total_examples = 0
    all_preds = []
    all_targets = []

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        if train:
            optimizer.zero_grad(set_to_none=True)

        preds = model(inputs)
        loss = criterion(preds, targets)

        if train:
            loss.backward()
            optimizer.step()

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_examples += batch_size
        all_preds.append(preds.detach().cpu().numpy())
        all_targets.append(targets.detach().cpu().numpy())

    metrics = compute_localization_metrics(
        np.concatenate(all_targets, axis=0),
        np.concatenate(all_preds, axis=0),
    )
    metrics["loss"] = total_loss / max(total_examples, 1)
    return metrics


def append_result_row(csv_path, row):
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow({field: row.get(field) for field in RESULT_FIELDS})


def finetune_one_scenario(args, scenario, device):
    train_path = os.path.join(args.data_dir, f"_{scenario}_train_data.pt")
    val_path = os.path.join(args.data_dir, f"_{scenario}_val_data.pt")
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Missing train data for {scenario}: {train_path}")
    if not os.path.exists(val_path):
        raise FileNotFoundError(f"Missing val data for {scenario}: {val_path}")

    train_dataset = UserLocalizationMovementDataset(train_path)
    val_dataset = UserLocalizationMovementDataset(val_path)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    backbone = create_backbone(args).to(device)
    checkpoint_path = resolve_checkpoint_path(args.checkpoint)
    state_dict = torch.load(checkpoint_path, map_location=device)
    backbone.load_state_dict(state_dict, strict=False)

    model = WiFoLocalizer(
        backbone=backbone,
        hidden_dim=args.head_hidden_dim,
        dropout=args.head_dropout,
        freeze_backbone=args.freeze_backbone,
    ).to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.SmoothL1Loss()

    scenario_dir = os.path.join(args.output_dir, scenario)
    os.makedirs(scenario_dir, exist_ok=True)
    best_val_mde = float("inf")
    best_metrics = None
    best_epoch = -1

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_metrics = run_epoch(model, val_loader, criterion, optimizer, device, train=False)

        print(
            f"[{scenario}] epoch {epoch}/{args.epochs} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_mde={train_metrics['mde_m']:.2f}m "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_mde={val_metrics['mde_m']:.2f}m "
            f"val_med={val_metrics['median_mde_m']:.2f}m "
            f"val_p90={val_metrics['p90_mde_m']:.2f}m"
        )

        if val_metrics["mde_m"] < best_val_mde:
            best_val_mde = val_metrics["mde_m"]
            best_epoch = epoch
            best_metrics = {
                "train_loss": train_metrics["loss"],
                "train_mde_m": train_metrics["mde_m"],
                "val_loss": val_metrics["loss"],
                "val_mde_m": val_metrics["mde_m"],
                "val_median_mde_m": val_metrics["median_mde_m"],
                "val_p90_mde_m": val_metrics["p90_mde_m"],
            }
            torch.save(
                {
                    "scenario": scenario,
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "freeze_backbone": args.freeze_backbone,
                    "args": vars(args),
                    "metrics": best_metrics,
                },
                os.path.join(scenario_dir, "best_localization_head.pt"),
            )

    row = {
        "scenario": scenario,
        "status": "ok",
        "best_epoch": best_epoch,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "train_loss": best_metrics["train_loss"],
        "train_mde_m": best_metrics["train_mde_m"],
        "val_loss": best_metrics["val_loss"],
        "val_mde_m": best_metrics["val_mde_m"],
        "val_median_mde_m": best_metrics["val_median_mde_m"],
        "val_p90_mde_m": best_metrics["val_p90_mde_m"],
        "checkpoint": checkpoint_path,
        "error": "",
    }
    return row


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a WiFo encoder for XY user localization from generated movement channels.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the pretrained WiFo checkpoint (.pkl or path stem without suffix).",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/home/blessedg/Pathformer/WiFo/dataset/blessed_task",
        help="Directory containing generated WiFo movement-channel train/val .pt files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/home/blessedg/Pathformer/WiFo/localization_finetune_runs",
        help="Directory for per-scenario localization checkpoints and summaries.",
    )
    parser.add_argument(
        "--results-csv",
        type=str,
        default="/home/blessedg/Pathformer/WiFo/localization_finetune_results.csv",
        help="CSV file to append per-scenario localization finetuning results.",
    )
    parser.add_argument("--scenario", type=str, default=None, help="Run only one scenario.")
    parser.add_argument("--scenarios", nargs="+", default=None, help="Optional explicit scenario list to run.")
    parser.add_argument("--all-scenarios", action="store_true", help="Run every scenario in ALL_SCENARIOS.")
    parser.add_argument("--num-splits", type=int, default=1, help="Split the selected scenario list into N interleaved shards.")
    parser.add_argument("--split-index", type=int, default=0, help="Which shard to run when using --num-splits.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--head-hidden-dim", type=int, default=256)
    parser.add_argument("--head-dropout", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device-id", type=str, default="0")
    parser.add_argument("--size", type=str, default="base")
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--t-patch-size", type=int, default=4)
    parser.add_argument("--pos-emb", type=str, default="SinCos_3D")
    parser.add_argument("--no-qkv-bias", type=int, default=0)
    parser.add_argument("--freeze-backbone", action="store_true", help="Freeze the WiFo encoder and train only the localization head.")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu")
    checkpoint_path = resolve_checkpoint_path(args.checkpoint)
    if args.scenarios:
        scenarios = args.scenarios
    elif args.all_scenarios:
        scenarios = ALL_SCENARIOS
    elif args.scenario:
        scenarios = [args.scenario]
    else:
        scenarios = ALL_SCENARIOS

    scenarios = split_scenarios(scenarios, args.num_splits, args.split_index)
    print(f"Running {len(scenarios)} scenarios on split {args.split_index + 1}/{args.num_splits}: {scenarios}")

    for scenario in scenarios:
        try:
            row = finetune_one_scenario(args, scenario, device)
        except Exception as exc:
            row = {
                "scenario": scenario,
                "status": "error",
                "best_epoch": None,
                "train_samples": None,
                "val_samples": None,
                "train_loss": None,
                "train_mde_m": None,
                "val_loss": None,
                "val_mde_m": None,
                "val_median_mde_m": None,
                "val_p90_mde_m": None,
                "checkpoint": checkpoint_path,
                "error": str(exc),
            }
            print(f"[{scenario}] failed: {exc}")
        append_result_row(args.results_csv, row)
        print(f"Appended results for {scenario} to {args.results_csv}")


if __name__ == "__main__":
    main()


# python /home/blessedg/Pathformer/WiFo/src/finetune_user_localization.py --scenario city_6_miami_3p5 --checkpoint ./weights/wifo_base
'''

python /home/blessedg/Pathformer/WiFo/src/finetune_user_localization.py \
  --scenario city_2_chicago_3p5 \
  --checkpoint /home/blessedg/Pathformer/WiFo/src/weights/wifo_base \
  --data-dir /home/blessedg/Pathformer/WiFo/dataset/blessed_task_user_loc \
  --results-csv /home/blessedg/Pathformer/WiFo/localization_finetune_results.csv
'''