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
    "num_classes",
    "train_loss",
    "train_top1_acc",
    "train_top3_acc",
    "val_loss",
    "val_top1_acc",
    "val_top3_acc",
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


def topk_accuracy(logits, targets, k):
    k = min(k, logits.shape[1])
    topk = torch.topk(logits, k=k, dim=1).indices
    return (topk == targets.unsqueeze(1)).any(dim=1).float().mean().item()


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


class BeamMovementDataset(Dataset):
    def __init__(self, pt_path):
        payload = torch.load(pt_path, map_location="cpu", weights_only=False)
        channels = payload["channels"].to(torch.complex64)
        labels = payload["labels"].long()

        if channels.dim() != 5 or channels.shape[1] != 1:
            raise ValueError(
                f"Expected generated movement channels with shape (N, 1, T, H, W), got {tuple(channels.shape)}"
            )

        self.inputs = torch.cat((channels.real, channels.imag), dim=1).float()
        self.labels = labels

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]


class BeamMLPHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class FrozenWiFoBeamClassifier(nn.Module):
    def __init__(self, backbone, hidden_dim, num_classes, dropout=0.1):
        super().__init__()
        self.backbone = backbone
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.head = BeamMLPHead(backbone.embed_dim, hidden_dim, num_classes, dropout=dropout)

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
        # print(f"tokens: {tokens.shape}")
        return tokens[:,-1, :]
        # return tokens.mean(dim=1)

    def forward(self, x):
        self.backbone.eval()
        with torch.no_grad():
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
    if train:
        model.head.train()
    else:
        model.head.eval()

    total_loss = 0.0
    total_top1 = 0.0
    total_top3 = 0.0
    total_examples = 0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        if train:
            optimizer.zero_grad(set_to_none=True)

        logits = model(inputs)
        loss = criterion(logits, targets)

        print("pred")
        print(logits)
        print("true")
        print(targets)

        if train:
            loss.backward()
            optimizer.step()

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_top1 += topk_accuracy(logits, targets, 1) * batch_size
        total_top3 += topk_accuracy(logits, targets, 3) * batch_size
        total_examples += batch_size

    return {
        "loss": total_loss / max(total_examples, 1),
        "top1_acc": total_top1 / max(total_examples, 1),
        "top3_acc": total_top3 / max(total_examples, 1),
    }


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

    train_dataset = BeamMovementDataset(train_path)
    val_dataset = BeamMovementDataset(val_path)
    # num_classes = int(max(train_dataset.labels.max().item(), val_dataset.labels.max().item()) + 1)
    num_classes = 64


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

    model = FrozenWiFoBeamClassifier(
        backbone=backbone,
        hidden_dim=args.head_hidden_dim,
        num_classes=num_classes,
        dropout=args.head_dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    scenario_dir = os.path.join(args.output_dir, scenario)
    os.makedirs(scenario_dir, exist_ok=True)
    best_top1 = -1.0
    best_metrics = None
    best_epoch = -1

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_metrics = run_epoch(model, val_loader, criterion, optimizer, device, train=False)

        print(
            f"[{scenario}] epoch {epoch}/{args.epochs} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_top1={train_metrics['top1_acc']:.4f} "
            f"train_top3={train_metrics['top3_acc']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_top1={val_metrics['top1_acc']:.4f} "
            f"val_top3={val_metrics['top3_acc']:.4f}"
        )

        if val_metrics["top1_acc"] > best_top1:
            best_top1 = val_metrics["top1_acc"]
            best_epoch = epoch
            best_metrics = {
                "train_loss": train_metrics["loss"],
                "train_top1_acc": train_metrics["top1_acc"],
                "train_top3_acc": train_metrics["top3_acc"],
                "val_loss": val_metrics["loss"],
                "val_top1_acc": val_metrics["top1_acc"],
                "val_top3_acc": val_metrics["top3_acc"],
            }
            torch.save(
                {
                    "scenario": scenario,
                    "epoch": epoch,
                    "head_state_dict": model.head.state_dict(),
                    "num_classes": num_classes,
                    "args": vars(args),
                    "metrics": best_metrics,
                },
                os.path.join(scenario_dir, "best_beam_head.pt"),
            )

    row = {
        "scenario": scenario,
        "status": "ok",
        "best_epoch": best_epoch,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "num_classes": num_classes,
        "train_loss": best_metrics["train_loss"],
        "train_top1_acc": best_metrics["train_top1_acc"],
        "train_top3_acc": best_metrics["train_top3_acc"],
        "val_loss": best_metrics["val_loss"],
        "val_top1_acc": best_metrics["val_top1_acc"],
        "val_top3_acc": best_metrics["val_top3_acc"],
        "checkpoint": checkpoint_path,
        "error": "",
    }
    return row


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a frozen WiFo encoder with an MLP beam-prediction head.")
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
        default="/home/blessedg/Pathformer/WiFo/beam_finetune_runs",
        help="Directory for per-scenario beam head checkpoints and summaries.",
    )
    parser.add_argument(
        "--results-csv",
        type=str,
        default="/home/blessedg/Pathformer/WiFo/beam_finetune_results.csv",
        help="CSV file to append per-scenario beam finetuning results.",
    )
    parser.add_argument("--scenario", type=str, default=None, help="Run only one scenario.")
    parser.add_argument("--scenarios", nargs="+", default=None, help="Optional explicit scenario list to run.")
    parser.add_argument("--all-scenarios", action="store_true", help="Run every scenario in ALL_SCENARIOS.")
    parser.add_argument("--num-splits", type=int, default=1, help="Split the selected scenario list into N interleaved shards.")
    parser.add_argument("--split-index", type=int, default=0, help="Which shard to run when using --num-splits.")
    parser.add_argument("--epochs", type=int, default=20)
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
                "num_classes": None,
                "train_loss": None,
                "train_top1_acc": None,
                "train_top3_acc": None,
                "val_loss": None,
                "val_top1_acc": None,
                "val_top3_acc": None,
                "checkpoint": checkpoint_path,
                "error": str(exc),
            }
            print(f"[{scenario}] failed: {exc}")
        append_result_row(args.results_csv, row)
        print(f"Appended results for {scenario} to {args.results_csv}")


if __name__ == "__main__":
    main()


#python /home/blessedg/Pathformer/WiFo/src/finetune_beam.py --scenario city_6_miami_3p5 --checkpoint ./weights/wifo_base