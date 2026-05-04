import argparse
import os
import warnings

import deepmimo as dm
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.dataloaders import PreTrainMySeqDataLoader
from multiscenario_direct_training_first_step_residual import (
    FirstStepResidualDataset,
    FirstStepResidualPathDecoder,
    load_best_checkpoint as load_residual_checkpoint,
    resolve_scenarios,
)
from multiscenario_direct_training_first_step_residual_corridor import (
    build_first_step_assignments_with_corridor,
)
from scene_feature_utils import SceneFeatureBank
from utils.utils import (
    ChannelParameters,
    MyChannelComputer,
    compute_single_array_response_torch,
    count_parameters,
)

warnings.filterwarnings("ignore", category=UserWarning)

csv_log_file = "beam_prediction_finetuning_foundation.csv"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Beam head finetuning for the corridor-concat foundation model."
    )
    parser.add_argument("scenarios", nargs="*", help="Optional scenario names to run.")
    parser.add_argument("--scenario", dest="scenario_flag", action="append")
    parser.add_argument("--scenario-file", type=str)
    parser.add_argument("--shard-index", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=None)
    parser.add_argument("--csv-log-file", type=str, default=csv_log_file)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--pretrained-checkpoint",
        type=str,
        default="/home/blessedg/Pathformer/checkpoints_first_step_residual_corridor_concat/first_step_residual_corridor_concat_27scenarios_44710a4a_best_model_checkpoint.pth",
    )
    parser.add_argument(
        "--beam-head-checkpoint-dir",
        type=str,
        default="/home/blessedg/Pathformer/checkpoints_beam_heads_foundation",
    )
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--n-clusters", type=int, default=25)
    parser.add_argument("--nearest-k", type=int, default=5)
    parser.add_argument("--corridor-k", type=int, default=5)
    parser.add_argument("--corridor-bins", type=int, default=8)
    parser.add_argument("--use-material-features", action="store_true")
    parser.add_argument("--no-material-features", dest="use_material_features", action="store_false")
    parser.set_defaults(use_material_features=True)
    return parser.parse_args()


def default_scenarios():
    return [
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


def compute_beam_label_from_channel(H, S):
    if not torch.is_tensor(H):
        H = torch.from_numpy(H)
    if not torch.is_tensor(S):
        S = torch.from_numpy(S)
    H = H.to(torch.complex64)
    S = S.to(torch.complex64)
    Y = S.conj().T @ H
    prx = torch.sum(torch.abs(Y) ** 2, dim=2)
    best = torch.argmax(prx, dim=1)
    return best, prx


def make_dft_codebook(B=8):
    params = ChannelParameters()
    az_t = np.linspace(-np.pi, np.pi, B, endpoint=False, dtype=np.float32)
    el_t = np.linspace(-np.pi, np.pi, B, endpoint=False, dtype=np.float32)
    az_new = []
    el_new = []
    for az in az_t:
        for el in el_t:
            az_new.append(az)
            el_new.append(el)
    az_new = torch.tensor(az_new).unsqueeze(1)
    el_new = torch.tensor(el_new).unsqueeze(1)
    array_response = compute_single_array_response_torch(params.bs_antenna, az_new, el_new)
    return array_response.squeeze(2).T


def build_corridor_loaders(dataset, batch_size, pad_value, args):
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
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=train_data.collate_fn)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, collate_fn=val_data.collate_fn)
    return train_data, val_data, train_loader, val_loader


class BeamHeadFinetuner(nn.Module):
    def __init__(self, backbone, hidden_dim=1024, n_beams=64):
        super().__init__()
        self.backbone = backbone
        self.beam_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, n_beams),
        )

    def forward(self, prompts, paths, interactions, path_padding_mask):
        h_paths, _ = self.backbone.backbone.forward_hidden(prompts, paths, interactions)
        valid_mask = path_padding_mask.bool()
        if valid_mask.size(1) != h_paths.size(1):
            valid_mask = valid_mask[:, : h_paths.size(1)]
        if valid_mask.size(1) > 0:
            valid_mask = valid_mask.clone()
            valid_mask[:, 0] = False
        valid_float = valid_mask.unsqueeze(-1).float()
        denom = valid_float.sum(dim=1).clamp(min=1.0)
        summary = (h_paths * valid_float).sum(dim=1) / denom
        return self.beam_head(summary)


def freeze_backbone(module):
    for param in module.parameters():
        param.requires_grad = False


def compute_beam_labels(paths, pad_value, mycomputer, S):
    paths_out = paths[:, 1:, :]
    delay_t = paths_out[:, :, 0].detach().cpu().numpy()
    power_t = paths_out[:, :, 1].detach().cpu().numpy()
    phase = np.rad2deg(paths_out[:, :, 2].detach().cpu().numpy())
    aod_az = paths_out[:, :, 5].detach().cpu().numpy()
    aod_el = paths_out[:, :, 6].detach().cpu().numpy()

    power_t = np.where(power_t == pad_value, 0, power_t)
    power_linear = 10 ** ((power_t / 0.01) / 10)
    delay_secs = delay_t / 1e6
    mask = delay_t == pad_value

    delay_secs = np.where(mask, np.nan, delay_secs)
    phase = np.where(mask, np.nan, phase)
    power_linear = np.where(mask, np.nan, power_linear)
    aod_az = np.where(mask, np.nan, aod_az)
    aod_el = np.where(mask, np.nan, aod_el)

    H = mycomputer.compute_channels(
        power_linear,
        delay_secs,
        phase,
        aod_az,
        aod_el,
        kwargs=None,
    )[:, 0, :, :]
    beam_label, _ = compute_beam_label_from_channel(H, S)
    return beam_label.long()


def unpack_batch(batch):
    prompts, paths, path_lengths, interactions, env, env_prop, path_padding_mask, first_step_baselines = batch
    return prompts, paths, interactions, path_padding_mask


@torch.no_grad()
def evaluate_beam_head(model, val_loader, pad_value, mycomputer, S, k_list=(1, 3)):
    model.eval()
    topk_correct = {k: 0 for k in k_list}
    total = 0
    for batch in tqdm(val_loader, desc="Eval [foundation]", leave=False):
        prompts, paths, interactions, path_padding_mask = unpack_batch(batch)
        prompts = prompts.to(device)
        paths = paths.to(device)
        interactions = interactions.to(device)
        path_padding_mask = path_padding_mask.to(device)

        labels = compute_beam_labels(paths, pad_value, mycomputer, S).to(device)
        logits = model(prompts, paths, interactions, path_padding_mask)
        max_k = max(k_list)
        topk = torch.topk(logits, k=max_k, dim=1).indices
        for k in k_list:
            topk_correct[k] += (topk[:, :k] == labels.unsqueeze(1)).any(dim=1).sum().item()
        total += prompts.size(0)

    return {f"top{k}_acc": topk_correct[k] / max(total, 1) for k in k_list}


def train_beam_head(model, train_loader, val_loader, pad_value, config, checkpoint_path):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.beam_head.parameters(), lr=config["LR"], weight_decay=config["WEIGHT_DECAY"])
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=3, mode="max")
    mycomputer = MyChannelComputer()
    S = make_dft_codebook(B=8)
    best_val_top1 = -1.0

    for epoch in range(config["epochs"]):
        model.train()
        train_losses = []
        train_correct = 0
        train_total = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [foundation Train]", leave=False)
        for batch in pbar:
            prompts, paths, interactions, path_padding_mask = unpack_batch(batch)
            prompts = prompts.to(device)
            paths = paths.to(device)
            interactions = interactions.to(device)
            path_padding_mask = path_padding_mask.to(device)

            labels = compute_beam_labels(paths, pad_value, mycomputer, S).to(device)
            logits = model(prompts, paths, interactions, path_padding_mask)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.beam_head.parameters(), max_norm=1.0)
            optimizer.step()

            preds = logits.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += prompts.size(0)
            train_losses.append(loss.item())
            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "train_acc": f"{train_correct / max(train_total, 1):.4f}",
                }
            )

        val_metrics = evaluate_beam_head(model, val_loader, pad_value, mycomputer, S, k_list=(1, 3))
        val_top1 = val_metrics["top1_acc"]
        scheduler.step(val_top1)
        print(
            f"Epoch {epoch:02d} [foundation] "
            f"train_loss={np.mean(train_losses):.4f} "
            f"train_acc={train_correct / max(train_total, 1):.4f} "
            f"val_top1={val_metrics['top1_acc']:.4f} "
            f"val_top3={val_metrics['top3_acc']:.4f}"
        )

        if val_top1 > best_val_top1:
            best_val_top1 = val_top1
            torch.save(
                {
                    "epoch": epoch,
                    "beam_head_state_dict": model.beam_head.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_val_top1": best_val_top1,
                    "model_family": "foundation_corridor_concat",
                },
                checkpoint_path,
            )
            print(f"  ✓ Saved beam head checkpoint to {checkpoint_path}")

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.beam_head.load_state_dict(checkpoint["beam_head_state_dict"])

    return evaluate_beam_head(model, val_loader, pad_value, mycomputer, S, k_list=(1, 3))


def build_foundation_objects(dataset, args, config):
    pad_value = config["PAD_VALUE"]
    train_data, val_data, train_loader, val_loader = build_corridor_loaders(dataset, args.batch_size, pad_value, args)
    prompt_dim = int(train_data.augmented_prompts[0].numel())
    backbone = FirstStepResidualPathDecoder(
        prompt_dim=prompt_dim,
        hidden_dim=512 * 2,
        n_layers=8 + 4,
        n_heads=8,
    ).to(device)
    load_residual_checkpoint(backbone, config["backbone_checkpoint_path"])
    freeze_backbone(backbone)
    model = BeamHeadFinetuner(backbone=backbone, hidden_dim=512 * 2, n_beams=64).to(device)
    return model, train_loader, val_loader


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

    os.makedirs(args.beam_head_checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.csv_log_file) or ".", exist_ok=True)
    print(f"Running foundation beam finetuning for {len(scenarios)} scenario(s): {scenarios}")
    print(f"Using foundation checkpoint: {args.pretrained_checkpoint}")

    for scenario in scenarios:
        print(f"\nBeam finetuning for {scenario} [foundation_corridor_concat]")
        dataset = dm.load(scenario)
        config = {
            "PAD_VALUE": 0,
            "LR": args.lr,
            "WEIGHT_DECAY": args.weight_decay,
            "epochs": args.epochs,
            "backbone_checkpoint_path": args.pretrained_checkpoint,
        }
        model, train_loader, val_loader = build_foundation_objects(dataset, args, config)
        print(f"foundation_corridor_concat trainable head parameters: {count_parameters(model.beam_head)}")
        head_checkpoint_path = os.path.join(
            args.beam_head_checkpoint_dir,
            f"beam_head_foundation_corridor_concat_{scenario}.pth",
        )

        if args.skip_train and os.path.exists(head_checkpoint_path):
            checkpoint = torch.load(head_checkpoint_path, map_location=device)
            model.beam_head.load_state_dict(checkpoint["beam_head_state_dict"])
            metrics = evaluate_beam_head(
                model,
                val_loader,
                config["PAD_VALUE"],
                MyChannelComputer(),
                make_dft_codebook(B=8),
                k_list=(1, 3),
            )
        elif args.skip_train:
            print(f"Skipping {scenario} [foundation_corridor_concat] because head checkpoint is missing: {head_checkpoint_path}")
            continue
        else:
            metrics = train_beam_head(
                model,
                train_loader,
                val_loader,
                config["PAD_VALUE"],
                config,
                head_checkpoint_path,
            )

        row = {
            "scenario": scenario,
            "model_family": "foundation_corridor_concat",
            "top1_acc": metrics["top1_acc"],
            "top3_acc": metrics["top3_acc"],
            "backbone_checkpoint_path": args.pretrained_checkpoint,
            "beam_head_checkpoint_path": head_checkpoint_path,
            "use_material_features": args.use_material_features,
            "n_clusters": args.n_clusters,
            "nearest_k": args.nearest_k,
            "corridor_k": args.corridor_k,
            "corridor_bins": args.corridor_bins,
        }
        pd.DataFrame([row]).to_csv(
            args.csv_log_file,
            mode="a",
            index=False,
            header=not os.path.exists(args.csv_log_file),
        )
        print(
            f"{scenario} [foundation_corridor_concat] | top1={metrics['top1_acc']:.4f}, "
            f"top3={metrics['top3_acc']:.4f}"
        )


if __name__ == "__main__":
    main()
