import argparse
import os
import warnings
from collections import defaultdict

import deepmimo as dm
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
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

csv_log_file = "beam_prediction_finetuning_foundation_newx32.csv"
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
    parser.add_argument("--label-match-decimals", type=int, default=4)
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


def make_dft_codebook(B=32, ant_params=None):
    """Build an easier azimuth codebook for the default 8x1 BS array."""
    if ant_params is None:
        ant_params = ChannelParameters().bs_antenna

    n_beams = int(B)

    # Sample uniformly in spatial frequency u = sin(phi), not in phi itself.
    u = torch.linspace(-0.95, 0.95, steps=n_beams, dtype=torch.float32).unsqueeze(0)
    azimuth = torch.asin(u)

    elevation = torch.full_like(azimuth, np.pi / 2)
    codebook = compute_single_array_response_torch(ant_params, elevation, azimuth)
    return codebook.squeeze(0)

# def make_dft_codebook(B=8, ant_params=None):
#     """Build a 64-beam azimuth sweep for the default 8x1 BS array."""
#     n_beams = int(B)
#     if ant_params is None:
#         ant_params = ChannelParameters().bs_antenna

#     azimuth = torch.linspace(-np.pi / 2, np.pi / 2, steps=n_beams, dtype=torch.float32).unsqueeze(0)
#     elevation = torch.full_like(azimuth, np.pi / 2)
#     codebook = compute_single_array_response_torch(ant_params, elevation, azimuth)
#     return codebook.squeeze(0)


def infer_mmwave_scenario(sub6_scenario):
    if "3p5" not in sub6_scenario:
        raise ValueError(f"Expected a sub-6 scenario containing '3p5', got: {sub6_scenario}")
    return sub6_scenario.replace("3p5", "28", 1)


# def make_dft_codebook(B=8):
#     params = ChannelParameters()
#     az_t = np.linspace(-np.pi, np.pi, B, endpoint=False, dtype=np.float32)
#     el_t = np.linspace(-np.pi, np.pi, B, endpoint=False, dtype=np.float32)
#     az_new = []
#     el_new = []
#     for az in az_t:
#         for el in el_t:
#             az_new.append(az)
#             el_new.append(el)
#     az_new = torch.tensor(az_new).unsqueeze(1)
#     el_new = torch.tensor(el_new).unsqueeze(1)
#     array_response = compute_single_array_response_torch(params.bs_antenna, az_new, el_new)
#     return array_response.squeeze(2).T


class BeamLabelDataset(Dataset):
    def __init__(self, base_dataset, beam_labels):
        self.base_dataset = base_dataset
        self.beam_labels = beam_labels.long().cpu()

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        return self.base_dataset[idx], self.beam_labels[idx]

    def collate_fn(self, batch):
        base_items = [item[0] for item in batch]
        labels = torch.stack([item[1] for item in batch], dim=0)
        collated = self.base_dataset.collate_fn(base_items)
        return (*collated, labels)


class CachedBeamFeatureDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features.float().cpu()
        self.labels = labels.long().cpu()

    def __len__(self):
        return self.features.size(0)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class IndexedDataset(Dataset):
    def __init__(self, base_dataset, indices):
        self.base_dataset = base_dataset
        self.indices = list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.base_dataset[self.indices[idx]]

    def collate_fn(self, batch):
        return self.base_dataset.collate_fn(batch)


def _make_loader(dataset, batch_size, shuffle, collate_fn=None):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)


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
    train_loader = _make_loader(train_data, batch_size, True, train_data.collate_fn)
    val_loader = _make_loader(val_data, batch_size, False, val_data.collate_fn)
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
        # return h_paths[:, 0]

    def classify_summary_features(self, summary_features):
        return self.beam_head(summary_features)

    def forward(self, prompts, paths, interactions, path_padding_mask):
        summary = self.extract_summary_features(prompts, paths, interactions, path_padding_mask)
        return self.classify_summary_features(summary)


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
    prompts, paths, path_lengths, interactions, env, env_prop, path_padding_mask, first_step_baselines, labels = batch
    return prompts, paths, interactions, path_padding_mask, labels


def _unpack_paths_from_batch(batch):
    if len(batch) == 8:
        prompts, paths, path_lengths, interactions, env, env_prop, path_padding_mask, _ = batch
    elif len(batch) == 7:
        prompts, paths, path_lengths, interactions, env, env_prop, path_padding_mask = batch
    else:
        raise ValueError(f"Unexpected batch structure with {len(batch)} elements")
    return paths


def _get_base_dataset(dataset):
    return dataset.base_dataset if hasattr(dataset, "base_dataset") else dataset


def _position_key(tx_pos, rx_pos, decimals):
    tx_key = tuple(np.round(np.asarray(tx_pos, dtype=np.float64), decimals=decimals).tolist())
    rx_key = tuple(np.round(np.asarray(rx_pos, dtype=np.float64), decimals=decimals).tolist())
    return tx_key + rx_key


def _dataset_position_keys(dataset, decimals):
    base_dataset = _get_base_dataset(dataset)
    df = base_dataset.dataset_filtered
    return [
        _position_key(tx_pos, rx_pos, decimals)
        for tx_pos, rx_pos in zip(df["tx_pos"], df["rx_pos"])
    ]


def precompute_beam_labels_for_dataset(dataset, batch_size, pad_value, mycomputer, S, desc):
    loader = _make_loader(dataset, batch_size, False, dataset.collate_fn)
    all_labels = []
    for batch in tqdm(loader, desc=desc, leave=False):
        paths = _unpack_paths_from_batch(batch)
        labels = compute_beam_labels(paths, pad_value, mycomputer, S)
        all_labels.append(labels.cpu())
    return torch.cat(all_labels, dim=0)


def attach_cached_beam_labels(dataset, batch_size, pad_value, mycomputer, S, desc):
    beam_labels = precompute_beam_labels_for_dataset(
        dataset=dataset,
        batch_size=batch_size,
        pad_value=pad_value,
        mycomputer=mycomputer,
        S=S,
        desc=desc,
    )
    return BeamLabelDataset(dataset, beam_labels)


def attach_position_matched_beam_labels(dataset, reference_dataset, reference_labels, decimals, desc):
    dataset_keys = _dataset_position_keys(dataset, decimals)
    reference_keys = _dataset_position_keys(reference_dataset, decimals)

    label_lookup = defaultdict(list)
    for key, label in zip(reference_keys, reference_labels.tolist()):
        label_lookup[key].append(int(label))

    matched_labels = []
    kept_indices = []
    dropped_count = 0
    for idx, key in enumerate(dataset_keys):
        if label_lookup[key]:
            kept_indices.append(idx)
            matched_labels.append(label_lookup[key].pop(0))
        else:
            dropped_count += 1

    if not matched_labels:
        raise RuntimeError(f"{desc}: no matched mmWave labels were found after position matching.")

    print(
        f"{desc}: matched {len(matched_labels)} labels, dropped {dropped_count} unmatched users "
        f"using rounded tx/rx positions with decimals={decimals}"
    )
    subset_dataset = IndexedDataset(dataset, kept_indices)
    return BeamLabelDataset(subset_dataset, torch.tensor(matched_labels, dtype=torch.long))


@torch.no_grad()
def precompute_backbone_features(model, dataset, batch_size, desc):
    model.eval()
    loader = _make_loader(dataset, batch_size, False, dataset.collate_fn)
    all_features = []
    all_labels = []
    for batch in tqdm(loader, desc=desc, leave=False):
        prompts, paths, interactions, path_padding_mask, labels = unpack_batch(batch)
        prompts = prompts.to(device)
        paths = paths.to(device)
        interactions = interactions.to(device)
        path_padding_mask = path_padding_mask.to(device)
        features = model.extract_summary_features(prompts, paths, interactions, path_padding_mask)
        all_features.append(features.cpu())
        all_labels.append(labels.cpu())
    return CachedBeamFeatureDataset(torch.cat(all_features, dim=0), torch.cat(all_labels, dim=0))


@torch.no_grad()
def evaluate_beam_head(model, val_loader, k_list=(1, 3)):
    model.eval()
    topk_correct = {k: 0 for k in k_list}
    total = 0
    for batch in tqdm(val_loader, desc="Eval [foundation]", leave=False):
        features, labels = batch
        features = features.to(device)
        labels = labels.to(device)
        logits = model.classify_summary_features(features)
        max_k = max(k_list)
        topk = torch.topk(logits, k=max_k, dim=1).indices
        for k in k_list:
            topk_correct[k] += (topk[:, :k] == labels.unsqueeze(1)).any(dim=1).sum().item()
        total += features.size(0)

    return {f"top{k}_acc": topk_correct[k] / max(total, 1) for k in k_list}


def train_beam_head(model, train_loader, val_loader, pad_value, config, checkpoint_path):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.beam_head.parameters(), lr=config["LR"], weight_decay=config["WEIGHT_DECAY"])
    # scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=3, mode="max")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=25,
        T_mult=1,
        eta_min=1e-8,
    )
    best_val_top1 = -1.0

    for epoch in range(config["epochs"]):
        model.train()
        train_losses = []
        train_correct = 0
        train_total = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [foundation Train]", leave=False)
        for batch in pbar:
            features, labels = batch
            features = features.to(device)
            labels = labels.to(device)
            logits = model.classify_summary_features(features)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.beam_head.parameters(), max_norm=1.0)
            optimizer.step()

            preds = logits.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += features.size(0)
            train_losses.append(loss.item())
            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "train_acc": f"{train_correct / max(train_total, 1):.4f}",
                }
            )

        val_metrics = evaluate_beam_head(model, val_loader, k_list=(1, 3))
        val_top1 = val_metrics["top1_acc"]
        # scheduler.step(val_top1)
        scheduler.step()

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

    return evaluate_beam_head(model, val_loader, k_list=(1, 3))


def build_foundation_objects(dataset, mmwave_dataset, args, config):
    pad_value = config["PAD_VALUE"]
    mycomputer = MyChannelComputer()
    # codebook = make_dft_codebook(B=8)
    codebook = make_dft_codebook(B=32)
    train_data, val_data, _, _ = build_corridor_loaders(dataset, args.batch_size, pad_value, args)
    mmwave_train_base = PreTrainMySeqDataLoader(
        mmwave_dataset,
        train=True,
        split_by="user",
        sort_by="power",
        normalizers=None,
        apply_normalizers=[],
        pad_value=pad_value,
        include_aod=True,
    )
    mmwave_val_base = PreTrainMySeqDataLoader(
        mmwave_dataset,
        train=False,
        split_by="user",
        sort_by="power",
        normalizers=None,
        apply_normalizers=[],
        pad_value=pad_value,
        include_aod=True,
    )
    prompt_dim = int(train_data.augmented_prompts[0].numel())
    backbone = FirstStepResidualPathDecoder(
        prompt_dim=prompt_dim,
        hidden_dim=512 * 2,
        n_layers=8 + 4,
        n_heads=8,
    ).to(device)
    load_residual_checkpoint(backbone, config["backbone_checkpoint_path"])
    mmwave_train_labels = precompute_beam_labels_for_dataset(
        dataset=mmwave_train_base,
        batch_size=args.batch_size,
        pad_value=pad_value,
        mycomputer=mycomputer,
        S=codebook,
        desc="Precompute mmWave train labels [foundation]",
    )
    mmwave_val_labels = precompute_beam_labels_for_dataset(
        dataset=mmwave_val_base,
        batch_size=args.batch_size,
        pad_value=pad_value,
        mycomputer=mycomputer,
        S=codebook,
        desc="Precompute mmWave val labels [foundation]",
    )
    train_data = attach_position_matched_beam_labels(
        dataset=train_data,
        reference_dataset=mmwave_train_base,
        reference_labels=mmwave_train_labels,
        decimals=args.label_match_decimals,
        desc="Attach train labels [foundation]",
    )
    val_data = attach_position_matched_beam_labels(
        dataset=val_data,
        reference_dataset=mmwave_val_base,
        reference_labels=mmwave_val_labels,
        decimals=args.label_match_decimals,
        desc="Attach val labels [foundation]",
    )
    freeze_backbone(backbone)
    model = BeamHeadFinetuner(backbone=backbone, hidden_dim=512 * 2, n_beams=config['n_beams']).to(device)
    cached_train = precompute_backbone_features(
        model=model,
        dataset=train_data,
        batch_size=args.batch_size,
        desc="Cache train features [foundation]",
    )
    cached_val = precompute_backbone_features(
        model=model,
        dataset=val_data,
        batch_size=args.batch_size,
        desc="Cache val features [foundation]",
    )
    train_loader = _make_loader(cached_train, args.batch_size, True)
    val_loader = _make_loader(cached_val, args.batch_size, False)
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
        mmwave_scenario = infer_mmwave_scenario(scenario)
        print(f"Using mmWave label scenario: {mmwave_scenario}")
        mmwave_dataset = dm.load(mmwave_scenario)
        config = {
            "PAD_VALUE": 0,
            "LR": args.lr,
            "WEIGHT_DECAY": args.weight_decay,
            "epochs": args.epochs,
            "backbone_checkpoint_path": args.pretrained_checkpoint,
            "n_beams": 32
        }
        model, train_loader, val_loader = build_foundation_objects(dataset, mmwave_dataset, args, config)
        print(f"foundation_corridor_concat trainable head parameters: {count_parameters(model.beam_head)}")
        head_checkpoint_path = os.path.join(
            args.beam_head_checkpoint_dir,
            f"beam_head_foundation_corridor_concat_{scenario}.pth",
        )

        if args.skip_train and os.path.exists(head_checkpoint_path):
            checkpoint = torch.load(head_checkpoint_path, map_location=device)
            model.beam_head.load_state_dict(checkpoint["beam_head_state_dict"])
            metrics = evaluate_beam_head(model, val_loader, k_list=(1, 3))
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
