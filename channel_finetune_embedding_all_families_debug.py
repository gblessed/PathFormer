import argparse
import os
import warnings
from typing import Literal

import deepmimo as dm
import generator_things.consts as c
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.dataloaders import PreTrainMySeqDataLoader
from models import PathDecoder
from multiscenario_direct_training_first_step_residual import (
    FirstStepResidualDataset,
    FirstStepResidualPathDecoder,
    build_first_step_assignments,
    load_best_checkpoint as load_residual_checkpoint,
    resolve_scenarios,
)
from multiscenario_direct_training_first_step_residual_corridor import (
    build_first_step_assignments_with_corridor,
)
from scene_feature_utils import SceneFeatureBank
from utils.utils import (
    ChannelParameters,
    count_parameters,
    load_best_checkpoint as load_direct_checkpoint,
)


warnings.filterwarnings("ignore", category=UserWarning)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_CSV_LOG = "/home/blessedg/Pathformer/logs/channel_finetune_embedding_all_families.csv"


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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune channel heads from frozen path-model embeddings for direct, first-step residual, and corridor families."
    )
    parser.add_argument("scenarios", nargs="*", help="Optional scenario names to run.")
    parser.add_argument("--scenario", dest="scenario_flag", action="append")
    parser.add_argument("--scenario-file", type=str)
    parser.add_argument("--shard-index", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=None)
    parser.add_argument("--csv-log-file", type=str, default=DEFAULT_CSV_LOG)
    parser.add_argument("--checkpoint-root-direct", type=str, default="/home/blessedg/Pathformer/base_no_env")
    parser.add_argument("--checkpoint-root-residual", type=str, default="/home/blessedg/Pathformer/checkpoints_first_step_residual")
    parser.add_argument("--checkpoint-root-corridor", type=str, default="/home/blessedg/Pathformer/checkpoints_first_step_residual_corridor")
    parser.add_argument("--channel-checkpoint-dir", type=str, default="/home/blessedg/Pathformer/checkpoints_channel_embedding_all_families")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--embed-pool", choices=["first", "last", "mean"], default="mean")
    parser.add_argument("--max-generated-steps", type=int, default=25)
    parser.add_argument("--n-clusters", type=int, default=25)
    parser.add_argument("--nearest-k", type=int, default=5)
    parser.add_argument("--corridor-k", type=int, default=5)
    parser.add_argument("--corridor-bins", type=int, default=8)
    parser.add_argument("--use-material-features", action="store_true")
    parser.add_argument("--no-material-features", dest="use_material_features", action="store_false")
    parser.set_defaults(use_material_features=True)
    return parser.parse_args()


def channel_numel():
    params = ChannelParameters()
    n_sc = np.array(params.ofdm[c.PARAMSET_OFDM_SC_SAMP]).size
    m_tx = int(np.prod(params.bs_antenna[c.PARAMSET_ANT_SHAPE]))
    m_rx = int(np.prod(params.ue_antenna[c.PARAMSET_ANT_SHAPE]))
    return 2 * m_rx * m_tx * n_sc


def normalize_dataset_layout(dataset, channels):
    if hasattr(dataset, "n_ue") and isinstance(dataset.n_ue, int):
        return [dataset], [channels]
    return list(dataset), list(channels)


def normalize_channel_tensor(channels):
    channels = torch.as_tensor(np.asarray(channels)).to(torch.complex64)
    if channels.dim() >= 1 and channels.shape[-1] == 1:
        channels = channels.squeeze(-1)
    if channels.dim() == 3:
        channels = channels.unsqueeze(1).unsqueeze(1)
    elif channels.dim() == 4:
        if channels.shape[1] == 1:
            channels = channels.unsqueeze(2)
        else:
            channels = channels.unsqueeze(1)
    return channels


def build_channel_targets(dataset, channels, train, seed=42, train_ratio=0.8):
    dataset_list, channel_list = normalize_dataset_layout(dataset, channels)
    targets = []

    for tx in range(len(dataset_list)):
        data_tx = dataset_list[tx]
        channel_tx = normalize_channel_tensor(channel_list[tx])
        n_ue = data_tx.n_ue

        indices = np.arange(n_ue)
        np.random.seed(seed + tx)
        np.random.shuffle(indices)

        split_idx = int(train_ratio * len(indices))
        if train:
            indices = indices[:split_idx]
        else:
            indices = indices[split_idx:]

        use_indices = data_tx.los != -1
        indices = [i for i in indices if use_indices[i]]
        if indices:
            targets.append(channel_tx[indices])

    if not targets:
        return torch.empty(0, 1, 1, 0, 0, dtype=torch.complex64)
    return torch.cat(targets, dim=0)


class ChannelTargetDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, channel_targets):
        self.base_dataset = base_dataset
        self.channel_targets = channel_targets

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        return self.base_dataset[idx], self.channel_targets[idx]

    def collate_fn(self, batch):
        base_items = [item[0] for item in batch]
        gt_channels = torch.stack([item[1] for item in batch], dim=0).to(torch.complex64)
        base_batch = self.base_dataset.collate_fn(base_items)
        return (*base_batch, gt_channels)


class ChannelHeadFromPathEmbeddings(nn.Module):
    def __init__(
        self,
        path_model: nn.Module,
        model_family: str,
        hidden_dim: int,
        out_numel: int,
        pool_mode: Literal["first", "last", "mean"] = "mean",
        max_generated_steps: int = 25,
    ):
        super().__init__()
        self.path_model = path_model
        self.model_family = model_family
        self.hidden_dim = hidden_dim
        self.out_numel = out_numel
        self.pool_mode = pool_mode
        self.max_generated_steps = max_generated_steps
        for param in self.path_model.parameters():
            param.requires_grad = False
        self.channel_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, out_numel),
        )

    def _forward_hidden(self, prompts, paths, interactions):
        if self.model_family == "direct":
            return self.path_model.forward_hidden(prompts, paths, interactions)[0]
        return self.path_model.backbone.forward_hidden(prompts, paths, interactions)[0]

    def _predict_next(self, prompts, paths, interactions, first_step_baseline):
        if self.model_family == "direct":
            outputs = self.path_model(prompts, paths, interactions)
        else:
            outputs = self.path_model(prompts, paths, interactions, first_step_baseline)

        delay = outputs[0][:, -1]
        power = outputs[1][:, -1]
        phase = outputs[4][:, -1]
        aoa_az = outputs[7][:, -1]
        aoa_el = outputs[10][:, -1]
        aod_az = outputs[13][:, -1]
        aod_el = outputs[16][:, -1]
        interaction = torch.sigmoid(outputs[-1][:, -1, :])
        next_path = torch.stack([delay, power, phase, aoa_az, aoa_el, aod_az, aod_el], dim=-1)
        return next_path, interaction

    def _rollout_hidden(self, prompts, steps, first_step_baseline=None):
        B = prompts.size(0)
        cur = torch.zeros(B, 1, 7, device=prompts.device, dtype=prompts.dtype)
        interactions = -1 * torch.ones(B, 1, 4, device=prompts.device, dtype=prompts.dtype)
        hidden_steps = []

        for _ in range(steps):
            h_paths = self._forward_hidden(prompts, cur, interactions)
            hidden_steps.append(h_paths[:, -1, :])
            next_path, next_interaction = self._predict_next(
                prompts,
                cur,
                interactions,
                first_step_baseline,
            )
            cur = torch.cat([cur, next_path.unsqueeze(1)], dim=1)
            interactions = torch.cat([interactions, next_interaction.unsqueeze(1)], dim=1)

        return torch.stack(hidden_steps, dim=1)

    def _pool(self, h_paths):
        if self.pool_mode == "first":
            return h_paths[:, 0, :]
        if self.pool_mode == "last":
            return h_paths[:, -1, :]
        return h_paths.mean(dim=1)

    def forward(self, prompts, paths_in=None, interactions_in=None, first_step_baseline=None):
        steps = paths_in.size(1) if paths_in is not None else self.max_generated_steps
        h_paths = self._rollout_hidden(prompts, steps, first_step_baseline)
        pooled = self._pool(h_paths)
        flat = self.channel_head(pooled)
        half = self.out_numel // 2
        real = flat[:, :half]
        imag = flat[:, half:]

        params = ChannelParameters()
        n_sc = np.array(params.ofdm[c.PARAMSET_OFDM_SC_SAMP]).size
        m_tx = int(np.prod(params.bs_antenna[c.PARAMSET_ANT_SHAPE]))
        m_rx = int(np.prod(params.ue_antenna[c.PARAMSET_ANT_SHAPE]))
        real = real.view(prompts.size(0), m_rx, m_tx, n_sc)
        imag = imag.view(prompts.size(0), m_rx, m_tx, n_sc)
        return torch.complex(real, imag).unsqueeze(1)


def checkpoint_specs_for_scenario(scenario, args):
    return [
        (
            "direct",
            os.path.join(args.checkpoint_root_direct, f"multiscenario_direct_{scenario}_best_model_checkpoint.pth"),
        ),
        (
            "first_step_residual",
            os.path.join(args.checkpoint_root_residual, f"first_step_residual_{scenario}_best_model_checkpoint.pth"),
        ),
        (
            "first_step_residual_corridor",
            os.path.join(args.checkpoint_root_corridor, f"first_step_residual_corridor_{scenario}_best_model_checkpoint.pth"),
        ),
    ]


def build_direct_loaders(dataset, args, pad_value):
    train_data = PreTrainMySeqDataLoader(dataset, train=True, split_by="user", sort_by="power", normalizers=None, apply_normalizers=[], pad_value=pad_value, include_aod=True)
    val_data = PreTrainMySeqDataLoader(dataset, train=False, split_by="user", sort_by="power", normalizers=None, apply_normalizers=[], pad_value=pad_value, include_aod=True)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=train_data.collate_fn)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, collate_fn=val_data.collate_fn)
    return train_data, val_data, train_loader, val_loader, None


def build_residual_loaders(dataset, args, pad_value):
    base_train = PreTrainMySeqDataLoader(dataset, train=True, split_by="user", sort_by="power", normalizers=None, apply_normalizers=[], pad_value=pad_value, include_aod=True)
    base_val = PreTrainMySeqDataLoader(dataset, train=False, split_by="user", sort_by="power", normalizers=None, apply_normalizers=[], pad_value=pad_value, include_aod=True)
    train_aug, train_baselines, val_aug, val_baselines = build_first_step_assignments(base_train, base_val, n_clusters=args.n_clusters)
    train_data = FirstStepResidualDataset(base_train, train_aug, train_baselines)
    val_data = FirstStepResidualDataset(base_val, val_aug, val_baselines)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=train_data.collate_fn)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, collate_fn=val_data.collate_fn)
    return train_data, val_data, train_loader, val_loader, int(train_aug[0].numel())


def build_corridor_loaders(dataset, args, pad_value):
    base_train = PreTrainMySeqDataLoader(dataset, train=True, split_by="user", sort_by="power", normalizers=None, apply_normalizers=[], pad_value=pad_value, include_aod=True)
    base_val = PreTrainMySeqDataLoader(dataset, train=False, split_by="user", sort_by="power", normalizers=None, apply_normalizers=[], pad_value=pad_value, include_aod=True)
    scene_bank = SceneFeatureBank.from_dataset(dataset, use_material_features=args.use_material_features)
    train_aug, train_baselines, val_aug, val_baselines = build_first_step_assignments_with_corridor(
        base_train,
        base_val,
        scene_bank,
        n_clusters=args.n_clusters,
        nearest_k=args.nearest_k,
        corridor_k=args.corridor_k,
        corridor_bins=args.corridor_bins,
    )
    train_data = FirstStepResidualDataset(base_train, train_aug, train_baselines)
    val_data = FirstStepResidualDataset(base_val, val_aug, val_baselines)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=train_data.collate_fn)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, collate_fn=val_data.collate_fn)
    return train_data, val_data, train_loader, val_loader, int(train_aug[0].numel())


def make_path_model(model_family, prompt_dim, checkpoint_path):
    if model_family == "direct":
        model = PathDecoder(hidden_dim=512, n_layers=8, n_heads=8, include_aod=True).to(device)
        load_direct_checkpoint(model, checkpoint_path=checkpoint_path)
        return model

    model = FirstStepResidualPathDecoder(
        prompt_dim=prompt_dim,
        hidden_dim=512,
        n_layers=8,
        n_heads=8,
    ).to(device)
    load_residual_checkpoint(model, checkpoint_path)
    return model


def unpack_batch(batch, model_family):
    if model_family == "direct":
        prompts, paths, path_lengths, interactions, env, env_prop, path_padding_mask, gt_channels = batch
        first_step_baseline = None
    else:
        prompts, paths, path_lengths, interactions, env, env_prop, path_padding_mask, first_step_baseline, gt_channels = batch
        first_step_baseline = first_step_baseline.to(device)

    prompts = prompts.to(device)
    paths = paths.to(device)
    interactions = interactions.to(device)
    path_padding_mask = path_padding_mask.to(device)
    gt_channels = gt_channels.to(device)
    return prompts, paths, interactions, path_padding_mask, first_step_baseline, gt_channels


def channel_loss_for_batch(model, batch, model_family, scale=1e6):
    prompts, paths, interactions, path_padding_mask, first_step_baseline, gt_ch = unpack_batch(batch, model_family)
    paths_in = paths[:, :-1, :]
    interactions_in = interactions[:, :-1, :]
    pred_ch = model(prompts, paths_in, interactions_in, first_step_baseline)
    print(f"pred_s: {pred_ch[0,0,0,0]}\n GT_s:{gt_ch[0,0,0,0]}")
    gt_s = gt_ch * scale
    pred_s = pred_ch * scale
    mse = ((gt_s.real - pred_s.real) ** 2 + (gt_s.imag - pred_s.imag) ** 2).mean()
    gt_norm_sq = (gt_s.real ** 2 + gt_s.imag ** 2).mean().clamp(min=1e-6)
    return (mse / (gt_norm_sq + 1e-10)).clamp(max=100.0)


def evaluate_channel_head(model, val_loader, model_family):
    model.eval()
    scale = 1e6
    nmses, nmse_logs, nmse_dbs, scores = [], [], [], []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Eval channel head [{model_family}]", leave=False):
            prompts, paths, interactions, path_padding_mask, first_step_baseline, gt_ch = unpack_batch(batch, model_family)
            paths_in = paths[:, :-1, :]
            interactions_in = interactions[:, :-1, :]
            pred_ch = model(prompts, paths_in, interactions_in, first_step_baseline)
            gt_s = gt_ch * scale
            pred_s = pred_ch * scale
            mse = ((gt_s.real - pred_s.real) ** 2 + (gt_s.imag - pred_s.imag) ** 2).mean(dim=(1, 2, 3, 4))
            gt_norm_sq = (gt_s.real ** 2 + gt_s.imag ** 2).mean(dim=(1, 2, 3, 4)).clamp(min=1e-6)
            nmse = (mse / (gt_norm_sq + 1e-10)).detach().cpu().numpy()
            nmse_log = np.log10(nmse + 1e-10)
            nmse_db = 10.0 * nmse_log
            score = 1.0 - ((nmse_db - (-20.0)) / (0.0 - (-20.0)))
            score = np.clip(score, 0.0, 1.0)
            nmse_logs.extend(nmse_log.tolist())
            nmse_dbs.extend(nmse_db.tolist())
            nmses.extend(nmse.tolist())
            scores.extend(score.tolist())

    return {
        "ch_nmse": float(np.mean(nmses)) if nmses else 0.0,
        "ch_nmse_log": float(np.mean(nmse_logs)) if nmse_logs else 0.0,
        "ch_nmse_log_std": float(np.std(nmse_logs)) if nmse_logs else 0.0,
        "avg_ch_nmse_dB": float(np.mean(nmse_dbs)) if nmse_dbs else 0.0,
        "avg_ch_nmse_dB_std": float(np.std(nmse_dbs)) if nmse_dbs else 0.0,
        "avg_ch_score": float(np.mean(scores)) if scores else 0.0,
        "avg_ch_score_std": float(np.std(scores)) if scores else 0.0,
        "n_eval": len(scores),
    }


def save_channel_head_checkpoint(model, path, epoch, best_val_loss, args, model_family, scenario):
    torch.save(
        {
            "epoch": epoch,
            "channel_head_state_dict": model.channel_head.state_dict(),
            "best_val_ch_loss": float(best_val_loss),
            "model_family": model_family,
            "scenario": scenario,
            "embed_pool": args.embed_pool,
            "max_generated_steps": args.max_generated_steps,
            "channel_numel": model.out_numel,
        },
        path,
    )


def load_channel_head_checkpoint(model, path):
    checkpoint = torch.load(path, map_location=device)
    model.channel_head.load_state_dict(checkpoint["channel_head_state_dict"])
    return checkpoint


def train_channel_head(model, train_loader, val_loader, model_family, args, checkpoint_path, scenario):
    optimizer = torch.optim.AdamW(model.channel_head.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=25,
        T_mult=1,
        eta_min=1e-8,
    )
    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        model.train()
        model.path_model.eval()
        train_losses = []
        for batch in tqdm(train_loader, desc=f"{scenario} [{model_family}] epoch {epoch} train", leave=False):
            loss = channel_loss_for_batch(model, batch, model_family)
            train_losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.channel_head.parameters(), args.grad_clip_norm)
            optimizer.step()

        scheduler.step()

        model.eval()
        model.path_model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"{scenario} [{model_family}] epoch {epoch} val", leave=False):
                val_losses.append(channel_loss_for_batch(model, batch, model_family).item())
        val_loss = float(np.mean(val_losses)) if val_losses else float("inf")
        train_loss = float(np.mean(train_losses)) if train_losses else float("inf")
        print(
            f"{scenario} [{model_family}] epoch {epoch:03d} "
            f"train_ch_loss={train_loss:.4f} val_ch_loss={val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_channel_head_checkpoint(
                model,
                checkpoint_path,
                epoch,
                best_val_loss,
                args,
                model_family,
                scenario,
            )
            print(f"  ✓ Saved best channel head only: {checkpoint_path}")

    return best_val_loss


def build_loaders_for_family(dataset, model_family, args, pad_value):
    if model_family == "direct":
        return build_direct_loaders(dataset, args, pad_value)
    if model_family == "first_step_residual":
        return build_residual_loaders(dataset, args, pad_value)
    if model_family == "first_step_residual_corridor":
        return build_corridor_loaders(dataset, args, pad_value)
    raise ValueError(f"Unknown model family: {model_family}")


def attach_channel_targets(train_data, val_data, args, channels):
    train_source = train_data.dataset if hasattr(train_data, "dataset") else train_data.base_dataset.dataset
    val_source = val_data.dataset if hasattr(val_data, "dataset") else val_data.base_dataset.dataset
    train_targets = build_channel_targets(train_source, channels, train=True)
    val_targets = build_channel_targets(val_source, channels, train=False)
    train_dataset = ChannelTargetDataset(train_data, train_targets)
    val_dataset = ChannelTargetDataset(val_data, val_targets)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=train_dataset.collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=val_dataset.collate_fn)
    return train_loader, val_loader


def run_scenario_family(dataset, channel_targets, scenario, model_family, path_checkpoint, args):
    if not os.path.exists(path_checkpoint):
        print(f"Skipping {scenario} [{model_family}] missing path checkpoint: {path_checkpoint}")
        return None

    pad_value = 0
    train_data, val_data, _, _, prompt_dim = build_loaders_for_family(dataset, model_family, args, pad_value)
    train_loader, val_loader = attach_channel_targets(train_data, val_data, args, channel_targets)
    if model_family == "direct":
        prompt_dim = None

    path_model = make_path_model(model_family, prompt_dim, path_checkpoint)
    model = ChannelHeadFromPathEmbeddings(
        path_model=path_model,
        model_family=model_family,
        hidden_dim=512,
        out_numel=channel_numel(),
        pool_mode=args.embed_pool,
        max_generated_steps=args.max_generated_steps,
    ).to(device)
    print(f"{scenario} [{model_family}] trainable channel parameters: {count_parameters(model)}")

    os.makedirs(args.channel_checkpoint_dir, exist_ok=True)
    channel_checkpoint = os.path.join(
        args.channel_checkpoint_dir,
        f"{model_family}_{scenario}_channel_head_best.pth",
    )

    if args.eval_only or args.skip_train:
        load_channel_head_checkpoint(model, channel_checkpoint)
    else:
        train_channel_head(model, train_loader, val_loader, model_family, args, channel_checkpoint, scenario)
        load_channel_head_checkpoint(model, channel_checkpoint)

    metrics = evaluate_channel_head(model, val_loader, model_family)
    metrics.update(
        {
            "scenario": scenario,
            "model_family": model_family,
            "path_checkpoint": path_checkpoint,
            "channel_checkpoint": channel_checkpoint,
            "embed_pool": args.embed_pool,
            "max_generated_steps": args.max_generated_steps,
            "eval_only": args.eval_only,
        }
    )
    print(
        f"{scenario} [{model_family}] | ch_nmse_dB={metrics['avg_ch_nmse_dB']:.4f}, "
        f"score={metrics['avg_ch_score']:.4f}"
    )
    return metrics


def run_scenario(scenario, args):
    dataset = dm.load(scenario)
    channel_dataset = dm.load(scenario)
    channel_dataset.compute_channels(ChannelParameters())
    channel_targets = channel_dataset.channels
    rows = []
    for model_family, path_checkpoint in checkpoint_specs_for_scenario(scenario, args):
        row = run_scenario_family(dataset, channel_targets, scenario, model_family, path_checkpoint, args)
        if row is not None:
            rows.append(row)
            pd.DataFrame([row]).to_csv(
                args.csv_log_file,
                mode="a",
                index=False,
                header=not os.path.exists(args.csv_log_file),
            )
    return rows


def main():
    args = parse_args()
    if not args.scenarios and not args.scenario_flag and not args.scenario_file:
        scenarios = default_scenarios()
    else:
        scenarios = resolve_scenarios(args)

    if args.shard_index is not None or args.num_shards is not None:
        if args.shard_index is None or args.num_shards is None:
            raise ValueError("Provide both --shard-index and --num-shards.")
        scenarios = [s for idx, s in enumerate(scenarios) if idx % args.num_shards == args.shard_index]

    os.makedirs(os.path.dirname(args.csv_log_file) or ".", exist_ok=True)
    print(f"Running channel-head finetuning for {len(scenarios)} scenario(s): {scenarios}")

    for scenario in scenarios:
        # try:
            run_scenario(scenario, args)
        # except Exception as exc:
        #     print(f"✗ Failed scenario {scenario}: {exc}")
        #     pd.DataFrame([{"scenario": scenario, "error": str(exc)}]).to_csv(
        #         args.csv_log_file,
        #         mode="a",
        #         index=False,
        #         header=not os.path.exists(args.csv_log_file),
        #     )


if __name__ == "__main__":
    main()
