import argparse
import ast
import os
from pathlib import Path

import deepmimo as dm
import numpy as np
import pandas as pd
import torch

from dataset.dataloaders import PreTrainMySeqDataLoader


ROOT = Path(__file__).resolve().parent
SOURCE_FILE = ROOT / "playground_std_fix1.py"
DEFAULT_CSV = str(ROOT / "playground_std_fix1_cluster_sweep_results.csv")
DEFAULT_CKPT_DIR = str(ROOT / "checkpoints_std_fix1_quick_sweep")


def parse_args():
    parser = argparse.ArgumentParser(description="Quick n_clusters sweep for playground_std_fix1.")
    parser.add_argument("scenario", nargs="?", default="city_47_chicago_3p5")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-seq-len", type=int, default=4)
    parser.add_argument("--cluster-values", type=int, nargs="+", default=[5, 10, 20, 30, 40, 50])
    parser.add_argument("--csv-log-file", default=DEFAULT_CSV)
    parser.add_argument("--checkpoint-dir", default=DEFAULT_CKPT_DIR)
    return parser.parse_args()


def load_fix1_namespace():
    source = SOURCE_FILE.read_text()
    tree = ast.parse(source, filename=str(SOURCE_FILE))
    kept_nodes = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef)):
            kept_nodes.append(node)
    module = ast.Module(body=kept_nodes, type_ignores=[])
    namespace = {"__file__": str(SOURCE_FILE), "__name__": "playground_std_fix1_defs"}
    exec(compile(module, filename=str(SOURCE_FILE), mode="exec"), namespace, namespace)
    return namespace


class LimitedSequenceDataset(torch.utils.data.Dataset):
    """Wrap a base dataset and keep only SOS + max_seq_len real steps."""

    def __init__(self, base_dataset, max_seq_len):
        self.base_dataset = base_dataset
        self.max_seq_len = max_seq_len
        self.pad_value = base_dataset.pad_value
        self.sort_by = base_dataset.sort_by
        self.dataset_filtered = base_dataset.dataset_filtered
        self.mins = getattr(base_dataset, "mins", None)
        self.maxs = getattr(base_dataset, "maxs", None)
        self.normalizers = getattr(base_dataset, "normalizers", None)
        self.apply_normalizers = getattr(base_dataset, "apply_normalizers", None)

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        prompt, paths, num_paths, interactions, environment, environment_material_props = self.base_dataset[idx]
        keep_len = min(paths.size(0), self.max_seq_len + 1)
        paths = paths[:keep_len]
        interactions = interactions[:keep_len]
        valid_paths = max(0, keep_len - 1)
        num_paths = torch.tensor([valid_paths], dtype=torch.float32) / 25.0
        return prompt, paths, num_paths, interactions, environment, environment_material_props

    def collate_fn(self, batch):
        return self.base_dataset.collate_fn(batch)


def make_config(args, scenario, n_clusters):
    return {
        "BATCH_SIZE": args.batch_size,
        "PAD_VALUE": 0,
        "USE_WANDB": False,
        "LR": args.lr,
        "epochs": args.epochs,
        "interaction_weight": 0.01,
        "experiment": f"noise_std_fix1_quick_sweep_{scenario}_clusters_{n_clusters}_seqlen_{args.max_seq_len}",
        "hidden_dim": 512,
        "n_layers": 8,
        "n_heads": 8,
        "use_cluster_conditioning": True,
        "n_clusters": n_clusters,
        "max_path_len_clusters": args.max_seq_len,
        "cluster_features": ["delay", "power"],
        "delay_only_loss": False,
        "TARGET_NOISE_PROB": 0.2,
        "TARGET_NOISE_PARAMS": None,
        "use_cluster_mlp_head": False,
        "pretrained_checkpoint": "checkpoints2/noise_enc_direct_city_47_chicago_3p5_interacaction_all_inter_str_dec_all_repeat_best_model_checkpoint.pth",
    }


def create_datasets(dataset, config, max_seq_len):
    train_base = PreTrainMySeqDataLoader(
        dataset,
        train=True,
        split_by="user",
        sort_by="power",
        pad_value=config["PAD_VALUE"],
        normalizers=None,
        apply_normalizers=[],
    )
    val_base = PreTrainMySeqDataLoader(
        dataset,
        train=False,
        split_by="user",
        sort_by="power",
        pad_value=config["PAD_VALUE"],
        normalizers=None,
        apply_normalizers=[],
    )
    train_data = LimitedSequenceDataset(train_base, max_seq_len=max_seq_len)
    val_data = LimitedSequenceDataset(val_base, max_seq_len=max_seq_len)
    return train_data, val_data


def load_best_val_loss(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    best_val_loss = checkpoint.get("best_val_loss")
    if isinstance(best_val_loss, torch.Tensor):
        best_val_loss = best_val_loss.item()
    return float(best_val_loss)


def run_single_experiment(args, scenario, dataset, namespace, n_clusters):
    config = make_config(args, scenario, n_clusters)
    train_data, val_data = create_datasets(dataset, config, max_seq_len=args.max_seq_len)

    model_cls = namespace["PathDecoderClusterEncoderAttentionFix1"]
    compute_clusters = namespace["compute_feature_kmeans_cluster_stats"]
    precompute_lookup = namespace["precompute_train_cluster_center_std_sequences"]
    count_parameters = namespace["count_parameters"]
    train_with_interactions = namespace["train_with_interactions"]

    cluster_lookup_data = None
    device = namespace["device"]
    model = model_cls(
        prompt_dim=6,
        hidden_dim=config["hidden_dim"],
        n_layers=config["n_layers"],
        n_heads=config["n_heads"],
        pad_value=config["PAD_VALUE"],
        cluster_feature_dim=2 * len(config["cluster_features"]),
    ).to(device)

    centers, stds = compute_clusters(
        train_data,
        feature_keys=config["cluster_features"],
        max_path_len=config["max_path_len_clusters"],
        n_clusters=n_clusters,
    )
    train_rx_pos, train_cluster_center_std, train_valid_len = precompute_lookup(
        train_data,
        cluster_centers=centers,
        cluster_stds=stds,
        feature_keys=config["cluster_features"],
        max_path_len=config["max_path_len_clusters"],
    )
    cluster_lookup_data = (train_rx_pos, train_cluster_center_std, train_valid_len)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["LR"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=max(1, min(25, config["epochs"])),
        T_mult=1,
        eta_min=1e-8,
    )

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.checkpoint_dir, f"{config['experiment']}_best_model_checkpoint.pth")

    train_loader = torch.utils.data.DataLoader(
        dataset=train_data,
        batch_size=config["BATCH_SIZE"],
        shuffle=True,
        collate_fn=train_data.collate_fn,
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=val_data,
        batch_size=config["BATCH_SIZE"],
        shuffle=False,
        collate_fn=val_data.collate_fn,
    )

    namespace["optimizer"] = optimizer
    namespace["scheduler"] = scheduler
    namespace["checkpoint_path"] = checkpoint_path

    print(f"Prepared cluster lookup for n_clusters={n_clusters}")
    print(f"Total trainable parameters: {count_parameters(model)}")
    train_with_interactions(
        model,
        train_loader,
        val_loader,
        config,
        train_data,
        cluster_lookup_data=cluster_lookup_data,
    )

    best_val_loss = load_best_val_loss(checkpoint_path)

    del model
    del optimizer
    del scheduler
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "scenario": scenario,
        "n_clusters": n_clusters,
        "epochs": config["epochs"],
        "max_seq_len": args.max_seq_len,
        "best_val_loss": best_val_loss,
        "checkpoint_path": checkpoint_path,
    }


def main():
    args = parse_args()
    namespace = load_fix1_namespace()
    namespace["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dm.download(args.scenario)
    dataset = dm.load(args.scenario)

    print(f"######### Quick cluster sweep on {args.scenario} #########")
    print(f"Using max_seq_len={args.max_seq_len}, epochs={args.epochs}, cluster_values={args.cluster_values}")

    results = []
    for n_clusters in args.cluster_values:
        print()
        print(f"===== Running n_clusters={n_clusters} =====")
        result = run_single_experiment(args, args.scenario, dataset, namespace, n_clusters)
        results.append(result)
        print(f"Completed n_clusters={n_clusters}: best_val_loss={result['best_val_loss']:.6f}")

    results_df = pd.DataFrame(results).sort_values("best_val_loss").reset_index(drop=True)
    print("\nSweep summary (sorted by best val loss):")
    print(results_df[["n_clusters", "best_val_loss", "max_seq_len", "epochs"]].to_string(index=False))
    results_df.to_csv(args.csv_log_file, index=False)
    print(f"\nSaved sweep summary to {args.csv_log_file}")


if __name__ == "__main__":
    main()
