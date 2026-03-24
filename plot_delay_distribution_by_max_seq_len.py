import argparse
from pathlib import Path

import deepmimo as dm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dataset.dataloaders import PreTrainMySeqDataLoader


DEFAULT_OUTPUT = Path(__file__).resolve().parent / "delay_distribution_by_max_seq_len.png"
DEFAULT_SUMMARY = Path(__file__).resolve().parent / "delay_distribution_by_max_seq_len_summary.csv"
DEFAULT_LENGTH_OUTPUT = Path(__file__).resolve().parent / "path_length_distribution.png"

METRIC_CONFIG = {
    "delay": {
        "path_column": 0,
        "label": "Delay",
        "unit": "us",
        "csv_prefix": "delay",
        "default_output_name": "delay_distribution_by_path_length.png",
        "default_summary_name": "delay_distribution_by_path_length_summary.csv",
    },
    "power": {
        "path_column": 1,
        "label": "Power",
        "unit": "scaled",
        "csv_prefix": "power",
        "default_output_name": "power_distribution_by_path_length.png",
        "default_summary_name": "power_distribution_by_path_length_summary.csv",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot per-path delay or power distributions grouped by actual path count and summarize the path-length distribution."
    )
    parser.add_argument("scenario", nargs="?", default="city_47_chicago_3p5")
    parser.add_argument("--metric", choices=sorted(METRIC_CONFIG), default="delay")
    parser.add_argument("--split", choices=["train", "val", "all"], default="all")
    parser.add_argument("--sort-by", choices=["power", "delay"], default="power")
    parser.add_argument("--pad-value", type=float, default=0.0)
    parser.add_argument("--bins", type=int, default=80)
    parser.add_argument("--alpha", type=float, default=0.65)
    parser.add_argument("--density", action="store_true")
    parser.add_argument("--log-x", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--length-output", type=Path, default=DEFAULT_LENGTH_OUTPUT)
    parser.add_argument("--summary-csv", type=Path, default=None)
    return parser.parse_args()


def build_datasets(dataset, split, sort_by, pad_value):
    common = dict(split_by="user", sort_by=sort_by, pad_value=pad_value, normalizers=None, apply_normalizers=[])
    if split == "train":
        return [("train", PreTrainMySeqDataLoader(dataset, train=True, **common))]
    if split == "val":
        return [("val", PreTrainMySeqDataLoader(dataset, train=False, **common))]
    return [
        ("train", PreTrainMySeqDataLoader(dataset, train=True, **common)),
        ("val", PreTrainMySeqDataLoader(dataset, train=False, **common)),
    ]


def resolve_output_paths(args):
    metric_cfg = METRIC_CONFIG[args.metric]
    if args.output is None:
        args.output = Path(__file__).resolve().parent / metric_cfg["default_output_name"]
    if args.summary_csv is None:
        args.summary_csv = Path(__file__).resolve().parent / metric_cfg["default_summary_name"]
    return metric_cfg


def collect_metric_groups_by_actual_path_length(datasets, metric_key):
    metric_cfg = METRIC_CONFIG[metric_key]
    grouped = {}
    path_length_records = []

    for split_name, seq_data in datasets:
        for idx in range(len(seq_data)):
            _, paths, path_lengths, _, _, _ = seq_data[idx]
            num_valid_paths = int(round(float(path_lengths.item()) * 25))
            if num_valid_paths <= 0:
                continue

            values = paths[1 : 1 + num_valid_paths, metric_cfg["path_column"]].cpu().numpy()
            grouped.setdefault(num_valid_paths, []).append(values)
            path_length_records.append(
                {
                    "split": split_name,
                    "path_length": num_valid_paths,
                    f"num_{metric_cfg['csv_prefix']}_values": int(values.size),
                }
            )

    grouped = {
        path_length: np.concatenate(values) if values else np.array([], dtype=np.float32)
        for path_length, values in sorted(grouped.items())
    }
    return grouped, pd.DataFrame(path_length_records)


def build_summary(grouped, path_length_df, metric_key):
    metric_cfg = METRIC_CONFIG[metric_key]
    prefix = metric_cfg["csv_prefix"]
    value_count_col = f"num_{prefix}_values"

    rows = []
    for path_length, values in grouped.items():
        if values.size == 0:
            rows.append(
                {
                    "path_length": path_length,
                    "num_samples": 0,
                    value_count_col: 0,
                    f"mean_{prefix}_{metric_cfg['unit']}": np.nan,
                    f"median_{prefix}_{metric_cfg['unit']}": np.nan,
                    f"std_{prefix}_{metric_cfg['unit']}": np.nan,
                    f"p90_{prefix}_{metric_cfg['unit']}": np.nan,
                    f"p95_{prefix}_{metric_cfg['unit']}": np.nan,
                    f"min_{prefix}_{metric_cfg['unit']}": np.nan,
                    f"max_{prefix}_{metric_cfg['unit']}": np.nan,
                }
            )
            continue
        rows.append(
            {
                "path_length": path_length,
                "num_samples": np.nan,
                value_count_col: int(values.size),
                f"mean_{prefix}_{metric_cfg['unit']}": float(np.mean(values)),
                f"median_{prefix}_{metric_cfg['unit']}": float(np.median(values)),
                f"std_{prefix}_{metric_cfg['unit']}": float(np.std(values)),
                f"p90_{prefix}_{metric_cfg['unit']}": float(np.percentile(values, 90)),
                f"p95_{prefix}_{metric_cfg['unit']}": float(np.percentile(values, 95)),
                f"min_{prefix}_{metric_cfg['unit']}": float(np.min(values)),
                f"max_{prefix}_{metric_cfg['unit']}": float(np.max(values)),
            }
        )

    summary_df = pd.DataFrame(rows).sort_values("path_length").reset_index(drop=True)
    if path_length_df.empty:
        return summary_df

    counts_df = (
        path_length_df.groupby("path_length", as_index=False)
        .agg(num_samples=("path_length", "size"), **{value_count_col: (value_count_col, "sum")})
    )
    split_counts = (
        path_length_df.groupby(["path_length", "split"], as_index=False)
        .agg(num_samples=("path_length", "size"), **{value_count_col: (value_count_col, "sum")})
    )

    sample_pivot = split_counts.pivot(index="path_length", columns="split", values="num_samples").fillna(0)
    sample_pivot.columns = [f"num_samples_{col}" for col in sample_pivot.columns]
    sample_pivot = sample_pivot.reset_index()

    value_pivot = split_counts.pivot(index="path_length", columns="split", values=value_count_col).fillna(0)
    value_pivot.columns = [f"{value_count_col}_{col}" for col in value_pivot.columns]
    value_pivot = value_pivot.reset_index()

    summary_df = summary_df.drop(columns=["num_samples", value_count_col])
    summary_df = summary_df.merge(counts_df, on="path_length", how="left")
    summary_df = summary_df.merge(sample_pivot, on="path_length", how="left")
    summary_df = summary_df.merge(value_pivot, on="path_length", how="left")
    return summary_df


def plot_metric_groups(grouped, args, metric_cfg):
    path_lengths = sorted(grouped)
    n = len(path_lengths)
    cols = min(3, max(1, n))
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows), squeeze=False)
    axes_flat = axes.flatten()
    unit_suffix = f" ({metric_cfg['unit']})" if metric_cfg["unit"] else ""

    for ax, path_length in zip(axes_flat, path_lengths):
        values = grouped[path_length]
        if values.size == 0:
            ax.text(0.5, 0.5, f"No valid {args.metric} values", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"path_length={path_length}")
            ax.set_xlabel(f"{metric_cfg['label']}{unit_suffix}")
            ax.set_ylabel("Density" if args.density else "Count")
            continue

        ax.hist(values, bins=args.bins, alpha=args.alpha, density=args.density, color="#1f77b4", edgecolor="white")
        ax.axvline(
            np.median(values),
            color="#d62728",
            linestyle="--",
            linewidth=1.5,
            label=f"median={np.median(values):.3f}{(' ' + metric_cfg['unit']) if metric_cfg['unit'] else ''}",
        )
        ax.axvline(
            np.mean(values),
            color="#2ca02c",
            linestyle=":",
            linewidth=1.5,
            label=f"mean={np.mean(values):.3f}{(' ' + metric_cfg['unit']) if metric_cfg['unit'] else ''}",
        )
        ax.set_title(f"path_length={path_length} | n={values.size}")
        ax.set_xlabel(f"{metric_cfg['label']}{unit_suffix}")
        ax.set_ylabel("Density" if args.density else "Count")
        if args.log_x:
            positive = values[values > 0]
            if positive.size > 0:
                ax.set_xscale("log")
        ax.legend()
        ax.grid(alpha=0.2)

    for ax in axes_flat[n:]:
        ax.axis("off")

    fig.suptitle(
        f"{metric_cfg['label']} distribution by actual path length\n"
        f"scenario={args.scenario}, split={args.split}, sort_by={args.sort_by}"
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def plot_path_length_distribution(path_length_df, args):
    counts = path_length_df["path_length"].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(counts.index.astype(int), counts.values, color="#ff7f0e", edgecolor="white", alpha=0.9)
    ax.set_title(
        f"Distribution of actual path lengths\n"
        f"scenario={args.scenario}, split={args.split}, sort_by={args.sort_by}"
    )
    ax.set_xlabel("Number of valid paths")
    ax.set_ylabel("Number of samples")
    ax.set_xticks(counts.index.astype(int))
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    return fig


def main():
    args = parse_args()
    metric_cfg = resolve_output_paths(args)

    dm.download(args.scenario)
    dataset = dm.load(args.scenario)
    datasets = build_datasets(dataset, args.split, args.sort_by, args.pad_value)
    grouped, path_length_df = collect_metric_groups_by_actual_path_length(datasets, args.metric)
    summary_df = build_summary(grouped, path_length_df, args.metric)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.length_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)

    metric_fig = plot_metric_groups(grouped, args, metric_cfg)
    metric_fig.savefig(args.output, dpi=200, bbox_inches="tight")
    plt.close(metric_fig)

    length_fig = plot_path_length_distribution(path_length_df, args)
    length_fig.savefig(args.length_output, dpi=200, bbox_inches="tight")
    plt.close(length_fig)

    summary_df.to_csv(args.summary_csv, index=False)

    print(summary_df.to_string(index=False))
    print(f"\nSaved {args.metric} figure to {args.output}")
    print(f"Saved path-length figure to {args.length_output}")
    print(f"Saved summary to {args.summary_csv}")


if __name__ == "__main__":
    main()
