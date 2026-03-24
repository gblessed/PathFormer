import argparse
from pathlib import Path

import deepmimo as dm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from dataset.dataloaders import PreTrainMySeqDataLoader


DEFAULT_OUTPUT = Path(__file__).resolve().parent / "first_timestep_user_kmeans.png"
DEFAULT_SUMMARY = Path(__file__).resolve().parent / "first_timestep_user_kmeans_summary.csv"
DEFAULT_ASSIGNMENTS = Path(__file__).resolve().parent / "first_timestep_user_kmeans_assignments.csv"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cluster users by first-path delay and power, then plot user and feature-space distributions."
    )
    parser.add_argument("scenario", nargs="?", default="city_47_chicago_3p5")
    parser.add_argument("--split", choices=["train", "val", "all"], default="all")
    parser.add_argument("--sort-by", choices=["power", "delay"], default="power")
    parser.add_argument("--pad-value", type=float, default=0.0)
    parser.add_argument("--n-clusters", type=int, default=4)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-init", type=int, default=10)
    parser.add_argument("--no-standardize", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--assignments-csv", type=Path, default=DEFAULT_ASSIGNMENTS)
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


def collect_first_timestep_features(datasets):
    records = []

    for split_name, seq_data in datasets:
        for idx in range(len(seq_data)):
            prompts, paths, path_lengths, _, _, _ = seq_data[idx]
            num_valid_paths = int(round(float(path_lengths.item()) * 25))
            if num_valid_paths < 1:
                continue

            first_path = paths[1].cpu().numpy()
            prompt_np = prompts.cpu().numpy()
            records.append(
                {
                    "split": split_name,
                    "sample_idx": idx,
                    "rx_x": float(prompt_np[3]),
                    "rx_y": float(prompt_np[4]),
                    "path_length": num_valid_paths,
                    "delay_us": float(first_path[0]),
                    "power_scaled": float(first_path[1]),
                }
            )

    df = pd.DataFrame(records)
    if df.empty:
        raise ValueError("No users with at least one valid path were found.")
    return df


def cluster_users(df, args):
    feature_cols = ["delay_us", "power_scaled"]
    features = df[feature_cols].to_numpy(dtype=np.float32)

    scaler = None
    features_for_kmeans = features
    if not args.no_standardize:
        scaler = StandardScaler()
        features_for_kmeans = scaler.fit_transform(features)

    if len(df) < args.n_clusters:
        raise ValueError(
            f"Requested n_clusters={args.n_clusters}, but only {len(df)} users with a first valid path were found."
        )

    kmeans = KMeans(n_clusters=args.n_clusters, random_state=args.random_state, n_init=args.n_init)
    labels = kmeans.fit_predict(features_for_kmeans)

    df = df.copy()
    df["cluster"] = labels.astype(int)

    centers_scaled = kmeans.cluster_centers_
    if scaler is not None:
        centers_original = scaler.inverse_transform(centers_scaled)
    else:
        centers_original = centers_scaled.copy()

    centers_df = pd.DataFrame(centers_original, columns=feature_cols)
    centers_df.insert(0, "cluster", np.arange(args.n_clusters, dtype=int))
    return df, centers_df


def build_cluster_summary(df, centers_df):
    summary = (
        df.groupby("cluster", as_index=False)
        .agg(
            num_users=("cluster", "size"),
            delay_us_mean=("delay_us", "mean"),
            delay_us_std=("delay_us", "std"),
            power_scaled_mean=("power_scaled", "mean"),
            power_scaled_std=("power_scaled", "std"),
            path_length_mean=("path_length", "mean"),
            path_length_std=("path_length", "std"),
            rx_x_mean=("rx_x", "mean"),
            rx_x_std=("rx_x", "std"),
            rx_y_mean=("rx_y", "mean"),
            rx_y_std=("rx_y", "std"),
        )
        .sort_values("cluster")
        .reset_index(drop=True)
    )
    summary = summary.merge(
        centers_df.rename(columns={"delay_us": "center_delay_us", "power_scaled": "center_power_scaled"}),
        on="cluster",
        how="left",
    )
    return summary


def plot_clusters(df, centers_df, args):
    clusters = np.sort(df["cluster"].unique())
    cmap = plt.get_cmap("tab10", len(clusters))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax_users, ax_features = axes

    for cluster in clusters:
        cluster_df = df[df["cluster"] == cluster]
        color = cmap(int(cluster))
        ax_users.scatter(
            cluster_df["rx_x"],
            cluster_df["rx_y"],
            s=18,
            alpha=0.75,
            color=color,
            label=f"cluster {cluster} (n={len(cluster_df)})",
        )
        ax_features.scatter(
            cluster_df["delay_us"],
            cluster_df["power_scaled"],
            s=20,
            alpha=0.75,
            color=color,
            label=f"cluster {cluster}",
        )

    ax_users.set_title("User distribution colored by cluster")
    ax_users.set_xlabel("RX x")
    ax_users.set_ylabel("RX y")
    ax_users.grid(alpha=0.2)
    ax_users.legend(loc="best", fontsize=8)

    ax_features.scatter(
        centers_df["delay_us"],
        centers_df["power_scaled"],
        s=180,
        c="black",
        marker="X",
        linewidths=1.0,
        edgecolors="white",
        label="cluster centers",
    )
    ax_features.set_title("First-path feature clusters")
    ax_features.set_xlabel("Delay (us)")
    ax_features.set_ylabel("Power (scaled)")
    ax_features.grid(alpha=0.2)
    ax_features.legend(loc="best", fontsize=8)

    fig.suptitle(
        f"KMeans clustering from first path delay/power\n"
        f"scenario={args.scenario}, split={args.split}, sort_by={args.sort_by}, n_clusters={args.n_clusters}"
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def main():
    args = parse_args()

    dm.download(args.scenario)
    dataset = dm.load(args.scenario)
    datasets = build_datasets(dataset, args.split, args.sort_by, args.pad_value)
    df = collect_first_timestep_features(datasets)
    clustered_df, centers_df = cluster_users(df, args)
    summary_df = build_cluster_summary(clustered_df, centers_df)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    args.assignments_csv.parent.mkdir(parents=True, exist_ok=True)

    fig = plot_clusters(clustered_df, centers_df, args)
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    plt.close(fig)

    clustered_df.to_csv(args.assignments_csv, index=False)
    summary_df.to_csv(args.summary_csv, index=False)

    print(summary_df.to_string(index=False))
    print(f"\nSaved figure to {args.output}")
    print(f"Saved cluster assignments to {args.assignments_csv}")
    print(f"Saved cluster summary to {args.summary_csv}")


if __name__ == "__main__":
    main()
