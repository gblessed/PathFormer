import argparse
import csv
import math
import os

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_FILES = {
    "k=1": "knn_baseline_results_kth_1.csv",
    "k=3": "knn_baseline_results_kth_3.csv",
    "k=5": "knn_baseline_results_kth_5.csv",
    "k=10": "knn_baseline_results_kth_10.csv",
}

DEFAULT_METRICS = [
    "delay_rmse",
    "power_rmse",
    "phase_rmse",
    "az_rmse",
    "el_rmse",
    "path_length_rmse",
]


def load_csv_rows(path):
    with open(path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def to_float(value):
    if value is None or value == "":
        return math.nan
    return float(value)


def build_metric_table(file_map, metrics):
    per_run = {}
    scenario_order = []
    seen = set()

    for label, path in file_map.items():
        rows = load_csv_rows(path)
        run_rows = {}
        for row in rows:
            scenario = row.get("scenario")
            if not scenario:
                continue
            if scenario not in seen:
                scenario_order.append(scenario)
                seen.add(scenario)
            run_rows[scenario] = {metric: to_float(row.get(metric)) for metric in metrics}
        per_run[label] = run_rows

    return scenario_order, per_run


def plot_metric_grid(scenarios, per_run, metrics, output_path):
    n_metrics = len(metrics)
    ncols = 2
    nrows = math.ceil(n_metrics / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 4.8 * nrows), constrained_layout=True)
    axes = np.array(axes).reshape(-1)

    labels = list(per_run.keys())
    x = np.arange(len(scenarios))
    width = 0.18 if len(labels) >= 4 else 0.22
    colors = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a"]

    for metric_idx, metric in enumerate(metrics):
        ax = axes[metric_idx]
        for run_idx, label in enumerate(labels):
            values = [
                per_run[label].get(scenario, {}).get(metric, math.nan)
                for scenario in scenarios
            ]
            offset = (run_idx - (len(labels) - 1) / 2.0) * width
            ax.bar(
                x + offset,
                values,
                width=width,
                label=label,
                color=colors[run_idx % len(colors)],
                alpha=0.9,
            )

        ax.set_title(metric.replace("_", " "))
        ax.set_xticks(x)
        ax.set_xticklabels(scenarios, rotation=35, ha="right")
        ax.grid(axis="y", linestyle="--", alpha=0.35)

    for ax in axes[n_metrics:]:
        ax.axis("off")

    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=len(labels), frameon=False)
    fig.suptitle("KNN Baseline Comparison Across kth Neighbour Choices", fontsize=16)

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved comparison plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot comparison of KNN kth-neighbour CSV results.")
    parser.add_argument(
        "--output",
        type=str,
        default="knn_comparison_plots/knn_kth_comparison.png",
        help="Output image path.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=DEFAULT_METRICS,
        help="Metric columns to plot.",
    )
    parser.add_argument("--k1", type=str, default=DEFAULT_FILES["k=1"], help="CSV for kth=1.")
    parser.add_argument("--k3", type=str, default=DEFAULT_FILES["k=3"], help="CSV for kth=3.")
    parser.add_argument("--k5", type=str, default=DEFAULT_FILES["k=5"], help="CSV for kth=5.")
    parser.add_argument("--k10", type=str, default=DEFAULT_FILES["k=10"], help="CSV for kth=10.")
    args = parser.parse_args()

    file_map = {
        "k=1": args.k1,
        "k=3": args.k3,
        "k=5": args.k5,
        "k=10": args.k10,
    }
    scenarios, per_run = build_metric_table(file_map, args.metrics)
    plot_metric_grid(scenarios, per_run, args.metrics, args.output)


if __name__ == "__main__":
    main()
