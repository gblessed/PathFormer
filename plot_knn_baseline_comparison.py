"""
Compare KNN baseline results: average_3 vs kth_1.
Plots grouped bar charts per metric and saves figures.
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_AVG = os.path.join(BASE_DIR, "knn_baseline_results_average_3.csv")
CSV_KTH = os.path.join(BASE_DIR, "knn_baseline_results_kth_1.csv")
OUT_DIR = os.path.join(BASE_DIR, "knn_comparison_plots")
os.makedirs(OUT_DIR, exist_ok=True)

# Metric columns to compare (exclude scenario and meta columns)
METRIC_COLS = [
    "delay_rmse", "power_rmse", "phase_rmse", "az_rmse", "el_rmse",
    "delay_mae", "power_mae", "phase_mae", "az_mae", "el_mae",
    "path_length_rmse", "path_length_mae",
]
METRIC_LABELS = [
    "Delay RMSE", "Power RMSE", "Phase RMSE", "Az RMSE", "El RMSE",
    "Delay MAE", "Power MAE", "Phase MAE", "Az MAE", "El MAE",
    "Path length RMSE", "Path length MAE",
]

def load_and_merge():
    df_avg = pd.read_csv(CSV_AVG)
    df_kth = pd.read_csv(CSV_KTH)
    # Drop rows with missing numeric metrics (e.g. error rows)
    df_avg = df_avg.dropna(subset=["delay_rmse"])
    df_kth = df_kth.dropna(subset=["delay_rmse"])
    return df_avg, df_kth

def plot_all_metrics_one_figure(df_avg, df_kth):
    """One figure with subplots for each metric."""
    n_metrics = len(METRIC_COLS)
    n_cols = 4
    n_rows = (n_metrics + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    axes = axes.flatten()

    scenarios = df_avg["scenario"].tolist()
    x = np.arange(len(scenarios))
    w = 0.35

    for i, (col, label) in enumerate(zip(METRIC_COLS, METRIC_LABELS)):
        ax = axes[i]
        v_avg = df_avg[col].values
        v_kth = df_kth[col].values
        ax.bar(x - w / 2, v_avg, w, label="k=3 average", color="steelblue")
        ax.bar(x + w / 2, v_kth, w, label="k=1 kth", color="coral", alpha=0.9)
        ax.set_ylabel(label)
        ax.set_xticks(x)
        ax.set_xticklabels([s.replace("city_", "").replace("_3p5", "").replace("_s", "") for s in scenarios], rotation=45, ha="right")
        ax.legend(loc="upper right", fontsize=7)
        ax.grid(axis="y", alpha=0.3)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("KNN baseline: k=3 average vs k=1 kth", fontsize=12)
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "knn_comparison_all_metrics.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")

def plot_per_metric(df_avg, df_kth):
    """One saved figure per metric."""
    scenarios = df_avg["scenario"].tolist()
    x = np.arange(len(scenarios))
    w = 0.35
    short_scenarios = [s.replace("city_", "").replace("_3p5", "").replace("_s", "") for s in scenarios]

    for col, label in zip(METRIC_COLS, METRIC_LABELS):
        fig, ax = plt.subplots(figsize=(10, 4))
        v_avg = df_avg[col].values
        v_kth = df_kth[col].values
        ax.bar(x - w / 2, v_avg, w, label="k=3 average", color="steelblue")
        ax.bar(x + w / 2, v_kth, w, label="k=1 kth", color="coral", alpha=0.9)
        ax.set_ylabel(label)
        ax.set_xlabel("Scenario")
        ax.set_xticks(x)
        ax.set_xticklabels(short_scenarios, rotation=45, ha="right")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        ax.set_title(f"KNN baseline: {label}")
        plt.tight_layout()
        safe_name = col.replace("_", "-")
        out_path = os.path.join(OUT_DIR, f"knn_comparison_{safe_name}.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {out_path}")

def main():
    df_avg, df_kth = load_and_merge()
    print(f"Scenarios: {list(df_avg['scenario'])}")
    plot_all_metrics_one_figure(df_avg, df_kth)
    plot_per_metric(df_avg, df_kth)
    print(f"Plots saved to: {OUT_DIR}")

if __name__ == "__main__":
    main()
