import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


MLP_PATH = Path("/home/blessedg/kayley/PathFormer/MLP_beam_pred_using_channel_resultsmmwave.csv")
PATHFORMER_PATH = Path("/home/blessedg/Pathformer/logs/beam_foundation_mmwave_all.csv")
LWM_PATH = Path("/home/blessedg/Pathformer/logs/lwm_beam_results_mmwave.csv")
WIFO_PATH = Path("/home/blessedg/Pathformer/WiFo/beam_finetune_results_parallel_merged_mmwave.csv")
OUTPUT_DIR = Path("/home/blessedg/Pathformer/logs/mmwave_beam_comparison")


def short_label(scenario: str) -> str:
    parts = scenario.split("_")
    if len(parts) >= 3:
        return parts[2]
    return scenario


def load_metric_frame() -> pd.DataFrame:
    mlp = pd.read_csv(MLP_PATH)[["scenario", "top1_mean", "top3_mean"]].rename(
        columns={"top1_mean": "MLP_top1", "top3_mean": "MLP_top3"}
    )
    pathformer = pd.read_csv(PATHFORMER_PATH)[["scenario", "top1_acc", "top3_acc"]].rename(
        columns={"top1_acc": "Pathformer_top1", "top3_acc": "Pathformer_top3"}
    )
    lwm = pd.read_csv(LWM_PATH)[["scenario", "top1_acc", "top3_acc"]].rename(
        columns={"top1_acc": "LWM_top1", "top3_acc": "LWM_top3"}
    )
    wifo = pd.read_csv(WIFO_PATH)[["scenario", "val_top1_acc", "val_top3_acc"]].rename(
        columns={"val_top1_acc": "Wifo_top1", "val_top3_acc": "Wifo_top3"}
    )

    common = set(mlp["scenario"]) & set(pathformer["scenario"]) & set(lwm["scenario"]) & set(wifo["scenario"])
    merged = (
        mlp[mlp["scenario"].isin(common)]
        .merge(pathformer[pathformer["scenario"].isin(common)], on="scenario", how="inner")
        .merge(lwm[lwm["scenario"].isin(common)], on="scenario", how="inner")
        .merge(wifo[wifo["scenario"].isin(common)], on="scenario", how="inner")
        .sort_values("scenario")
        .reset_index(drop=True)
    )
    merged["scenario_label"] = merged["scenario"].map(short_label)
    return merged


def plot_metric(df: pd.DataFrame, metric: str, output_path: Path) -> None:
    methods = ["MLP", "Pathformer", "LWM", "Wifo"]
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e"]

    x = range(len(df))
    width = 0.2
    offsets = [-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width]

    fig, ax = plt.subplots(figsize=(14, 6))
    for method, color, offset in zip(methods, colors, offsets):
        ax.bar(
            [i + offset for i in x],
            df[f"{method}_{metric}"],
            width=width,
            label=method,
            color=color,
        )

    ax.set_xticks(list(x))
    ax.set_xticklabels(df["scenario_label"], rotation=30, ha="right")
    ax.set_ylabel("Validation Accuracy")
    ax.set_xlabel("Scenario")
    ax.set_ylim(0, 1.0)
    ax.set_title(f"MMWave Beam Prediction {metric.upper()} Validation Accuracy")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_metric_frame()
    df.to_csv(OUTPUT_DIR / "shared_scenarios_metrics.csv", index=False)
    plot_metric(df, "top1", OUTPUT_DIR / "shared_scenarios_top1.png")
    plot_metric(df, "top3", OUTPUT_DIR / "shared_scenarios_top3.png")
    print(f"Saved merged metrics to {OUTPUT_DIR / 'shared_scenarios_metrics.csv'}")
    print(f"Saved top-1 plot to {OUTPUT_DIR / 'shared_scenarios_top1.png'}")
    print(f"Saved top-3 plot to {OUTPUT_DIR / 'shared_scenarios_top3.png'}")
    print("Shared scenarios:", ", ".join(df["scenario"].tolist()))


if __name__ == "__main__":
    main()
