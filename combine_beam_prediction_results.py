from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path("/home/blessedg/Pathformer")
LOG_DIR = ROOT / "logs"

LWM_PATH = LOG_DIR / "lwm_beam_results.csv"
FOUNDATION_PATH = LOG_DIR / "beam_foundation_all.csv"
SINGLE_PATH = ROOT / "beam_prediction_finetuning_all_families.csv"
ZERO_SHOT_PATH = ROOT / "zeero_beam_prediction_finetuning.csv"


def load_results() -> pd.DataFrame:
    lwm = pd.read_csv(LWM_PATH).copy()
    lwm = lwm[["scenario", "top1_acc", "top3_acc", "test_f1"]]
    lwm["method"] = "LWM Finetune"
    lwm["source_file"] = str(LWM_PATH)

    foundation = pd.read_csv(FOUNDATION_PATH).copy()
    foundation = foundation[["scenario", "top1_acc", "top3_acc", "model_family"]]
    foundation["method"] = "PathFormer Multienvironment"
    foundation["source_file"] = str(FOUNDATION_PATH)
    foundation = foundation.drop(columns=["model_family"])
    foundation["test_f1"] = pd.NA

    single = pd.read_csv(SINGLE_PATH).copy()
    single = single[single["model_family"] == "first_step_residual_corridor"].copy()
    single = single[["scenario", "top1_acc", "top3_acc", "model_family"]]
    single["method"] = "PathFormer Single Scenario"
    single["source_file"] = str(SINGLE_PATH)
    single = single.drop(columns=["model_family"])
    single["test_f1"] = pd.NA

    zero_shot = pd.read_csv(ZERO_SHOT_PATH).copy()
    zero_shot = zero_shot[zero_shot["model_family"] == "first_step_residual_corridor"].copy()
    zero_shot = zero_shot[["scenario", "beam_acc"]]
    zero_shot = zero_shot.rename(columns={"beam_acc": "top1_acc"})
    zero_shot["top3_acc"] = pd.NA
    zero_shot["test_f1"] = pd.NA
    zero_shot["method"] = "PathFormer Zero Shot"
    zero_shot["source_file"] = str(ZERO_SHOT_PATH)

    combined = pd.concat([lwm, foundation, single, zero_shot], ignore_index=True)
    combined["top1_acc"] = pd.to_numeric(combined["top1_acc"], errors="coerce")
    combined["top3_acc"] = pd.to_numeric(combined["top3_acc"], errors="coerce")
    combined = (
        combined.sort_values(["method", "scenario", "top1_acc", "top3_acc"], ascending=[True, True, False, False])
        .drop_duplicates(subset=["method", "scenario"], keep="first")
        .reset_index(drop=True)
    )
    return combined


def build_common_wide(long_df: pd.DataFrame) -> pd.DataFrame:
    common = sorted(set.intersection(*[
        set(long_df.loc[long_df["method"] == method, "scenario"])
        for method in long_df["method"].unique()
    ]))

    common_df = long_df[long_df["scenario"].isin(common)].copy()
    wide = common_df.pivot(index="scenario", columns="method", values=["top1_acc", "top3_acc"])
    wide.columns = [f"{metric}_{method.lower().replace(' ', '_')}" for metric, method in wide.columns]
    wide = wide.reset_index().sort_values("scenario")
    return wide


def summarize_common(wide_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    method_map = {
        "LWM Finetune": "lwm_finetune",
        "PathFormer Multienvironment": "pathformer_multienvironment",
        "PathFormer Single Scenario": "pathformer_single_scenario",
        "PathFormer Zero Shot": "pathformer_zero_shot",
    }
    for label, prefix in method_map.items():
        rows.append({
            "method": label,
            "n_scenarios": int(len(wide_df)),
            "top1_mean": float(wide_df[f"top1_acc_{prefix}"].mean()),
            "top1_std": float(wide_df[f"top1_acc_{prefix}"].std(ddof=1)),
            "top3_mean": float(wide_df[f"top3_acc_{prefix}"].mean()) if f"top3_acc_{prefix}" in wide_df.columns else float("nan"),
            "top3_std": float(wide_df[f"top3_acc_{prefix}"].std(ddof=1)) if f"top3_acc_{prefix}" in wide_df.columns else float("nan"),
        })
    return pd.DataFrame(rows)


def plot_common(wide_df: pd.DataFrame, summary_df: pd.DataFrame, out_path: Path) -> None:
    method_labels = {
        "lwm_finetune": "LWM Finetune",
        "pathformer_multienvironment": "PathFormer Multienvironment",
        "pathformer_single_scenario": "PathFormer Single Scenario",
        "pathformer_zero_shot": "PathFormer Zero Shot",
    }
    colors = {
        "lwm_finetune": "#4c78a8",
        "pathformer_multienvironment": "#f58518",
        "pathformer_single_scenario": "#54a24b",
        "pathformer_zero_shot": "#e45756",
    }

    sort_col = "top1_acc_pathformer_single_scenario"
    plot_df = wide_df.sort_values(sort_col).reset_index(drop=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)

    # Left panel: per-scenario top-1 comparison on common scenarios
    x = range(len(plot_df))
    for prefix, label in method_labels.items():
        axes[0].plot(
            x,
            plot_df[f"top1_acc_{prefix}"],
            marker="o",
            linewidth=1.8,
            markersize=4,
            label=label,
            color=colors[prefix],
        )
    axes[0].set_title("Top-1 Accuracy on Common Scenarios")
    axes[0].set_xlabel("Scenarios")
    axes[0].set_ylabel("Top-1 Accuracy")
    axes[0].grid(True, alpha=0.25)

    # Right panel: average top-1/top-3 with std bars
    summary_plot = summary_df.copy()
    x_pos = range(len(summary_plot))
    width = 0.36
    top1_means = summary_plot["top1_mean"].to_numpy()
    top1_std = summary_plot["top1_std"].to_numpy()
    top3_means = summary_plot["top3_mean"].to_numpy()
    top3_std = summary_plot["top3_std"].to_numpy()
    bar_colors = []
    for label in summary_plot["method"]:
        for key, pretty in method_labels.items():
            if pretty == label:
                bar_colors.append(colors[key])
                break

    axes[1].bar([i - width / 2 for i in x_pos], top1_means, width=width, yerr=top1_std, capsize=4, color=bar_colors, alpha=0.9, label="Top-1")
    top3_positions = []
    top3_values = []
    top3_errors = []
    top3_colors = []
    for i, (mean, std, color) in enumerate(zip(top3_means, top3_std, bar_colors)):
        if pd.notna(mean):
            top3_positions.append(i + width / 2)
            top3_values.append(mean)
            top3_errors.append(std)
            top3_colors.append(color)
    axes[1].bar(top3_positions, top3_values, width=width, yerr=top3_errors, capsize=4, color=top3_colors, alpha=0.45, label="Top-3")
    axes[1].set_xticks(list(x_pos))
    axes[1].set_xticklabels(summary_plot["method"], rotation=10, ha="right")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Average Accuracy Across Common Scenarios")
    axes[1].grid(True, axis="y", alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, labels, loc="lower right", frameon=False)
    axes[1].legend(loc="lower right", frameon=False)

    fig.suptitle("Beam Prediction Comparison: LWM vs PathFormer", fontsize=14)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    long_df = load_results()
    long_out = LOG_DIR / "beam_prediction_comparison_long.csv"
    long_df.to_csv(long_out, index=False)

    wide_df = build_common_wide(long_df)
    wide_out = LOG_DIR / "beam_prediction_comparison_common_scenarios.csv"
    wide_df.to_csv(wide_out, index=False)

    summary_df = summarize_common(wide_df)
    summary_out = LOG_DIR / "beam_prediction_comparison_summary.csv"
    summary_df.to_csv(summary_out, index=False)

    plot_out = LOG_DIR / "beam_prediction_comparison.png"
    plot_common(wide_df, summary_df, plot_out)

    print(f"Saved long comparison to {long_out}")
    print(f"Saved common-scenario comparison to {wide_out}")
    print(f"Saved summary to {summary_out}")
    print(f"Saved plot to {plot_out}")
    print(f"Common scenarios: {len(wide_df)}")


if __name__ == "__main__":
    main()
