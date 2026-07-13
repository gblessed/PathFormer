from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path("/home/blessedg/Pathformer")
LOG_DIR = ROOT / "logs"

FOUNDATION_PATH = ROOT / "foundation_model_eval.csv"
SINGLE_PATH = LOG_DIR / "corridor_results.csv"
FINETUNE_DIR = LOG_DIR / "foundation_finetune_per_scenario"

TRAINED_SCENARIOS = {
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
}

METRICS = [
    ("delay_mae", "Delay MAE", "lower"),
    ("power_mae", "Power MAE", "lower"),
    ("avg_aod_az_mae", "AoD Az MAE", "lower"),
    ("avg_aod_el_mae", "AoD El MAE", "lower"),
    ("interaction_accuracy", "Interaction Accuracy", "higher"),
    ("interaction_f1", "Interaction F1", "higher"),
]

METHOD_LABELS = {
    "foundation_zero_shot": "Foundation Zero-shot",
    "single_scenario": "Single Scenario",
    "foundation_finetuned": "Foundation + Finetune",
}

METHOD_COLORS = {
    "foundation_zero_shot": "#4c78a8",
    "single_scenario": "#f58518",
    "foundation_finetuned": "#54a24b",
}


def add_split(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["split"] = df["scenario"].apply(lambda s: "ID" if s in TRAINED_SCENARIOS else "OOD")
    return df


def load_foundation() -> pd.DataFrame:
    df = pd.read_csv(FOUNDATION_PATH).copy()
    df["method"] = "foundation_zero_shot"
    return add_split(df)


def load_single() -> pd.DataFrame:
    df = pd.read_csv(SINGLE_PATH).copy()
    df["method"] = "single_scenario"
    return add_split(df)


def load_finetuned() -> pd.DataFrame:
    frames = []
    for path in sorted(FINETUNE_DIR.glob("*.csv")):
        df = pd.read_csv(path)
        if not df.empty:
            frames.append(df.iloc[[0]])
    if not frames:
        raise RuntimeError(f"No finetune CSV files found in {FINETUNE_DIR}")
    df = pd.concat(frames, ignore_index=True)
    df["method"] = "foundation_finetuned"
    return add_split(df)


def save_long_table(found_df: pd.DataFrame, single_df: pd.DataFrame, finetune_df: pd.DataFrame) -> pd.DataFrame:
    combined = pd.concat([found_df, single_df, finetune_df], ignore_index=True, sort=False)
    out_path = LOG_DIR / "foundation_single_finetuned_comparison_long.csv"
    combined.to_csv(out_path, index=False)
    return combined


def make_summary(long_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method in ["foundation_zero_shot", "single_scenario", "foundation_finetuned"]:
        for split in ["ID", "OOD", "ALL"]:
            if split == "ALL":
                group = long_df[long_df["method"] == method]
            else:
                group = long_df[(long_df["method"] == method) & (long_df["split"] == split)]
            if group.empty:
                continue
            row = {
                "method": method,
                "method_label": METHOD_LABELS[method],
                "split": split,
                "n_scenarios": int(len(group)),
            }
            for metric, _, _ in METRICS:
                row[f"{metric}_mean"] = float(group[metric].mean())
                row[f"{metric}_std"] = float(group[metric].std(ddof=1)) if len(group) > 1 else 0.0
            rows.append(row)
    summary = pd.DataFrame(rows)
    out_path = LOG_DIR / "foundation_single_finetuned_summary.csv"
    summary.to_csv(out_path, index=False)
    return summary


def make_common_tables(found_df: pd.DataFrame, single_df: pd.DataFrame, finetune_df: pd.DataFrame) -> dict:
    outputs = {}

    common_fs = sorted(set(found_df["scenario"]) & set(single_df["scenario"]))
    fs = pd.merge(
        found_df[found_df["scenario"].isin(common_fs)],
        single_df[single_df["scenario"].isin(common_fs)],
        on=["scenario", "split"],
        suffixes=("_foundation", "_single"),
    )
    fs_path = LOG_DIR / "foundation_vs_single_common.csv"
    fs.to_csv(fs_path, index=False)
    outputs["foundation_vs_single"] = fs

    common_ff = sorted(set(found_df["scenario"]) & set(finetune_df["scenario"]))
    ff = pd.merge(
        found_df[found_df["scenario"].isin(common_ff)],
        finetune_df[finetune_df["scenario"].isin(common_ff)],
        on=["scenario", "split"],
        suffixes=("_foundation", "_finetuned"),
    )
    ff_path = LOG_DIR / "foundation_vs_finetuned_common.csv"
    ff.to_csv(ff_path, index=False)
    outputs["foundation_vs_finetuned"] = ff

    common_all = sorted(set(common_fs) & set(common_ff))
    all3 = (
        found_df[found_df["scenario"].isin(common_all)][["scenario", "split"] + [m for m, _, _ in METRICS]]
        .rename(columns={m: f"{m}_foundation" for m, _, _ in METRICS})
        .merge(
            single_df[single_df["scenario"].isin(common_all)][["scenario"] + [m for m, _, _ in METRICS]]
            .rename(columns={m: f"{m}_single" for m, _, _ in METRICS}),
            on="scenario",
        )
        .merge(
            finetune_df[finetune_df["scenario"].isin(common_all)][["scenario"] + [m for m, _, _ in METRICS]]
            .rename(columns={m: f"{m}_finetuned" for m, _, _ in METRICS}),
            on="scenario",
        )
    )
    all3_path = LOG_DIR / "foundation_single_finetuned_common.csv"
    all3.to_csv(all3_path, index=False)
    outputs["all_three"] = all3
    return outputs


def plot_split_summary(summary_df: pd.DataFrame, out_path: Path) -> None:
    metrics_to_plot = METRICS[:4]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    axes = axes.flatten()
    x_labels = ["ID", "OOD"]
    method_order = ["foundation_zero_shot", "single_scenario", "foundation_finetuned"]
    width = 0.24

    for ax, (metric, title, _) in zip(axes, metrics_to_plot):
        for i, method in enumerate(method_order):
            vals = []
            errs = []
            for split in x_labels:
                sub = summary_df[(summary_df["method"] == method) & (summary_df["split"] == split)]
                vals.append(sub[f"{metric}_mean"].iloc[0] if not sub.empty else float("nan"))
                errs.append(sub[f"{metric}_std"].iloc[0] if not sub.empty else float("nan"))
            positions = [j + (i - 1) * width for j in range(len(x_labels))]
            ax.bar(
                positions,
                vals,
                width=width,
                yerr=errs,
                capsize=4,
                color=METHOD_COLORS[method],
                alpha=0.9,
                label=METHOD_LABELS[method],
            )
        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.25)
        ax.set_ylabel("MAE")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Foundation vs Single-Scenario vs Finetuned: ID/OOD Summary", y=1.06, fontsize=14)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_finetune_improvement(common_ff: pd.DataFrame, out_path: Path) -> None:
    metric = "avg_aod_az_mae"
    title = "AoD Az MAE Improvement from Finetuning"
    df = common_ff.copy()
    df["improvement"] = df[f"{metric}_foundation"] - df[f"{metric}_finetuned"]
    df = df.sort_values("improvement")

    colors = df["split"].map({"ID": "#4c78a8", "OOD": "#e45756"}).tolist()

    plt.figure(figsize=(13, 6))
    plt.bar(range(len(df)), df["improvement"], color=colors, alpha=0.9)
    plt.axhline(0.0, color="black", linewidth=1)
    plt.xticks(range(len(df)), df["scenario"], rotation=75, ha="right")
    plt.ylabel("Foundation MAE - Finetuned MAE")
    plt.title(title)
    plt.grid(True, axis="y", alpha=0.25)
    from matplotlib.patches import Patch
    plt.legend(handles=[Patch(color="#4c78a8", label="ID"), Patch(color="#e45756", label="OOD")], frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def write_markdown(summary_df: pd.DataFrame, common_tables: dict) -> None:
    out = LOG_DIR / "foundation_single_finetuned_summary.md"
    lines = []
    lines.append("# Foundation vs Single-Scenario vs Finetuned\n")
    lines.append("## Scenario Counts\n")
    for split in ["ID", "OOD", "ALL"]:
        sub = summary_df[summary_df["split"] == split]
        if not sub.empty:
            counts = ", ".join(f"{row.method_label}: {int(row.n_scenarios)}" for row in sub.itertuples())
            lines.append(f"- `{split}`: {counts}")
    lines.append("\n## Key Means\n")
    cols = ["method_label", "split", "n_scenarios"] + [f"{m}_mean" for m, _, _ in METRICS]
    lines.append("```text")
    lines.append(summary_df[cols].to_string(index=False))
    lines.append("```")
    lines.append("\n## Common Scenario Overlaps\n")
    lines.append(f"- Foundation vs Single: `{len(common_tables['foundation_vs_single'])}` scenarios")
    lines.append(f"- Foundation vs Finetuned: `{len(common_tables['foundation_vs_finetuned'])}` scenarios")
    lines.append(f"- All three: `{len(common_tables['all_three'])}` scenarios")
    out.write_text("\n".join(lines))


def main() -> None:
    found_df = load_foundation()
    single_df = load_single()
    finetune_df = load_finetuned()

    long_df = save_long_table(found_df, single_df, finetune_df)
    summary_df = make_summary(long_df)
    common_tables = make_common_tables(found_df, single_df, finetune_df)

    plot_split_summary(summary_df, LOG_DIR / "foundation_single_finetuned_id_ood_summary.png")
    plot_finetune_improvement(common_tables["foundation_vs_finetuned"], LOG_DIR / "foundation_finetune_improvement_aod_az.png")
    write_markdown(summary_df, common_tables)

    print("Saved:")
    print(LOG_DIR / "foundation_single_finetuned_comparison_long.csv")
    print(LOG_DIR / "foundation_single_finetuned_summary.csv")
    print(LOG_DIR / "foundation_vs_single_common.csv")
    print(LOG_DIR / "foundation_vs_finetuned_common.csv")
    print(LOG_DIR / "foundation_single_finetuned_common.csv")
    print(LOG_DIR / "foundation_single_finetuned_id_ood_summary.png")
    print(LOG_DIR / "foundation_finetune_improvement_aod_az.png")
    print(LOG_DIR / "foundation_single_finetuned_summary.md")


if __name__ == "__main__":
    main()
