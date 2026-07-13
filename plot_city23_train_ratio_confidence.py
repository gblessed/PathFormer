import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


LOG_DIR = Path("/home/blessedg/Pathformer/logs")
SCENARIO = "city_23_beijing_3p5"

FAMILY_SPECS = {
    "direct": "direct_train_ratio_city_23_beijing_3p5_r*_seed*.csv",
    "first_step_residual": "first_step_train_ratio_city_23_beijing_3p5_r*_seed*.csv",
    "first_step_residual_corridor": "corridor_train_ratio_city_23_beijing_3p5_r*_seed*.csv",
}

METRICS = [
    ("delay_mae", "Delay MAE"),
    ("power_mae", "Power MAE"),
    ("avg_az_mae", "AoA Az MAE"),
    ("avg_el_mae", "AoA El MAE"),
    ("avg_aod_az_mae", "AoD Az MAE"),
    ("avg_aod_el_mae", "AoD El MAE"),
    ("interaction_accuracy", "Interaction Accuracy"),
    ("interaction_f1", "Interaction F1"),
]

FAMILY_LABELS = {
    "direct": "Direct",
    "first_step_residual": "First-step Residual",
    "first_step_residual_corridor": "Corridor Residual",
}

FAMILY_COLORS = {
    "direct": "#1f77b4",
    "first_step_residual": "#ff7f0e",
    "first_step_residual_corridor": "#2ca02c",
}


def load_seed_rows() -> pd.DataFrame:
    frames = []
    for family, pattern in FAMILY_SPECS.items():
        for path in sorted(LOG_DIR.glob(pattern)):
            df = pd.read_csv(path)
            if df.empty:
                continue
            row = df.iloc[0].copy()
            row["model_family"] = family
            row["source_file"] = path.name
            frames.append(pd.DataFrame([row]))
    if not frames:
        raise RuntimeError("No seed-sweep CSV files found.")
    combined = pd.concat(frames, ignore_index=True)
    combined["train_percent"] = combined["train_ratio"] * 100.0
    return combined


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (family, ratio), group in df.groupby(["model_family", "train_ratio"], sort=True):
        summary = {
            "scenario": SCENARIO,
            "model_family": family,
            "train_ratio": ratio,
            "train_percent": ratio * 100.0,
            "n_runs": int(len(group)),
            "seeds": ",".join(str(int(s)) for s in sorted(group["seed"].tolist())),
        }
        for metric, _ in METRICS:
            vals = pd.to_numeric(group[metric], errors="coerce").dropna()
            n = len(vals)
            mean = float(vals.mean()) if n else math.nan
            std = float(vals.std(ddof=1)) if n > 1 else 0.0 if n == 1 else math.nan
            sem = float(std / math.sqrt(n)) if n > 0 and not math.isnan(std) else math.nan
            ci95 = float(1.96 * sem) if n > 1 else 0.0 if n == 1 else math.nan
            summary[f"{metric}_mean"] = mean
            summary[f"{metric}_std"] = std
            summary[f"{metric}_sem"] = sem
            summary[f"{metric}_ci95"] = ci95
            summary[f"{metric}_ci95_low"] = mean - ci95 if n else math.nan
            summary[f"{metric}_ci95_high"] = mean + ci95 if n else math.nan
        rows.append(summary)
    return pd.DataFrame(rows).sort_values(["model_family", "train_ratio"]).reset_index(drop=True)


def plot(summary_df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), constrained_layout=True)
    axes = axes.flatten()

    for ax, (metric, title) in zip(axes, METRICS):
        for family in FAMILY_SPECS:
            fam_df = summary_df[summary_df["model_family"] == family].sort_values("train_ratio")
            if fam_df.empty:
                continue
            x = fam_df["train_percent"].to_numpy()
            y = fam_df[f"{metric}_mean"].to_numpy()
            ci = fam_df[f"{metric}_ci95"].to_numpy()

            ax.plot(
                x,
                y,
                marker="o",
                linewidth=2,
                color=FAMILY_COLORS[family],
                label=FAMILY_LABELS[family],
            )
            ax.fill_between(
                x,
                y - ci,
                y + ci,
                alpha=0.18,
                color=FAMILY_COLORS[family],
            )

        ax.set_title(title)
        ax.set_xlabel("Train Users (%)")
        ax.grid(True, alpha=0.25)

    for ax in axes[:6]:
        ax.set_ylabel("MAE")
    axes[6].set_ylabel("Accuracy")
    axes[7].set_ylabel("F1")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.03))
    # fig.suptitle(
    #     f"Train-Ratio Sweep on {SCENARIO} (mean ± 95% CI over seeds)",
    #     y=1.08,
    #     fontsize=14,
    # )
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    raw_df = load_seed_rows()
    summary_df = summarize(raw_df)

    raw_path = LOG_DIR / "city_23_beijing_3p5_train_ratio_seed_sweep_raw.csv"
    summary_path = LOG_DIR / "city_23_beijing_3p5_train_ratio_seed_sweep_summary.csv"
    plot_path = LOG_DIR / "city_23_beijing_3p5_train_ratio_seed_sweep_confidence.pdf"

    raw_df.to_csv(raw_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    plot(summary_df, plot_path)

    print(f"Saved raw rows to {raw_path}")
    print(f"Saved summary to {summary_path}")
    print(f"Saved plot to {plot_path}")

    missing = summary_df[summary_df["n_runs"] < 3][["model_family", "train_ratio", "n_runs", "seeds"]]
    if not missing.empty:
        print("\nPoints with fewer than 3 runs:")
        print(missing.to_string(index=False))


if __name__ == "__main__":
    main()
