import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


WIFO_PATH = Path("/home/blessedg/Pathformer/WiFo/src/experiments/temporal_0.5Test_Dataset_all_scenarios_14_Task_short_FewRatio_0.0_base_/results_summary.csv")
LWM_PATH = Path("/home/blessedg/Pathformer/logs/lwm_channel_interpolation_last_16_subcarriers.csv")
OUTPUT_DIR = Path("/home/blessedg/Pathformer/logs/wifo_vs_lwm_channel_estimation")


def short_label(name: str) -> str:
    parts = name.split("_")
    return parts[2] if len(parts) >= 3 else name


def build_merged() -> pd.DataFrame:
    wifo = pd.read_csv(WIFO_PATH)[["dataset", "last_nmse", "last_nmse_db", "nmse", "nmse_db"]].rename(
        columns={
            "dataset": "scenario",
            "last_nmse": "WiFo_last16_nmse",
            "last_nmse_db": "WiFo_last16_nmse_db",
            "nmse": "WiFo_full_nmse",
            "nmse_db": "WiFo_full_nmse_db",
        }
    )
    lwm = pd.read_csv(LWM_PATH)[["scenario", "nmse", "nmse_db", "score"]].rename(
        columns={
            "nmse": "LWM_last16_nmse",
            "nmse_db": "LWM_last16_nmse_db",
            "score": "LWM_score",
        }
    )
    common = sorted(set(wifo["scenario"]) & set(lwm["scenario"]))
    merged = (
        wifo[wifo["scenario"].isin(common)]
        .merge(lwm[lwm["scenario"].isin(common)], on="scenario", how="inner")
        .sort_values("scenario")
        .reset_index(drop=True)
    )
    merged["scenario_label"] = merged["scenario"].map(short_label)
    merged["nmse_winner"] = merged.apply(
        lambda row: "WiFo" if row["WiFo_last16_nmse"] < row["LWM_last16_nmse"] else "LWM",
        axis=1,
    )
    return merged


def plot_nmse_db(merged: pd.DataFrame, out_path: Path) -> None:
    x = range(len(merged))
    width = 0.36
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.bar([i - width / 2 for i in x], merged["WiFo_last16_nmse_db"], width=width, label="WiFo", color="#ff7f0e")
    ax.bar([i + width / 2 for i in x], merged["LWM_last16_nmse_db"], width=width, label="LWM", color="#2ca02c")
    ax.set_xticks(list(x))
    ax.set_xticklabels(merged["scenario_label"], rotation=30, ha="right")
    ax.set_ylabel("NMSE (dB)")
    ax.set_xlabel("Scenario")
    ax.set_title("Last-16-Subcarrier Channel Estimation Comparison")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_nmse(merged: pd.DataFrame, out_path: Path) -> None:
    x = range(len(merged))
    width = 0.36
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.bar([i - width / 2 for i in x], merged["WiFo_last16_nmse"], width=width, label="WiFo", color="#ff7f0e")
    ax.bar([i + width / 2 for i in x], merged["LWM_last16_nmse"], width=width, label="LWM", color="#2ca02c")
    ax.set_xticks(list(x))
    ax.set_xticklabels(merged["scenario_label"], rotation=30, ha="right")
    ax.set_ylabel("NMSE")
    ax.set_xlabel("Scenario")
    ax.set_title("Last-16-Subcarrier Channel Estimation Comparison")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    merged = build_merged()
    csv_path = OUTPUT_DIR / "wifo_lwm_last16_comparison.csv"
    merged.to_csv(csv_path, index=False)
    plot_path = OUTPUT_DIR / "wifo_lwm_last16_nmse_db.png"
    plot_nmse_path = OUTPUT_DIR / "wifo_lwm_last16_nmse.png"
    plot_nmse_db(merged, plot_path)
    plot_nmse(merged, plot_nmse_path)
    print(f"Saved merged comparison to {csv_path}")
    print(f"Saved plot to {plot_path}")
    print(f"Saved plot to {plot_nmse_path}")
    print(merged.to_string(index=False))


if __name__ == "__main__":
    main()
