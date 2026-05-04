import csv
from pathlib import Path

import matplotlib.pyplot as plt


SCENARIO = "city_47_chicago_3p5"
LOG_DIR = Path("/home/blessedg/Pathformer/logs")

MODEL_FILES = {
    "direct": {
        0.3: LOG_DIR / "direct_train_ratio_ablate0_3.csv",
        0.4: LOG_DIR / "direct_train_ratio_ablate0_4.csv",
        0.5: LOG_DIR / "direct_train_ratio_ablate0_5.csv",
        0.6: LOG_DIR / "direct_train_ratio_ablate0_6.csv",
        0.7: LOG_DIR / "direct_train_ratio_ablate0_7.csv",
        0.8: LOG_DIR / "multiscenario_direct_city_47_chicago_3p5.csv",
    },
    "first_step_residual": {
        0.3: LOG_DIR / "first_step_train_ratio_ablate0_3.csv",
        0.4: LOG_DIR / "first_step_train_ratio_ablate0_4.csv",
        0.5: LOG_DIR / "first_step_train_ratio_ablate0_5.csv",
        0.6: LOG_DIR / "first_step_train_ratio_ablate0_6.csv",
        0.7: LOG_DIR / "first_step_train_ratio_ablate0_7.csv",
        0.8: LOG_DIR / "first_step_city_47_chicago_3p5.csv",
    },
    "first_step_residual_corridor": {
        0.3: LOG_DIR / "train_ratio_ablate0_3.csv",
        0.4: LOG_DIR / "train_ratio_ablate0_4.csv",
        0.5: LOG_DIR / "train_ratio_ablate0_5.csv",
        0.6: LOG_DIR / "train_ratio_ablate0_6.csv",
        0.7: LOG_DIR / "train_ratio_ablate0_7.csv",
        0.8: LOG_DIR / "corridor_city_47_chicago_3p5.csv",
    },
}

RMSE_COLUMNS = [
    "delay_rmse",
    "power_rmse",
    "az_rmse",
    "el_rmse",
    "aod_az_rmse",
    "aod_el_rmse",
    "path_length_rmse",
]

ALL_OUTPUT = LOG_DIR / "city_47_chicago_3p5_train_ratio_combined_all_metrics.csv"
RMSE_OUTPUT = LOG_DIR / "city_47_chicago_3p5_train_ratio_combined_rmse.csv"
PLOT_OUTPUT = LOG_DIR / "city_47_chicago_3p5_train_ratio_rmse_comparison.png"


def read_single_row(csv_path: Path) -> dict:
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise ValueError(f"No rows found in {csv_path}")
    return rows[-1]


def build_records() -> list[dict]:
    records = []
    for model_family, ratio_files in MODEL_FILES.items():
        for train_ratio, csv_path in sorted(ratio_files.items()):
            row = read_single_row(csv_path)
            record = {
                "scenario": row.get("scenario", SCENARIO),
                "model_family": model_family,
                "train_ratio": train_ratio,
                "train_percent": train_ratio * 100.0,
                "source_file": str(csv_path),
            }
            record.update(row)
            records.append(record)
    return records


def write_csv(records: list[dict], output_path: Path, fieldnames: list[str]) -> None:
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record.get(key, "") for key in fieldnames})


def plot_rmse(records: list[dict]) -> None:
    family_order = list(MODEL_FILES.keys())
    color_map = {
        "direct": "#1f77b4",
        "first_step_residual": "#ff7f0e",
        "first_step_residual_corridor": "#2ca02c",
    }
    title_map = {
        "delay_rmse": "Delay RMSE",
        "power_rmse": "Power RMSE",
        "az_rmse": "AoA Azimuth RMSE",
        "el_rmse": "AoA Elevation RMSE",
        "aod_az_rmse": "AoD Azimuth RMSE",
        "aod_el_rmse": "AoD Elevation RMSE",
        "path_length_rmse": "Path Length RMSE",
    }

    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    axes = axes.flatten()

    for idx, metric in enumerate(RMSE_COLUMNS):
        ax = axes[idx]
        for family in family_order:
            family_records = sorted(
                (r for r in records if r["model_family"] == family),
                key=lambda r: float(r["train_ratio"]),
            )
            xs = [float(r["train_percent"]) for r in family_records]
            ys = [float(r[metric]) for r in family_records]
            ax.plot(
                xs,
                ys,
                marker="o",
                linewidth=2,
                markersize=6,
                label=family,
                color=color_map[family],
            )
        ax.set_title(title_map[metric])
        ax.set_xlabel("Train Users (%)")
        ax.set_ylabel("RMSE")
        ax.grid(True, alpha=0.3)

    for idx in range(len(RMSE_COLUMNS), len(axes)):
        axes[idx].axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.suptitle(f"RMSE vs Train User Percentage for {SCENARIO}", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(PLOT_OUTPUT, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    records = build_records()
    all_fields = [
        "scenario",
        "model_family",
        "train_ratio",
        "train_percent",
        "source_file",
    ]
    for record in records:
        for key in record:
            if key not in all_fields:
                all_fields.append(key)

    rmse_fields = [
        "scenario",
        "model_family",
        "train_ratio",
        "train_percent",
        "source_file",
        *RMSE_COLUMNS,
    ]

    write_csv(records, ALL_OUTPUT, all_fields)
    write_csv(records, RMSE_OUTPUT, rmse_fields)
    plot_rmse(records)

    print(f"Wrote {ALL_OUTPUT}")
    print(f"Wrote {RMSE_OUTPUT}")
    print(f"Wrote {PLOT_OUTPUT}")


if __name__ == "__main__":
    main()
