import csv
from pathlib import Path


LOG_DIR = Path("/home/blessedg/Pathformer/logs")

MODEL_FILES = {
    "direct": LOG_DIR / "direct_results.csv",
    "first_step_residual": LOG_DIR / "first_residual_results.csv",
    "first_step_residual_corridor": LOG_DIR / "corridor_results.csv",
}

MAE_METRICS = [
    ("delay_mae", "delay_mae_std"),
    ("power_mae", "power_mae_std"),
    ("avg_az_mae", "avg_az_mae_std"),
    ("avg_el_mae", "avg_el_mae_std"),
    ("avg_aod_az_mae", "avg_aod_az_mae_std"),
    ("avg_aod_el_mae", "avg_aod_el_mae_std"),
]

INTERACTION_METRICS = [
    ("interaction_accuracy", "interaction_accuracy_std"),
    ("interaction_f1", "interaction_f1_std"),
]

LONG_OUTPUT = LOG_DIR / "all_scenarios_model_family_mae_long.csv"
WIDE_OUTPUT = LOG_DIR / "all_scenarios_model_family_mae_wide.csv"
SUMMARY_OUTPUT = LOG_DIR / "all_scenarios_model_family_mae_summary.md"


def round_str(value: str) -> str:
    if value == "" or value is None:
        return ""
    return f"{float(value):.3f}"


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def build_scenario_map() -> dict[str, dict[str, dict]]:
    scenario_map: dict[str, dict[str, dict]] = {}
    for family, path in MODEL_FILES.items():
        for row in read_rows(path):
            scenario = row["scenario"]
            scenario_map.setdefault(scenario, {})
            scenario_map[scenario][family] = row
    return dict(sorted(scenario_map.items()))


def build_long_records(scenario_map: dict[str, dict[str, dict]]) -> list[dict]:
    records = []
    for scenario, family_rows in scenario_map.items():
        for family in MODEL_FILES:
            row = family_rows.get(family)
            if row is None:
                continue
            record = {
                "scenario": scenario,
                "model_family": family,
            }
            for mean_col, std_col in MAE_METRICS:
                record[mean_col] = round_str(row.get(mean_col, ""))
                record[std_col] = round_str(row.get(std_col, ""))
            for mean_col, std_col in INTERACTION_METRICS:
                record[mean_col] = round_str(row.get(mean_col, ""))
                record[std_col] = round_str(row.get(std_col, ""))
            records.append(record)
    return records


def write_long_csv(records: list[dict]) -> None:
    fieldnames = ["scenario", "model_family"]
    for mean_col, std_col in MAE_METRICS:
        fieldnames.extend([mean_col, std_col])
    for mean_col, std_col in INTERACTION_METRICS:
        fieldnames.extend([mean_col, std_col])

    with LONG_OUTPUT.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def write_wide_csv(scenario_map: dict[str, dict[str, dict]]) -> None:
    fieldnames = ["scenario"]
    for family in MODEL_FILES:
        for mean_col, std_col in MAE_METRICS:
            fieldnames.append(f"{family}_{mean_col}")
            fieldnames.append(f"{family}_{std_col}")
        for mean_col, std_col in INTERACTION_METRICS:
            fieldnames.append(f"{family}_{mean_col}")
            fieldnames.append(f"{family}_{std_col}")

    rows = []
    for scenario, family_rows in scenario_map.items():
        out_row = {"scenario": scenario}
        for family in MODEL_FILES:
            row = family_rows.get(family)
            for mean_col, std_col in MAE_METRICS:
                out_row[f"{family}_{mean_col}"] = round_str(row.get(mean_col, "")) if row else ""
                out_row[f"{family}_{std_col}"] = round_str(row.get(std_col, "")) if row else ""
            for mean_col, std_col in INTERACTION_METRICS:
                out_row[f"{family}_{mean_col}"] = round_str(row.get(mean_col, "")) if row else ""
                out_row[f"{family}_{std_col}"] = round_str(row.get(std_col, "")) if row else ""
        rows.append(out_row)

    with WIDE_OUTPUT.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def average(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def write_summary(scenario_map: dict[str, dict[str, dict]]) -> None:
    metric_labels = {
        "delay_mae": "Delay",
        "power_mae": "Power",
        "avg_az_mae": "AoA Az",
        "avg_el_mae": "AoA El",
        "avg_aod_az_mae": "AoD Az",
        "avg_aod_el_mae": "AoD El",
    }

    lines = [
        "# MAE Summary Across Scenarios",
        "",
        f"- Scenarios found: {len(scenario_map)}",
        f"- Source files: {', '.join(str(path) for path in MODEL_FILES.values())}",
        "",
        "## Average Per-Scenario MAE and MAE Std by Model Family",
        "",
        "| Model Family | Delay | Delay Std | Power | Power Std | AoA Az | AoA Az Std | AoA El | AoA El Std | AoD Az | AoD Az Std | AoD El | AoD El Std | Acc. | Acc. Std | F1 | F1 Std |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for family in MODEL_FILES:
        family_rows = [family_rows[family] for family_rows in scenario_map.values() if family in family_rows]
        formatted = []
        for mean_col, std_col in MAE_METRICS:
            mean_avg = average([float(row[mean_col]) for row in family_rows])
            std_avg = average([float(row[std_col]) for row in family_rows])
            formatted.extend([f"{mean_avg:.3f}", f"{std_avg:.3f}"])
        for mean_col, std_col in INTERACTION_METRICS:
            mean_avg = average([float(row[mean_col]) for row in family_rows])
            std_avg = average([float(row[std_col]) for row in family_rows])
            formatted.extend([f"{mean_avg:.3f}", f"{std_avg:.3f}"])
        lines.append(f"| {family} | " + " | ".join(formatted) + " |")

    lines.extend(
        [
            "",
            "## Scenario Coverage",
            "",
            "| Scenario | Direct | First Step Residual | Corridor |",
            "| --- | --- | --- | --- |",
        ]
    )

    for scenario, family_rows in scenario_map.items():
        lines.append(
            f"| {scenario} | "
            f"{'yes' if 'direct' in family_rows else 'no'} | "
            f"{'yes' if 'first_step_residual' in family_rows else 'no'} | "
            f"{'yes' if 'first_step_residual_corridor' in family_rows else 'no'} |"
        )

    SUMMARY_OUTPUT.write_text("\n".join(lines) + "\n")


def main() -> None:
    scenario_map = build_scenario_map()
    long_records = build_long_records(scenario_map)
    write_long_csv(long_records)
    write_wide_csv(scenario_map)
    write_summary(scenario_map)
    print(f"Wrote {LONG_OUTPUT}")
    print(f"Wrote {WIDE_OUTPUT}")
    print(f"Wrote {SUMMARY_OUTPUT}")


if __name__ == "__main__":
    main()
