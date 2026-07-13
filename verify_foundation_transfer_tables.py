import argparse
from pathlib import Path

import pandas as pd


TRAIN_SCENARIOS = {
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
    "city_16_sanfrancisco_3p5_lwm",
    "city_17_seattle_3p5_s",
    "city_18_denver_3p5",
    "city_19_oklahoma_3p5_s",
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
    "delay_mae",
    "power_mae",
    "avg_az_mae",
    "avg_el_mae",
    "avg_aod_az_mae",
    "avg_aod_el_mae",
    "interaction_accuracy",
    "interaction_f1",
]


def _load_per_scenario_dir(directory: Path, allowed_scenarios: set[str]) -> pd.DataFrame:
    rows = []
    for csv_path in sorted(directory.glob("*.csv")):
        df = pd.read_csv(csv_path)
        # Some logs contain repeated identical rows; use the final logged row.
        row = df.iloc[-1].copy()
        if row["scenario"] in allowed_scenarios:
            rows.append(row)
    out = pd.DataFrame(rows).sort_values("scenario").reset_index(drop=True)
    out["split"] = out["scenario"].apply(lambda s: "seen" if s in TRAIN_SCENARIOS else "unseen")
    return out


def _summary_rows(df: pd.DataFrame, model_label: str, split_label: str) -> list[dict]:
    rows = []
    for metric in METRICS:
        rows.append(
            {
                "model": model_label,
                "split": split_label,
                "metric": metric,
                "n_scenarios": int(len(df)),
                "mean": float(df[metric].mean()),
                "std_sample": float(df[metric].std(ddof=1)),
            }
        )
    return rows


def build_outputs(
    foundation_skip_dir: Path,
    foundation_finetune_dir: Path,
    scenario_table_csv: Path,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    base = pd.read_csv(scenario_table_csv)
    base = base[base["model_family"] == "first_step_residual_corridor"].copy()
    base = base.sort_values("scenario").reset_index(drop=True)
    base["split"] = base["scenario"].apply(lambda s: "seen" if s in TRAIN_SCENARIOS else "unseen")
    allowed_scenarios = set(base["scenario"].unique())

    foundation_raw = _load_per_scenario_dir(foundation_skip_dir, allowed_scenarios)
    foundation_finetune = _load_per_scenario_dir(foundation_finetune_dir, allowed_scenarios)

    summary_rows = []
    for label, df in [
        ("foundation_raw", foundation_raw),
        ("foundation_finetune", foundation_finetune),
        ("scenario_rag", base),
    ]:
        summary_rows.extend(_summary_rows(df, label, "all"))
        summary_rows.extend(_summary_rows(df[df["split"] == "seen"], label, "seen"))
        summary_rows.extend(_summary_rows(df[df["split"] == "unseen"], label, "unseen"))
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / "foundation_transfer_summary.csv", index=False)

    scenario_compare = foundation_finetune[["scenario", *METRICS]].merge(
        base[["scenario", *METRICS]],
        on="scenario",
        suffixes=("_foundation_finetune", "_scenario_rag"),
    )
    for metric in METRICS:
        ft_col = f"{metric}_foundation_finetune"
        rag_col = f"{metric}_scenario_rag"
        scenario_compare[f"{metric}_delta"] = scenario_compare[ft_col] - scenario_compare[rag_col]
        if metric in {"interaction_accuracy", "interaction_f1"}:
            scenario_compare[f"{metric}_foundation_finetune_better"] = scenario_compare[f"{metric}_delta"] > 0
        else:
            scenario_compare[f"{metric}_foundation_finetune_better"] = scenario_compare[f"{metric}_delta"] < 0
    scenario_compare.to_csv(output_dir / "foundation_finetune_vs_scenario_rag_per_scenario.csv", index=False)

    win_rows = []
    for metric in METRICS:
        better_col = f"{metric}_foundation_finetune_better"
        win_rows.append(
            {
                "metric": metric,
                "n_scenarios": int(len(scenario_compare)),
                "foundation_finetune_better_count": int(scenario_compare[better_col].sum()),
                "foundation_finetune_mean": float(scenario_compare[f"{metric}_foundation_finetune"].mean()),
                "scenario_rag_mean": float(scenario_compare[f"{metric}_scenario_rag"].mean()),
                "mean_delta": float(scenario_compare[f"{metric}_delta"].mean()),
            }
        )
    win_df = pd.DataFrame(win_rows)
    win_df.to_csv(output_dir / "foundation_finetune_vs_scenario_rag_win_counts.csv", index=False)

    unseen_compare = foundation_raw[foundation_raw["split"] == "unseen"][["scenario", *METRICS]].merge(
        base[base["split"] == "unseen"][["scenario", *METRICS]],
        on="scenario",
        suffixes=("_foundation_raw", "_scenario_rag"),
    )
    unseen_compare.to_csv(output_dir / "foundation_raw_vs_scenario_rag_unseen_per_scenario.csv", index=False)

    note_lines = [
        "Verification outputs for the foundation-transfer tables in the Infocom draft.",
        "",
        "Inputs:",
        f"- Raw foundation eval dir: {foundation_skip_dir}",
        f"- Foundation finetune dir: {foundation_finetune_dir}",
        f"- Scenario-trained +RAG table source: {scenario_table_csv}",
        "",
        "Assumptions used in the paper:",
        "- Train scenarios are the 27 scenarios listed in verify_foundation_transfer_tables.py.",
        "- Unseen scenarios are the 4 scenarios present in the 31-scenario main table but absent from that training list.",
        "- If a per-scenario CSV contains duplicate identical rows, the final row is used.",
        "",
        "Generated files:",
        "- foundation_transfer_summary.csv: means/stds for raw foundation, finetuned foundation, and scenario-trained +RAG across all/seen/unseen splits.",
        "- foundation_finetune_vs_scenario_rag_per_scenario.csv: per-scenario comparison between finetuned foundation and scenario-trained +RAG.",
        "- foundation_finetune_vs_scenario_rag_win_counts.csv: win counts and average deltas used in the paper text.",
        "- foundation_raw_vs_scenario_rag_unseen_per_scenario.csv: unseen-scenario raw-foundation versus scenario-trained +RAG values.",
    ]
    (output_dir / "README.txt").write_text("\n".join(note_lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute the foundation-transfer averages used in the paper draft.")
    parser.add_argument(
        "--foundation-skip-dir",
        type=Path,
        default=Path("/home/blessedg/Pathformer/logs/foundation_skip_train_eval_per_scenario"),
    )
    parser.add_argument(
        "--foundation-finetune-dir",
        type=Path,
        default=Path("/home/blessedg/Pathformer/logs/foundation_finetune_per_scenario"),
    )
    parser.add_argument(
        "--scenario-table-csv",
        type=Path,
        default=Path("/home/blessedg/Pathformer/logs/all_scenarios_model_family_mae_long.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/blessedg/Pathformer/logs/paper_foundation_verification"),
    )
    args = parser.parse_args()

    build_outputs(
        foundation_skip_dir=args.foundation_skip_dir,
        foundation_finetune_dir=args.foundation_finetune_dir,
        scenario_table_csv=args.scenario_table_csv,
        output_dir=args.output_dir,
    )
    print(f"Saved verification files to {args.output_dir}")


if __name__ == "__main__":
    main()
