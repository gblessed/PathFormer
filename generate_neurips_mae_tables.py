import csv
from pathlib import Path


LOG_DIR = Path("/home/blessedg/Pathformer/logs")
WIDE_CSV = LOG_DIR / "all_scenarios_model_family_mae_wide.csv"

MAIN_TEX = LOG_DIR / "neurips_main_mae_table.tex"
APP_DELAY_POWER_TEX = LOG_DIR / "neurips_appendix_delay_power_mae_table.tex"
APP_AOD_TEX = LOG_DIR / "neurips_appendix_aod_mae_table.tex"
APP_AOA_TEX = LOG_DIR / "neurips_appendix_aoa_mae_table.tex"
APP_INTERACTION_TEX = LOG_DIR / "neurips_appendix_interaction_table.tex"


def esc(text: str) -> str:
    return text.replace("_", "\\_").replace("ã", "\\~a")


def fmt(value: str) -> str:
    return value if value else "--"


def avg(rows: list[dict], family: str, metric: str) -> tuple[float, float]:
    means = [float(r[f"{family}_{metric}"]) for r in rows if r[f"{family}_{metric}"]]
    stds = [float(r[f"{family}_{metric}_std"]) for r in rows if r[f"{family}_{metric}_std"]]
    return sum(means) / len(means), sum(stds) / len(stds)


def bold_if_best(value: float, candidates: list[float]) -> str:
    formatted = f"{value:.3f}"
    if value == min(candidates):
        return f"\\textbf{{{formatted}}}"
    return formatted


def bold_if_highest(value: float, candidates: list[float]) -> str:
    formatted = f"{value:.3f}"
    if value == max(candidates):
        return f"\\textbf{{{formatted}}}"
    return formatted


def write_main_table(rows: list[dict]) -> None:
    families = [
        ("Direct", "direct"),
        ("Residual", "first_step_residual"),
        ("Corridor", "first_step_residual_corridor"),
    ]
    metrics = [
        ("Delay", "delay_mae"),
        ("Power", "power_mae"),
        ("AoA Az", "avg_az_mae"),
        ("AoA El", "avg_el_mae"),
        ("AoD Az", "avg_aod_az_mae"),
        ("AoD El", "avg_aod_el_mae"),
        ("Int. Acc.", "interaction_accuracy"),
        ("Int. F1", "interaction_f1"),
    ]

    aggregates = {}
    for _, family_key in families:
        aggregates[family_key] = {}
        for _, metric_key in metrics:
            aggregates[family_key][metric_key] = avg(rows, family_key, metric_key)

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Cross-scenario MAE comparison over 32 scenarios. We report the average per-scenario MAE $\\pm$ average per-scenario MAE standard deviation. Lower is better.}",
        "\\label{tab:main_mae_results}",
        "\\scriptsize",
        "\\resizebox{\\columnwidth}{!}{%",
        "\\begin{tabular}{lcccccccc}",
        "\\toprule",
        "Model & Delay & Power & AoA Az & AoA El & AoD Az & AoD El & Int. Acc. & Int. F1 \\\\",
        "\\midrule",
    ]

    for family_label, family_key in families:
        row = [family_label]
        for _, metric_key in metrics:
            mean_candidates = [aggregates[f][metric_key][0] for _, f in families]
            std_candidates = [aggregates[f][metric_key][1] for _, f in families]
            mean_val, std_val = aggregates[family_key][metric_key]
            if metric_key in {"interaction_accuracy", "interaction_f1"}:
                row.append(
                    f"{bold_if_highest(mean_val, mean_candidates)} $\\pm$ {bold_if_best(std_val, std_candidates)}"
                )
            else:
                row.append(
                    f"{bold_if_best(mean_val, mean_candidates)} $\\pm$ {bold_if_best(std_val, std_candidates)}"
                )
        lines.append(" & ".join(row) + " \\\\")

    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "}",
            "\\end{table}",
        ]
    )
    MAIN_TEX.write_text("\n".join(lines) + "\n")


def scenario_rows(rows: list[dict], metric_pairs: list[tuple[str, str]]) -> list[str]:
    out = []
    for row in rows:
        cells = [f"\\texttt{{{esc(row['scenario'])}}}"]
        for metric in metric_pairs:
            for family in [
                "direct",
                "first_step_residual",
                "first_step_residual_corridor",
            ]:
                mean_key = f"{family}_{metric[0]}"
                std_key = f"{family}_{metric[1]}"
                cells.append(f"{fmt(row[mean_key])} $\\pm$ {fmt(row[std_key])}")
        out.append(" & ".join(cells) + " \\\\")
    return out


def write_appendix_table(
    output_path: Path,
    rows: list[dict],
    caption: str,
    label: str,
    group_titles: list[str],
    metric_pairs: list[tuple[str, str]],
) -> None:
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\scriptsize",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{lcccccc}",
        "\\toprule",
        f" & \\multicolumn{{3}}{{c}}{{{group_titles[0]}}} & \\multicolumn{{3}}{{c}}{{{group_titles[1]}}} \\\\",
        "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
        "Scenario & Direct & Residual & Corridor & Direct & Residual & Corridor \\\\",
        "\\midrule",
    ]
    lines.extend(scenario_rows(rows, metric_pairs))
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}%",
            "}",
            "\\end{table*}",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    with WIDE_CSV.open() as f:
        rows = list(csv.DictReader(f))

    write_main_table(rows)
    write_appendix_table(
        APP_DELAY_POWER_TEX,
        rows,
        "Per-scenario delay and power MAE ($\\pm$ MAE std). Missing entries indicate unavailable runs in the consolidated results files.",
        "tab:appendix_delay_power_mae",
        ["Delay", "Power"],
        [("delay_mae", "delay_mae_std"), ("power_mae", "power_mae_std")],
    )
    write_appendix_table(
        APP_AOD_TEX,
        rows,
        "Per-scenario AoD azimuth and elevation MAE ($\\pm$ MAE std).",
        "tab:appendix_aod_mae",
        ["AoD Az", "AoD El"],
        [("avg_aod_az_mae", "avg_aod_az_mae_std"), ("avg_aod_el_mae", "avg_aod_el_mae_std")],
    )
    write_appendix_table(
        APP_AOA_TEX,
        rows,
        "Per-scenario AoA azimuth and elevation MAE ($\\pm$ MAE std).",
        "tab:appendix_aoa_mae",
        ["AoA Az", "AoA El"],
        [("avg_az_mae", "avg_az_mae_std"), ("avg_el_mae", "avg_el_mae_std")],
    )
    write_appendix_table(
        APP_INTERACTION_TEX,
        rows,
        "Per-scenario interaction classification accuracy and F1 ($\\pm$ std). Higher is better.",
        "tab:appendix_interaction",
        ["Interaction Accuracy", "Interaction F1"],
        [("interaction_accuracy", "interaction_accuracy_std"), ("interaction_f1", "interaction_f1_std")],
    )

    for path in [MAIN_TEX, APP_DELAY_POWER_TEX, APP_AOD_TEX, APP_AOA_TEX, APP_INTERACTION_TEX]:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
