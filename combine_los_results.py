from pathlib import Path

import pandas as pd


ROOT = Path("/home/blessedg/Pathformer")
LOG_DIR = ROOT / "logs"
LOS_DIR = LOG_DIR / "los_comparisons"

FOUNDATION_PATH = LOS_DIR / "los_foundation_all.csv"
SINGLE_PATH = LOS_DIR / "mlp_los.csv"
WIFO_PATH = LOS_DIR / "wifo_los_finetune_results.csv"


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]
    return df


def _normalize_scenario(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["scenario"] = df["scenario"].astype(str).str.strip()
    df = df[df["scenario"].str.lower() != "nan"]
    return df


def _first_matching_column(columns, prefix):
    for col in columns:
        if str(col).strip().startswith(prefix):
            return col
    raise KeyError(f"No column starts with {prefix!r}")


def _extract_tab_prefixed_value(series: pd.Series) -> pd.Series:
    return series.astype(str).str.split("\t").str[0].str.strip()


def load_results() -> dict:
    wifo = _clean_columns(pd.read_csv(WIFO_PATH, index_col=False))
    wifo = wifo[wifo["status"] == "ok"].copy()
    wifo = wifo.rename(
        columns={
            "val_accuracy": "accuracy",
            "val_f1score": "f1",
        }
    )
    wifo = wifo[["scenario", "accuracy", "f1"]]
    wifo = _normalize_scenario(wifo)

    single = _clean_columns(pd.read_csv(SINGLE_PATH))
    single_acc_col = _first_matching_column(single.columns, "test accuracy")
    single_f1_col = _first_matching_column(single.columns, "f1 score")
    single["accuracy"] = pd.to_numeric(single[single_acc_col], errors="coerce")
    single["f1"] = pd.to_numeric(_extract_tab_prefixed_value(single[single_f1_col]), errors="coerce")
    single = single[["scenario", "accuracy", "f1"]]
    single = _normalize_scenario(single)

    foundation = _clean_columns(pd.read_csv(FOUNDATION_PATH))
    foundation = foundation.sort_values(
        ["scenario", "finetuned_test_f1", "finetuned_test_acc", "zero_shot_test_f1", "zero_shot_test_acc"],
        ascending=[True, False, False, False, False],
    )
    foundation = foundation.drop_duplicates(subset=["scenario"], keep="first").copy()

    foundation_finetune = foundation.rename(
        columns={
            "finetuned_test_acc": "accuracy",
            "finetuned_test_f1": "f1",
        }
    )
    foundation_finetune = foundation_finetune[["scenario", "accuracy", "f1"]]
    foundation_finetune = _normalize_scenario(foundation_finetune)

    foundation_zero = foundation.rename(
        columns={
            "zero_shot_test_acc": "accuracy",
            "zero_shot_test_f1": "f1",
        }
    )
    foundation_zero = foundation_zero[["scenario", "accuracy", "f1"]]
    foundation_zero = _normalize_scenario(foundation_zero)

    return {
        "wifo_finetune": wifo,
        "pathformer_multienvironment": foundation_finetune,
        "mlp": single,
        "pathformer_zero_shot": foundation_zero,
    }


def build_common_wide(frames: dict) -> pd.DataFrame:
    common = None
    for df in frames.values():
        scenarios = set(df["scenario"])
        common = scenarios if common is None else common & scenarios
    common = sorted(common)

    wide = pd.DataFrame({"scenario": pd.Series(common, dtype="string")})
    for prefix, df in frames.items():
        subset = df[df["scenario"].isin(common)].copy()
        subset = subset.rename(
            columns={
                "accuracy": f"accuracy_{prefix}",
                "f1": f"f1_{prefix}",
            }
        )
        wide = wide.merge(subset, on="scenario", how="left")

    return wide.sort_values("scenario").reset_index(drop=True)


def main() -> None:
    frames = load_results()
    wide_df = build_common_wide(frames)
    out_path = LOG_DIR / "los_prediction_comparison_common_scenarios.csv"
    wide_df.to_csv(out_path, index=False)
    print(f"Saved common-scenario comparison to {out_path}")
    print(f"Common scenarios: {len(wide_df)}")


if __name__ == "__main__":
    main()
