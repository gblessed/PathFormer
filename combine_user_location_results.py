from pathlib import Path

import pandas as pd


ROOT = Path("/home/blessedg/Pathformer")
LOG_DIR = ROOT / "logs"
LOC_DIR = LOG_DIR / "user_location_comparsions"

LWM_PATH = LOC_DIR / "lwm_user_loc_results_mmwave.csv"
WIFO_PATH = LOC_DIR / "localization_finetune_results_parallel_merged.csv"
MLP_PATH = LOC_DIR / "MLP_location_pred_using_channel_results.csv"
PATHFORMER_PATH = LOC_DIR / "foundation_localization_null_all.csv"


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]
    return df


def _normalize_scenarios(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["scenario"] = df["scenario"].astype(str).str.strip()
    df = df[df["scenario"].notna()]
    df = df[df["scenario"].str.lower() != "nan"]
    return df


def _scenario_from_dataset_path(path_str: str) -> str:
    name = Path(str(path_str)).name
    if name.startswith("_"):
        name = name[1:]
    for suffix in ("_train_data.pt", "_val_data.pt", "_test_data.pt"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def load_results() -> dict:
    mlp = _clean_columns(pd.read_csv(MLP_PATH))
    mlp = mlp.rename(
        columns={
            "mean error": "mean_error",
            "median error": "median_error",
            "90th percentile": "p90_error",
        }
    )
    mlp["scenario"] = mlp["scenario"].map(_scenario_from_dataset_path)
    mlp = mlp[["scenario", "mean_error", "median_error", "p90_error"]]
    mlp = _normalize_scenarios(mlp)

    wifo = _clean_columns(pd.read_csv(WIFO_PATH))
    wifo = wifo[wifo["status"] == "ok"].copy()
    wifo = wifo.rename(
        columns={
            "val_mde_m": "mean_error",
            "val_median_mde_m": "median_error",
            "val_p90_mde_m": "p90_error",
        }
    )
    wifo = wifo[["scenario", "mean_error", "median_error", "p90_error"]]
    wifo = _normalize_scenarios(wifo)

    lwm = _clean_columns(pd.read_csv(LWM_PATH))
    lwm = lwm.rename(
        columns={
            "mde_m": "mean_error",
            "median_mde_m": "median_error",
            "p90_mde_m": "p90_error",
        }
    )
    lwm = lwm[["scenario", "mean_error", "median_error", "p90_error"]]
    lwm = _normalize_scenarios(lwm)

    pathformer = _clean_columns(pd.read_csv(PATHFORMER_PATH))
    pathformer["scenario"] = pathformer["scenarios"]
    pathformer = pathformer.rename(
        columns={
            "mde_m": "mean_error",
            "median_mde_m": "median_error",
            "p90_mde_m": "p90_error",
        }
    )
    pathformer = pathformer[["scenario", "mean_error", "median_error", "p90_error"]]
    pathformer = _normalize_scenarios(pathformer)

    return {
        "lwm_finetune": lwm,
        "wifo_finetune": wifo,
        "mlp": mlp,
        "pathformer": pathformer,
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
                "mean_error": f"mean_error_{prefix}",
                "median_error": f"median_error_{prefix}",
                "p90_error": f"p90_error_{prefix}",
            }
        )
        wide = wide.merge(subset, on="scenario", how="left")

    return wide.sort_values("scenario").reset_index(drop=True)


def main() -> None:
    frames = load_results()
    wide_df = build_common_wide(frames)
    out_path = LOG_DIR / "user_location_prediction_comparison_common_scenarios.csv"
    wide_df.to_csv(out_path, index=False)
    print(f"Saved common-scenario comparison to {out_path}")
    print(f"Common scenarios: {len(wide_df)}")


if __name__ == "__main__":
    main()
