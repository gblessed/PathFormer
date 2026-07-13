from pathlib import Path

import pandas as pd


INPUT_CSV = Path("/home/blessedg/Pathformer/logs/autoreg_timing.csv")
OUTPUT_CSV = Path("/home/blessedg/Pathformer/logs/autoreg_timing_summary.csv")


def main():
    df = pd.read_csv(INPUT_CSV)

    numeric_columns = [
        column
        for column in df.columns
        if column not in {"model_family", "checkpoint_path", "scenario"}
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    summary = (
        df.groupby("model_family", as_index=False)
        .agg(
            parameter_count=("parameter_count", "first"),
            max_generate=("max_generate", "first"),
            warmup_batches=("warmup_batches", "first"),
            num_scenarios=("scenario", "count"),
            avg_inference_time_per_sample_ms=("avg_inference_time_per_sample_ms", "mean"),
            avg_codebook_lookup_time_per_sample_ms=("avg_codebook_lookup_time_per_sample_ms", "mean"),
            avg_feature_retrieval_time_per_sample_ms=("avg_feature_retrieval_time_per_sample_ms", "mean"),
        )
    )

    summary["avg_total_time_per_sample_ms"] = (
        summary["avg_inference_time_per_sample_ms"]
        + summary["avg_codebook_lookup_time_per_sample_ms"]
        + summary["avg_feature_retrieval_time_per_sample_ms"]
    )

    summary.to_csv(OUTPUT_CSV, index=False, float_format="%.6f")
    print(f"Wrote {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
