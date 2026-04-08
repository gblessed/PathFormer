import argparse
from pathlib import Path

import deepmimo as dm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dataset.dataloaders import PreTrainMySeqDataLoader


DEFAULT_OUTPUT = Path(__file__).resolve().parent / "cumulative_power_with_timesteps.png"
DEFAULT_SUMMARY = Path(__file__).resolve().parent / "cumulative_power_with_timesteps_summary.csv"
MAX_TIMESTEPS = 25


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot the distribution of cumulative total received power as paths accumulate over timesteps."
    )
    parser.add_argument("scenario", nargs="?", default="city_47_chicago_3p5")
    parser.add_argument("--split", choices=["train", "val", "all"], default="all")
    parser.add_argument("--sort-by", choices=["power", "delay"], default="power")
    parser.add_argument("--pad-value", type=float, default=0.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument(
        "--power-unit",
        choices=["mw", "dbm"],
        default="dbm",
        help="Plot cumulative power either in linear mW or converted back to dBm.",
    )
    parser.add_argument(
        "--fractional",
        action="store_true",
        help="Also plot cumulative power fraction relative to each sample's final total power.",
    )
    return parser.parse_args()


def build_datasets(dataset, split, sort_by, pad_value):
    common = dict(split_by="user", sort_by=sort_by, pad_value=pad_value, normalizers=None, apply_normalizers=[])
    if split == "train":
        return [("train", PreTrainMySeqDataLoader(dataset, train=True, **common))]
    if split == "val":
        return [("val", PreTrainMySeqDataLoader(dataset, train=False, **common))]
    return [
        ("train", PreTrainMySeqDataLoader(dataset, train=True, **common)),
        ("val", PreTrainMySeqDataLoader(dataset, train=False, **common)),
    ]


def scaled_power_to_mw(power_scaled):
    power_dbm = power_scaled / 0.01
    return np.power(10.0, power_dbm / 10.0)


def mw_to_dbm(power_mw):
    return 10.0 * np.log10(np.maximum(power_mw, 1e-12))


def collect_cumulative_power(datasets):
    cumulative_rows = []
    cumulative_power = []
    cumulative_fraction = []

    for split_name, seq_data in datasets:
        for idx in range(len(seq_data)):
            prompts, paths, path_lengths, _, _, _ = seq_data[idx]
            num_valid_paths = int(round(float(path_lengths.item()) * MAX_TIMESTEPS))
            if num_valid_paths <= 0:
                continue

            prompt_np = prompts.cpu().numpy()
            powers_scaled = paths[1 : 1 + num_valid_paths, 1].cpu().numpy().astype(np.float64)
            powers_mw = scaled_power_to_mw(powers_scaled)
            cumulative_mw = np.cumsum(powers_mw)
            total_mw = cumulative_mw[-1]
            # print(f"{total_mw}")
            cumulative_ratio = cumulative_mw / max(total_mw, 1e-12)

            cumulative_power.append(cumulative_mw)
            cumulative_fraction.append(cumulative_ratio)

            for timestep, (cum_mw, frac) in enumerate(zip(cumulative_mw, cumulative_ratio), start=1):
                cumulative_rows.append(
                    {
                        "split": split_name,
                        "sample_idx": idx,
                        "tx_x": float(prompt_np[0]),
                        "tx_y": float(prompt_np[1]),
                        "tx_z": float(prompt_np[2]),
                        "rx_x": float(prompt_np[3]),
                        "rx_y": float(prompt_np[4]),
                        "rx_z": float(prompt_np[5]),
                        "num_valid_paths": num_valid_paths,
                        "timestep": timestep,
                        "cumulative_power_mw": float(cum_mw),
                        "cumulative_power_dbm": float(mw_to_dbm(cum_mw)),
                        "cumulative_power_fraction": float(frac),
                    }
                )

    if not cumulative_rows:
        raise ValueError("No users with at least one valid path were found.")

    return pd.DataFrame(cumulative_rows), cumulative_power, cumulative_fraction


def build_summary(cumulative_df):
    grouped = cumulative_df.groupby("timestep")
    summary_df = grouped.agg(
        num_samples=("timestep", "size"),
        cumulative_power_mw_mean=("cumulative_power_mw", "mean"),
        cumulative_power_mw_median=("cumulative_power_mw", "median"),
        cumulative_power_mw_std=("cumulative_power_mw", "std"),
        cumulative_power_mw_p10=("cumulative_power_mw", lambda x: np.percentile(x, 10)),
        cumulative_power_mw_p90=("cumulative_power_mw", lambda x: np.percentile(x, 90)),
        cumulative_power_dbm_mean=("cumulative_power_dbm", "mean"),
        cumulative_power_dbm_median=("cumulative_power_dbm", "median"),
        cumulative_power_dbm_std=("cumulative_power_dbm", "std"),
        cumulative_power_dbm_p10=("cumulative_power_dbm", lambda x: np.percentile(x, 10)),
        cumulative_power_dbm_p90=("cumulative_power_dbm", lambda x: np.percentile(x, 90)),
        cumulative_power_fraction_mean=("cumulative_power_fraction", "mean"),
        cumulative_power_fraction_median=("cumulative_power_fraction", "median"),
        cumulative_power_fraction_p10=("cumulative_power_fraction", lambda x: np.percentile(x, 10)),
        cumulative_power_fraction_p90=("cumulative_power_fraction", lambda x: np.percentile(x, 90)),
    )
    return summary_df.reset_index()


def plot_cumulative_power(summary_df, args):
    x = summary_df["timestep"].to_numpy()
    if args.power_unit == "mw":
        y_mean = summary_df["cumulative_power_mw_mean"].to_numpy()
        y_median = summary_df["cumulative_power_mw_median"].to_numpy()
        y_p10 = summary_df["cumulative_power_mw_p10"].to_numpy()
        y_p90 = summary_df["cumulative_power_mw_p90"].to_numpy()
        ylabel = "Cumulative Total Received Power (mW)"
    else:
        y_mean = summary_df["cumulative_power_dbm_mean"].to_numpy()
        y_median = summary_df["cumulative_power_dbm_median"].to_numpy()
        y_p10 = summary_df["cumulative_power_dbm_p10"].to_numpy()
        y_p90 = summary_df["cumulative_power_dbm_p90"].to_numpy()
        ylabel = "Cumulative Total Received Power (dBm)"

    nrows = 2 if args.fractional else 1
    fig, axes = plt.subplots(nrows, 1, figsize=(10, 5 * nrows), sharex=True)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    ax = axes[0]
    ax.fill_between(x, y_p10, y_p90, color="#9ecae1", alpha=0.4, label="10th-90th percentile")
    ax.plot(x, y_median, color="#08519c", linewidth=2.5, label="median")
    ax.plot(x, y_mean, color="#cb181d", linewidth=2.0, linestyle="--", label="mean")
    ax.set_ylabel(ylabel)
    ax.set_title("Cumulative total received power vs timestep")
    ax.grid(alpha=0.25)
    ax.legend()

    if args.fractional:
        frac_mean = summary_df["cumulative_power_fraction_mean"].to_numpy()
        frac_median = summary_df["cumulative_power_fraction_median"].to_numpy()
        frac_p10 = summary_df["cumulative_power_fraction_p10"].to_numpy()
        frac_p90 = summary_df["cumulative_power_fraction_p90"].to_numpy()

        ax_frac = axes[1]
        ax_frac.fill_between(x, frac_p10, frac_p90, color="#a1d99b", alpha=0.4, label="10th-90th percentile")
        ax_frac.plot(x, frac_median, color="#238b45", linewidth=2.5, label="median")
        ax_frac.plot(x, frac_mean, color="#756bb1", linewidth=2.0, linestyle="--", label="mean")
        ax_frac.set_ylabel("Cumulative Power Fraction")
        ax_frac.set_xlabel("Timestep")
        ax_frac.set_ylim(0.0, 1.02)
        ax_frac.grid(alpha=0.25)
        ax_frac.legend()
    else:
        ax.set_xlabel("Timestep")

    fig.suptitle(
        f"Cumulative received power distribution\n"
        f"scenario={args.scenario}, split={args.split}, sort_by={args.sort_by}"
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def main():
    args = parse_args()

    dm.download(args.scenario)
    dataset = dm.load(args.scenario)
    datasets = build_datasets(dataset, args.split, args.sort_by, args.pad_value)
    cumulative_df, _, _ = collect_cumulative_power(datasets)
    summary_df = build_summary(cumulative_df)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)

    fig = plot_cumulative_power(summary_df, args)
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    plt.close(fig)

    summary_df.to_csv(args.summary_csv, index=False)

    print(summary_df.to_string(index=False))
    print(f"\nSaved figure to {args.output}")
    print(f"Saved summary to {args.summary_csv}")


if __name__ == "__main__":
    main()
