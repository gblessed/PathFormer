import argparse
import os
import warnings

import deepmimo as dm
import pandas as pd

from dataset.dataloaders import PreTrainMySeqDataLoader

warnings.filterwarnings("ignore", category=UserWarning)


def default_scenarios():
    return [
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
        "city_17_seattle_3p5_s",
        "city_18_denver_3p5",
        "city_19_oklahoma_3p5_s",
        "city_16_sanfrancisco_3p5_lwm",
        "city_23_beijing_3p5",
        "city_31_barcelona_3p5",
        "city_35_san_francisco_3p5",
        "city_47_chicago_3p5",
        "city_89_nairobi_3p5",
        "city_91_xiangyang_3p5",
        "city_92_sãopaulo_3p5",
        "boston5g_3p5",
        "city_86_ankara_3p5",
    ]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Count valid train/test multipaths for each default scenario and total them."
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--sort-by", type=str, default="power", choices=["power", "delay"])
    parser.add_argument("--include-aod", action="store_true")
    parser.add_argument("--csv-log-file", type=str, default=None)
    return parser.parse_args()


def count_valid_multipaths(seq_dataset):
    total_valid_paths = 0
    for idx in range(len(seq_dataset)):
        _, paths, _, _, _, _ = seq_dataset[idx]
        total_valid_paths += max(int(paths.size(0)) - 1, 0)
    return total_valid_paths


def build_split_dataset(dataset, train, args):
    return PreTrainMySeqDataLoader(
        dataset,
        train=train,
        split_by="user",
        train_ratio=args.train_ratio,
        sort_by=args.sort_by,
        normalizers=None,
        apply_normalizers=[],
        pad_value=0,
        include_aod=args.include_aod,
    )


def main():
    args = parse_args()
    rows = []
    total_train_samples = 0
    total_test_samples = 0
    total_train_tokens = 0
    total_test_tokens = 0

    scenarios = default_scenarios()
    print(f"Counting valid multipaths for {len(scenarios)} scenario(s)")

    for scenario in scenarios:
        dataset = dm.load(scenario)
        train_data = build_split_dataset(dataset, train=True, args=args)
        test_data = build_split_dataset(dataset, train=False, args=args)

        train_tokens = count_valid_multipaths(train_data)
        test_tokens = count_valid_multipaths(test_data)
        train_samples = len(train_data)
        test_samples = len(test_data)

        total_train_samples += train_samples
        total_test_samples += test_samples
        total_train_tokens += train_tokens
        total_test_tokens += test_tokens

        row = {
            "scenario": scenario,
            "train_samples": train_samples,
            "test_samples": test_samples,
            "train_valid_multipaths": train_tokens,
            "test_valid_multipaths": test_tokens,
        }
        rows.append(row)
        print(
            f"{scenario}: train_samples={train_samples}, test_samples={test_samples}, "
            f"train_tokens={train_tokens}, test_tokens={test_tokens}"
        )

    total_row = {
        "scenario": "TOTAL",
        "train_samples": total_train_samples,
        "test_samples": total_test_samples,
        "train_valid_multipaths": total_train_tokens,
        "test_valid_multipaths": total_test_tokens,
    }
    rows.append(total_row)

    df = pd.DataFrame(rows)
    print("\nTotals")
    print(
        f"train_samples={total_train_samples}, test_samples={total_test_samples}, "
        f"train_tokens={total_train_tokens}, test_tokens={total_test_tokens}"
    )

    if args.csv_log_file:
        os.makedirs(os.path.dirname(args.csv_log_file) or ".", exist_ok=True)
        df.to_csv(args.csv_log_file, index=False)
        print(f"Saved counts to {args.csv_log_file}")


if __name__ == "__main__":
    main()
