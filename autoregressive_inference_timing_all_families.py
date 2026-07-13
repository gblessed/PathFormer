import argparse
import os
import time
import warnings

import deepmimo as dm
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader

from dataset.dataloaders import PreTrainMySeqDataLoader
from models import PathDecoder
from multiscenario_direct_training_first_step_residual import (
    FirstStepResidualDataset,
    FirstStepResidualPathDecoder,
    generate_paths_first_step_residual_batch,
    load_best_checkpoint as load_residual_checkpoint,
    resolve_scenarios,
)
from scene_feature_utils import SceneFeatureBank
from utils.utils import (
    count_parameters,
    generate_paths_no_env_batch,
    load_best_checkpoint as load_direct_checkpoint,
)

warnings.filterwarnings("ignore", category=UserWarning)

csv_log_file = "autoregressive_inference_timing_all_families.csv"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TIMING_COLUMNS = [
    "model_family",
    "checkpoint_path",
    "parameter_count",
    "max_generate",
    "warmup_batches",
    "avg_inference_time_per_sample_sec",
    "avg_inference_time_per_sample_ms",
    "timed_samples",
    "timed_batches",
    "avg_codebook_lookup_time_per_sample_sec",
    "avg_codebook_lookup_time_per_sample_ms",
    "avg_feature_retrieval_time_per_sample_sec",
    "avg_feature_retrieval_time_per_sample_ms",
    "scenario",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run autoregressive generation for all model families and report average inference time."
    )
    parser.add_argument("scenarios", nargs="*", help="Optional scenario names to run.")
    parser.add_argument("--scenario", dest="scenario_flag", action="append")
    parser.add_argument("--scenario-file", type=str)
    parser.add_argument("--shard-index", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=None)
    parser.add_argument("--csv-log-file", type=str, default=csv_log_file)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-generate", type=int, default=25)
    parser.add_argument("--warmup-batches", type=int, default=1)
    parser.add_argument("--n-clusters", type=int, default=25)
    parser.add_argument("--nearest-k", type=int, default=5)
    parser.add_argument("--corridor-k", type=int, default=5)
    parser.add_argument("--corridor-bins", type=int, default=8)
    parser.add_argument("--use-material-features", action="store_true")
    parser.add_argument("--no-material-features", dest="use_material_features", action="store_false")
    parser.add_argument("--checkpoint-root-direct", type=str, default="/home/blessedg/Pathformer/base_no_env")
    parser.add_argument("--checkpoint-root-residual", type=str, default="/home/blessedg/Pathformer/checkpoints_first_step_residual")
    parser.add_argument("--checkpoint-root-corridor", type=str, default="/home/blessedg/Pathformer/checkpoints_first_step_residual_corridor")
    parser.set_defaults(use_material_features=True)
    return parser.parse_args()


def default_scenarios():
    return [
        "city_47_chicago_3p5",
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
        "city_89_nairobi_3p5",
        "city_91_xiangyang_3p5",
        "city_92_sãopaulo_3p5",
        "boston5g_3p5",
        "city_86_ankara_3p5",
        "city_72_capetown_3p5",
        "city_84_baoding_3p5",
        "city_95_delhi_3p5",
        "city_96_osaka_3p5",
        "city_88_tongshan_3p5",
    ]


def checkpoint_specs_for_scenario(args, scenario):
    return [
        {
            "model_family": "direct",
            "backbone_checkpoint_path": os.path.join(
                args.checkpoint_root_direct,
                f"multiscenario_direct_{scenario}_best_model_checkpoint.pth",
            ),
        },
        {
            "model_family": "first_step_residual",
            "backbone_checkpoint_path": os.path.join(
                args.checkpoint_root_residual,
                f"first_step_residual_{scenario}_best_model_checkpoint.pth",
            ),
        },
        {
            "model_family": "first_step_residual_corridor",
            "backbone_checkpoint_path": os.path.join(
                args.checkpoint_root_corridor,
                f"first_step_residual_corridor_{scenario}_best_model_checkpoint.pth",
            ),
        },
    ]


def maybe_sync_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _make_loader(dataset, batch_size, collate_fn, num_workers):
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


def build_direct_val_loader(dataset, args, pad_value):
    val_data = PreTrainMySeqDataLoader(
        dataset,
        train=False,
        split_by="user",
        sort_by="power",
        normalizers=None,
        apply_normalizers=[],
        pad_value=pad_value,
        include_aod=True,
    )
    return val_data, _make_loader(val_data, args.batch_size, val_data.collate_fn, args.num_workers), {
        "avg_codebook_lookup_time_per_sample_sec": 0.0,
        "avg_codebook_lookup_time_per_sample_ms": 0.0,
        "avg_feature_retrieval_time_per_sample_sec": 0.0,
        "avg_feature_retrieval_time_per_sample_ms": 0.0,
    }


def _extract_first_step_metadata(seq_dataset):
    samples = []
    for idx in range(len(seq_dataset)):
        prompt, paths, *_ = seq_dataset[idx]
        tx_key = tuple(prompt[:3].numpy().tolist())
        rx_pos = prompt[3:].numpy().astype(np.float32)
        first_target = paths[1, :2].numpy().astype(np.float32) if paths.shape[0] > 1 else np.zeros(2, dtype=np.float32)
        samples.append({"tx_key": tx_key, "rx_pos": rx_pos, "prompt": prompt.numpy().astype(np.float32), "first_target": first_target})
    return samples


def _compute_cluster_stats(targets, labels, centers):
    stds = np.zeros_like(centers, dtype=np.float32)
    for k in range(centers.shape[0]):
        members = targets[labels == k]
        if len(members) > 0:
            stds[k] = members.std(axis=0).astype(np.float32)
    return stds


def build_residual_val_loader_with_timing(dataset, args, pad_value):
    base_train = PreTrainMySeqDataLoader(
        dataset,
        train=True,
        split_by="user",
        sort_by="power",
        normalizers=None,
        apply_normalizers=[],
        pad_value=pad_value,
        include_aod=True,
    )
    base_val = PreTrainMySeqDataLoader(
        dataset,
        train=False,
        split_by="user",
        sort_by="power",
        normalizers=None,
        apply_normalizers=[],
        pad_value=pad_value,
        include_aod=True,
    )

    train_meta = _extract_first_step_metadata(base_train)
    val_meta = _extract_first_step_metadata(base_val)

    train_groups = {}
    for idx, sample in enumerate(train_meta):
        train_groups.setdefault(sample["tx_key"], {"indices": [], "rx_pos": [], "targets": []})
        train_groups[sample["tx_key"]]["indices"].append(idx)
        train_groups[sample["tx_key"]]["rx_pos"].append(sample["rx_pos"])
        train_groups[sample["tx_key"]]["targets"].append(sample["first_target"])

    val_groups = {}
    for idx, sample in enumerate(val_meta):
        val_groups.setdefault(sample["tx_key"], {"indices": [], "rx_pos": []})
        val_groups[sample["tx_key"]]["indices"].append(idx)
        val_groups[sample["tx_key"]]["rx_pos"].append(sample["rx_pos"])

    val_aug_prompts = [None] * len(val_meta)
    val_baselines = [None] * len(val_meta)
    total_codebook_lookup_sec = 0.0
    val_lookup_samples = 0

    for tx_key, group in train_groups.items():
        rx_pos = np.stack(group["rx_pos"], axis=0).astype(np.float32)
        targets = np.stack(group["targets"], axis=0).astype(np.float32)
        k_eff = min(args.n_clusters, len(targets))
        kmeans = KMeans(n_clusters=k_eff, random_state=42, n_init=10)
        labels = kmeans.fit_predict(targets)
        centers = kmeans.cluster_centers_.astype(np.float32)
        stds = _compute_cluster_stats(targets, labels, centers)

        if tx_key not in val_groups:
            continue

        t0 = time.perf_counter()
        val_rx = np.stack(val_groups[tx_key]["rx_pos"], axis=0).astype(np.float32)
        dists = np.sum((val_rx[:, None, :] - rx_pos[None, :, :]) ** 2, axis=2)
        nearest_train_idx = np.argmin(dists, axis=1)
        assigned_labels = labels[nearest_train_idx]
        for local_idx, dataset_idx in enumerate(val_groups[tx_key]["indices"]):
            sample = val_meta[dataset_idx]
            baseline = centers[assigned_labels[local_idx]]
            std = stds[assigned_labels[local_idx]]
            val_aug_prompts[dataset_idx] = torch.from_numpy(
                np.concatenate([sample["prompt"], baseline, std], axis=0).astype(np.float32)
            )
            val_baselines[dataset_idx] = torch.from_numpy(baseline.astype(np.float32))
        total_codebook_lookup_sec += time.perf_counter() - t0
        val_lookup_samples += len(val_groups[tx_key]["indices"])

    for idx in range(len(val_aug_prompts)):
        if val_aug_prompts[idx] is None:
            sample = val_meta[idx]
            zeros = np.zeros(4, dtype=np.float32)
            val_aug_prompts[idx] = torch.from_numpy(np.concatenate([sample["prompt"], zeros], axis=0).astype(np.float32))
            val_baselines[idx] = torch.zeros(2, dtype=torch.float32)

    val_data = FirstStepResidualDataset(base_val, val_aug_prompts, val_baselines)
    metrics = {
        "avg_codebook_lookup_time_per_sample_sec": total_codebook_lookup_sec / max(val_lookup_samples, 1),
        "avg_codebook_lookup_time_per_sample_ms": 1000.0 * total_codebook_lookup_sec / max(val_lookup_samples, 1),
        "avg_feature_retrieval_time_per_sample_sec": 0.0,
        "avg_feature_retrieval_time_per_sample_ms": 0.0,
    }
    return val_data, _make_loader(val_data, args.batch_size, val_data.collate_fn, args.num_workers), metrics


def _fit_standardizer(tensor_list):
    arr = np.stack([t.numpy() for t in tensor_list], axis=0).astype(np.float32)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def _apply_standardizer(tensor_list, mean, std):
    out = []
    for tensor in tensor_list:
        arr = tensor.numpy().astype(np.float32)
        out.append(torch.from_numpy(((arr - mean) / std).astype(np.float32)))
    return out


def _extract_corridor_metadata(seq_dataset, scene_bank, args, measure_feature_time):
    samples = []
    feature_time_sec = 0.0
    for idx in range(len(seq_dataset)):
        prompt, paths, *_ = seq_dataset[idx]
        tx_pos = np.asarray(seq_dataset.dataset_filtered["tx_pos"][idx], dtype=np.float32)
        rx_pos = np.asarray(seq_dataset.dataset_filtered["rx_pos"][idx], dtype=np.float32)
        tx_key = tuple(tx_pos.tolist())
        first_target = paths[1, :2].numpy().astype(np.float32) if paths.shape[0] > 1 else np.zeros(2, dtype=np.float32)
        if measure_feature_time:
            t0 = time.perf_counter()
        scene_features = scene_bank.build_feature_vector(
            tx_pos=tx_pos,
            rx_pos=rx_pos,
            nearest_k=args.nearest_k,
            corridor_k=args.corridor_k,
            radii=(25.0, 50.0, 100.0),
            corridor_bins=args.corridor_bins,
        )
        if measure_feature_time:
            feature_time_sec += time.perf_counter() - t0
        samples.append(
            {
                "tx_key": tx_key,
                "rx_pos": rx_pos,
                "prompt": prompt.numpy().astype(np.float32),
                "first_target": first_target,
                "scene_features": scene_features,
            }
        )
    return samples, feature_time_sec


def build_corridor_val_loader_with_timing(dataset, args, pad_value):
    base_train = PreTrainMySeqDataLoader(
        dataset,
        train=True,
        split_by="user",
        sort_by="power",
        normalizers=None,
        apply_normalizers=[],
        pad_value=pad_value,
        include_aod=True,
    )
    base_val = PreTrainMySeqDataLoader(
        dataset,
        train=False,
        split_by="user",
        sort_by="power",
        normalizers=None,
        apply_normalizers=[],
        pad_value=pad_value,
        include_aod=True,
    )

    scene_bank = SceneFeatureBank.from_dataset(dataset, use_material_features=args.use_material_features)
    train_meta, _ = _extract_corridor_metadata(base_train, scene_bank, args, measure_feature_time=False)
    val_meta, val_feature_time_sec = _extract_corridor_metadata(base_val, scene_bank, args, measure_feature_time=True)

    train_groups = {}
    for idx, sample in enumerate(train_meta):
        train_groups.setdefault(sample["tx_key"], {"indices": [], "rx_pos": [], "targets": []})
        train_groups[sample["tx_key"]]["indices"].append(idx)
        train_groups[sample["tx_key"]]["rx_pos"].append(sample["rx_pos"])
        train_groups[sample["tx_key"]]["targets"].append(sample["first_target"])

    val_groups = {}
    for idx, sample in enumerate(val_meta):
        val_groups.setdefault(sample["tx_key"], {"indices": [], "rx_pos": []})
        val_groups[sample["tx_key"]]["indices"].append(idx)
        val_groups[sample["tx_key"]]["rx_pos"].append(sample["rx_pos"])

    train_aug_prompts = [None] * len(train_meta)
    val_aug_prompts = [None] * len(val_meta)
    val_baselines = [None] * len(val_meta)
    total_codebook_lookup_sec = 0.0
    val_lookup_samples = 0

    for tx_key, group in train_groups.items():
        rx_pos = np.stack(group["rx_pos"], axis=0).astype(np.float32)
        targets = np.stack(group["targets"], axis=0).astype(np.float32)
        k_eff = min(args.n_clusters, len(targets))
        kmeans = KMeans(n_clusters=k_eff, random_state=42, n_init=10)
        labels = kmeans.fit_predict(targets)
        centers = kmeans.cluster_centers_.astype(np.float32)
        stds = _compute_cluster_stats(targets, labels, centers)

        for local_idx, dataset_idx in enumerate(group["indices"]):
            sample = train_meta[dataset_idx]
            baseline = centers[labels[local_idx]]
            std = stds[labels[local_idx]]
            train_aug_prompts[dataset_idx] = torch.from_numpy(
                np.concatenate([sample["prompt"], baseline, std, sample["scene_features"]], axis=0).astype(np.float32)
            )

        if tx_key not in val_groups:
            continue

        t0 = time.perf_counter()
        val_rx = np.stack(val_groups[tx_key]["rx_pos"], axis=0).astype(np.float32)
        dists = np.sum((val_rx[:, None, :] - rx_pos[None, :, :]) ** 2, axis=2)
        nearest_train_idx = np.argmin(dists, axis=1)
        assigned_labels = labels[nearest_train_idx]
        for local_idx, dataset_idx in enumerate(val_groups[tx_key]["indices"]):
            sample = val_meta[dataset_idx]
            baseline = centers[assigned_labels[local_idx]]
            std = stds[assigned_labels[local_idx]]
            val_aug_prompts[dataset_idx] = torch.from_numpy(
                np.concatenate([sample["prompt"], baseline, std, sample["scene_features"]], axis=0).astype(np.float32)
            )
            val_baselines[dataset_idx] = torch.from_numpy(baseline.astype(np.float32))
        total_codebook_lookup_sec += time.perf_counter() - t0
        val_lookup_samples += len(val_groups[tx_key]["indices"])

    for idx in range(len(train_aug_prompts)):
        if train_aug_prompts[idx] is None:
            sample = train_meta[idx]
            zeros = np.zeros(4, dtype=np.float32)
            train_aug_prompts[idx] = torch.from_numpy(
                np.concatenate([sample["prompt"], zeros, sample["scene_features"]], axis=0).astype(np.float32)
            )

    for idx in range(len(val_aug_prompts)):
        if val_aug_prompts[idx] is None:
            sample = val_meta[idx]
            zeros = np.zeros(4, dtype=np.float32)
            val_aug_prompts[idx] = torch.from_numpy(
                np.concatenate([sample["prompt"], zeros, sample["scene_features"]], axis=0).astype(np.float32)
            )
            val_baselines[idx] = torch.zeros(2, dtype=torch.float32)

    prompt_mean, prompt_std = _fit_standardizer(train_aug_prompts)
    val_aug_prompts = _apply_standardizer(val_aug_prompts, prompt_mean, prompt_std)

    val_data = FirstStepResidualDataset(base_val, val_aug_prompts, val_baselines)
    metrics = {
        "avg_codebook_lookup_time_per_sample_sec": total_codebook_lookup_sec / max(val_lookup_samples, 1),
        "avg_codebook_lookup_time_per_sample_ms": 1000.0 * total_codebook_lookup_sec / max(val_lookup_samples, 1),
        "avg_feature_retrieval_time_per_sample_sec": val_feature_time_sec / max(len(val_meta), 1),
        "avg_feature_retrieval_time_per_sample_ms": 1000.0 * val_feature_time_sec / max(len(val_meta), 1),
    }
    return val_data, _make_loader(val_data, args.batch_size, val_data.collate_fn, args.num_workers), metrics


def _time_generation_call(run_fn, warmup_batches):
    timed_sec = 0.0
    timed_samples = 0
    processed_batches = 0
    for batch_size, fn in run_fn:
        maybe_sync_cuda()
        t0 = time.perf_counter()
        fn()
        maybe_sync_cuda()
        elapsed = time.perf_counter() - t0
        if processed_batches >= warmup_batches:
            timed_sec += elapsed
            timed_samples += batch_size
        processed_batches += 1
    return timed_sec, timed_samples, processed_batches


def collect_direct_timing(model, val_loader, args):
    model.eval()
    calls = []
    with torch.no_grad():
        for prompts, paths, path_lengths, interactions, env, env_prop, path_padding_mask in val_loader:
            prompts = prompts.to(device)
            calls.append(
                (
                    prompts.size(0),
                    lambda prompts=prompts: generate_paths_no_env_batch(
                        model, prompts, max_steps=args.max_generate
                    ),
                )
            )
    total_sec, total_samples, total_batches = _time_generation_call(calls, args.warmup_batches)
    return {
        "avg_inference_time_per_sample_sec": total_sec / max(total_samples, 1),
        "avg_inference_time_per_sample_ms": 1000.0 * total_sec / max(total_samples, 1),
        "timed_samples": total_samples,
        "timed_batches": max(total_batches - args.warmup_batches, 0),
    }


def collect_residual_timing(model, val_loader, args):
    model.eval()
    calls = []
    with torch.no_grad():
        for prompts, paths, path_lengths, interactions, env, env_prop, path_padding_mask, first_step_baselines in val_loader:
            prompts = prompts.to(device)
            first_step_baselines = first_step_baselines.to(device)
            calls.append(
                (
                    prompts.size(0),
                    lambda prompts=prompts, first_step_baselines=first_step_baselines: generate_paths_first_step_residual_batch(
                        model, prompts, first_step_baselines, max_steps=args.max_generate
                    ),
                )
            )
    total_sec, total_samples, total_batches = _time_generation_call(calls, args.warmup_batches)
    return {
        "avg_inference_time_per_sample_sec": total_sec / max(total_samples, 1),
        "avg_inference_time_per_sample_ms": 1000.0 * total_sec / max(total_samples, 1),
        "timed_samples": total_samples,
        "timed_batches": max(total_batches - args.warmup_batches, 0),
    }


def evaluate_model_family(dataset, model_family, checkpoint_path, args):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Missing checkpoint for {model_family}: {checkpoint_path}")

    pad_value = 0
    if model_family == "direct":
        _, val_loader, prep_metrics = build_direct_val_loader(dataset, args, pad_value)
        model = PathDecoder(hidden_dim=512, n_layers=8, n_heads=8, include_aod=True).to(device)
        load_direct_checkpoint(model, checkpoint_path=checkpoint_path)
        timing_metrics = collect_direct_timing(model, val_loader, args)
    elif model_family == "first_step_residual":
        _, val_loader, prep_metrics = build_residual_val_loader_with_timing(dataset, args, pad_value)
        model = FirstStepResidualPathDecoder(
            prompt_dim=10, hidden_dim=512, n_layers=8, n_heads=8, include_aod=True
        ).to(device)
        load_residual_checkpoint(model, checkpoint_path)
        timing_metrics = collect_residual_timing(model, val_loader, args)
    elif model_family == "first_step_residual_corridor":
        val_data, val_loader, prep_metrics = build_corridor_val_loader_with_timing(dataset, args, pad_value)
        prompt_dim = int(val_data.augmented_prompts[0].numel())
        model = FirstStepResidualPathDecoder(
            prompt_dim=prompt_dim, hidden_dim=512, n_layers=8, n_heads=8, include_aod=True
        ).to(device)
        load_residual_checkpoint(model, checkpoint_path)
        timing_metrics = collect_residual_timing(model, val_loader, args)
    else:
        raise ValueError(f"Unknown model family: {model_family}")

    row = {
        "model_family": model_family,
        "checkpoint_path": checkpoint_path,
        "parameter_count": count_parameters(model),
        "max_generate": args.max_generate,
        "warmup_batches": args.warmup_batches,
    }
    row.update(timing_metrics)
    row.update(prep_metrics)
    return row


def main():
    args = parse_args()
    if not args.scenarios and not args.scenario_flag and not args.scenario_file:
        scenarios = default_scenarios()
    else:
        scenarios = resolve_scenarios(args)
    if not scenarios:
        print("No scenarios selected.")
        return

    os.makedirs(os.path.dirname(args.csv_log_file) or ".", exist_ok=True)
    print(f"Running autoregressive timing for {len(scenarios)} scenario(s): {scenarios}")

    for scenario in scenarios:
        print(f"\nAutoregressive timing for {scenario}")
        dataset = dm.load(scenario)
        for spec in checkpoint_specs_for_scenario(args, scenario):
            model_family = spec["model_family"]
            checkpoint_path = spec["backbone_checkpoint_path"]
            if not os.path.exists(checkpoint_path):
                print(f"Skipping {scenario} [{model_family}] because checkpoint is missing: {checkpoint_path}")
                continue

            row = evaluate_model_family(dataset, model_family, checkpoint_path, args)
            row["scenario"] = scenario
            pd.DataFrame([row], columns=TIMING_COLUMNS).to_csv(
                args.csv_log_file,
                mode="a",
                index=False,
                header=not os.path.exists(args.csv_log_file),
            )

            msg = (
                f"{scenario} [{model_family}] | "
                f"inference={row['avg_inference_time_per_sample_ms']:.3f} ms/sample"
            )
            if "avg_codebook_lookup_time_per_sample_ms" in row:
                msg += f", codebook_lookup={row['avg_codebook_lookup_time_per_sample_ms']:.3f} ms/sample"
            if "avg_feature_retrieval_time_per_sample_ms" in row:
                msg += f", feature_retrieval={row['avg_feature_retrieval_time_per_sample_ms']:.3f} ms/sample"
            print(msg)


if __name__ == "__main__":
    main()
