import argparse
import os
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch

from utils.utils import ChannelParameters, MyChannelComputer
from utils.utils import generate_paths_no_env_batch
from multiscenario_direct_training_first_step_residual import generate_paths_first_step_residual_batch
import deepmimo as dm
import zero_shot_channel_estimation_all_families as zsc


import torch

from models import PathDecoder
from multiscenario_direct_training_first_step_residual import (
    FirstStepResidualPathDecoder,
    load_best_checkpoint as load_residual_checkpoint,
)
from utils.utils import count_parameters, load_best_checkpoint as load_direct_checkpoint

warnings.filterwarnings("ignore", category=UserWarning)

DEFAULT_CSV_LOG = "/home/blessedg/Pathformer/logs/pathformer_pilot_geometry_channel_estimation.csv"


def select_pilot_indices(total_entries, pilot_count, seed):
    pilot_count = int(max(1, min(pilot_count, total_entries)))
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(total_entries, size=pilot_count, replace=False))

def select_last_n_pilot_indices(n, shape):
    indices = np.arange(shape[0]* shape[1])
    indices = indices.reshape((shape[0], shape[1]))
    return indices.reshape(-1)




def reconstruct_from_sparse_pilots(basis, channel, pilot_idx, ridge_lambda=1e-4, noisy=False, snr_db=None):
    basis = np.asarray(basis, dtype=np.complex64)
    channel = np.asarray(channel, dtype=np.complex64)
    pilot_idx = np.asarray(pilot_idx, dtype=np.int64)

    if basis.ndim != 2:
        raise ValueError(f"basis must be 2D [n_entries, n_paths], got {basis.shape}")
    if channel.ndim != 1:
        raise ValueError(f"channel must be 1D [n_entries], got {channel.shape}")
    if basis.shape[0] != channel.shape[0]:
        raise ValueError(f"basis/channel entry mismatch: {basis.shape[0]} vs {channel.shape[0]}")

    n_entries, n_paths = basis.shape
    if n_paths == 0:
        return np.zeros(n_entries, dtype=np.complex64), np.zeros(0, dtype=np.complex64)

    pilot_basis = basis[pilot_idx]
    pilot_channel = channel[pilot_idx]
    if noisy and snr_db:
        pilot_channel = add_complex_awgn(pilot_channel, snr_db, np.random.default_rng())
    # print(f"pilot_basis {pilot_basis.shape} pilot_channel {pilot_channel.shape}")
    lhs = pilot_basis.conj().T @ pilot_basis
    lhs = lhs + float(ridge_lambda) * np.eye(n_paths, dtype=np.complex64)
    rhs = pilot_basis.conj().T @ pilot_channel

    try:
        coeffs = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        coeffs = np.linalg.lstsq(lhs, rhs, rcond=None)[0]

    return basis @ coeffs, coeffs


def parse_int_list(value):
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def _finite_path_mask(power, delay, aod_az, aod_el):
    return np.isfinite(power) & np.isfinite(delay) & np.isfinite(aod_az) & np.isfinite(aod_el) & (power > 0)


def _top_power_indices(power, valid_mask, top_k):
    valid_idx = np.flatnonzero(valid_mask)
    if valid_idx.size == 0:
        return valid_idx
    order = np.argsort(power[valid_idx])[::-1]
    return valid_idx[order[: min(top_k, valid_idx.size)]]


def power_scaled_to_linear(power_scaled):
    return 10 ** ((np.clip(power_scaled, -15000.0, 500.0) / 0.01) / 10)


def predicted_inputs_from_generated(generated, support_top_k):
    pred_np = generated.detach().cpu().numpy()
    B, T_gen = pred_np.shape[:2]
    power_all = power_scaled_to_linear(pred_np[:, :, 1])
    delay_all = pred_np[:, :, 0] / 1e6
    phase_all = np.rad2deg(pred_np[:, :, 2])
    aod_az_all = pred_np[:, :, 5]
    aod_el_all = pred_np[:, :, 6]

    power = np.full((B, T_gen), np.nan, dtype=np.float32)
    delay = np.full((B, T_gen), np.nan, dtype=np.float32)
    phase = np.full((B, T_gen), np.nan, dtype=np.float32)
    aod_az = np.full((B, T_gen), np.nan, dtype=np.float32)
    aod_el = np.full((B, T_gen), np.nan, dtype=np.float32)

    for b in range(B):
        valid_mask = _finite_path_mask(power_all[b], delay_all[b], aod_az_all[b], aod_el_all[b])
        valid_mask &= delay_all[b] > 0
        support_idx = _top_power_indices(power_all[b], valid_mask, support_top_k)
        if support_idx.size == 0:
            continue
        n = support_idx.size
        power[b, :n] = power_all[b, support_idx]
        delay[b, :n] = delay_all[b, support_idx]
        phase[b, :n] = phase_all[b, support_idx]
        aod_az[b, :n] = aod_az_all[b, support_idx]
        aod_el[b, :n] = aod_el_all[b, support_idx]

    return power, delay, phase, aod_az, aod_el


def build_geometry_basis(computer, power, delay, aod_az, aod_el, support_top_k):
    valid_mask = _finite_path_mask(power, delay, aod_az, aod_el)
    support_idx = _top_power_indices(power, valid_mask, support_top_k)
    n_paths = int(support_idx.size)
    if n_paths == 0:
        return None, support_idx

    unit_power = np.full((n_paths, n_paths), np.nan, dtype=np.float32)
    unit_delay = np.full((n_paths, n_paths), np.nan, dtype=np.float32)
    zero_phase = np.full((n_paths, n_paths), np.nan, dtype=np.float32)
    basis_aod_az = np.full((n_paths, n_paths), np.nan, dtype=np.float32)
    basis_aod_el = np.full((n_paths, n_paths), np.nan, dtype=np.float32)

    for row, path_idx in enumerate(support_idx):
        unit_power[row, row] = 1.0
        unit_delay[row, row] = delay[path_idx]
        zero_phase[row, row] = 0.0
        basis_aod_az[row, row] = aod_az[path_idx]
        basis_aod_el[row, row] = aod_el[path_idx]

    basis_channels = computer.compute_channels(
        unit_power,
        unit_delay,
        zero_phase,
        basis_aod_az,
        basis_aod_el,
    )
    # print(f"basis_channels.shape: {basis_channels.shape} {(basis_channels.reshape(n_paths, -1).T).shape}")
    return basis_channels.reshape(n_paths, -1).T, support_idx


def nmse_metrics(gt_flat, pred_flat):
    scale = 1e6
    gt_s = gt_flat * scale
    pred_s = pred_flat * scale
    mse = np.mean(np.abs(gt_s - pred_s) ** 2)
    norm = max(float(np.mean(np.abs(gt_s) ** 2)), 1e-6)
    nmse = float(mse / (norm + 1e-10))
    nmse_db = float(10.0 * np.log10(nmse + 1e-10))
    score = float(np.clip(1.0 - ((nmse_db - (-20.0)) / 20.0), 0.0, 1.0))
    return nmse, np.log10(nmse + 1e-10), nmse_db, score

def add_complex_awgn(signal, snr_db, rng):
    signal = np.asarray(signal, dtype=np.complex64)
    signal_power = np.mean(np.abs(signal) ** 2)
    snr_linear = 10 ** (snr_db / 10.0)
    noise_power = signal_power / max(snr_linear, 1e-12)
    noise_std = np.sqrt(noise_power / 2.0)
    noise = noise_std * (
        rng.standard_normal(signal.shape) + 1j * rng.standard_normal(signal.shape)
    )
    return (signal + noise).astype(np.complex64)

def summarize_metric_rows(rows):
    grouped = defaultdict(lambda: {"nmse": [], "log": [], "db": [], "score": []})
    for row in rows:
        key = (row["method"], row["pilot_count"])
        grouped[key]["nmse"].append(row["nmse"])
        grouped[key]["log"].append(row["nmse_log10"])
        grouped[key]["db"].append(row["nmse_db"])
        grouped[key]["score"].append(row["score"])

    summary = []
    for (method, pilot_count), vals in grouped.items():
        summary.append(
            {
                "method": method,
                "pilot_count": pilot_count,
                "nmse_raw": float(np.mean(vals["nmse"])),
                "avg_ch_nmse_log10": float(np.mean(vals["log"])),
                "std_ch_nmse_log10": float(np.std(vals["log"])),
                "avg_ch_nmse_dB": float(np.mean(vals["db"])),
                "std_ch_nmse_dB": float(np.std(vals["db"])),
                "avg_ch_score": float(np.mean(vals["score"])),
                "std_ch_score": float(np.std(vals["score"])),
                "n_eval": len(vals["db"]),
            }
        )
    return summary


def load_model_and_val_loader(dataset, scenario, model_family, checkpoint_path, args, zsc):

    pad_value = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_family == "direct":
        _, val_loader = zsc.build_direct_val_loader(dataset, args.batch_size, pad_value)
        model = PathDecoder(hidden_dim=512, n_layers=8, n_heads=8, include_aod=True).to(device)
        print(f"{model_family} parameters: {count_parameters(model)}")
        load_direct_checkpoint(model, checkpoint_path=checkpoint_path)
    elif model_family == "first_step_residual":
        _, val_loader = zsc.build_residual_val_loader(dataset, args.batch_size, pad_value, args.n_clusters)
        model = FirstStepResidualPathDecoder(prompt_dim=10, hidden_dim=512, n_layers=8, n_heads=8).to(device)
        print(f"{model_family} parameters: {count_parameters(model)}")
        load_residual_checkpoint(model, checkpoint_path)
    elif model_family == "first_step_residual_corridor":
        val_data, val_loader = zsc.build_corridor_val_loader(dataset, args.batch_size, pad_value, args)
        prompt_dim = int(val_data.augmented_prompts[0].numel())
        model = FirstStepResidualPathDecoder(prompt_dim=prompt_dim, hidden_dim=512, n_layers=8, n_heads=8).to(device)
        print(f"{model_family} parameters: {count_parameters(model)}")
        load_residual_checkpoint(model, checkpoint_path)
    else:
        raise ValueError(f"Unknown model family: {model_family}")
    return model, val_loader, pad_value, device


def evaluate_pilot_geometry(model, model_family, val_loader, pad_value, args, zsc):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    computer = MyChannelComputer()
    params = ChannelParameters()
    sample_rows = []
    sample_offset = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Pilot geometry [{model_family}]", leave=False):
            if model_family == "direct":
                prompts, paths, path_lengths, interactions, env, env_prop, path_padding_mask = batch
                prompts = prompts.to(device)
                generated, _, _ = generate_paths_no_env_batch(model, prompts, max_steps=args.max_generate)
            else:
                prompts, paths, path_lengths, interactions, env, env_prop, path_padding_mask, first_step_baselines = batch
                prompts = prompts.to(device)
                first_step_baselines = first_step_baselines.to(device)
                generated, _, _ = generate_paths_first_step_residual_batch(
                    model,
                    prompts,
                    first_step_baselines,
                    max_steps=args.max_generate,
                )

            paths_out = paths[:, 1:, :]
            gt_inputs, _, valid_paths = zsc._batch_channel_inputs(
                paths_out,
                generated,
                path_lengths,
                phase_source="predicted",
                pad_value=pad_value,
            )
            if valid_paths == 0:
                continue
            
            pred_inputs = predicted_inputs_from_generated(generated, args.support_top_k)
            gt_ch = computer.compute_channels(*gt_inputs, params=params)
            print(f"gt_ch {gt_ch.shape}")
            pred_ch = computer.compute_channels(*pred_inputs, params=params)
            B = gt_ch.shape[0]

            for b in range(B):
                n = args.pilot_counts[0]
                shape = gt_ch.shape[2:]
                gt_select = gt_ch[b][:,:, n:]
                pred_select = pred_ch[b][:,:, n:]
                gt_flat = gt_select.reshape(-1)
                pred_flat = pred_select.reshape(-1)



                # gt_flat = gt_ch[b].reshape(-1)
                # pred_flat = pred_ch[b].reshape(-1)

                if not np.any(np.abs(gt_flat) > 0):
                    continue

                nmse, nmse_log, nmse_db, score = nmse_metrics(gt_flat, pred_flat)
                sample_rows.append(
                    {
                        "method": "raw_pathformer_pred_phase",
                        "pilot_count": 0,
                        "nmse": nmse,
                        "nmse_log10": nmse_log,
                        "nmse_db": nmse_db,
                        "score": score,
                    }
                )
                gt_flat = gt_ch[b].reshape(-1)
                pred_flat = pred_ch[b].reshape(-1)
                
                pred_basis, pred_support = build_geometry_basis(
                    computer,
                    pred_inputs[0][b],
                    pred_inputs[1][b],
                    pred_inputs[3][b],
                    pred_inputs[4][b],
                    args.support_top_k,
                )
                # print(f"pred_basis {pred_basis.shape} {gt_flat.shape}")
                oracle_basis = None
                if args.include_oracle_support:
                    oracle_basis, _ = build_geometry_basis(
                        computer,
                        gt_inputs[0][b],
                        gt_inputs[1][b],
                        gt_inputs[3][b],
                        gt_inputs[4][b],
                        args.support_top_k,
                    )

                for pilot_count in args.pilot_counts:
                    if not args.use_last_n:
                        pilot_idx = select_pilot_indices(
                            total_entries=gt_flat.size,
                            pilot_count=pilot_count,
                            seed=args.seed + sample_offset + 1009 * pilot_count,
                        )
                    else:
                        pilot_idx = select_last_n_pilot_indices(n, shape)

                    # pilot_idx = {}
                    if pred_basis is not None:
                        recon, _ = reconstruct_from_sparse_pilots(
                            pred_basis,
                            gt_flat,
                            pilot_idx,
                            ridge_lambda=args.ridge_lambda,
                        )
                        # print(f"coeff {_} ")

                        nmse, nmse_log, nmse_db, score = nmse_metrics(gt_flat, recon)
                        sample_rows.append(
                            {
                                "method": "pilot_ls_pathformer_support",
                                "pilot_count": int(pilot_count),
                                "nmse": nmse,
                                "nmse_log10": nmse_log,
                                "nmse_db": nmse_db,
                                "score": score,
                            }
                        )
                    if oracle_basis is not None:
                        recon, _ = reconstruct_from_sparse_pilots(
                            oracle_basis,
                            gt_flat,
                            pilot_idx,
                            ridge_lambda=args.ridge_lambda,
                        )
                        nmse, nmse_log, nmse_db, score = nmse_metrics(gt_flat, recon)
                        sample_rows.append(
                            {
                                "method": "pilot_ls_oracle_support",
                                "pilot_count": int(pilot_count),
                                "nmse": nmse,
                                "nmse_log10": nmse_log,
                                "nmse_db": nmse_db,
                                "score": score,
                            }
                        )
                    recon, _ = reconstruct_from_sparse_pilots(
                        pred_basis,
                        gt_flat,
                        pilot_idx,
                        ridge_lambda=args.ridge_lambda,
                        noisy = True,
                        snr_db = args.noise_snr_db

                    )
                    nmse, nmse_log, nmse_db, score = nmse_metrics(gt_flat, recon)
                    sample_rows.append(
                        {
                            "method": "noisy_pilot_ls_pathformer_support",
                            "pilot_count": int(pilot_count),
                            "nmse": nmse,
                            "nmse_log10": nmse_log,
                            "nmse_db": nmse_db,
                            "score": score,
                        }
                    )                    

                sample_offset += 1
                if args.max_eval_samples and sample_offset >= args.max_eval_samples:
                    return summarize_metric_rows(sample_rows)

    return summarize_metric_rows(sample_rows)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inference-valid channel estimation: PathFormer geometry support + sparse pilot LS complex gains."
    )
    parser.add_argument("scenarios", nargs="*", help="Optional scenario names to run.")
    parser.add_argument("--scenario", dest="scenario_flag", action="append")
    parser.add_argument("--scenario-file", type=str)
    parser.add_argument("--shard-index", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=None)
    parser.add_argument("--csv-log-file", type=str, default=DEFAULT_CSV_LOG)
    parser.add_argument("--model-family", choices=["direct", "first_step_residual", "first_step_residual_corridor"], default="first_step_residual_corridor")
    parser.add_argument("--checkpoint-root-direct", type=str, default="/home/blessedg/Pathformer/base_no_env")
    parser.add_argument("--checkpoint-root-residual", type=str, default="/home/blessedg/Pathformer/checkpoints_first_step_residual")
    parser.add_argument("--checkpoint-root-corridor", type=str, default="/home/blessedg/Pathformer/checkpoints_first_step_residual_corridor")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-generate", type=int, default=26)
    parser.add_argument("--support-top-k", type=int, default=12)
    parser.add_argument("--pilot-counts", type=parse_int_list, default=[8, 16, 32, 64, 128])
    parser.add_argument("--use-last-n", action= 'store_true')
    parser.add_argument("--ridge-lambda", type=float, default=1e-3)
    parser.add_argument("--max-eval-samples", type=int, default=0)
    parser.add_argument("--include-oracle-support", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--n-clusters", type=int, default=25)
    parser.add_argument("--nearest-k", type=int, default=5)
    parser.add_argument("--corridor-k", type=int, default=5)
    parser.add_argument("--corridor-bins", type=int, default=8)
    parser.add_argument("--use-material-features", action="store_true")
    parser.add_argument("--no-material-features", dest="use_material_features", action="store_false")
    parser.add_argument("--noise-snr-db", type=float, default=20.0)
    parser.set_defaults(use_material_features=True)
    return parser.parse_args()


def checkpoint_for_family(scenario, args):
    if args.model_family == "direct":
        return os.path.join(args.checkpoint_root_direct, f"multiscenario_direct_{scenario}_best_model_checkpoint.pth")
    if args.model_family == "first_step_residual":
        return os.path.join(args.checkpoint_root_residual, f"first_step_residual_{scenario}_best_model_checkpoint.pth")
    return os.path.join(args.checkpoint_root_corridor, f"first_step_residual_corridor_{scenario}_best_model_checkpoint.pth")


def main():

    args = parse_args()
    scenarios = zsc.resolve_scenarios(args) if (args.scenarios or args.scenario_flag or args.scenario_file) else ["city_47_chicago_3p5"]
    os.makedirs(os.path.dirname(args.csv_log_file) or ".", exist_ok=True)
    print(f"Running pilot-geometry channel estimation for {scenarios}")
    print(f"model_family={args.model_family}, support_top_k={args.support_top_k}, pilots={args.pilot_counts}")

    all_rows = []
    for scenario in scenarios:
        checkpoint_path = checkpoint_for_family(scenario, args)
        if not os.path.exists(checkpoint_path):
            print(f"Skipping {scenario}: missing checkpoint {checkpoint_path}")
            continue

        print(f"\nScenario: {scenario}")
        dataset = dm.load(scenario)
        model, val_loader, pad_value, _ = load_model_and_val_loader(
            dataset,
            scenario,
            args.model_family,
            checkpoint_path,
            args,
            zsc,
        )
        rows = evaluate_pilot_geometry(model, args.model_family, val_loader, pad_value, args, zsc)
        for row in rows:
            row.update(
                {
                    "scenario": scenario,
                    "model_family": args.model_family,
                    "checkpoint_path": checkpoint_path,
                    "support_top_k": args.support_top_k,
                    "ridge_lambda": args.ridge_lambda,
                    "max_generate": args.max_generate,
                    "max_eval_samples": args.max_eval_samples,
                }
            )
        if rows:
            pd.DataFrame(rows).to_csv(
                args.csv_log_file,
                mode="a",
                index=False,
                header=not os.path.exists(args.csv_log_file),
            )
            all_rows.extend(rows)
            for row in sorted(rows, key=lambda r: (r["method"], r["pilot_count"])):
                print(
                    f"{scenario} [{row['method']}, pilots={row['pilot_count']}] "
                    f"nmse_dB={row['avg_ch_nmse_dB']:.4f}, score={row['avg_ch_score']:.4f}, n={row['n_eval']}"
                )

    print(f"\nSaved {len(all_rows)} rows to {args.csv_log_file}")


if __name__ == "__main__":
    main()
