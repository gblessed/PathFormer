# %%
# %%
# !pip install DeepMIMO==4.0.0b10

# %%
# %%
# =============================================================================
# 1. IMPORTS AND WARNINGS SETUP
#    - Load necessary PyTorch modules, utilities, and suppress UserWarnings
# =============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import torch
from tqdm import tqdm
import math
# from utils import (generate_channels_and_labels, tokenizer_train, tokenizer, make_sample, nmse_loss,
                #    create_train_dataloader, patch_maker, count_parameters, train_lwm)
from collections import defaultdict
import numpy as np
# import pretrained_model  # Assuming this contains the LWM model definition
import matplotlib.pyplot as plt
import warnings
import os
import bisect
# from collections import defaultdict
from tqdm import tqdm
warnings.filterwarnings("ignore", category=UserWarning)
# from utils import *
import deepmimo as dm
from sklearn.metrics import mean_squared_error
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from models import PathDecoder, GPTPathDecoder
from dataset.dataloaders import PreTrainMySeqDataLoader
from k_means_utils import *
from models import PathDecoderClusterEncoderAttention
from utils.utils import add_noise_to_paths

from tqdm import tqdm
import torch
import numpy as np
import pandas as pd
import os
from sklearn.cluster import KMeans
import gc
csv_log_file = "cluster_noise_results.csv"

# %%
# scenario = 'city_89_nairobi_3p5'
scenario = 'city_0_newyork_3p5'

dm.download(scenario)
dataset = dm.load(scenario, )

# %%
dataset.scene.plot()


# %%
dm.info()


# %%
config = {
    "BATCH_SIZE":128,
    "PAD_VALUE": 0,
    "USE_WANDB": False,
    "LR":2e-5,
    "epochs" : 100,
    "interaction_weight": 0.01,
    "experiment": f"{scenario}_interacaction_all_inter_str_dec_all_repeat",
    "hidden_dim": 512,
    "n_layers": 8,
    "n_heads": 8,
    "use_delay_kmeans": True,
    "n_clusters": 4,
    "max_path_len_clusters": 26,
    "cluster_features": ["delay", "power"],
    "delay_only_loss": False,
    "TARGET_NOISE_PROB": 0.2,
    "TARGET_NOISE_PARAMS": None,
    # "pretrained_checkpoint": "checkpoints2/noise_enc_direct_city_47_chicago_3p5_interacaction_all_inter_str_dec_all_repeat_best_model_checkpoint.pth",
}




# %%



# %%


# %%
train_data  = PreTrainMySeqDataLoader(dataset, train=True, split_by="user", sort_by="power", pad_value=config["PAD_VALUE"])

train_loader = torch.utils.data.DataLoader(
    dataset     = train_data,
    batch_size  = config['BATCH_SIZE'],
    shuffle     = True,
    collate_fn= train_data.collate_fn
    )
val_data  = PreTrainMySeqDataLoader(dataset, train=False, split_by="user", sort_by="power", pad_value=config["PAD_VALUE"])
val_loader = torch.utils.data.DataLoader(
    dataset     = val_data,
    batch_size  = config['BATCH_SIZE'],
    shuffle     = False,
    collate_fn= val_data.collate_fn
    )

for item in train_loader:
    print(f"Prompt shape: {item[0].shape}, Paths shape: {item[1].shape}, Num paths shape: {item[2].shape}")
    
    break


# %%
print("No. of Train Points   : ", train_data.__len__())
print("Batch Size           : ", config["BATCH_SIZE"])
print("Train Batches        : ", train_loader.__len__())
print("No. of Train Points   : ", val_data.__len__())
print("Val Batches          : ", val_loader.__len__())

def compute_stop_metrics(path_count, targets, pad_value=0):
    """

    Args:

    """

    rmse = np.sqrt(mean_squared_error(path_count.cpu().numpy(), targets.squeeze().cpu().numpy()))
    
    return rmse 



def evaluate_model(model, val_loader, max_generate=26, log_to_wandb=False, pad_value=0, data_stats=None,
                   cluster_lookup_data=None):
    """
    cluster_lookup_data: (train_rx_pos, train_tx_idx, train_cluster_center_std, train_valid_len) for NN lookup by (x,y)
    """
    model.eval()
    device = next(model.parameters()).device

    delay_errors = []
    power_errors = []
    phase_errors = []
    path_length_rmses = []



    delay_maes = []
    power_maes = []
    phase_maes = []
    path_length_maes = []
    # AoA metrics
    az_errors = []
    el_errors = []
    az_maes = []
    el_maes = []
    with torch.no_grad():
        outer_bar = tqdm(val_loader, desc="Evaluating (batches)", leave=True)

        for prompts, paths, path_lengths, interactions, env, env_prop, path_padding_mask in outer_bar:
            prompts = prompts.cuda()
            paths = paths.cuda()
            path_lengths = path_lengths.cuda()
            path_padding_mask = path_padding_mask.cuda()
            # For cluster lookup use tx_idx=0 per sample when not provided (single-Tx)
            batch_tx_idx = torch.zeros(prompts.size(0), dtype=torch.long, device=prompts.device)
         
            # Inner tqdm to show per-sample progress
            inner_bar = tqdm(range(prompts.size(0)), 
                             desc="   Processing samples", 
                             leave=False)


            for b in inner_bar:
                cluster_center_std_b, cluster_pad_b = None, None
                if cluster_lookup_data is not None:
                    prom_b = prompts[b:b+1]
                    tx_b = batch_tx_idx[b:b+1]
                    train_rx_pos, train_tx_idx, train_cluster_center_std, train_valid_len = cluster_lookup_data
                    cluster_center_std_b, cluster_pad_b = lookup_cluster_center_std_by_position(
                        prom_b, tx_b, train_rx_pos, train_tx_idx, train_cluster_center_std, train_valid_len, device
                    )
                generated, path_lengths_pred, inter_str_pred = generate_paths_no_env(
                    model, prompts[b], max_steps=max_generate,
                    cluster_center_std=cluster_center_std_b, cluster_pad_mask=cluster_pad_b
                )
                # print("generated",generated[:])

                generated = generated.cuda()
                # Use path length for valid positions (mask-based; pad value is 0)
                n_valid = int(round(path_lengths[b].item() * 25))
                gt = paths[b][1:1 + n_valid, :5]

                T = min(len(gt), len(generated))
                pred = generated[:T]
                gt = gt[:T]

                delay_pred = pred[:,0]
                delay = gt[:,0]

                power_pred = pred[:,1]
                power = gt[:,1]

                phase_pred = pred[:,2]
                phase = gt[:,2]

                aoa_az_pred = pred[:,3]
                aoa_az = gt[:,3]
              
                aoa_el_pred = pred[:,4]
                aoa_el = gt[:,4]

                # Denormalize when data_stats provided (model was trained on normalized data)
                if data_stats is not None:
                    def denorm(x, key):
                        if key not in data_stats:
                            return x
                        s, m = data_stats[key]["std"], data_stats[key]["mean"]
                        return x * s + m
                    delay_pred = denorm(pred[:,0], "delay")
                    delay = denorm(gt[:,0], "delay")
                    power_pred = denorm(pred[:,1], "power")
                    power = denorm(gt[:,1], "power")
                    phase_pred = denorm(pred[:,2], "phase")
                    phase = denorm(gt[:,2], "phase")
                    aoa_az_pred = denorm(pred[:,3], "aoa_az")
                    aoa_az = denorm(gt[:,3], "aoa_az")
                    aoa_el_pred = denorm(pred[:,4], "aoa_el")
                    aoa_el = denorm(gt[:,4], "aoa_el")


                # ---- Compute Metrics ----
                delay_rmse = torch.mean((delay_pred - delay)**2).sqrt().item()
                delay_mae = torch.mean(torch.abs(delay_pred - delay)).item()

                power_rmse = torch.mean((power_pred/0.01 -power/0.01)**2).sqrt().item()
                power_mae = torch.mean((torch.abs(power_pred/0.01 -power/0.01))).item()


                # Phase errors
                y_hat_angles = (phase_pred / (np.pi/180))
                y_angles = (phase / (np.pi/180))
                phase_circular_dist = (y_hat_angles - y_angles + 180) % 360 - 180
                phase_rmse = torch.mean(phase_circular_dist**2).sqrt().item()
                phase_mae = torch.mean(torch.abs(phase_circular_dist)).item()

                # AoA azimuth errors
                y_hat_az = (aoa_az_pred / (np.pi/180))
                y_az = (aoa_az / (np.pi/180))

                az_circular_dist = (y_hat_az - y_az + 180) % 360 - 180
                az_rmse = torch.mean(az_circular_dist**2).sqrt().item()
                az_mae = torch.mean(torch.abs(az_circular_dist)).item()

                # AoA elevation errors
                y_hat_el = (aoa_el_pred / (np.pi/180))
                y_el = (aoa_el/ (np.pi/180))
                el_circular_dist = (y_hat_el - y_el + 180) % 360 - 180
                el_rmse = torch.mean(el_circular_dist**2).sqrt().item()
                el_mae = torch.mean(torch.abs(el_circular_dist)).item()

                # Path length RMSE (path_lengths in [0,1]; pathcounts raw; use same scale)
                pl_pred = path_lengths_pred.squeeze()
                pl_gt = path_lengths[b].squeeze()
                length_rmse = (torch.mean((pl_pred - pl_gt)**2)).sqrt().item()
                length_mae = (torch.mean(torch.abs(pl_pred - pl_gt))).item()


                # Save metrics
                delay_errors.append(delay_rmse)
                power_errors.append(power_rmse)
                phase_errors.append(phase_rmse)
                path_length_rmses.append(length_rmse)
                # AoA
                az_errors.append(az_rmse)
                el_errors.append(el_rmse)

                delay_maes.append(delay_mae)
                power_maes.append(power_mae)
                phase_maes.append(phase_mae)
                path_length_maes.append(length_mae)
                az_maes.append(az_mae)
                el_maes.append(el_mae)
                # Show live metric values in tqdm
                inner_bar.set_postfix({
                    "delay_rmse": f"{delay_rmse:.3f}",
                    "power_rmse": f"{power_rmse:.3f}",
                    "phase_rmse": f"{phase_rmse:.3f}",
                    "az_rmse": f"{az_rmse:.3f}",
                    "el_rmse": f"{el_rmse:.3f}",
                    "length_rmse": f"{length_rmse:.3f}",
                    "delay_mae": f"{delay_mae:.3f}",
                    "power_mae": f"{power_mae:.3f}",
                    "phase_mae": f"{phase_mae:.3f}",
                    "az_mae": f"{az_mae:.3f}",
                    "el_mae": f"{el_mae:.3f}",
                    "length_mae": f"{length_mae:.3f}"
                })

                # wandb logging
                if log_to_wandb:
                    wandb.log({
                        "test_delay_rmse": delay_rmse,
                        "test_power_rmse": power_rmse,
                        "test_phase_circ_err": phase_rmse,
                        "test_stop_length_rmse": length_rmse,
                        "test_az_rmse": az_rmse,
                        "test_el_rmse": el_rmse,
                        "test_delay_mae": delay_mae,
                        "test_power_mae": power_mae,
                        "test_phase_circ_err_mae": phase_mae,
                        "test_az_mae": az_mae,
                        "test_el_mae": el_mae,
                        "test_stop_length_mae": length_mae,
                    })
            # print("Batch evaluation complete.")
            
            # print("\n================= Up toBATCH EVALUATION RESULTS =================")
            # print(f"Avg Delay RMSE           : {np.mean(delay_errors):.4f} µs")
            # print(f"Avg Power RMSE           : {np.mean(power_errors):.4f} dB")
            # print(f"Avg Phase RMSE           : {np.mean(phase_errors):.4f} degrees")
            # print(f"Avg Path Length RMSE     : {np.mean(path_length_rmses):.4f}")
            # print(f"Avg Delay MAE           : {np.mean(delay_maes):.4f} µs")
            # print(f"Avg Power MAE           : {np.mean(power_maes):.4f} dB")
            # print(f"Avg Phase MAE           : {np.mean(phase_maes):.4f} degrees")
            # print(f"Avg Path Length MAE     : {np.mean(path_length_maes):.4f}")
            # print("============================================================")
            

    # ---- Final Aggregated Results ----
    avg_delay = np.mean(delay_errors)
    avg_power = np.mean(power_errors)
    avg_phase = np.mean(phase_errors)
    avg_az = np.mean(az_errors) if len(az_errors) > 0 else 0.0
    avg_el = np.mean(el_errors) if len(el_errors) > 0 else 0.0
    avg_path_length_rmse = np.mean(path_length_rmses)
   
    avg_delay_mae = np.mean(delay_maes)
    avg_power_mae = np.mean(power_maes)
    avg_phase_mae = np.mean(phase_maes)
    avg_az_mae = np.mean(az_maes)
    avg_el_mae = np.mean(el_maes)
    avg_path_length_mae= np.mean(path_length_maes)

    print("\n=================  Final EVALUATION RESULTS =================")
    print(f"Delay RMSE           : {avg_delay:.4f} µs")
    print(f"Power RMSE           : {avg_power:.4f} dB")
    print(f"Phase RMSE           : {avg_phase:.4f} degrees")
    print(f"AoA Azimuth RMSE     : {avg_az:.4f} degrees")
    print(f"AoA Elevation RMSE   : {avg_el:.4f} degrees")
    print(f"Path Length RMSE     : {avg_path_length_rmse:.4f}")
        
    print(f"Delay MAE           : {avg_delay_mae:.4f} µs")
    print(f"Power MAE           : {avg_power_mae:.4f} dB")
    print(f"Phase MAE           : {avg_phase_mae:.4f} degrees")
    print(f"AoA Azimuth MAE     : {avg_az_mae:.4f} degrees")
    print(f"AoA Elevation MAE   : {avg_el_mae:.4f} degrees")
    print(f"Path Length MAE     : {avg_path_length_mae:.4f}")
    print("=====================================================\n")

    if log_to_wandb:
        wandb.run.summary["test_delay_rmse"] = avg_delay
        wandb.run.summary["test_power_rmse"] = avg_power
        wandb.run.summary["test_phase_circ_err"] = avg_phase
        wandb.run.summary["test_path_length_rmse"] = avg_path_length_rmse
        
        wandb.run.summary["test_delay_mae"] = avg_delay_mae
        wandb.run.summary["test_power_mae"] = avg_power_mae
        wandb.run.summary["test_phase_circ_err_mae"] = avg_phase_mae
        wandb.run.summary["test_path_length_mae"] = avg_path_length_mae

    return avg_delay, avg_power, avg_phase, avg_az, avg_el, avg_path_length_rmse, avg_delay_mae, avg_power_mae, avg_phase_mae,avg_az_mae, avg_el_mae, avg_path_length_mae 




def show_example(model, val_loader, sample_index=0, k=25, plot=True, cluster_lookup_data=None):
    model.eval()
    batch = next(iter(val_loader))
    prompts, paths, path_lengths, interactions = batch[0], batch[1], batch[2], batch[3]
    path_padding_mask = batch[6] if len(batch) > 6 else None
    
    prompts = prompts.cuda()
    paths = paths.cuda()
    path_lengths = path_lengths.cuda()
    device = next(model.parameters()).device
    batch_tx_idx = torch.zeros(prompts.size(0), dtype=torch.long, device=device)
    
    cluster_center_std_b, cluster_pad_b = None, None
    if cluster_lookup_data is not None:
        prom_b = prompts[sample_index:sample_index+1]
        tx_b = batch_tx_idx[sample_index:sample_index+1]
        train_rx_pos, train_tx_idx, train_cluster_center_std, train_valid_len = cluster_lookup_data
        cluster_center_std_b, cluster_pad_b = lookup_cluster_center_std_by_position(
            prom_b, tx_b, train_rx_pos, train_tx_idx, train_cluster_center_std, train_valid_len, device
        )
    pred_paths, path_lengths_pred, inter_str_pred = generate_paths_no_env(
        model, prompts[sample_index], cluster_center_std=cluster_center_std_b, cluster_pad_mask=cluster_pad_b
    )
    

    pred = pred_paths[0]  # (T,3)
    n_valid = int(round(path_lengths[sample_index].item() * 25))
    gt = paths[sample_index][1:1 + n_valid, :3]  # Extract only 3D components (T,3)

    print("\n--- Ground Truth Length {} ".format(len(gt)))

    print("\n--- Model Predict Length (first {} paths) ---".format(path_lengths_pred.item()))
    
    
    print(gt[:k])
    print(pred[:k])

    if plot:
        T = min(len(gt), len(pred))
        # print("len_path", len(pred), "actual = ", T)
        pred = pred[:T]
        gt = gt[:T]

        fig, axs = plt.subplots(3,1, figsize=(10,12))

        axs[0].plot(gt[:,0].cpu(), label="GT Delay", marker='o')
        axs[0].plot(pred[:,0].cpu(), label="Pred Delay", marker='x')
        axs[0].set_title("Path Delay (µs)")
        axs[0].legend()

        axs[1].plot(gt[:,1].cpu()*0.01, label="GT Power", marker='o')
        axs[1].plot(pred[:,1].cpu()*0.01, label="Pred Power", marker='x')
        axs[1].set_title("Path Power dB")
        axs[1].legend()

        axs[2].plot(gt[:,2].cpu()/(np.pi/180), label="GT Phase", marker='o')
        axs[2].plot(pred[:,2].cpu()/(np.pi/180), label="Pred Phase", marker='x')
        axs[2].set_title("Path Phase (degrees)")

        axs[2].legend()

        plt.tight_layout()
        plt.show()


# def evaluate_generation(val_loader, n_samples=3):
#     model.eval()
#     for i, (prompts, paths) in enumerate(val_loader):
#         if i >= n_samples:
#             break
#         pred, path_lengths_pred = generate_paths_no_env(model, prompts[0])  # autoregressive generation
#         print(f"path lengths pred: {path_lengths_pred[0]}")
#         print(f"\nSample {i}")
#         print("GT paths (first 5):")
#         print(paths[0][:5])
#         print("Predicted paths (first 5):")
#         print(pred[0][:5])

# # %%
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _convert_feature_value(key, value):
    if key == "delay":
        return float(value) * 1e6
    if key == "power":
        return float(value) * 0.01
    if key in ["phase", "aoa_az", "aoa_el"]:
        return float(value) * (np.pi / 180.0)
    return float(value)


def _normalize_feature_value(dataloader_obj, key, value):
    if getattr(dataloader_obj, "normalizers", None) and getattr(dataloader_obj, "apply_normalizers", None):
        if key in dataloader_obj.apply_normalizers:
            stat = dataloader_obj.normalizers.get(key, None)
            if stat is not None:
                return (value - stat["mean"]) / (stat["std"] + 1e-8)
    return value


def _sorted_path_indices(df, idx, sort_by):
    delays_raw = np.array(df["delay"][idx])
    powers_raw = np.array(df["power"][idx])
    if sort_by == "power":
        return np.argsort(-powers_raw)
    return np.argsort(delays_raw)


def compute_feature_kmeans_cluster_stats(dataloader_obj, feature_keys, max_path_len=26, n_clusters=4, random_state=42):
    """
    Cluster selected path features at each path step, per Tx.
    Returns:
      centers: (n_tx, max_path_len, n_clusters, F)
      stds:    (n_tx, max_path_len, n_clusters, F)
    """
    df = dataloader_obj.dataset_filtered
    n_samples = len(df["delay"])
    n_features = len(feature_keys)
    if n_samples == 0:
        z = np.zeros((1, max_path_len, n_clusters, n_features), dtype=np.float32)
        return z, z.copy()

    tx_indices = np.array(df.get("tx_idx", [0] * n_samples))
    if tx_indices.size == 0:
        tx_indices = np.zeros(n_samples, dtype=np.int64)
    n_tx = max(1, int(tx_indices.max()) + 1)

    step_vectors = [[[] for _ in range(max_path_len)] for _ in range(n_tx)]
    for idx in range(n_samples):
        tx = int(tx_indices[idx])
        indices = _sorted_path_indices(df, idx, dataloader_obj.sort_by)
        step = 0
        for path_idx in indices:
            if step >= max_path_len:
                break
            vec = []
            broken = False
            for k in feature_keys:
                v = np.array(df[k][idx])[path_idx]
                if np.isnan(v):
                    broken = True
                    break
                vv = _convert_feature_value(k, v)
                vv = _normalize_feature_value(dataloader_obj, k, vv)
                vec.append(vv)
            if broken:
                break
            step_vectors[tx][step].append(np.array(vec, dtype=np.float32))
            step += 1

    centers = np.zeros((n_tx, max_path_len, n_clusters, n_features), dtype=np.float32)
    stds = np.zeros((n_tx, max_path_len, n_clusters, n_features), dtype=np.float32)

    for tx in range(n_tx):
        for t in range(max_path_len):
            arr = np.array(step_vectors[tx][t], dtype=np.float32)
            if len(arr) == 0:
                continue
            if len(arr) < n_clusters:
                reps = [arr[min(i, len(arr) - 1)] for i in range(n_clusters)]
                c = np.stack(reps, axis=0)
                s = np.stack([np.std(arr, axis=0)] * n_clusters, axis=0)
            else:
                km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
                labels = km.fit_predict(arr)
                c = km.cluster_centers_.astype(np.float32)
                s = np.zeros_like(c)
                for k in range(n_clusters):
                    pts = arr[labels == k]
                    if len(pts) > 0:
                        s[k] = np.std(pts, axis=0)
            centers[tx, t] = c
            stds[tx, t] = s
    return centers, stds


def precompute_train_cluster_center_std_sequences(dataloader_obj, cluster_centers, cluster_stds, feature_keys, max_path_len=26):
    """
    For each training sample, select nearest cluster center per step and concatenate center+std.
    Returns:
      train_rx_pos (N,2), train_tx_idx (N,), train_cluster_center_std (N,max_path_len,2F), train_valid_len (N,)
    """
    df = dataloader_obj.dataset_filtered
    n_samples = len(df["delay"])
    mins = np.array(dataloader_obj.mins)
    maxs = np.array(dataloader_obj.maxs)
    norm = getattr(dataloader_obj, "normalizers", None)
    apply_norm = getattr(dataloader_obj, "apply_normalizers", None) or []
    n_features = len(feature_keys)

    train_rx_pos = []
    train_tx_idx = []
    train_center_std = []
    valid_lens = []

    for idx in range(n_samples):
        tx = int(df["tx_idx"][idx])
        rx_raw = np.array(df["rx_pos"][idx], dtype=np.float32)
        rx_xy = rx_raw[:2]
        if norm and "pos" in apply_norm:
            rx_xy = (rx_xy - norm["rx_pos"]["mean"][:2]) / (norm["rx_pos"]["std"][:2] + 1e-8)
        else:
            rx_xy = (rx_xy - mins[:2]) / (maxs[:2] - mins[:2] + 1e-8)
        train_rx_pos.append(rx_xy)
        train_tx_idx.append(tx)

        seq = np.zeros((max_path_len, 2 * n_features), dtype=np.float32)
        indices = _sorted_path_indices(df, idx, dataloader_obj.sort_by)
        step = 0
        for path_idx in indices:
            if step >= max_path_len:
                break
            vec = []
            broken = False
            for k in feature_keys:
                v = np.array(df[k][idx])[path_idx]
                if np.isnan(v):
                    broken = True
                    break
                vv = _convert_feature_value(k, v)
                vv = _normalize_feature_value(dataloader_obj, k, vv)
                vec.append(vv)
            if broken:
                break
            vec = np.array(vec, dtype=np.float32)
            cc = cluster_centers[tx, step]
            dist = np.linalg.norm(cc - vec[None, :], axis=1)
            nn = int(np.argmin(dist))
            seq[step, :n_features] = cluster_centers[tx, step, nn]
            seq[step, n_features:] = cluster_stds[tx, step, nn]
            step += 1
        valid_lens.append(step)
        train_center_std.append(seq)

    return (
        np.array(train_rx_pos, dtype=np.float32),
        np.array(train_tx_idx, dtype=np.int64),
        np.array(train_center_std, dtype=np.float32),
        np.array(valid_lens, dtype=np.int64),
    )


def lookup_cluster_center_std_by_position(prompts, tx_idx, train_rx_pos, train_tx_idx, train_cluster_center_std, train_valid_len, device, prompt_rx_slice=(3, 5)):
    """
    NN lookup by (x,y): retrieve per-user cluster center+std sequence.
    """
    B = prompts.shape[0]
    rx_query = prompts[:, prompt_rx_slice[0]:prompt_rx_slice[1]].detach().cpu().numpy()
    max_T = train_cluster_center_std.shape[1]
    D = train_cluster_center_std.shape[2]
    batch = np.zeros((B, max_T, D), dtype=np.float32)
    pad_mask = np.ones((B, max_T), dtype=bool)
    for b in range(B):
        tx_b = int(tx_idx[b].item()) if torch.is_tensor(tx_idx[b]) else int(tx_idx[b])
        pos_b = rx_query[b]
        mask = (train_tx_idx == tx_b)
        if not np.any(mask):
            continue
        pos_subset = train_rx_pos[mask]
        idx_subset = np.where(mask)[0]
        dist = np.sqrt(np.sum((pos_subset - pos_b) ** 2, axis=1))
        nn_idx = idx_subset[np.argmin(dist)]
        batch[b] = train_cluster_center_std[nn_idx]
        valid = train_valid_len[nn_idx]
        pad_mask[b, :valid] = False
        pad_mask[b, valid:] = True
    return (
        torch.tensor(batch, dtype=torch.float32, device=device),
        torch.tensor(pad_mask, dtype=torch.bool, device=device),
    )


class PathDecoderClusterEncoderAdapter(nn.Module):
    """
    Learns center+std embeddings and injects them into encoder conditioning via prompt augmentation.
    """
    def __init__(self, backbone, hidden_dim, cluster_feature_dim, prompt_dim=6):
        super().__init__()
        self.backbone = backbone
        self.hidden_dim = hidden_dim
        self.cluster_feature_dim = cluster_feature_dim
        self.cluster_embed = nn.Sequential(
            nn.Linear(cluster_feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.cluster_to_prompt = nn.Linear(hidden_dim, prompt_dim)

    def forward(self, prompts, paths, interactions, cluster_center_std=None, cluster_pad_mask=None):
        B = prompts.size(0)
        if cluster_center_std is None:
            cluster_summary = torch.zeros(B, self.hidden_dim, device=prompts.device, dtype=prompts.dtype)
        else:
            emb = self.cluster_embed(cluster_center_std)  # (B, Tc, H)
            if cluster_pad_mask is not None:
                valid = (~cluster_pad_mask).float().unsqueeze(-1)
                denom = valid.sum(dim=1).clamp_min(1.0)
                cluster_summary = (emb * valid).sum(dim=1) / denom
            else:
                cluster_summary = emb.mean(dim=1)
        prompts_aug = prompts + self.cluster_to_prompt(cluster_summary)
        return self.backbone(prompts_aug, paths, interactions)




def generate_paths_no_env(model, prompt, max_steps=25, stop_threshold=0.5, cluster_center_std=None, cluster_pad_mask=None):
    """
    Local autoregressive generation with optional cluster center+std conditioning.
    """
    model.eval()
    device = next(model.parameters()).device
    prompt = prompt.unsqueeze(0).to(device)
    cur = torch.zeros(1, 1, 5, device=device)
    inter_str = -1 * torch.ones(1, 1, 4, device=device)
    outputs = []
    outputs_inter_str = []
    for _ in range(max_steps):
        d, p, s, c, ph, az_s, az_c, az, el_s, el_c, el, pathcounts, inter_str_logits = model(
            prompt, cur, inter_str, cluster_center_std=cluster_center_std, cluster_pad_mask=cluster_pad_mask
        )
        d_t = d[:, -1]
        p_t = p[:, -1]
        ph_t = ph[:, -1]
        az_t = az[:, -1]
        el_t = el[:, -1]
        inter_logits_t = inter_str_logits[:, -1]
        inter_pred_t = (torch.sigmoid(inter_logits_t) > 0.5).float()
        outputs.append(torch.stack([d_t, p_t, ph_t, az_t, el_t], dim=-1))
        outputs_inter_str.append(inter_pred_t)
        next_path = torch.stack([d_t, p_t, ph_t, az_t, el_t], dim=-1).unsqueeze(1)
        cur = torch.cat([cur, next_path], dim=1)
        inter_str = torch.cat([inter_str, inter_pred_t.unsqueeze(1)], dim=1)
    return (torch.stack(outputs, dim=1).squeeze(0).detach().cpu(),
            pathcounts,
            torch.stack(outputs_inter_str, dim=1).squeeze(0).detach().cpu())


def train_with_interactions(model, train_loader, val_loader, config, train_data, task=None, cluster_lookup_data=None):
    """
    cluster_lookup_data: (train_rx_pos, train_tx_idx, train_cluster_center_std, train_valid_len) for NN lookup by (x,y)
    """
    device = next(model.parameters()).device
    best_val_loss = float('inf')


    for epoch in range(config["epochs"]):
        # -------------------- TRAINING --------------------
        model.train()
        train_losses = []
        train_loss_delay = []
        train_loss_power = []
        train_loss_phase = []
        train_loss_path_length = []
        train_loss_interaction = []  # NEW
        train_path_length_rmse = []
        train_loss_az = []
        train_loss_el = []
        train_ch_nmse = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]", leave=False)
        for prompts, paths, path_lengths, interactions, env, env_prop, path_padding_mask in pbar:
            prompts = prompts.cuda()
            paths = paths.cuda()
            path_lengths = path_lengths.cuda()
            interactions = interactions.cuda()
            path_padding_mask = path_padding_mask.cuda()
            batch_tx_idx = torch.zeros(prompts.size(0), dtype=torch.long, device=prompts.device)
            
            cluster_center_std, cluster_pad_mask = None, None
            if cluster_lookup_data is not None:
                train_rx_pos, train_tx_idx, train_cluster_center_std, train_valid_len = cluster_lookup_data
                cluster_center_std, cluster_pad_mask = lookup_cluster_center_std_by_position(
                    prompts, batch_tx_idx, train_rx_pos, train_tx_idx, train_cluster_center_std, train_valid_len, device
                )

            paths_in = paths[:, :-1, :]
            p_noise = config.get("TARGET_NOISE_PROB", 0.0)
            if p_noise > 0:
                paths_in = add_noise_to_paths(paths_in, path_padding_mask[:, :-1], p_noise=p_noise,
                                              noise_params=config.get("TARGET_NOISE_PARAMS"))
            interactions_in = interactions[:, :-1, :]

            paths_out = paths[:, 1:, :]
            interactions_out = interactions[:, 1:, :]

            (delay_pred, power_pred, phase_sin_pred, phase_cos_pred, phase_pred,
             az_sin_pred, az_cos_pred, az_pred, el_sin_pred, el_cos_pred, el_pred,
             path_length_pred, interaction_logits) = model(
                prompts, paths_in, interactions_in,
                cluster_center_std=cluster_center_std, cluster_pad_mask=cluster_pad_mask
            )

            (total_loss, loss_delay, loss_power, loss_phase,
             loss_az, loss_el, loss_path_length, loss_interaction,loss_channel) = masked_loss(
                delay_pred, power_pred, phase_sin_pred, phase_cos_pred,phase_pred,
                az_sin_pred, az_cos_pred, az_pred, el_sin_pred, el_cos_pred,el_pred,
                path_length_pred, interaction_logits, paths_out, path_lengths,
                interactions_out, finetune=task, pad_value=train_data.pad_value,
                interaction_weight=config.get("interaction_weight", 0.1),
                delay_only=config.get("delay_only_loss", False),
                path_padding_mask=path_padding_mask
            )
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            path_length_rmse = compute_stop_metrics(path_length_pred.detach().squeeze(-1), 
                                                    path_lengths)
            ch_nmse = 0
            if epoch >= 0:
                pass
                # pred_power_linear = 10**( ((power_pred.cpu().detach().numpy())/0.01)/10)
                # pred_delay_secs = delay_pred.cpu().detach().numpy()/ 1e6


                # delay_t = paths_out[:, :, 0].cpu().detach().numpy()
                # power_t = paths_out[:, :, 1].cpu().detach().numpy()
                # phase = paths_out[:, :, 2].cpu().detach().numpy()
                # az = paths_out[:, :, 3].cpu().detach().numpy()
                # el = paths_out[:, :, 4].cpu().detach().numpy()
                # power_linear = 10**( (power_t/0.01)/10)
                # delay_secs = delay_t/ 1e6

                # predicted_channels = mycomputer.compute_channels(pred_power_linear,pred_delay_secs, phase_pred.cpu().detach().numpy(), az_pred.cpu().detach().numpy(), el_pred.cpu().detach().numpy(),kwargs=None  )
                # gt_channels = mycomputer.compute_channels(power_linear,delay_secs, phase, az, el ,kwargs=None )
   


                # ch_nmse = compute_channel_nmse(predicted_channels, gt_channels)
            train_ch_nmse.append(ch_nmse)
            train_losses.append(total_loss.item())
            train_loss_delay.append(loss_delay.item())
            train_loss_power.append(loss_power.item())
            train_loss_phase.append(loss_phase.item())
            train_loss_path_length.append(loss_path_length.item())
            # track aoa losses
            # if 'train_loss_az' not in locals():
            #     train_loss_az = []
            #     train_loss_el = []
            train_loss_az.append(loss_az.item())
            train_loss_el.append(loss_el.item())
            train_loss_interaction.append(loss_interaction.item())  # NEW
            train_path_length_rmse.append(path_length_rmse)
            current_lr = optimizer.param_groups[0]["lr"]
            pbar.set_postfix({
                "loss": f"{total_loss.item():.4f}",
                "delay": f"{loss_delay.item():.4f}",
                
                "power": f"{loss_power.item():.4f}",
                "phase": f"{loss_phase.item():.4f}",
                "az": f"{loss_az.item():.4f}",
                "el": f"{loss_el.item():.4f}",
                "inter": f"{loss_interaction.item():.4f}",  # NEW
                "path_rmse": f"{path_length_rmse:.4f}",
                "ch_nmse":f"{ch_nmse:.4f}",
                "lr": f"{current_lr:.2e}"
            })
        scheduler.step()
        try:
            attn = getattr(model.decoder.layers[-1], "cross_attn_weights", None) if hasattr(model, "decoder") else None
        except Exception:
            attn = None
        # print("Train delay_pred->",delay_pred[0])
        # print("Train actual->",paths_out[0, :, 0])
        avg_train_loss = np.mean(train_losses)
        avg_train_delay = np.mean(train_loss_delay)
        avg_train_power = np.mean(train_loss_power)
        avg_train_phase = np.mean(train_loss_phase)
        avg_train_az = np.mean(train_loss_az) 
        avg_train_el = np.mean(train_loss_el)
        avg_train_path_length = np.mean(train_loss_path_length)
        avg_train_interaction = np.mean(train_loss_interaction)  # NEW
        avg_train_path_length_rmse = np.mean(train_path_length_rmse)

        # -------------------- VALIDATION --------------------
        model.eval()
        val_losses = []
        val_loss_delay = []
        val_loss_power = []
        val_loss_phase = []
        val_loss_path_length = []
        val_loss_interaction = []  # NEW
        val_path_length_rmse = []
        val_loss_az = []
        val_loss_el = []

        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]", leave=False)
            # prepare val aoa loss lists

            for prompts, paths, path_lengths, interactions, env, env_prop, path_padding_mask in pbar:
                prompts = prompts.cuda()
                paths = paths.cuda()
                path_lengths = path_lengths.cuda()
                interactions = interactions.cuda()
                path_padding_mask = path_padding_mask.cuda()
                batch_tx_idx = torch.zeros(prompts.size(0), dtype=torch.long, device=prompts.device)

                cluster_center_std, cluster_pad_mask = None, None
                if cluster_lookup_data is not None:
                    train_rx_pos, train_tx_idx, train_cluster_center_std, train_valid_len = cluster_lookup_data
                    cluster_center_std, cluster_pad_mask = lookup_cluster_center_std_by_position(
                        prompts, batch_tx_idx, train_rx_pos, train_tx_idx, train_cluster_center_std, train_valid_len, device
                    )

                paths_in = paths[:, :-1, :]
                interactions_in = interactions[:, :-1, :]

                paths_out = paths[:, 1:, :]
                interactions_out = interactions[:, 1:, :]

                (delay_pred, power_pred, phase_sin_pred, phase_cos_pred, phase_pred,
                 az_sin_pred, az_cos_pred, az_pred, el_sin_pred, el_cos_pred, el_pred,
                 path_length_pred, interaction_logits) = model(
                    prompts, paths_in, interactions_in,
                    cluster_center_std=cluster_center_std, cluster_pad_mask=cluster_pad_mask
                )
                
                try:
                    if hasattr(model, "decoder") and hasattr(model.decoder, "layers"):
                        attn = getattr(model.decoder.layers[-1], "cross_attn_weights", None)
                    elif hasattr(model, "backbone") and hasattr(model.backbone, "decoder"):
                        attn = getattr(model.backbone.decoder.layers[-1], "cross_attn_weights", None)
                    else:
                        attn = None
                except Exception:
                    attn = None

                (total_loss, loss_delay, loss_power, loss_phase,
                loss_az, loss_el, loss_path_length, loss_interaction,loss_channel) = masked_loss(
                    delay_pred, power_pred, phase_sin_pred, phase_cos_pred,phase_pred,
                    az_sin_pred, az_cos_pred, az_pred, el_sin_pred, el_cos_pred,el_pred,
                    path_length_pred, interaction_logits, paths_out, path_lengths,
                    interactions_out, finetune=task, pad_value=train_data.pad_value,
                    interaction_weight=config.get("interaction_weight", 0.1),
                    delay_only=config.get("delay_only_loss", False),
                    path_padding_mask=path_padding_mask
                )

                path_length_rmse = compute_stop_metrics(path_length_pred.detach().squeeze(-1), 
                                                       path_lengths)

                val_losses.append(total_loss.item())
                val_loss_delay.append(loss_delay.item())
                val_loss_power.append(loss_power.item())
                val_loss_phase.append(loss_phase.item())
                val_loss_az.append(loss_az.item())
                val_loss_el.append(loss_el.item())
                val_loss_path_length.append(loss_path_length.item())
                val_loss_interaction.append(loss_interaction.item())  # NEW
                val_path_length_rmse.append(path_length_rmse)

                pbar.set_postfix({
                    "val_loss": f"{total_loss.item():.4f}",
                    "inter": f"{loss_interaction.item():.4f}",  # NEW
                })
        # print("Val delay_pred->",delay_p 

        
        # print("val attn->",attn[0, 1,])
        avg_val_loss = np.mean(val_losses)
        avg_val_delay = np.mean(val_loss_delay)
        avg_val_power = np.mean(val_loss_power)
        avg_val_phase = np.mean(val_loss_phase)
        avg_val_az = np.mean(val_loss_az) 
        avg_val_el = np.mean(val_loss_el)
        avg_val_path_length = np.mean(val_loss_path_length)
        avg_val_interaction = np.mean(val_loss_interaction)  # NEW
        avg_val_path_length_rmse = np.mean(val_path_length_rmse)

        # scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        # -------------------- CHECKPOINT SAVING --------------------
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': torch.tensor(best_val_loss),
            }, checkpoint_path)
            print(f"  ✓ Best checkpoint saved (val_loss: {best_val_loss:.4f})")

        if config.get("USE_WANDB", False):
            import wandb
            wandb.log({
                "train_loss": avg_train_loss,
                "train_loss_delay": avg_train_delay,
                "train_loss_power": avg_train_power,
                "train_loss_phase": avg_train_phase,
                "train_loss_az": avg_train_az,
                "train_loss_el": avg_train_el,
                "train_loss_path_length": avg_train_path_length,
                "train_loss_interaction": avg_train_interaction,  # NEW
                "train_path_length_rmse": avg_train_path_length_rmse,

                "val_loss": avg_val_loss,
                "val_loss_delay": avg_val_delay,
                "val_loss_power": avg_val_power,
                "val_loss_phase": avg_val_phase,
                "val_loss_az": avg_val_az,
                "val_loss_el": avg_val_el,
                "val_loss_path_length": avg_val_path_length,
                "val_loss_interaction": avg_val_interaction,  # NEW
                "val_path_length_rmse": avg_val_path_length_rmse,
                "epoch": epoch,
                "lr": current_lr,
            })

        print(f"\nEpoch {epoch:02d}")
        print(f"  Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"    Delay: {avg_train_delay:.4f} (val: {avg_val_delay:.4f})")
        print(f"    Power: {avg_train_power:.4f} (val: {avg_val_power:.4f})")
        print(f"    Phase: {avg_train_phase:.4f} (val: {avg_val_phase:.4f})")
        print(f"    Az: {avg_train_az:.4f} (val: {avg_val_az:.4f})")
        print(f"    El: {avg_train_el:.4f} (val: {avg_val_el:.4f})")

        print(f"    Interaction: {avg_train_interaction:.4f} (val: {avg_val_interaction:.4f})")  # NEW
        print(f"    PathLength: {avg_train_path_length:.4f} (val: {avg_val_path_length:.4f})")  # NEW

        print(f"  LR: {current_lr:.3e}")

# model = PathDecoder().to(device)




# %%
if config["USE_WANDB"]:
    import wandb

    wandb.init(
        project="deepmimo-path-decoder",
        config = config
        # config={
        #     "batch_size": train_loader.batch_size,
        #     "split_type": train_data.split_by,
        # }
    )





# %%


# %%
all_scenarios = ['city_47_chicago_3p5', 'city_23_beijing_3p5', 'city_91_xiangyang_3p5', 'city_17_seattle_3p5_s', 'city_12_fortworth_3p5', 'city_92_sãopaulo_3p5', 'city_35_san_francisco_3p5', 'city_10_florida_villa_7gp_1758095156175', 'city_19_oklahoma_3p5_s', 'city_74_chiyoda_3p5']

for scenario in all_scenarios[9:]:
# %%
    dataset = dm.load(scenario, )
    print(f"######### Training on Scenario {scenario}  #########")
    config = {
        "BATCH_SIZE": 128,
        "PAD_VALUE": 0,
        "USE_WANDB": False,
        "LR": 2e-5,
        "epochs": 5,
        "interaction_weight": 0.01,
        "experiment": f"delay_only_enc_direct_{scenario}_interacaction_all_inter_str_dec_all_repeat",
        "hidden_dim": 512,
        "n_layers": 8,
        "n_heads": 8,
        "use_delay_kmeans": True,
        "n_clusters": 5,
        "max_path_len_clusters": 25,
        # "cluster_features": ["delay", "power", "aoa_az", "aoa_el", "phase"],
        "cluster_features": ["delay", "power"],

        "delay_only_loss": False,
        "TARGET_NOISE_PROB": 0.2,
        "TARGET_NOISE_PARAMS": None,
        # "pretrained_checkpoint": "checkpoints2/noise_enc_direct_city_47_chicago_3p5_interacaction_all_inter_str_dec_all_repeat_best_model_checkpoint.pth",
    }

    train_data  = PreTrainMySeqDataLoader(dataset, train=True, split_by="user", sort_by="power", pad_value=config["PAD_VALUE"], normalizers=None, apply_normalizers=[])
    val_data  = PreTrainMySeqDataLoader(dataset, train=False, split_by="user", sort_by="power", pad_value=config["PAD_VALUE"], normalizers=None, apply_normalizers=[])

    n_clusters = config.get("n_clusters", 4)
    max_path_len_clusters = config.get("max_path_len_clusters", 26)
    cluster_features = config.get("cluster_features", ["delay", "power", "aoa_az", "aoa_el", "phase"])
    cluster_centers = None
    cluster_stds = None
    cluster_lookup_data = None

    # backbone = PathDecoder(
    #     prompt_dim=6,
    #     hidden_dim=config["hidden_dim"],
    #     n_layers=config["n_layers"],
    #     n_heads=config["n_heads"],
    #     pad_value=config["PAD_VALUE"],
    # ).to(device)
    # model = PathDecoderClusterEncoderAdapter(
    #     backbone=backbone,
    #     hidden_dim=config["hidden_dim"],
    #     cluster_feature_dim=2 * len(cluster_features),
    #     prompt_dim=6,
    # ).to(device)
    model = PathDecoderClusterEncoderAttention(hidden_dim=config["hidden_dim"], cluster_feature_dim=2 * len(cluster_features), prompt_dim=6).to(device)

    if config.get("use_delay_kmeans", False):
        if not hasattr(train_data, "mins"):
            train_data.mins = np.array([0.0, 0.0, 0.0])
            train_data.maxs = np.array([1.0, 1.0, 1.0])
        if "tx_idx" not in train_data.dataset_filtered:
            train_data.dataset_filtered["tx_idx"] = [0] * len(train_data.dataset_filtered["delay"])
        cluster_centers, cluster_stds = compute_feature_kmeans_cluster_stats(
            train_data,
            feature_keys=cluster_features,
            max_path_len=max_path_len_clusters,
            n_clusters=n_clusters,
        )
        train_rx_pos, train_tx_idx, train_cluster_center_std, train_valid_len = precompute_train_cluster_center_std_sequences(
            train_data,
            cluster_centers=cluster_centers,
            cluster_stds=cluster_stds,
            feature_keys=cluster_features,
            max_path_len=max_path_len_clusters,
        )
        cluster_lookup_data = (train_rx_pos, train_tx_idx, train_cluster_center_std, train_valid_len)
        print(f"Precomputed train cluster center+std for NN lookup by (x,y), features={cluster_features}")
    print("Total trainable parameters:", count_parameters(model))
    # print(f"train_tx_idx: {train_tx_idx.shape} {train_tx_idx[:6]}")
    # print(f"train_rx_pos: {train_rx_pos.shape} {train_rx_pos[:6]}")
    # print(f"train_valid_len: {train_valid_len.shape} {train_valid_len.max()}")
    # print(f"train_step_means: {train_step_means.shape} {train_step_means[0].T}")



    optimizer = torch.optim.AdamW(model.parameters(), lr=config["LR"])
    # scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=2, mode="min")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=25,      # Restart every 10 epochs
        T_mult=1,    # Double the period after each restart
        eta_min=1e-8 # Minimum LR
    )

    # Initialize best checkpoint tracking (based on path_length loss)

    # scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=2, mode="min")

    checkpoint_path = f"{config['experiment']}_best_model_checkpoint.pth"
    os.makedirs("checkpoints2", exist_ok=True)
    checkpoint_path = os.path.join("checkpoints2", checkpoint_path)

    train_loader = torch.utils.data.DataLoader(
        dataset     = train_data,
        batch_size  = config['BATCH_SIZE'],
        shuffle     = True,
        collate_fn= train_data.collate_fn
        )
    val_loader = torch.utils.data.DataLoader(
        dataset     = val_data,
        batch_size  = config['BATCH_SIZE'],
        shuffle     = False,
        collate_fn= val_data.collate_fn
        )

    # Train
    train_with_interactions(model, train_loader, val_loader, config, train_data, cluster_lookup_data=cluster_lookup_data)
    
    # %% [markdown]
    # 
    # evaluate_generation(train_loader)
    # 

    # %%
    def load_best_checkpoint(model, checkpoint_path="checkpoints2/best_model_checkpoint.pth"):
        """
        Load the best model checkpoint saved during training.
        
        Args:
            model: The model instance to load the checkpoint into
            checkpoint_path: Path to the checkpoint file
        
        Returns:
            epoch: Epoch at which best checkpoint was saved
            best_val_loss: Best validation loss achieved
        """
        if not os.path.exists(checkpoint_path):
            print(f"Warning: Checkpoint not found at {checkpoint_path}")
            return None, None
        
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint['epoch']
        best_avg_val_loss = checkpoint['best_val_loss']
        
        print(f"✓ Loaded best checkpoint from epoch {epoch} (val_loss: {best_avg_val_loss:.4f})")
        return epoch, best_avg_val_loss
    # torch.serialization.add_safe_globals([np._core.multiarray.scalar])
    # torch.serialization.add_safe_globals([np.dtype])

    # Load best checkpoint for inference/evaluation
    best_epoch, best_loss = load_best_checkpoint(model, checkpoint_path=checkpoint_path)

    # %%
    checkpoint_path

    # %%
    results = evaluate_model(model, val_loader, pad_value=config["PAD_VALUE"], data_stats=None,
                             cluster_lookup_data=cluster_lookup_data)
    # print(results)
    avg_delay, avg_power, avg_phase, avg_az, avg_el, avg_path_length_rmse, avg_delay_mae, avg_power_mae, avg_phase_mae, avg_az_mae, avg_el_mae, avg_path_length_mae  = results
    # (avg_delay, avg_power, avg_phase, avg_path_length_rmse, 
    #  avg_delay_mae, avg_power_mae, avg_phase_mae, avg_path_length_mae) = results
    scenario_row = {
            "scenario": scenario,
            "delay_rmse": avg_delay,
            "power_rmse": avg_power,
            "phase_rmse": avg_phase,
            "az_rmse": avg_az,
            "el_rmse": avg_el,
            "phase_rmse": avg_phase,
            "path_length_rmse": avg_path_length_rmse,
            "delay_mae": avg_delay_mae,
            "power_mae": avg_power_mae,
            "phase_mae": avg_phase_mae,
            "avg_az_mae": avg_az_mae,
            "avg_el_mae": avg_el_mae,
            "path_length_mae": avg_path_length_mae,
            "best_val_loss": best_loss.item() if hasattr(best_loss, 'item') else best_loss
        }

        # 4. Append to CSV
    df = pd.DataFrame([scenario_row])
    # header=not os.path.exists(...) ensures the header is only written once
    df.to_csv(csv_log_file, mode='a', index=False, header=not os.path.exists(csv_log_file))

    print(f"✓ Results for {scenario} saved to {csv_log_file}")
    del dataset, train_loader, val_loader, model

    gc.collect()
    torch.cuda.empty_cache()
    # %%
    # show_example(model, val_loader, sample_index=24)





# %%
best_epoch, best_loss = load_best_checkpoint(model, checkpoint_path=checkpoint_path)

# %%
checkpoint_path

# %%
results = evaluate_model(model, val_loader, pad_value=config["PAD_VALUE"], data_stats=None,
                            cluster_lookup_data=cluster_lookup_data)

# %%


# %%



# %%


# %%



