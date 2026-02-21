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
from models_play import PathDecoder, GPTPathDecoder, PathDecoderDelta
from dataset.dataloaders_play import PreTrainMySeqDataLoader
from utils.utils_play import *

from tqdm import tqdm
import torch
import numpy as np
import pandas as pd
import os

csv_log_file = "final_scenario_results.csv"

# %%
scenario = 'city_89_nairobi_3p5'
# scenario = 'city_0_newyork_3p5'

dm.download(scenario)
dataset = dm.load(scenario, )

# %%
dataset.scene.plot()


# %%
dm.info()


# %%
config = {
    "BATCH_SIZE":128,
    "PAD_VALUE": 0,  # Padding value; use path_padding_mask for loss masking (avoids large sentinel in representations)
    "USE_WANDB": False,
    "LR":2e-5,
    "epochs" : 100,
    "interaction_weight": 0.01,  # Weight for interaction loss
    "experiment": f"{scenario}_interacaction_all_inter_str_dec_all_repeat",
    "hidden_dim": 512,
    "n_layers": 8,
    "n_heads": 8,
}




# %%



# %%


# %%
# Paths sorted by delay (ascending) for encoder-first + decoder-deltas training
train_data  = PreTrainMySeqDataLoader(dataset, train=True, split_by="user", sort_by="delay")

train_loader = torch.utils.data.DataLoader(
    dataset     = train_data,
    batch_size  = config['BATCH_SIZE'],
    shuffle     = True,
    collate_fn= train_data.collate_fn
    )
val_data  = PreTrainMySeqDataLoader(dataset, train=False, split_by="user", sort_by="delay")
val_loader = torch.utils.data.DataLoader(
    dataset     = val_data,
    batch_size  = config['BATCH_SIZE'],
    shuffle     = False,
    collate_fn= val_data.collate_fn
    )

for item in train_loader:
    print(f"Prompt shape: {item[0].shape}, Paths shape: {item[1].shape}, Num paths shape: {item[2].shape}, path_padding_mask: {item[6].shape}")
    
    break


# %%
print("No. of Train Points   : ", train_data.__len__())
print("Batch Size           : ", config["BATCH_SIZE"])
print("Train Batches        : ", train_loader.__len__())
print("No. of Train Points   : ", val_data.__len__())
print("Val Batches          : ", val_loader.__len__())

def compute_stop_metrics(path_count, targets, pad_value=500):
    """

    Args:

    """

    rmse = np.sqrt(mean_squared_error(path_count.cpu().numpy(), targets.squeeze().cpu().numpy()))
    
    return rmse


def make_delta_targets(paths_out):
    """
    Build targets for encoder-first-step + decoder-deltas training:
    - First step (t=0): actual values (encoder predicts first path metrics).
    - Subsequent steps (t>=1): differences from previous step (decoder predicts deltas).
    paths_out: (B, T, 5) — delay, power, phase, aoa_az, aoa_el.
    Returns: (B, T, 5) same shape; first row = actuals, rest = deltas.
    """
    targets = paths_out.clone()
    T = paths_out.size(1)
    if T > 1:
        targets[:, 1:, :] = paths_out[:, 1:, :] - paths_out[:, :-1, :]
    return targets 



def evaluate_model(model, val_loader, max_generate=26, log_to_wandb=False):
    model.eval()

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
         
            # Inner tqdm to show per-sample progress
            inner_bar = tqdm(range(prompts.size(0)), 
                             desc="   Processing samples", 
                             leave=False)


            for b in inner_bar:
                generated, path_lengths_pred, inter_str_pred = generate_paths_no_env(model, prompts[b], max_steps=max_generate)

                generated = generated.cuda()
                # generate_paths_no_env returns actuals (encoder first step + decoder deltas accumulated)
                pred_actuals = generated  # (T, 5) actual values

                # ground truth: delay, power, phase, aoa_az, aoa_el (use path length, not pad value, so delay=0 is safe)
                n_valid = int(round(path_lengths[b].item() * 25))
                gt = paths[b][1:1 + n_valid, :5]

                T = min(len(gt), len(pred_actuals))
                pred = pred_actuals[:T]
                gt = gt[:T]

                # Un-normalize to original scale for RMSE on actual values
                # delay_pred = (pred[:,0]  * (data_stats["delay"]["max"] - data_stats["delay"]["min"]) ) + data_stats["delay"]["min"]
                # delay = (gt[:,0] * (data_stats["delay"]["max"] -  data_stats["delay"]["min"])) + data_stats["delay"]["min"]


                # power_pred = (pred[:,1]  * (data_stats["power"]["max"]  - data_stats["power"]["min"] )) + data_stats["power"]["min"]
                # power = (gt[:,1] * (data_stats["power"]["max"]  - data_stats["power"]["min"])) + data_stats["power"]["min"]

                # phase_pred = (pred[:,2]  * (data_stats["phase"]["max"] - data_stats["phase"]["min"] )) + data_stats["phase"]["min"]
                # phase = (gt[:,2] * (data_stats["phase"]["max"] - data_stats["phase"]["min"])) + data_stats["phase"]["min"]


                # aoa_az_pred = (pred[:,3]  * (data_stats["aoa_az"]["max"] - data_stats["aoa_az"]["min"]) ) + data_stats["aoa_az"]["min"]
                # aoa_az = (gt[:,3] * (data_stats["aoa_az"]["max"] - data_stats["aoa_az"]["min"])) + data_stats["aoa_az"]["min"]
              
                # aoa_el_pred = (pred[:,4]  * (data_stats["aoa_el"]["max"] -  data_stats["aoa_el"]["min"])) + data_stats["aoa_el"]["min"]
                # aoa_el = (gt[:,4] *(data_stats["aoa_el"]["max"] -  data_stats["aoa_el"]["min"])) + data_stats["aoa_el"]["min"]

                delay_pred = pred[:,0]  
                # delay_pred = torch.cumsum(delay_pred, dim=0)
                delay = gt[:,0]


                power_pred = pred[:,1]  
                # power_pred = torch.cumsum(power_pred, dim=0)
                power = gt[:,1] 

                phase_pred = pred[:,2]
                # phase_pred = torch.cumsum(phase_pred, dim=0)

                phase = gt[:,2]


                aoa_az_pred = pred[:,3]  
                # aoa_az_pred = torch.cumsum(aoa_az_pred, dim=0)

                aoa_az = gt[:,3] 
              
                aoa_el_pred = pred[:,4]
                # aoa_el_pred = torch.cumsum(aoa_el_pred, dim=0)

                aoa_el = gt[:,4] 
            

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

                # Path length RMSE
                # print(path_lengths_pred, path_lengths[b],)
                length_rmse = (torch.mean( (path_lengths_pred - path_lengths[b])**2)).sqrt().item()
                length_mae = (torch.mean(torch.abs(path_lengths_pred - path_lengths[b]))).item()


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




def show_example(model, val_loader, sample_index=0, k=25, plot=True):
    model.eval()
    batch = next(iter(val_loader))
    prompts, paths, path_lengths, interactions = batch[0], batch[1], batch[2], batch[3]
    
    prompts = prompts.cuda()
    paths = paths.cuda()
    
    pred_paths, path_lengths_pred,inter_str_pred= generate_paths_no_env(model, prompts[sample_index])
    

    pred = pred_paths[0]  # (T,3)
    n_valid = int(round(path_lengths[sample_index].item() * 25))
    gt = paths[sample_index][1:1 + n_valid, :3]  # use path length so delay=0 is safe

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


def evaluate_generation(val_loader, n_samples=3):
    model.eval()
    for i, (prompts, paths) in enumerate(val_loader):
        if i >= n_samples:
            break
        pred, path_lengths_pred = generate_paths_no_env(model, prompts[0])  # autoregressive generation
        print(f"path lengths pred: {path_lengths_pred[0]}")
        print(f"\nSample {i}")
        print("GT paths (first 5):")
        print(paths[0][:5])
        print("Predicted paths (first 5):")
        print(pred[0][:5])

# %%
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_with_interactions(model, train_loader, val_loader, config, train_data, task=None):
    """
    Modified training loop with interaction prediction.
    """

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
        for prompts, paths, path_lengths, interactions, env, env_prop, path_padding_mask in pbar:  # path_padding_mask: (B, T), True=valid
            prompts = prompts.cuda()
            paths = paths.cuda()
            path_lengths = path_lengths.cuda()
            interactions = interactions.cuda()
            path_padding_mask = path_padding_mask.cuda()
            
            paths_in = paths[:, :-1, :]
            interactions_in = interactions[:, :-1, :]

            paths_out = paths[:, 1:, :]
            interactions_out = interactions[:, 1:, :]
            # Encoder predicts first step (actuals); decoder predicts deltas for subsequent steps
            # print(f"paths_out= {paths_out[0]} {paths_out.shape}")

            paths_out = make_delta_targets(paths_out)
            # print(f"paths_out_delta= {paths_out[0]}")

            (delay_pred, power_pred, phase_sin_pred, phase_cos_pred, phase_pred,
             az_sin_pred, az_cos_pred, az_pred, el_sin_pred, el_cos_pred, el_pred,
             path_length_pred, interaction_logits) = model(prompts, paths_in, interactions_in)
            


            
            (total_loss, loss_delay, loss_power, loss_phase,
             loss_az, loss_el, loss_path_length, loss_interaction,loss_channel) = masked_loss(
                delay_pred, power_pred, phase_sin_pred, phase_cos_pred,phase_pred,
                az_sin_pred, az_cos_pred, az_pred, el_sin_pred, el_cos_pred,el_pred,
                path_length_pred, interaction_logits, paths_out, path_lengths,
                interactions_out, finetune=task, pad_value=train_data.pad_value,
                interaction_weight=config.get("interaction_weight", 0.1),
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

                paths_in = paths[:, :-1, :]
                interactions_in = interactions[:, :-1, :]

                paths_out = paths[:, 1:, :]
                interactions_out = interactions[:, 1:, :]
                paths_out = make_delta_targets(paths_out)

                (delay_pred, power_pred, phase_sin_pred, phase_cos_pred, phase_pred,
                 az_sin_pred, az_cos_pred, az_pred, el_sin_pred, el_cos_pred, el_pred,
                 path_length_pred, interaction_logits) = model(prompts, paths_in, interactions_in)
                
                

                (total_loss, loss_delay, loss_power, loss_phase,
                loss_az, loss_el, loss_path_length, loss_interaction,loss_channel) = masked_loss(
                    delay_pred, power_pred, phase_sin_pred, phase_cos_pred,phase_pred,
                    az_sin_pred, az_cos_pred, az_pred, el_sin_pred, el_cos_pred,el_pred,
                    path_length_pred, interaction_logits, paths_out, path_lengths,
                    interactions_out, finetune=task, pad_value=train_data.pad_value,
                    interaction_weight=config.get("interaction_weight", 0.1),
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
all_scenarios = ['city_47_chicago_3p5', 'city_23_beijing_3p5', 'city_91_xiangyang_3p5', 'city_17_seattle_3p5_s', 'city_12_fortworth_3p5', 'city_92_sãopaulo_3p5', 'city_35_san_francisco_3p5', 'city_10_florida_villa_7gp_1758095156175', 'city_19_oklahoma_3p5_s', 'city_74_chiyoda_3p5']

for scenario in all_scenarios[:1]:
# %%
    # model = GPTPathDecoder().to(device)
    model = PathDecoderDelta(hidden_dim=config["hidden_dim"], n_layers = config["n_layers"], n_heads=config["n_heads"]).to(device)

    print("Total trainable parameters:", count_parameters(model))
    dataset = dm.load(scenario, )
    print(f"######### Training on Scenario {scenario}  #########")
    config = {
        "BATCH_SIZE":128,
        "PAD_VALUE": 0,
        "USE_WANDB": False,
        "LR":2e-5,
        "epochs" : 100,
        "interaction_weight": 0.01,  # Weight for interaction loss
        "experiment": f"enc_direct_{scenario}_interacaction_all_inter_str_dec_all_repeat",
        "hidden_dim": 512,
        "n_layers": 8,
        "n_heads": 8,
        "sort_by": "delay"
    }

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
    train_data  = PreTrainMySeqDataLoader(dataset, train=True, split_by="user", sort_by=config["sort_by"], apply_normalizers =["power"] , pad_value=config["PAD_VALUE"])                                                                              
    data_stats = get_dataset_statistics(train_data)
    
    train_data  = PreTrainMySeqDataLoader(dataset, train=True, split_by="user", sort_by=config["sort_by"], normalizers = data_stats, apply_normalizers =["power"] , pad_value=config["PAD_VALUE"])
    train_loader = torch.utils.data.DataLoader(
        dataset     = train_data,
        batch_size  = config['BATCH_SIZE'],
        shuffle     = True,
        collate_fn= train_data.collate_fn
        )
    val_data  = PreTrainMySeqDataLoader(dataset, train=False, split_by="user", sort_by=config["sort_by"], normalizers = data_stats, apply_normalizers =["power"], pad_value=config["PAD_VALUE"])
    val_loader = torch.utils.data.DataLoader(
        dataset     = val_data,
        batch_size  = config['BATCH_SIZE'],
        shuffle     = False,
        collate_fn= val_data.collate_fn
        )

    # Train
    # train_with_interactions(model, train_loader, val_loader, config, train_data)
    
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
    results = evaluate_model(model, val_loader)
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


    # %%
    # show_example(model, val_loader, sample_index=24)



