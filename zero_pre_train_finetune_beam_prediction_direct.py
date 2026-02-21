# %%
# %%
# !pip install DeepMIMO==4.0.0b10

# %%
# %%
# =============================================================================
# 1. IMPORTS AND WARNINGS SETUP
#    - Load necessary PyTorch modules, utilities, and suppress UserWarnings
# =============================================================================

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
from models import GPTPathDecoder, PathDecoderEnv, PathFormerBeamPredictor,PathDecoder
from dataset.dataloaders import MySeqDataLoader,PreTrainMySeqDataLoader
from utils.utils import *
import pandas as pd
# %%
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
                # ground truth: delay, power, phase, aoa_az, aoa_el
                # gt = paths[b][1:, :5]

                # Mask padded values
                # valid_mask = (gt[:,0] != train_data.pad_value)
                n_valid = int(round(path_lengths[b].item() * 25))
                gt = paths[b][1:1 + n_valid, :5]

                T = min(len(gt), len(generated))
                pred = generated[:T]
                gt = gt[:T]

   
                # delay_pred = (pred[:,0]  * data_stats["delay"]["std"] ) + data_stats["delay"]["mean"]
                # delay = (gt[:,0] * data_stats["delay"]["std"]) + data_stats["delay"]["mean"]


                # power_pred = (pred[:,1]  * data_stats["power"]["std"] ) + data_stats["power"]["mean"]
                # power = (gt[:,1] * data_stats["power"]["std"]) + data_stats["power"]["mean"]

                # phase_pred = (pred[:,2]  * data_stats["phase"]["std"] ) + data_stats["phase"]["mean"]
                # phase = (gt[:,2] * data_stats["phase"]["std"]) + data_stats["phase"]["mean"]


                # aoa_az_pred = (pred[:,3]  * data_stats["aoa_az"]["std"] ) + data_stats["aoa_az"]["mean"]
                # aoa_az = (gt[:,3] * data_stats["aoa_az"]["std"]) + data_stats["aoa_az"]["mean"]
              
                # aoa_el_pred = (pred[:,4]  * data_stats["aoa_el"]["std"] ) + data_stats["aoa_el"]["mean"]
                # aoa_el = (gt[:,4] * data_stats["aoa_el"]["std"]) + data_stats["aoa_el"]["mean"]

                delay_pred = pred[:,0]  
                # delay_pred = torch.cumsum(delay_pred, dim=0)
                delay = gt[:,0]


                power_pred = pred[:,1]  
                power = gt[:,1] 
                phase_pred = pred[:,2]
                phase = gt[:,2]
                aoa_az_pred = pred[:,3]  
                aoa_az = gt[:,3]           
                aoa_el_pred = pred[:,4]    
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
            print("Batch evaluation complete.")
            
            print("\n================= Up toBATCH EVALUATION RESULTS =================")
            print(f"Avg Delay RMSE           : {np.mean(delay_errors):.4f} µs")
            print(f"Avg Power RMSE           : {np.mean(power_errors):.4f} dB")
            print(f"Avg Phase RMSE           : {np.mean(phase_errors):.4f} degrees")
            print(f"Avg Path Length RMSE     : {np.mean(path_length_rmses):.4f}")
            print(f"Avg Delay MAE           : {np.mean(delay_maes):.4f} µs")
            print(f"Avg Power MAE           : {np.mean(power_maes):.4f} dB")
            print(f"Avg Phase MAE           : {np.mean(phase_maes):.4f} degrees")
            print(f"Avg Path Length MAE     : {np.mean(path_length_maes):.4f}")
            print("============================================================")
            break

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


def generate_paths(model, prompt, env, env_prop, max_steps=25, stop_threshold=0.5):
    """
    Generate paths autoregressively.
    """
    model.eval()
    prompt = prompt.unsqueeze(0).cuda()  # (1, prompt_dim)
    env = env.unsqueeze(0).cuda()  # (1, prompt_dim)
    env_prop = env_prop.unsqueeze(0).cuda()  # (1, prompt_dim)
    # Start with SOS tokens (delay, power, phase, aoa_az, aoa_el)
    cur = torch.zeros(1, 1, 5).cuda()  # (1, 1, 5)
    inter_str = -1 * torch.ones(1, 1, 4).cuda()  # (1, 1, 4) - interaction labels

    outputs = []
    outputs_inter_str = []

    for t in range(max_steps):
        # Forward pass - unpack expanded outputs (including aoa preds)
        d, p, s, c, ph, az_s, az_c, az, el_s, el_c, el, pathcounts, inter_str_logits = model(prompt, cur, inter_str, env_prop, env,  pre_train = False)

        # Get last timestep predictions
        d_t = d[:, -1]           # (1,)
        p_t = p[:, -1]           # (1,)
        ph_t = ph[:, -1]         # (1,)
        az_t = az[:, -1]
        el_t = el[:, -1]
        inter_logits_t = inter_str_logits[:, -1]  # (1, 4)

        # Convert logits to binary predictions
        inter_pred_t = (torch.sigmoid(inter_logits_t) > 0.5).float()  # (1, 4) - binary [0, 1]

        # Store outputs (delay, power, phase, aoa_az, aoa_el)
        outputs.append(torch.stack([d_t, p_t, ph_t, az_t, el_t], dim=-1))
        outputs_inter_str.append(inter_pred_t)

        # Append predictions for next iteration
        next_path = torch.stack([d_t, p_t, ph_t, az_t, el_t], dim=-1).unsqueeze(1)  # (1, 1, 5)
        cur = torch.cat([cur, next_path], dim=1)

        # Use binary predictions for interactions
        inter_str = torch.cat([inter_str, inter_pred_t.unsqueeze(1)], dim=1)  # (1, t+2, 4)

    return (torch.stack(outputs, dim=1).squeeze(0).detach().cpu(),  # (T, 5)
            pathcounts, 
            torch.stack(outputs_inter_str, dim=1).squeeze(0).detach().cpu())  # (T, 4)


# %%
# %%
# scenario = 'city_89_nairobi_3p5'
scenario = 'city_0_newyork_3p5'

csv_log_file = "zeero_beam_prediction_finetuning.csv"
dm.download(scenario)
dataset = dm.load(scenario, )

# %%
dataset.scene.plot()


# %%
dm.info()

def make_dft_codebook(B=8, dtype=np.complex64):
    params = ChannelParameters()

    az_t = np.linspace(-np.pi, np.pi, B, endpoint=False, dtype=np.float32)
    el_t = np.linspace(-np.pi, np.pi, B, endpoint=False, dtype=np.float32)
    az_new = []
    el_new = []
    for az in az_t:
        for el in el_t:
            az_new.append(az)
            el_new.append(el)
    az_new = torch.tensor(az_new).unsqueeze(1)
    el_new = torch.tensor(el_new).unsqueeze(1)
    array_response = compute_single_array_response_torch(params.bs_antenna,  az_new, el_new)

    return array_response.squeeze(2).T ## Tx x (B**2) 
    
def compute_beam_label_from_channel(H, S):
    """
    H: np.array shape (Nt, K)  dtype complex
    S: np.array shape (Nt, B)  dtype complex (codebook)
    returns: best_beam_index (int), Prx array (B,)
    """
    # Compute Y = S^H * H  -> shape (B, K)
    # Using conjugate transpose of S
    Y = np.conj(S.T) @ H   # (B, K)
    Prx = torch.sum(torch.abs(Y)**2, dim=2)  # (B,)
    best = torch.argmax(Prx, dim=1)
    return best, Prx
# %%
config = {
    "BATCH_SIZE":64,
    "PAD_VALUE": 500,
    "USE_WANDB": False,
    "LR":2e-5,
    "task": "beam_prediction",
    "epochs" : 20,
    "interaction_weight": 0.01,  # Weight for interaction loss
    "experiment": f"{scenario}_interacaction_all_inter_str_dec_all_aod",
    "hidden_dim": 512,
    "n_layers": 8,
    "n_heads": 8,
}


# %%
train_data  = PreTrainMySeqDataLoader(dataset, train=True, split_by="user", sort_by="power", normalizers=None)

train_loader = torch.utils.data.DataLoader(
    dataset     = train_data,
    batch_size  = config['BATCH_SIZE'],
    shuffle     = True,
    collate_fn= train_data.collate_fn
    )

val_data  = PreTrainMySeqDataLoader(dataset, train=False, split_by="user", sort_by="power")
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

# %%


@torch.no_grad()
@torch.no_grad()
def zero_shot_beam_eval_angle_only(
    model, val_loader, S, params, config, device
):
    """
    Zero-shot beam prediction using ONLY predicted AoA angles.
    No channel reconstruction.
    No teacher forcing.
    No beam classifier training.
    """

    model.eval()
    total = 0
    correct = 0

    for prompts, paths, path_lengths, interactions, env, env_prop, _ in tqdm(val_loader):
        prompts = prompts.to(device)
        env = env.to(device)
        env_prop = env_prop.to(device)

        B = prompts.shape[0]

        # ---- Generate first path only ----
        az_pred = []
        el_pred = []

        for b in range(B):
            generated, _, _ = generate_paths_no_env(
                model,
                prompts[b],
                max_steps=1   # ONLY first path
            )
            az_pred.append(generated[0, 3].item())
            el_pred.append(generated[0, 4].item())

        az_pred = torch.tensor(az_pred, device=device)
       
        el_pred = torch.tensor(el_pred, device=device)
        # print("az_pred", az_pred)
        # print("el_pred", el_pred)
        # ---- Build predicted steering vectors ----
        a_pred = compute_single_array_response_torch(
            params.bs_antenna,
            az_pred.unsqueeze(1),
            el_pred.unsqueeze(1)
        ).squeeze(2).cpu()   # (B, Nt)
        
        # ---- Beam scoring ----
        # score[b, k] = | s_k^H a_pred[b] |^2
        scores = torch.abs(S.conj().T @ a_pred.T) ** 2  # (B, B^2)

        beam_pred = torch.argmax(scores, dim=0)

        # ---- Oracle beam from ground truth channel ----
        paths_out = paths[:, 1:, :]

        delay_gt = paths_out[:, :, 0].cpu().numpy()
        power_gt = paths_out[:, :, 1].cpu().numpy()
        phase_gt = paths_out[:, :, 2].cpu().numpy()
        az_gt = paths_out[:, :, 3].cpu().numpy()
        el_gt = paths_out[:, :, 4].cpu().numpy()

        y_hat_az = (az_pred.cpu() / (np.pi/180))
        y_az = (paths_out[:, :, 3][:,0 ].cpu() / (np.pi/180))
        print(f"y_hat_az {y_hat_az[:10]}")
        print(f"y_az {y_az[:10]}")

        az_circular_dist = (y_hat_az - y_az + 180) % 360 - 180
        az_rmse = torch.mean(az_circular_dist**2).sqrt().item()
        az_mae = torch.mean(torch.abs(az_circular_dist)).item()
        print(f"az_rmse {az_rmse}")
        print(f"az_mae {az_mae}")


        mask = delay_gt == config["PAD_VALUE"]

        power_gt = np.where(mask, 0, power_gt)
        power_linear_gt = 10 ** ((power_gt / 0.01) / 10)
        delay_sec_gt = delay_gt / 1e6

        delay_sec_gt = np.where(mask, np.nan, delay_sec_gt)
        phase_gt = np.where(mask, np.nan, phase_gt)
        az_gt = np.where(mask, np.nan, az_gt)
        el_gt = np.where(mask, np.nan, el_gt)

        H_gt = mycomputer.compute_channels(
            power_linear_gt,
            delay_sec_gt,
            phase_gt,
            az_gt,
            el_gt,
            kwargs=None
        )[:, 0, :, :]   # (B, Nt, K)

        beam_gt, _ = compute_beam_label_from_channel(H_gt, S)

        # ---- Accuracy ----
        correct += (beam_pred.cpu() == beam_gt.cpu()).sum().item()
        total += B

    acc = correct / total
    print(f"\n Zero-Shot Beam Prediction Accuracy (AoA only) = {acc:.4f}")
    return acc

# %%
mycomputer = MyChannelComputer()
S = make_dft_codebook()



# %%
all_scenarios = ['city_47_chicago_3p5','city_10_florida_villa_7gp_1758095156175',  'city_23_beijing_3p5', 'city_91_xiangyang_3p5', 'city_17_seattle_3p5_s', 'city_12_fortworth_3p5', 'city_92_sãopaulo_3p5', 'city_35_san_francisco_3p5',  'city_19_oklahoma_3p5_s', 'city_74_chiyoda_3p5']



for scenario in all_scenarios:
    print(f"Beam prediction for {scenario}")
    config = {
        "BATCH_SIZE":128,
        "PAD_VALUE": 0,
        "USE_WANDB": False,
        "LR":2e-3,
        "epochs" : 100,
        "interaction_weight": 0.01,  # Weight for interaction loss
        "experiment": f"true_enc_direct_{scenario}_interacaction_all_inter_str_dec_all_repeat",
        "base_experiment": f"noise_enc_direct_{scenario}_interacaction_all_inter_str_dec_all_repeat",
        "hidden_dim": 512,
        "n_layers": 8,
        "n_heads": 8,
        "pre_train":False,
        "task":"beam_prediction"
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = f"{config['experiment']}_best_model_checkpoint.pth"
    os.makedirs("checkpoints2", exist_ok=True)
    checkpoint_path = os.path.join("checkpoints2", checkpoint_path)
    os.makedirs("checkpoints_channel", exist_ok=True)
    checkpoint_finetune = f"{config['task']}_{scenario}_best_model_checkpoint.pth"
    checkpoint_finetune_path = os.path.join("checkpoints_channel", checkpoint_finetune)

    dataset = dm.load(scenario, )
    train_data  = PreTrainMySeqDataLoader(dataset, train=True, split_by="user", sort_by="power", normalizers = None, apply_normalizers =[], pad_value=config["PAD_VALUE"] )
    train_loader = torch.utils.data.DataLoader(
        dataset     = train_data,
        batch_size  = config['BATCH_SIZE'],
        shuffle     = True,
        collate_fn= train_data.collate_fn
        )
    val_data  = PreTrainMySeqDataLoader(dataset, train=False, split_by="user", sort_by="power", normalizers = None, apply_normalizers =[], pad_value=config["PAD_VALUE"] )
    val_loader = torch.utils.data.DataLoader(
        dataset     = val_data,
        batch_size  = config['BATCH_SIZE'],
        shuffle     = False,
        collate_fn= val_data.collate_fn
        )


    backbone_model = PathDecoder(hidden_dim=config["hidden_dim"], n_layers = config["n_layers"], n_heads=config["n_heads"]).to(device)
    # model = PathDecoder().to(device)
    # model = GPTPathDecoder().to(device)
    best_val_loss = float('inf')
    base_model_checkpoint_path = f"{config['base_experiment']}_best_model_checkpoint.pth"
    base_model_checkpoint_path = os.path.join("checkpoints2", base_model_checkpoint_path)

    print("backbone_model parameters:", count_parameters(backbone_model))
    best_epoch, best_loss = load_best_checkpoint(backbone_model, checkpoint_path=base_model_checkpoint_path)


    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    #     optimizer, 
    #     T_0=25,      # Restart every 10 epochs
    #     T_mult=1,    # Double the period after each restart
    #     eta_min=1e-8 # Minimum LR
    # )

    # Initialize best checkpoint tracking (based on path_length loss)
    results = evaluate_model(backbone_model, val_loader)
    # scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=2, mode="min")
    mycomputer = MyChannelComputer()



    val_acc = zero_shot_beam_eval_angle_only(
    model=backbone_model,
    val_loader=val_loader,
    S=S,
    params= ChannelParameters(),
    config=config,
    device=device
    )

    print("Zero-shot beam accuracy:", val_acc)

    scenario_row = {
        "scenario": scenario,
        "val_acc": val_acc}
    df = pd.DataFrame([scenario_row])
    # header=not os.path.exists(...) ensures the header is only written once
    df.to_csv(csv_log_file, mode='a', index=False, header=not os.path.exists(csv_log_file))


# %%



