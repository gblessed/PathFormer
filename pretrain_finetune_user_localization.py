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
from models import PathDecoder, GPTPathDecoder, PathDecoderEnv, PathFormerLocalizer
from dataset.dataloaders import PreTrainMySeqDataLoader
from utils.utils import *

from tqdm import tqdm
import torch
import numpy as np
import pandas as pd
import os



# %%
scenario = 'city_89_nairobi_3p5'
# scenario = 'city_0_newyork_3p5'
csv_log_file = "user_localization_finetuning.csv"

dm.download(scenario)
dataset = dm.load(scenario, )

# %%
dataset.scene.plot()


# %%
dm.info()


# %%
config = {
    "BATCH_SIZE":128,
    "PAD_VALUE": 500,
    "USE_WANDB": False,
    "LR":2e-5,
    "epochs" : 50,
    "interaction_weight": 0.01,  # Weight for interaction loss
    "experiment": f"true_enc_pre_mixed_train_all_scenarios_interaction_weight_0.01_better_scheduler",
    "base_experiment": f"true_enc_direct_{scenario}_interacaction_all_inter_str_dec_all_repeat",
    "hidden_dim": 512,
    "n_layers": 8,
    "n_heads": 8,
}




# %%



# %%


# %%
train_data  = PreTrainMySeqDataLoader(dataset, train=True, split_by="user", sort_by="power")

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
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_with_interactions(model, train_loader, val_loader, config, train_data, task=None):
    """
    Modified training loop with interaction prediction.
    """

    best_val_loss = float('inf')
    once = False

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
        if (not (once)) and (epoch+1 > config["unfreezing"]):
            print("="*30)
            print("unfreezing all")
            unfreeze_all(model)
            once = True
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]", leave=False)
        for prompts, paths, path_lengths, interactions, env, env_prop in pbar:  # NEW: added interactions
            prompts = prompts.cuda()
            paths = paths.cuda()
            path_lengths = path_lengths.cuda()
            interactions = interactions.cuda()  # NEW
            env = env.cuda()
            env_prop = env_prop.cuda()
            paths_in = paths[:, :-1, :]
            interactions_in = interactions[:, :-1, :]

            paths_out = paths[:, 1:, :]
            interactions_out = interactions[:, 1:, :]  # NEW: shift targets

            (delay_pred, power_pred, phase_sin_pred, phase_cos_pred, phase_pred,
             az_sin_pred, az_cos_pred, az_pred, el_sin_pred, el_cos_pred, el_pred,
             path_length_pred, interaction_logits) = model(prompts, paths_in, interactions_in,env_prop, env, pre_train=False )

            (total_loss, loss_delay, loss_power, loss_phase,
             loss_az, loss_el, loss_path_length, loss_interaction,loss_channel) = masked_loss(
                delay_pred, power_pred, phase_sin_pred, phase_cos_pred,phase_pred,
                az_sin_pred, az_cos_pred, az_pred, el_sin_pred, el_cos_pred,el_pred,
                path_length_pred, interaction_logits, paths_out, path_lengths,
                interactions_out, finetune=task, pad_value=train_data.pad_value,
                interaction_weight=config.get("interaction_weight", 0.1)
            )
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            scheduler.step()
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

            for prompts, paths, path_lengths, interactions, env, env_prop in pbar:  # NEW
                prompts = prompts.cuda()
                paths = paths.cuda()
                path_lengths = path_lengths.cuda()
                interactions = interactions.cuda()  # NEW
                env = env.cuda()
                env_prop = env_prop.cuda()
                paths_in = paths[:, :-1, :]
                interactions_in = interactions[:, :-1, :]

                paths_out = paths[:, 1:, :]
                interactions_out = interactions[:, 1:, :]  # NEW: shift targets

                (delay_pred, power_pred, phase_sin_pred, phase_cos_pred, phase_pred,
                 az_sin_pred, az_cos_pred, az_pred, el_sin_pred, el_cos_pred, el_pred,
                 path_length_pred, interaction_logits) = model(prompts, paths_in, interactions_in,env_prop, env, pre_train=False )
                
                (total_loss, loss_delay, loss_power, loss_phase,
                loss_az, loss_el, loss_path_length, loss_interaction,loss_channel) = masked_loss(
                    delay_pred, power_pred, phase_sin_pred, phase_cos_pred,phase_pred,
                    az_sin_pred, az_cos_pred, az_pred, el_sin_pred, el_cos_pred,el_pred,
                    path_length_pred, interaction_logits, paths_out, path_lengths,
                    interactions_out, finetune=task, pad_value=train_data.pad_value,
                    interaction_weight=config.get("interaction_weight", 0.1)
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

def finetune_localization(model, train_loader, val_loader, config):
    # Loss: Mean Squared Error on coordinates
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["LR"])
    
    # Optional: Freeze backbone for the first few epochs to protect pre-trained weights
    # for param in model.backbone.parameters(): param.requires_grad = False
    best_val_mde = float('inf')
    for epoch in range(config["epochs"]):
        model.train()
        epoch_losses = []
        mdes = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Localization]")
        for prompts, paths, _, interactions, env, env_prop in pbar:
            # The 'prompts' tensor contains the Ground Truth [x, y, z] in the first 3 columns
            # print(f"prompts {prompts}")
            gt_coords = prompts[:, 3:5].cuda()
            
            paths, interactions = paths.cuda(), interactions.cuda()
            env, env_prop = env.cuda(), env_prop.cuda()
            
            # Predict
            pred_coords = model(paths, interactions, env_prop, env)
            
            loss = criterion(pred_coords, gt_coords)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pred_coords = model(paths.cuda(), interactions.cuda(), env_prop.cuda(), env.cuda())
            # gt_coords =  (  torch.tensor(data_stats["rx_pos"]["std"])[:2].cuda() * gt_coords) +  torch.tensor(data_stats["rx_pos"]["mean"][:2]).cuda()
            # gt_coords =(gt_coords  * (torch.tensor(val_data.maxs[:2]).cuda() - torch.tensor(val_data.mins[:2]).cuda())) + torch.tensor(val_data.mins[:2]).cuda()

            # pred_coords =  (torch.tensor(data_stats["rx_pos"]["std"])[:2].cuda() * pred_coords) +  torch.tensor(data_stats["rx_pos"]["mean"][:2]).cuda()
            # pred_coords = (pred_coords * torch.tensor(val_data.maxs[:2]).cuda() - torch.tensor(val_data.mins[:2]).cuda()) +torch.tensor(val_data.mins[:2]).cuda()
            # Calculate Mean Distance Error (in meters) for tracking
            mde = torch.linalg.norm(pred_coords - gt_coords, dim=1).mean().item()
            mdes.append(mde)
            epoch_losses.append(loss.item())
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "mde": f"{mde:.2f}"})
            
        train_mean_mde = np.mean(mdes)
        avg_train_loss = np.mean(epoch_losses)

        # Evaluate on Validation
        val_mean_mde = evaluate_localization(model, val_loader)
        if val_mean_mde < best_val_mde:
            best_val_mde = val_mean_mde
        if config.get("USE_WANDB", False):
                wandb.log({"train_loss": avg_train_loss, "train_mde": train_mean_mde, "val_mean_mde": val_mean_mde})
    return best_val_mde
def evaluate_localization(model, val_loader):
    model.eval()
    all_errors = []
    with torch.no_grad():
        for prompts, paths, _, interactions, env, env_prop in val_loader:
            gt_coords = prompts[:, 3:5].cuda()
            pred_coords = model(paths.cuda(), interactions.cuda(), env_prop.cuda(), env.cuda())
            # print(f"gt {gt_coords[:10]}")
            
            # gt_coords =  (  torch.tensor(data_stats["rx_pos"]["std"])[:2].cuda() * gt_coords) +  torch.tensor(data_stats["rx_pos"]["mean"][:2]).cuda()
            # gt_coords =(gt_coords  * (torch.tensor(val_data.maxs[:2]).cuda() - torch.tensor(val_data.mins[:2]).cuda())) + torch.tensor(val_data.mins[:2]).cuda()

            # pred_coords =  (torch.tensor(data_stats["rx_pos"]["std"])[:2].cuda() * pred_coords) +  torch.tensor(data_stats["rx_pos"]["mean"][:2]).cuda()
            # pred_coords = (pred_coords * torch.tensor(val_data.maxs[:2]).cuda() - torch.tensor(val_data.mins[:2]).cuda()) +torch.tensor(val_data.mins[:2]).cuda()

            error = torch.linalg.norm(pred_coords - gt_coords, dim=1)
            all_errors.extend(error.cpu().numpy())
            
    # print(f"\n>>> Validation MDE: {np.mean(all_errors):.4f} meters")
    # print(f"pred {pred_coords[:10]}")
    # print(f"pred {gt_coords[:10]}")


    return np.mean(all_errors)


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

def freeze_backbone(model):
    for param in model.parameters():
        param.requires_grad = False
def freeze_for_finetuning(model):
    # 1. First, disable gradients for EVERY parameter in the model
    for param in model.parameters():
        param.requires_grad = False

    # 2. UNFREEZE Encoder / Embedding components
    # (The parts that process the input before the decoder)
    model.prompt_to_prefix.requires_grad_(True)
    # model.path_in.requires_grad_(True)
    # model.pos_emb.requires_grad_(True)
    # model.environment_embed.requires_grad_(True)
    # model.environment_prop_embed.requires_grad_(True)

    # 3. UNFREEZE Cross-Attention weights only inside the Decoder
    # In PyTorch, cross-attention is 'multihead_attn' 
    # self-attention is 'self_attn'
    for name, param in model.decoder.named_parameters():
        if "multihead_attn" in name:
            param.requires_grad = True
            print(f"Unfrozen: {name}")

    # Note: 'self_attn', 'linear1', 'linear2', 'norm1', 'norm2', 'norm3' 
    # stay frozen inside the decoder layers.

    # 4. EXPLICITLY FREEZE the Linear Heads (just to be safe)
    model.out.requires_grad_(False)
    model.interaction_head.requires_grad_(False)
    model.pathcount_head.requires_grad_(False)

def unfreeze_all(model):
    # 1. First, disable gradients for EVERY parameter in the model
    for param in model.parameters():
        param.requires_grad = True


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
# %%
all_scenarios = ['city_47_chicago_3p5','city_10_florida_villa_7gp_1758095156175',  'city_23_beijing_3p5', 'city_91_xiangyang_3p5', 'city_17_seattle_3p5_s', 'city_12_fortworth_3p5', 'city_92_sãopaulo_3p5', 'city_35_san_francisco_3p5',  'city_19_oklahoma_3p5_s', 'city_74_chiyoda_3p5']

for scenario in all_scenarios:
# %%
    # model = GPTPathDecoder().to(device)
    backbone_model = PathDecoderEnv(hidden_dim=config["hidden_dim"], n_layers = config["n_layers"], n_heads=config["n_heads"])

    print("backbone_modelparameters:", count_parameters(backbone_model))
    dataset = dm.load(scenario, )
    print(f"######### Training on Scenario {scenario}  #########")
    config = {
        "BATCH_SIZE":128,
        "PAD_VALUE": 500,
        "USE_WANDB": False,
        "LR":2e-4,
        # "unfreezing":50,
        "epochs" : 100,
        "interaction_weight": 0.01,  # Weight for interaction loss
        "base_experiment": f"true_enc_pre_mixed_train_all_scenarios_interaction_weight_0.01_better_scheduler",
        "experiment": f"finetune_{scenario}_interaction_weight_0.01",
        "finetune_scenario": f"{scenario}",
        "experiment": f"localization",
        "hidden_dim": 512,
        "n_layers": 8,
        "n_heads": 8,
        "pre_train":False
    }


    # Initialize best checkpoint tracking (based on path_length loss)

    # scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=2, mode="min")

    best_val_loss = float('inf')
    base_model_checkpoint_path = f"{config['base_experiment']}_best_model_checkpoint.pth"
    base_model_checkpoint_path = os.path.join("checkpoints20M", base_model_checkpoint_path)
    checkpoint_path = f"{config['experiment']}_best_model_checkpoint.pth"

    os.makedirs("checkpoints2", exist_ok=True)
    checkpoint_path = os.path.join("checkpoints2", checkpoint_path)
    train_data  = PreTrainMySeqDataLoader(dataset, train=True, split_by="user", sort_by="power", normalizers=None)
    data_stats = get_dataset_statistics(train_data)
    # Print a nice summary
    for feature, val in data_stats.items():
        print(f"--- {feature} ---")
        for stat_name, stat_val in val.items():
            print(f"  {stat_name}: {stat_val}")


    train_data  = PreTrainMySeqDataLoader(dataset, train=True, split_by="user", sort_by="power", normalizers=None)
    
    train_loader = torch.utils.data.DataLoader(
        dataset     = train_data,
        batch_size  = config['BATCH_SIZE'],
        shuffle     = True,
        collate_fn= train_data.collate_fn
        )


    # --- Execution ---
    # Assuming you have initialized: train_data = PreTrainMySeqDataLoader(...)

    val_data  = PreTrainMySeqDataLoader(dataset, train=False, split_by="user", sort_by="power", normalizers=None)

            
    val_loader = torch.utils.data.DataLoader(
        dataset     = val_data,
        batch_size  = config['BATCH_SIZE'],
        shuffle     = False,
        collate_fn= val_data.collate_fn
        )
    best_epoch, best_loss = load_best_checkpoint(backbone_model, checkpoint_path=base_model_checkpoint_path)
    freeze_backbone(backbone_model)
    model  = PathFormerLocalizer(backbone_model).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["LR"])
    # scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=2, mode="min")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=25,      # Restart every 10 epochs
        T_mult=1,    # Double the period after each restart
        eta_min=1e-8 # Minimum LR
    )
    print("Trainable parameters:", count_parameters(model))
    # unfreeze_all(model)s
    # Train
    
    
    val_mde = finetune_localization(model, train_loader, val_loader, config)
    scenario_row = {
            "scenario": scenario,
            "val_mde": val_mde,}
    df = pd.DataFrame([scenario_row])
    # header=not os.path.exists(...) ensures the header is only written once
    df.to_csv(csv_log_file, mode='a', index=False, header=not os.path.exists(csv_log_file))


    
    # %% [markdown]
    # 
    # evaluate_generation(train_loader)
    # 


    # torch.serialization.add_safe_globals([np._core.multiarray.scalar])
    # torch.serialization.add_safe_globals([np.dtype])

    # Load best checkpoint for inference/evaluation
  



    # %%
    # show_example(model, val_loader, sample_index=24)



