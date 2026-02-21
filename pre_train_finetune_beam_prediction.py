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
from models_play import GPTPathDecoder, PathDecoderEnv, PathFormerBeamPredictor
from dataset.dataloaders_play import MySeqDataLoader,PreTrainMySeqDataLoader
from utils.utils import *
import pandas as pd
# %%


# %%
# %%
# scenario = 'city_89_nairobi_3p5'
scenario = 'city_0_newyork_3p5'

csv_log_file = "beam_prediction_finetuning.csv"
dm.download(scenario)
dataset = dm.load(scenario, )

# %%
dataset.scene.plot()


# %%
dm.info()


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



# %%
mycomputer = MyChannelComputer()
S = make_dft_codebook()

# %%
def train_beam_predictor(model, train_loader, val_loader, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # Freeze backbone initially
    for p in model.backbone.parameters():
        p.requires_grad = False
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(config['epochs']):
        model.train()
        totals = []
        for prompts, paths, path_lengths, interactions, env, env_prop, beam_label in tqdm(train_loader):
            paths = paths.to(device); interactions = interactions.to(device)
            env = env.to(device); env_prop = env_prop.to(device)
            labels = beam_label.to(device).long()
            logits = model(paths, interactions, env_prop, env)
            loss = criterion(logits, labels)
            optimizer.zero_grad(); 
            loss.backward(); optimizer.step()
            totals.append(loss.item())
        sched.step()
        # Validation
        val_acc_top1, val_acc_top3 = evaluate_beam_predictor(model, val_loader, k_list=(1,3))
        print(f"Epoch {epoch} loss {np.mean(totals):.4f} val_top1 {val_acc_top1:.4f} top3 {val_acc_top3:.4f}")

    # Optionally unfreeze some backbone layers and fine-tune with smaller LR


def task_finetune(model, train_loader, val_loader, config, train_data,):
    """
    Modified training loop with interaction prediction.
    """
    best_val_loss = float('inf')
    best_val_acc = 0

    criterion = nn.CrossEntropyLoss()

    for epoch in range(config["epochs"]):
        # -------------------- TRAINING --------------------
        
        model.train()
        train_losses = []
        val_losses = []
        train_correct = 0
        val_correct = 0
        total = 0
        val_total =0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]", leave=False)
        for prompts, paths, path_lengths, interactions, env, env_prop, path_padding_mask in pbar:  # NEW: added interactions
            prompts = prompts.cuda()
            paths = paths.cuda()
            path_lengths = path_lengths.cuda()
            interactions = interactions.cuda()  # NEW
            env_prop = env_prop.cuda()
            env = env.cuda()
            # path_padding_mask = path_padding_mask.cuda()

            paths_in = paths[:, :-1, :]
            interactions_in = interactions[:, :-1, :]

            paths_out = paths[:, 1:, :]
            interactions_out = interactions[:, 1:, :]  # NEW: shift targets

            logits = model(prompts, paths, interactions, env_prop, env)



            
           
            # path_length_rmse = compute_stop_metrics(path_length_pred.detach().squeeze(-1), 
            #                                         path_lengths)
            path_length_rmse = 0
            # ch_nmse = 0
            # if epoch >= 0:

            delay_t = paths_out[:, :, 0].cpu().detach().numpy()
            power_t = paths_out[:, :, 1].cpu().detach().numpy()
            phase = paths_out[:, :, 2].cpu().detach().numpy()
            phase = np.rad2deg(phase)
            az = paths_out[:, :, 3].cpu().detach().numpy()
            el = paths_out[:, :, 4].cpu().detach().numpy()
            power_t = np.where(power_t==config["PAD_VALUE"], 0, power_t)
            power_linear = 10**( (power_t/0.01)/10)



            delay_secs = delay_t/ 1e6
            # mask = path_padding_mask[:, 1:]
            mask= delay_secs == config["PAD_VALUE"]/ 1e6
            delay_secs = np.where(mask, np.nan, delay_secs)
            phase = np.where(mask, np.nan, phase)


            power_linear = np.where(mask, np.nan, power_linear)
            phase = np.where(mask, np.nan, phase)
            az = np.where(mask, np.nan, az)
            el = np.where(mask, np.nan, el)


            
            
            H = mycomputer.compute_channels(power_linear,delay_secs, phase, az, el ,kwargs=None )
            # print("channel->",H.shape)
            H  = H[:,0, :, :]
            beam_label, scores = compute_beam_label_from_channel(H, S)
            labels = beam_label.to(device).long()
            total_loss = criterion(logits, labels)
            
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()


            preds = logits.argmax(dim=1).cpu()
            train_correct += (preds == beam_label).sum().item()
            total += prompts.size(0)
            train_acc = train_correct / total

            train_losses.append(total_loss.item())
            current_lr = optimizer.param_groups[0]["lr"]
            pbar.set_postfix({
                "loss": f"{total_loss.item():.4f}",
                "train_acc" : f"{train_acc:.4f}",
                "lr": f"{current_lr:.2e}"
            })
        # print("pred_channel",predicted_channels[0])
        # print("gt_channel",gt_channels[0])

        avg_train_loss = np.mean(train_losses)

        # print("")
        # -------------------- VALIDATION --------------------
        model.eval()
        total = 0
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]", leave=False)
            # prepare val aoa loss lists
            val_loss_az = []
            val_loss_el = []
            for prompts, paths, path_lengths, interactions, env, env_prop, path_padding_mask in pbar:  # NEW: added interactions
                prompts = prompts.cuda()
                paths = paths.cuda()
                path_lengths = path_lengths.cuda()
                interactions = interactions.cuda()  # NEW
                env_prop = env_prop.cuda()
                env = env.cuda()
                # path_padding_mask = path_padding_mask.cuda()
                
                paths_in = paths[:, :-1, :]
                interactions_in = interactions[:, :-1, :]

                paths_out = paths[:, 1:, :]
                interactions_out = interactions[:, 1:, :]  # NEW: shift targets

                logits = model(prompts, paths, interactions, env_prop, env)
                

                delay_t = paths_out[:, :, 0].cpu().detach().numpy()
                power_t = paths_out[:, :, 1].cpu().detach().numpy()
                phase = paths_out[:, :, 2].cpu().detach().numpy()
                phase = np.rad2deg(phase)
                az = paths_out[:, :, 3].cpu().detach().numpy()
                el = paths_out[:, :, 4].cpu().detach().numpy()
                power_t = np.where(power_t==config["PAD_VALUE"], 0, power_t)
                power_linear = 10**( (power_t/0.01)/10)



                delay_secs = delay_t/ 1e6
                mask = delay_secs == config["PAD_VALUE"]/ 1e6
                # mask = path_padding_mask[:, 1:]
                delay_secs = np.where(mask, np.nan, delay_secs)
                phase = np.where(mask, np.nan, phase)


                power_linear = np.where(mask, np.nan, power_linear)
                phase = np.where(mask, np.nan, phase)
                az = np.where(mask, np.nan, az)
                el = np.where(mask, np.nan, el)


                
                
                H = mycomputer.compute_channels(power_linear,delay_secs, phase, az, el ,kwargs=None )
                # print("channel->",H.shape)
                H  = H[:,0, :, :]
                beam_label, scores = compute_beam_label_from_channel(H, S)
                labels = beam_label.to(device).long()
                val_loss = criterion(logits, labels)

                preds = logits.argmax(dim=1).cpu()
                val_correct += (preds == beam_label).sum().item()
                val_total += prompts.size(0)
                val_acc = val_correct / val_total
                val_losses.append(val_loss.item())

                pbar.set_postfix({
                    "val_loss": f"{val_loss.item():.4f}",
                    "val_acc": f"{val_acc:.4f}",
                })

        avg_val_loss = np.mean(val_losses)

        scheduler.step(avg_val_loss)
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
            }, checkpoint_finetune_path)
            print(f"  ✓ Best checkpoint saved (val_loss: {best_val_loss:.4f})")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        if config.get("USE_WANDB", False):
            import wandb
            wandb.log({
                "train_loss": avg_train_loss,
                "train_acc": train_acc,
                "val_loss": avg_val_loss,
                "val_acc": val_acc,
                "epoch": epoch,
                "lr": current_lr,
            })

        print(f"\nEpoch {epoch:02d}")
        print(f"  Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"  Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")


        print(f"  LR: {current_lr:.3e}")
        # scheduler.step()

    return best_val_acc

# best_epoch, best_loss = load_best_checkpoint(model, checkpoint_path=checkpoint_path)

# =============================================================================
# PARAMETER FREEZING STRATEGY
# =============================================================================

# Define a function to toggle the backbone
def set_backbone_trainable(model, trainable=True):
    # 'transformer' or 'blocks' usually refers to the GPT backbone
    # Adjust based on your GPTPathDecoder architecture
    for name, param in model.named_parameters():
  
        if "transformer" in name or "layers" in name: 
            param.requires_grad = trainable
            


def freeze_backbone(model):
    for param in model.parameters():
        param.requires_grad = False

# %%
all_scenarios = ['city_47_chicago_3p5','city_10_florida_villa_7gp_1758095156175',  'city_23_beijing_3p5', 'city_91_xiangyang_3p5', 'city_17_seattle_3p5_s', 'city_12_fortworth_3p5', 'city_92_sãopaulo_3p5', 'city_35_san_francisco_3p5',  'city_19_oklahoma_3p5_s', 'city_74_chiyoda_3p5']



for scenario in all_scenarios:
    print(f"Beam prediction for {scenario}")
    config = {
        "BATCH_SIZE":128,
        "PAD_VALUE": 500,
        "USE_WANDB": False,
        "LR":2e-3,
        "epochs" : 100,
        "interaction_weight": 0.01,  # Weight for interaction loss
        "experiment": f"true_enc_direct_{scenario}_interacaction_all_inter_str_dec_all_repeat",
        "base_experiment": f"true_enc_direct_{scenario}_interacaction_all_inter_str_dec_all_repeat",
        "hidden_dim": 512,
        "n_layers": 8,
        "n_heads": 8,
        "pre_train":False,
        "task":"beam_prediction"
    }
    
    dataset = dm.load(scenario, )
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


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = f"{config['experiment']}_best_model_checkpoint.pth"
    os.makedirs("checkpoints2", exist_ok=True)
    checkpoint_path = os.path.join("checkpoints2", checkpoint_path)
    os.makedirs("checkpoints_channel", exist_ok=True)
    checkpoint_finetune = f"{config['task']}_{scenario}_best_model_checkpoint.pth"
    checkpoint_finetune_path = os.path.join("checkpoints_channel", checkpoint_finetune)



    backbone_model = PathDecoderEnv(hidden_dim=config["hidden_dim"], n_layers = config["n_layers"], n_heads=config["n_heads"]).to(device)
    # model = PathDecoder().to(device)
    # model = GPTPathDecoder().to(device)
    best_val_loss = float('inf')
    base_model_checkpoint_path = f"{config['base_experiment']}_best_model_checkpoint.pth"
    base_model_checkpoint_path = os.path.join("checkpoints2", base_model_checkpoint_path)

    print("backbone_model parameters:", count_parameters(backbone_model))
    best_epoch, best_loss = load_best_checkpoint(backbone_model, checkpoint_path=base_model_checkpoint_path)
    freeze_backbone(backbone_model)

    checkpoint_path = f"{config['experiment']}_best_model_checkpoint.pth"
    model  = PathFormerBeamPredictor(backbone_model).to(device)
    print("Trainable parameters:", count_parameters(model))
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


    optimizer = torch.optim.AdamW(model.parameters(), lr=config["LR"])
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=3, mode="min")
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    #     optimizer, 
    #     T_0=25,      # Restart every 10 epochs
    #     T_mult=1,    # Double the period after each restart
    #     eta_min=1e-8 # Minimum LR
    # )

    # Initialize best checkpoint tracking (based on path_length loss)

    # scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=2, mode="min")
    mycomputer = MyChannelComputer()




    os.makedirs("checkpoints2", exist_ok=True)
    checkpoint_path = os.path.join("checkpoints2", checkpoint_path)
    # Phase 1: Freeze backbone, train only the projection heads
    print("Phase 1: Freezing transformer backbone. Training output heads only...")
    set_backbone_trainable(model, trainable=False)

    # Verify which parameters are training
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters (Heads only): {trainable_params}")
    # Train
    val_acc = task_finetune(model, train_loader, val_loader, config, train_data)
    scenario_row = {
        "scenario": scenario,
        "val_acc": val_acc}
    df = pd.DataFrame([scenario_row])
    # header=not os.path.exists(...) ensures the header is only written once
    df.to_csv(csv_log_file, mode='a', index=False, header=not os.path.exists(csv_log_file))


# %%



