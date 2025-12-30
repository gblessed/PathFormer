import torch
from sklearn.metrics import mean_squared_error
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import numpy as np


def generate_paths(model, prompt, max_steps=25, stop_threshold=0.5):
    """
    Generate paths autoregressively.
    """
    model.eval()
    prompt = prompt.unsqueeze(0).cuda()  # (1, prompt_dim)

    # Start with SOS tokens (delay, power, phase, aoa_az, aoa_el)
    cur = torch.zeros(1, 1, 5).cuda()  # (1, 1, 5)
    inter_str = -1 * torch.ones(1, 1, 4).cuda()  # (1, 1, 4) - interaction labels

    outputs = []
    outputs_inter_str = []

    for t in range(max_steps):
        # Forward pass - unpack expanded outputs (including aoa preds)
        d, p, s, c, ph, az_s, az_c, az, el_s, el_c, el, pathcounts, inter_str_logits = model(prompt, cur, inter_str)

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



def masked_loss(delay_pred, power_pred, phase_sin_pred, phase_cos_pred, 
                az_sin_pred, az_cos_pred, el_sin_pred, el_cos_pred,
                path_length_predict, interaction_logits, targets, path_length_targets,
                interaction_targets, pad_value=500, interaction_weight=0.1):
    """
    Added interaction prediction loss as auxiliary task.
    
    Args:
        interaction_logits: (B, T, 4) - logits for [R, D, S, T]
        interaction_targets: (B, T, 4) - binary labels, -1 for invalid
        interaction_weight: weight for interaction loss
    """
    delay_t = targets[:, :, 0]
    power_t = targets[:, :, 1]
    phase_t = targets[:, :, 2]
    az_t = targets[:, :, 3]
    el_t = targets[:, :, 4]

    sinp = torch.sin(phase_t)
    cosp = torch.cos(phase_t)

    sin_az_t = torch.sin(az_t)
    cos_az_t = torch.cos(az_t)

    sin_el_t = torch.sin(el_t)
    cos_el_t = torch.cos(el_t)
    
    # Mask for valid paths
    mask = (delay_t != pad_value)

    # Existing losses
    loss_delay = ((delay_pred - delay_t)**2)[mask].mean()
    loss_power = ((power_pred - power_t)**2)[mask].mean()
    loss_sin = ((phase_sin_pred - sinp)**2)[mask].mean()
    loss_cos = ((phase_cos_pred - cosp)**2)[mask].mean()
    loss_phase = (loss_sin + loss_cos) / 2

    # AoA losses
    loss_az_sin = ((az_sin_pred - sin_az_t)**2)[mask].mean()
    loss_az_cos = ((az_cos_pred - cos_az_t)**2)[mask].mean()
    loss_az = (loss_az_sin + loss_az_cos) / 2

    loss_el_sin = ((el_sin_pred - sin_el_t)**2)[mask].mean()
    loss_el_cos = ((el_cos_pred - cos_el_t)**2)[mask].mean()
    loss_el = (loss_el_sin + loss_el_cos) / 2

    loss_path_length = ((path_length_targets - path_length_predict)**2).mean() * 0.0
    
    # NEW: Multi-label interaction loss
    # Mask: valid interactions (not -1)
    interaction_mask = (interaction_targets[:, :, 0] != -1)  # (B, T)
    
    if interaction_mask.any():
        # Binary cross-entropy for multi-label classification
        valid_logits = interaction_logits[interaction_mask]  # (N, 4)
        valid_targets = interaction_targets[interaction_mask]  # (N, 4)
        
        loss_interaction = F.binary_cross_entropy_with_logits(
            valid_logits,
            valid_targets,
            reduction='mean'
        )
    else:
        loss_interaction = torch.tensor(0.0, device=delay_pred.device)
    
    total_loss = (loss_delay + loss_power + loss_phase + loss_az + loss_el +
                  loss_path_length + interaction_weight * loss_interaction)

    # total_loss = (loss_delay + 
    #              + interaction_weight * loss_interaction)
     
    return (total_loss, loss_delay, loss_power, loss_phase, 
        loss_az, loss_el, loss_path_length, loss_interaction)

def compute_stop_metrics(path_count, targets, pad_value=500):
    """

    Args:

    """

    rmse = np.sqrt(mean_squared_error(path_count.cpu().numpy(), targets.squeeze().cpu().numpy()))
    
    return rmse 


def show_example(model, val_loader, sample_index=0, k=25, plot=True, pad_value = 500):
    model.eval()
    prompts, paths, path_lengths,interactions = next(iter(val_loader))

    prompts = prompts.cuda()
    paths = paths.cuda()

    pred_paths, path_lengths_pred,inter_str_pred= generate_paths(model, prompts[sample_index])


    pred = pred_paths[0]  # (T,3)
    gt = paths[sample_index][1:, :3]  # Extract only 3D components (T,3)

    valid = (gt[:,0] != pad_value)
    gt = gt[valid]

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


def evaluate_generation(model, val_loader, n_samples=3):
    model.eval()
    for i, (prompts, paths) in enumerate(val_loader):
        if i >= n_samples:
            break
        pred, path_lengths_pred = generate_paths(model, prompts[0])  # autoregressive generation
        print(f"path lengths pred: {path_lengths_pred[0]}")
        print(f"\nSample {i}")
        print("GT paths (first 5):")
        print(paths[0][:5])
        print("Predicted paths (first 5):")
        print(pred[0][:5])

# %%
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


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