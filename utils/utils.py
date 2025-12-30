import torch
from sklearn.metrics import mean_squared_error
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

def generate_paths(model, prompt, max_steps=25, stop_threshold=0.5):
    """
    Generate paths autoregressively.
    """
    model.eval()
    prompt = prompt.unsqueeze(0).cuda()  # (1, prompt_dim)

    # Start with SOS tokens
    cur = torch.zeros(1, 1, 3).cuda()  # (1, 1, 3) - delay, power, phase
    inter_str = -1 * torch.ones(1, 1, 4).cuda()  # (1, 1, 4) - interaction labels

    outputs = []
    outputs_inter_str = []

    for t in range(max_steps):
        # Forward pass
        d, p, s, c, ph, pathcounts, inter_str_logits = model(prompt, cur, inter_str)

        # Get last timestep predictions
        d_t = d[:, -1]           # (1,)
        p_t = p[:, -1]           # (1,)
        ph_t = ph[:, -1]         # (1,)
        inter_logits_t = inter_str_logits[:, -1]  # (1, 4)

        # **FIX: Convert logits to binary predictions**
        inter_pred_t = (torch.sigmoid(inter_logits_t) > 0.5).float()  # (1, 4) - binary [0, 1]

        # Store outputs
        outputs.append(torch.stack([d_t, p_t, ph_t], dim=-1))
        outputs_inter_str.append(inter_pred_t)

        # Append predictions for next iteration
        next_path = torch.stack([d_t, p_t, ph_t], dim=-1).unsqueeze(1)  # (1, 1, 3)
        cur = torch.cat([cur, next_path], dim=1)

        # **FIX: Use binary predictions, not logits**
        inter_str = torch.cat([inter_str, inter_pred_t.unsqueeze(1)], dim=1)  # (1, t+2, 4)

    return (torch.stack(outputs, dim=1).squeeze(0).detach().cpu(),  # (T, 3)
            pathcounts,
            torch.stack(outputs_inter_str, dim=1).squeeze(0).detach().cpu())  # (T, 4)




def masked_loss(delay_pred, power_pred, sin_pred, cos_pred, phase_pred,
                path_length_predict, interaction_logits, targets, path_length_targets,
                interaction_targets, pad_value=500, interaction_weight=0.1):
    """
    Added interaction prediction loss as auxiliary task.

    Args:
        interaction_logits: (B, T, 4) - logits for [R, D, S, T]
        interaction_targets: (B, T, 4) - binary labels, -1 for invalid
        interaction_weight: weight for interaction loss
    """
    delay_t, power_t, phase_t = targets[:, :, 0], targets[:, :, 1], targets[:, :, 2]
    sinp = torch.sin(phase_t)
    cosp = torch.cos(phase_t)

    # Mask for valid paths
    mask = (delay_t != pad_value)

    # Existing losses
    loss_delay = ((delay_pred - delay_t)**2)[mask].mean()
    loss_power = ((power_pred - power_t)**2)[mask].mean()
    loss_sin = ((sin_pred - sinp)**2)[mask].mean()
    loss_cos = ((cos_pred - cosp)**2)[mask].mean()
    loss_phase = (loss_sin + loss_cos) / 2

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

    total_loss = (loss_delay + loss_power + loss_phase +
                  loss_path_length + interaction_weight * loss_interaction)

    # total_loss = (loss_delay +
    #              + interaction_weight * loss_interaction)

    return (total_loss, loss_delay, loss_power, loss_phase,
            loss_path_length, loss_interaction)

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
