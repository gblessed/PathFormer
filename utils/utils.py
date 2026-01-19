import torch
from sklearn.metrics import mean_squared_error
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import numpy as np
from typing import Dict
from generator_things.geometry import *
from generator_things import consts as c
from generator_things.channel import _generate_MIMO_channel, ChannelParameters, generate_MIMO_channel_torch

from typing import Tuple, Optional
from numpy.typing import NDArray

criterion =  torch.nn.MSELoss()
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
        d, p, s, c, ph, az_s, az_c, az, el_s, el_c, el, pathcounts, inter_str_logits = model(prompt, cur, inter_str, env, env_prop, pre_train = False)

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



def generate_paths_no_env(model, prompt, max_steps=25, stop_threshold=0.5):
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
def masked_loss_pre_train(delay_pred, power_pred, sin_pred, cos_pred, phase_pred,
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

def masked_loss(delay_pred, power_pred, phase_sin_pred, phase_cos_pred, phase_pred,
                az_sin_pred, az_cos_pred, az_pred, el_sin_pred, el_cos_pred,el_pred,
                path_length_predict, interaction_logits, targets, path_length_targets,
                interaction_targets, finetune=None, pad_value=500, interaction_weight=0.1):
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
    channel_loss = 0
    params = ChannelParameters()
    if finetune == "channel_estimation":
        delay_secs = delay_t/ 1e6
        mask = delay_secs == (pad_value / 1e6)
        power_t = power_t.masked_fill(mask, 0)
        phase_t = torch.rad2deg(phase_t)
        power_linear = 10**( (power_t/0.01)/10)
        power_linear = power_linear.masked_fill(mask, torch.nan)
        default_dopplers = torch.zeros_like(power_linear)
        array_response = compute_single_array_response_torch(params.bs_antenna,  az_t, el_t)

       
        gt_channel = generate_MIMO_channel_torch(array_response, power_linear, delay_secs, phase_t, default_dopplers, ofdm_params= params.ofdm, freq_domain=params.freq_domain )
        # print("gt_channel", gt_channel[0])


        ###############
        delay_pred_secs = delay_pred/ 1e6
        power_pred = power_pred.masked_fill(mask, 0)
        power_linear_pred = 10**( (power_pred/0.01)/10)
        power_linear_pred = power_linear_pred.masked_fill(mask, torch.nan)
        default_dopplers = torch.zeros_like(power_linear_pred)
       
        phase_pred = torch.rad2deg(phase_pred)
        array_response_pred = compute_single_array_response_torch(params.bs_antenna,  az_pred, el_pred)
        pred_channel = generate_MIMO_channel_torch(array_response_pred, power_linear_pred, delay_pred_secs, phase_pred, default_dopplers, ofdm_params= params.ofdm, freq_domain=params.freq_domain )
        
        
        gt_channel = gt_channel * 1e6
        print("gt_channel",  gt_channel[0])
        pred_channel = pred_channel * 1e6
        print("pred_channel", pred_channel[0])
        # print("pred_channel",  pred_channel[0])
        channel_loss = ((gt_channel.real - pred_channel.real )**2).mean() + ((gt_channel.imag - pred_channel.imag )**2).mean()
        channel_loss = channel_loss/ 1e3
        # channel_loss =  1e6 * channel_loss
        # print("channel_loss", channel_loss)
        total_loss += channel_loss
        # az_t = az_t.masked_fill(mask, 0)
        # el_t = el_t.masked_fill(mask, 0)

        # ch_loss = np.linalg.norm(channel_pred - gt_channel)**2
    # total_loss = (loss_delay + 
    #              + interaction_weight * loss_interaction)
     
    return (total_loss, loss_delay, loss_power, loss_phase, 
        loss_az, loss_el, loss_path_length, loss_interaction, channel_loss)


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

    pred = pred_paths  # (T,3)
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





def nmse(y_true, y_pred):
    """
    Calculate the Normalized Mean Squared Error (NMSE) between true and predicted values.

    Args:
        y_true (array-like): Ground truth values.
        y_pred (array-like): Predicted values.

    Returns:
        float: The NMSE value, computed as the mean squared error divided by the mean
               squared magnitude of the true values.
    """

    return np.mean(np.abs(y_true - y_pred)**2) / np.mean(np.abs(y_true)**2)

def pow2db(nmse):
    """
    Convert a Normalized Mean Squared Error (NMSE) value to decibels (dB).

    Args:
        nmse (float): The NMSE value to convert.

    Returns:
        float: The NMSE value in decibels, calculated as 10 * log10(nmse).
    """
    return 10 * np.log10(nmse)


def compute_channel_nmse( predicted_channels, gt_channels):
    
    # pred_delay = pred[:, 0]/ 1e6
    # pred_power = pred[:, 1]/0.01
    # pred_power = 10**(pred_power/10)
    # pred_phase = pred[:, 2]


    # gt_delay = gt[:, 0]/ 1e6
    # gt_power = gt[:, 1]/0.01
    # gt_power = 10**(gt_power/10)
    # gt_phase = gt[:, 2]
    # predicted_channels = mycomputer.compute_channels(pred_power,pred_delay, pred_phase, kwargs=None )
    # gt_channels = mycomputer.compute_channels(gt_power,gt_delay, gt_phase, kwargs=None )

    return pow2db(nmse(predicted_channels, gt_channels))

# ChannelParameters()
# _ = dataset._compute_channels(ch_params)

class MyChannelComputer:

    def _compute_single_array_response(self, ant_params: Dict, theta: np.ndarray, 
                                        phi: np.ndarray) -> np.ndarray:
        """Internal method to compute array response for a single antenna array.
        
        Args:
            ant_params: Antenna parameters dictionary
            theta: Elevation angles array
            phi: Azimuth angles array
            
        Returns:
            Array response matrix
        """
        # Use attribute access for antenna parameters
        kd = 2 * np.pi * ant_params.spacing
        ant_ind = ant_indices(ant_params[c.PARAMSET_ANT_SHAPE]) # tuple complications..
       
        return array_response_batch(ant_ind=ant_ind, theta=theta, phi=phi, kd=kd)


    def set_channel_params(self, params: Optional[ChannelParameters] = None) -> None:
        """Set channel generation parameters.
        
        Args:
            params: Channel generation parameters. If None, uses default parameters.
        """
        if params is None:
            params = ChannelParameters()
            
        # params.validate(dataset.n_ue)
        
        # Create a deep copy of the parameters to ensure isolation
        # old_params = (super().__getitem__(c.CH_PARAMS_PARAM_NAME) 
        #               if c.CH_PARAMS_PARAM_NAME in super().keys() else None)
        old_params = None
        self.ch_params = params.deepcopy()
        
        # If rotation has changed, clear rotated angles cache
        if old_params is not None:
            old_bs_rot = old_params.bs_antenna[c.PARAMSET_ANT_ROTATION]
            old_ue_rot = old_params.ue_antenna[c.PARAMSET_ANT_ROTATION]
            new_bs_rot = params.bs_antenna[c.PARAMSET_ANT_ROTATION]
            new_ue_rot = params.ue_antenna[c.PARAMSET_ANT_ROTATION]
            if not np.array_equal(old_bs_rot, new_bs_rot) or not np.array_equal(old_ue_rot, new_ue_rot):
                self._clear_cache_rotated_angles()
        
        return params
    def _compute_array_response_product(self, aod_az=None, aod_el=None) -> np.ndarray:
        """Internal method to compute product of TX and RX array responses.
        args:
        aod_az , aod_el (Num_ue x path_length) in radians

        Returns:
            Array response product matrix

        """
        # Get antenna parameters from channel parameters
        tx_ant_params = self.ch_params.bs_antenna # base case is 8x1
        rx_ant_params = self.ch_params.ue_antenna # base case is 1x1
        

        array_response_TX = self._compute_single_array_response(
            tx_ant_params, aod_az, aod_el)
            


        return array_response_TX[:, None, :, :]
        return array_response_RX[:, :, None, :] * array_response_TX[:, None, :, :]
    

    def compute_channels(self, _power_linear_ant_gain, delay, phase, aod_az=None, aod_el=None, params: Optional[ChannelParameters] = None, use_doppler = False, **kwargs) -> np.ndarray:
        """Compute MIMO channel matrices for all users.
        
        This is the main public method for computing channel matrices. It handles all the
        necessary preprocessing steps including:
        - Antenna pattern application
        - Field of view filtering
        - Array response computation
        - OFDM processing (if enabled)
        
        The computed channel will be cached and accessible as dataset.channel
        or dataset['channel'] after this call.
        
        Args:
            params: Channel generation parameters. If None, uses default parameters.
                    See ChannelParameters class for details.
            **kwargs: Additional keyword arguments to pass to ChannelParameters constructor
                    if params is None. Ignored if params is provided. 
                    If provided, overrides existing channel parameters (e.g. set_channel_params).
            
        Returns:
            numpy.ndarray: MIMO channel matrix with shape [n_users, n_rx_ant, n_tx_ant, n_subcarriers]
                        if freq_domain=True, otherwise [n_users, n_rx_ant, n_tx_ant, n_paths]
        """
        if params is None:
            if kwargs:
                params = ChannelParameters(**kwargs)
            else:
                params = self.ch_params if self.ch_params is not None else ChannelParameters()

        self.set_channel_params(params)

        # np.random.seed(1001)
        
        # Compute array response product
        array_response_product = self._compute_array_response_product(aod_az, aod_el)
        
        n_paths_to_gen = params.num_paths
        
        # Whether to enable the doppler shift per path in the channel
        n_paths = np.min((n_paths_to_gen, delay.shape[-1]))
        default_doppler = np.zeros((delay.shape[0], n_paths))
        # use_doppler = self.hasattr('doppler')

        if params[c.PARAMSET_DOPPLER_EN] and not use_doppler:
            all_obj_vel = np.array([obj.vel for obj in self.scene.objects])
            # Enable doppler if any velocity component is non-zero
            use_doppler = self.tx_vel.any() or self.rx_vel.any() or all_obj_vel.any()
            if not use_doppler:
                print("No doppler in channel generation because all velocities are zero")

        dopplers = self.doppler[..., :n_paths] if use_doppler else default_doppler

        channel = _generate_MIMO_channel(
            array_response_product=array_response_product[..., :n_paths],
            powers=_power_linear_ant_gain[..., :n_paths],
            delays=delay[..., :n_paths],
            phases=phase[..., :n_paths],
            dopplers=dopplers,
            ofdm_params=params.ofdm,
            freq_domain=params.freq_domain,
        )

        # self[c.CHANNEL_PARAM_NAME] = channel  # Cache the result

        return channel
    





def compute_single_array_response_torch(ant_params: Dict, theta: torch.Tensor, 
                                        phi: torch.Tensor) -> torch.Tensor:
        """PyTorch version of the single array response calculator."""
        # Calculate kd (wavenumber * spacing)
        kd = 2 * torch.pi * ant_params['spacing']
        
        # Determine device from input tensors
        device = theta.device
        
        # Generate antenna indices
        # Assuming PARAMSET_ANT_SHAPE is a key like 'ant_shape' returning (Mx, My)
        ant_shape = ant_params[c.PARAMSET_ANT_SHAPE]
        ant_ind = ant_indices_torch(ant_shape, device=device)
       
        return array_response_batch_torch(ant_ind=ant_ind, theta=theta, phi=phi, kd=kd)



def ant_indices_torch(panel_size: Tuple[int, int], device="cuda") -> torch.Tensor:
    """Generate antenna element indices for a rectangular panel in PyTorch."""
    Mx, My = panel_size
    # Create coordinate grid for y and z (assuming x is constant 0 for a planar array)
    # Using torch.meshgrid for a cleaner implementation than tile/repeat
    y_coords = torch.arange(Mx, device=device)
    z_coords = torch.arange(My, device=device)
    
    # meshgrid creates the 2D panel structure
    Y, Z = torch.meshgrid(y_coords, z_coords, indexing='ij')
    X = torch.zeros_like(Y)
    
    # Flatten and stack to get (N, 3)
    return torch.stack([X.flatten(), Y.flatten(), Z.flatten()], dim=1).float()

def _array_response_phase_torch(theta: torch.Tensor, phi: torch.Tensor, kd: float) -> torch.Tensor:
    """Calculate the phase components of the array response in PyTorch."""
    # Ensure complex type for the imaginary unit multiplication
    gamma_x = 1j * kd * torch.sin(theta) * torch.cos(phi)
    gamma_y = 1j * kd * torch.sin(theta) * torch.sin(phi)
    gamma_z = 1j * kd * torch.cos(theta)
    
    return torch.stack([gamma_x, gamma_y, gamma_z], dim=-1)

def array_response_batch_torch(ant_ind: torch.Tensor, theta: torch.Tensor, phi: torch.Tensor, kd: float) -> torch.Tensor:
    """Calculate vectorized array response vectors in PyTorch."""
    batch_size, n_paths = theta.shape
    n_ant = ant_ind.shape[0]
    device = theta.device
    
    # 1. Handle NaNs with a mask
    valid_mask = ~torch.isnan(theta) # [batch_size, n_paths]
    
    # 2. Compute phases for valid entries
    # theta[valid_mask] flattens the tensor to (n_valid_paths,)
    gamma = _array_response_phase_torch(theta[valid_mask], phi[valid_mask], kd) # [n_valid_paths, 3]
    
    # 3. Initialize complex output tensor
    # Note: Use complex64 or complex128 depending on your precision needs
    result = torch.zeros((batch_size, n_ant, n_paths), dtype=torch.complex64, device=device)
    
    # 4. Get indices of valid paths to map back later
    # nonzero(as_tuple=True) is the PyTorch equivalent of np.nonzero
    batch_idx, path_idx = torch.nonzero(valid_mask, as_tuple=True)
    
    # 5. Compute responses: exp(ant_ind @ gamma.T)
    # Using torch.matmul for the matrix product
    # result: [n_ant, n_valid_paths]
    # print(ant_ind[0], gamma.T [0])
    # valid_responses = torch.exp(torch.matmul(ant_ind, gamma.T))
    # Cast ant_ind to match gamma's complex dtype
    valid_responses = torch.exp(torch.matmul(ant_ind.to(gamma.dtype), gamma.T))
    
    # 6. Scatter valid responses back into the batch structure
    # We transpose valid_responses to [n_valid_paths, n_ant] to fit the indexing
    result[batch_idx, :, path_idx] = valid_responses.T
    
    return result