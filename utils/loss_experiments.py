import torch
import torch.nn.functional as F

from utils.utils import (
    ChannelParameters,
    compute_single_array_response_torch,
    generate_MIMO_channel_torch,
)


def _build_topk_weight_mask(
    mask,
    max_k=5,
    weighting_mode="exponential",
    decay=0.5,
    explicit_weights=None,
):
    """
    Build a per-time-step weight mask limited to the first `max_k` valid positions.

    Args:
        mask: bool tensor of shape (B, T), True where a path is valid.
        max_k: keep at most the first K valid time steps.
        weighting_mode: "uniform", "exponential", or "explicit".
        decay: exponential decay base if weighting_mode == "exponential".
        explicit_weights: optional list/tuple like [8.0, 4.0, 2.0, 1.0, 1.0]

    Returns:
        weighted_mask: float tensor of shape (B, T)
    """
    device = mask.device
    dtype = torch.float32
    mask_float = mask.float()
    B, T = mask.shape

    time_index = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
    topk_mask = mask & (time_index < max_k)

    if weighting_mode == "uniform":
        step_weights = torch.ones(T, device=device, dtype=dtype)
    elif weighting_mode == "exponential":
        step_weights = decay ** torch.arange(T, device=device, dtype=dtype)
    elif weighting_mode == "explicit":
        if explicit_weights is None or len(explicit_weights) == 0:
            raise ValueError("explicit_weights must be provided when weighting_mode='explicit'.")
        step_weights = torch.ones(T, device=device, dtype=dtype)
        explicit_tensor = torch.tensor(explicit_weights, device=device, dtype=dtype)
        upto = min(T, explicit_tensor.numel())
        step_weights[:upto] = explicit_tensor[:upto]
        if explicit_tensor.numel() < T:
            step_weights[explicit_tensor.numel():] = explicit_tensor[-1]
    else:
        raise ValueError(f"Unknown weighting_mode: {weighting_mode}")

    return topk_mask.float() * step_weights.unsqueeze(0) * mask_float


def _weighted_mean(sq_err, weighted_mask):
    denom = weighted_mask.sum().clamp(min=1e-8)
    return (sq_err * weighted_mask).sum() / denom


def masked_loss_topk_weighted(
    delay_pred,
    power_pred,
    phase_sin_pred,
    phase_cos_pred,
    phase_pred,
    az_sin_pred,
    az_cos_pred,
    az_pred,
    el_sin_pred,
    el_cos_pred,
    el_pred,
    path_length_predict,
    interaction_logits,
    targets,
    path_length_targets,
    interaction_targets,
    finetune=None,
    pad_value=500,
    interaction_weight=0.1,
    path_padding_mask=None,
    top_k=5,
    weighting_mode="exponential",
    decay=0.5,
    explicit_weights=None,
    verbose=False,
):
    """
    Experimental loss: only the first `top_k` valid path steps contribute to the
    regression losses, with optional stronger weighting toward earlier steps.

    This keeps the return signature aligned with `utils.utils.masked_loss`.

    Example:
        total_loss, *metrics = masked_loss_topk_weighted(
            ...,
            top_k=5,
            weighting_mode="explicit",
            explicit_weights=[8.0, 4.0, 2.0, 1.0, 1.0],
        )
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

    if path_padding_mask is not None:
        base_mask = path_padding_mask[:, 1:]
    else:
        base_mask = delay_t != pad_value

    weighted_mask = _build_topk_weight_mask(
        base_mask,
        max_k=top_k,
        weighting_mode=weighting_mode,
        decay=decay,
        explicit_weights=explicit_weights,
    ).to(delay_pred.dtype)

    if verbose:
        used = int((weighted_mask > 0).sum().item())
        total = int(base_mask.sum().item())
        print(
            f"[topk loss] using up to first {top_k} valid paths, "
            f"kept {used}/{total} valid positions, mode={weighting_mode}"
        )

    loss_delay = _weighted_mean((delay_pred - delay_t) ** 2, weighted_mask)
    loss_power = _weighted_mean((power_pred - power_t) ** 2, weighted_mask)
    loss_sin = _weighted_mean((phase_sin_pred - sinp) ** 2, weighted_mask)
    loss_cos = _weighted_mean((phase_cos_pred - cosp) ** 2, weighted_mask)
    loss_phase = (loss_sin + loss_cos) / 2

    loss_az_sin = _weighted_mean((az_sin_pred - sin_az_t) ** 2, weighted_mask)
    loss_az_cos = _weighted_mean((az_cos_pred - cos_az_t) ** 2, weighted_mask)
    loss_az = (loss_az_sin + loss_az_cos) / 2

    loss_el_sin = _weighted_mean((el_sin_pred - sin_el_t) ** 2, weighted_mask)
    loss_el_cos = _weighted_mean((el_cos_pred - cos_el_t) ** 2, weighted_mask)
    loss_el = (loss_el_sin + loss_el_cos) / 2

    loss_path_length = ((path_length_targets - path_length_predict) ** 2).mean() * 0.0

    interaction_mask = interaction_targets[:, :, 0] != -1
    if interaction_mask.any():
        valid_logits = interaction_logits[interaction_mask]
        valid_targets = interaction_targets[interaction_mask]
        loss_interaction = F.binary_cross_entropy_with_logits(
            valid_logits,
            valid_targets,
            reduction="mean",
        )
    else:
        loss_interaction = torch.tensor(0.0, device=delay_pred.device)

    total_loss = (
        loss_delay
        + loss_power
        + loss_phase
        + loss_az
        + loss_el
        + loss_path_length
        + interaction_weight * loss_interaction
    )

    channel_loss = 0
    params = ChannelParameters()
    if finetune == "channel_estimation":
        delay_secs = delay_t / 1e6
        if path_padding_mask is not None:
            channel_mask = ~path_padding_mask[:, 1:]
        else:
            channel_mask = delay_secs == (pad_value / 1e6)

        power_t_masked = power_t.masked_fill(channel_mask, 0)
        phase_degs = torch.rad2deg(phase_t)
        power_linear = 10 ** ((power_t_masked / 0.01) / 10)
        power_linear = power_linear.masked_fill(channel_mask, torch.nan)
        default_dopplers = torch.zeros_like(power_linear)
        array_response = compute_single_array_response_torch(params.bs_antenna, az_t, el_t)

        gt_channel = generate_MIMO_channel_torch(
            array_response,
            power_linear,
            delay_secs,
            phase_degs,
            default_dopplers,
            ofdm_params=params.ofdm,
            freq_domain=params.freq_domain,
        )

        delay_pred_secs = delay_pred / 1e6
        power_pred_masked = power_pred.masked_fill(channel_mask, 0)
        power_pred_clamped = power_pred_masked.clamp(-15000.0, 500.0)
        power_linear_pred = 10 ** ((power_pred_clamped / 0.01) / 10)
        power_linear_pred = power_linear_pred.masked_fill(channel_mask, torch.nan)
        default_dopplers = torch.zeros_like(power_linear_pred)
        phase_pred_degs = torch.rad2deg(phase_pred)
        array_response_pred = compute_single_array_response_torch(params.bs_antenna, az_pred, el_pred)

        pred_channel = generate_MIMO_channel_torch(
            array_response_pred,
            power_linear_pred,
            delay_pred_secs,
            phase_pred_degs,
            default_dopplers,
            ofdm_params=params.ofdm,
            freq_domain=params.freq_domain,
        )

        gt_channel = torch.nan_to_num(gt_channel, nan=0.0)
        pred_channel = torch.nan_to_num(pred_channel, nan=0.0)
        channel_loss = (
            (gt_channel.real - pred_channel.real) ** 2
            + (gt_channel.imag - pred_channel.imag) ** 2
        ).mean()
        total_loss = total_loss + channel_loss

    return (
        total_loss,
        loss_delay,
        loss_power,
        loss_phase,
        loss_az,
        loss_el,
        loss_path_length,
        loss_interaction,
        channel_loss,
    )
