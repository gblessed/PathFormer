import numpy as np
import torch

from utils.utils import ChannelParameters, compute_single_array_response_torch


def make_azimuth_codebook(n_beams=64, ant_params=None):
    """Build a 64-beam azimuth sweep for the default 8x1 BS array."""
    if ant_params is None:
        ant_params = ChannelParameters().bs_antenna

    azimuth = torch.linspace(-np.pi / 2, np.pi / 2, steps=n_beams, dtype=torch.float32).unsqueeze(0)
    elevation = torch.full_like(azimuth, np.pi / 2)
    codebook = compute_single_array_response_torch(ant_params, elevation, azimuth)
    return codebook.squeeze(0)


def compute_beam_label_from_channel(H, S):
    if not torch.is_tensor(H):
        H = torch.from_numpy(H)
    if not torch.is_tensor(S):
        S = torch.from_numpy(S)

    H = H.to(torch.complex64)
    S = S.to(torch.complex64)
    Y = S.conj().T @ H
    prx = torch.sum(torch.abs(Y) ** 2, dim=2)
    best = torch.argmax(prx, dim=1)
    return best, prx
