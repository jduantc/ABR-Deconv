"""Reconstruct an ABRR from EEG using DPS"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data import WAVEFORM_LENGTH
from diffusion import clean_from_v, cosine_schedule
from model import Denoiser


def stimulus_matrix(pulses, response_length=WAVEFORM_LENGTH, onset_index=50):
    """Use the pulse timings to turn one ABR into an EEG signal."""
    pulses = torch.as_tensor(pulses, dtype=torch.float32)
    if pulses.ndim != 1 or len(pulses) == 0:
        raise ValueError("pulses must be a nonempty one-ddimensional array")
    # Our ABR starts 50 samples before the pulse, so keep that offset here.
    eeg_times = torch.arange(len(pulses))[:, None]
    abr_times = torch.arange(response_length)[None, :] - onset_index
    indices = eeg_times - abr_times
    valid = (indices >= 0) & (indices < len(pulses))
    return pulses[indices.clamp(0, len(pulses) - 1)] * valid


def load_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = Denoiser()
    model.load_state_dict(checkpoint["model"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def reconstruct(model, eeg, pulses, noise_std, guidance=0.1, seed=1):
    """Start with some noise and use the EEG to guide the ABR reconstruction

    EEG and noise_std must be in the model's amplitude unit
    The pulse train contains event weights, not EEG voltaages.
    """
    if noise_std <= 0 or guidance < 0:
        raise ValueError("noise_std must be positive and guidance nonnegative")
    eeg = torch.as_tensor(eeg, dtype=torch.float32)
    matrix = stimulus_matrix(pulses)
    if eeg.ndim != 1 or len(eeg) != len(matrix):
        raise ValueError("eeg and pulses must be 1D arrays of the same length")
    alpha_bar = cosine_schedule()
    generator = torch.Generator().manual_seed(seed)
    current = torch.randn((1, WAVEFORM_LENGTH), generator=generator)

    for t in range(len(alpha_bar) - 1, -1, -1):
        current = current.detach().requires_grad_(True)
        step = torch.tensor([t], dtype=torch.long)
        predicted_v = model(current, step)
        clean = clean_from_v(current, predicted_v, step, alpha_bar)
        predicted_eeg = matrix @ clean[0]

        # See how well this ABR explains the EEG, then get a direction to fix it
        residual = (eeg - predicted_eeg) / noise_std
        loss = torch.linalg.vector_norm(residual) / np.sqrt(len(eeg))
        gradient = torch.autograd.grad(loss, current)[0]

        # Take one step towards less noise and nudge it using the EEG.
        a = alpha_bar[t]
        previous_a = alpha_bar[t - 1] if t > 0 else torch.tensor(1.0)
        alpha = a / previous_a
        beta = 1 - alpha
        mean = (previous_a.sqrt() * beta / (1 - a)) * clean
        mean = mean + (alpha.sqrt() * (1 - previous_a) / (1 - a)) * current
        variance = beta * (1 - previous_a) / (1 - a)
        noise = torch.randn(current.shape, generator=generator) if t > 0 else 0
        current = mean + variance.sqrt() * noise - guidance * gradient

    return current.detach().numpy()[0]
