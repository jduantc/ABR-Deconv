"""Cosine noise schedule and the v-prediction training formulas."""

import math

import torch


DIFFUSION_STEPS = 200


def cosine_schedule(steps=DIFFUSION_STEPS):
    time = torch.arange(steps + 1, dtype=torch.float64) / steps
    signal = torch.cos(((time + 0.008) / 1.008) * math.pi / 2) ** 2
    beta = (1 - signal[1:] / signal[:-1]).clamp(max=0.999)
    return torch.cumprod(1 - beta, dim=0).float()


def make_training_pair(clean, timesteps, noise, alpha_bar):
    signal_weight = alpha_bar[timesteps, None].sqrt()
    noise_weight = (1 - alpha_bar[timesteps, None]).sqrt()
    noisy = signal_weight * clean + noise_weight * noise
    target_v = signal_weight * noise - noise_weight * clean
    return noisy, target_v


def clean_from_v(noisy, predicted_v, timesteps, alpha_bar):
    signal_weight = alpha_bar[timesteps, None].sqrt()
    noise_weight = (1 - alpha_bar[timesteps, None]).sqrt()
    return signal_weight * noisy - noise_weight * predicted_v
