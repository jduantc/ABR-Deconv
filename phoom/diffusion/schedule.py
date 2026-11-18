# cosine_schedule, make_training_pair, clean_from_v from Leo/diffusion.py

import math

import torch
import torch.nn.functional as F


def cosine_schedule(steps=200):
    time = torch.arange(steps + 1, dtype=torch.float64) / steps
    signal = torch.cos(((time + 0.008) / 1.008) * math.pi / 2) ** 2
    beta = (1 - signal[1:] / signal[:-1]).clamp(max=0.999)
    return torch.cumprod(1 - beta, dim=0).float()


def cosine_abar(T=1000, s=0.008):
    t = torch.arange(T + 1, dtype=torch.float64) / T
    f = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    abar = (f / f[0]).clamp(1e-5, 0.9999)
    return abar[1:].float()


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


def schedule(network):
    if network == "unet":
        return cosine_abar(), torch.arange(4, 1000, 5)
    return cosine_schedule(), torch.arange(200)


def training_loss(network, model, clean, timesteps, noise, alpha_bar):
    if network == "unet":
        a = alpha_bar[timesteps, None]
        noisy = a.sqrt() * clean + (1 - a).sqrt() * noise
        return F.mse_loss(model(noisy[:, None, :], timesteps)[:, 0], noise)
    noisy, target_v = make_training_pair(clean, timesteps, noise, alpha_bar)
    return (model(noisy, timesteps) - target_v).square().mean()


def predict_clean(network, model, noisy, timesteps, alpha_bar):
    if network == "unet":
        a = alpha_bar[timesteps, None]
        return (noisy - (1 - a).sqrt() * model(noisy[:, None, :], timesteps)[:, 0]) / a.sqrt()
    return clean_from_v(noisy, model(noisy, timesteps), timesteps, alpha_bar)
