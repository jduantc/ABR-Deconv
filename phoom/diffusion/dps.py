# dps loop from reconstruct.py (Leo)

import argparse

import numpy as np
import torch

from data.synthetic import NOISE_UV, test_recordings
from diffusion.schedule import predict_clean, schedule
from models.mlp import Denoiser
from models.unet import UNet1D

# Tuned on separate recordings for the early-stopped networks, one value per noise level.
GUIDANCE = {"mlp": [10, 10 ** 1.5, 100], "unet": [10, 100, 10 ** 2.5]}


def load(path, device):
    checkpoint = torch.load(path, map_location=device)
    network = checkpoint["recipe"]
    model = (UNet1D() if network == "unet" else Denoiser()).to(device)
    model.load_state_dict(checkpoint["state"])
    model.eval().requires_grad_(False)
    return network, model, checkpoint["scale"]


def dps(network, model, scale, eeg_uv, matrix, noise_uv, guidance, samples=32, seed=0):
    """eeg_uv (recordings, time) in uV, matrix (time, 256), noise_uv the noise standard deviation.
    The model works on waveforms divided by scale. Returns (recordings, samples, 256) in uV."""
    device = next(model.parameters()).device
    alpha_bar, levels = schedule(network)
    alpha_bar, levels = alpha_bar.to(device), levels.to(device)
    matrix = torch.as_tensor(matrix, dtype=torch.float64, device=device)
    eeg = torch.as_tensor(eeg_uv, dtype=torch.float64, device=device).repeat_interleave(samples, dim=0)
    generator = torch.Generator(device=device).manual_seed(seed)
    current = torch.randn((len(eeg), matrix.shape[1]), device=device, generator=generator)

    for k in range(len(levels) - 1, -1, -1):
        t = levels[k]
        current = current.detach().requires_grad_(True)
        clean = predict_clean(network, model, current, t.expand(len(current)), alpha_bar)

        # How well each waveform explains its EEG, then the direction that improves it.
        residual = (eeg - scale * clean.double() @ matrix.T) / noise_uv
        loss = torch.linalg.vector_norm(residual, dim=1) / np.sqrt(matrix.shape[0])
        gradient = torch.autograd.grad(loss.sum(), current)[0]

        # One step towards less noise, nudged by the EEG.
        a = alpha_bar[t]
        previous_a = alpha_bar[levels[k - 1]] if k > 0 else torch.ones((), device=device)
        alpha = a / previous_a
        beta = 1 - alpha
        mean = (previous_a.sqrt() * beta / (1 - a)) * clean.detach()
        mean = mean + (alpha.sqrt() * (1 - previous_a) / (1 - a)) * current.detach()
        variance = beta * (1 - previous_a) / (1 - a)
        noise = torch.randn(current.shape, device=device, generator=generator) if k > 0 else 0
        current = mean + variance.sqrt() * noise - guidance * gradient

    return (current.detach() * scale).view(-1, samples, matrix.shape[1]).cpu().numpy()


def test_errors(network, model, scale):
    """RMSE in uV of the mean of 32 samples on the 12 test recordings, at each noise level."""
    truth, matrix, eeg = test_recordings()
    errors = {}
    for noise_uv, guidance in zip(NOISE_UV, GUIDANCE[network]):
        estimate = dps(network, model, scale, eeg[noise_uv], matrix, noise_uv, guidance).mean(1)
        errors[noise_uv] = float(np.sqrt(np.mean((estimate - truth) ** 2, axis=-1)).mean())
    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    args = parser.parse_args()
    network, model, scale = load(args.checkpoint, "cuda" if torch.cuda.is_available() else "cpu")
    for noise_uv, error in test_errors(network, model, scale).items():
        print(f"noise {noise_uv} uV: RMSE {error:.5f} uV")

if __name__ == "__main__":
    main()


