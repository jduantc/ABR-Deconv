"""Residual network that predicts v from a noisy ABR and a diffusion step."""

import math

import torch
from torch import nn

from data import WAVEFORM_LENGTH


class Denoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Linear(WAVEFORM_LENGTH + 64, 256)
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(256),
                    nn.SiLU(),
                    nn.Linear(256, 256),
                    nn.SiLU(),
                    nn.Linear(256, 256),
                )
                for _ in range(3)
            ]
        )
        self.output = nn.Sequential(
            nn.LayerNorm(256), nn.SiLU(), nn.Linear(256, WAVEFORM_LENGTH)
        )
        self.register_buffer(
            "freq", torch.exp(-math.log(10_000) * torch.arange(32) / 31)
        )

    def forward(self, noisy, timesteps):
        # Sine and cosine features tell the network which noise level it sees.
        angles = timesteps.float()[:, None] * self.freq[None, :]
        features = torch.cat([noisy, torch.sin(angles), torch.cos(angles)], dim=-1)
        hidden = self.input(features)
        for block in self.blocks:
            hidden = hidden + 0.5 * block(hidden)
        return self.output(hidden)


class ConvDenoiser(nn.Module):
    """Dilated temporal residual CNN; same input, timestep and v target as MLP."""

    def __init__(self, channels=16):
        super().__init__()
        self.register_buffer(
            "freq", torch.exp(-math.log(10_000) * torch.arange(32) / 31)
        )
        self.input = nn.Conv1d(1, channels, 5, padding=2)
        self.time = nn.Linear(64, channels)
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(4, channels),
                    nn.SiLU(),
                    nn.Conv1d(channels, channels, 3, padding=d, dilation=d),
                    nn.SiLU(),
                    nn.Conv1d(channels, channels, 1),
                )
                for d in (1, 2, 4, 8, 16, 32)
            ]
        )
        self.output = nn.Conv1d(channels, 1, 5, padding=2)

    def forward(self, noisy, timesteps):
        angles = timesteps.float()[:, None] * self.freq[None]
        time = self.time(torch.cat([angles.sin(), angles.cos()], -1))[:, :, None]
        h = self.input(noisy[:, None]) + time
        for block in self.blocks:
            h = h + 0.5 * block(h)
        return self.output(h)[:, 0]


class UNetBlock(nn.Module):
    """Residual convolution block with timestep-dependent scale and shift."""

    def __init__(self, cin, cout, time_dim=64):
        super().__init__()
        self.norm1 = nn.GroupNorm(4, cin)
        self.conv1 = nn.Conv1d(cin, cout, 3, padding=1)
        self.norm2 = nn.GroupNorm(4, cout)
        self.condition = nn.Linear(time_dim, 2 * cout)
        self.conv2 = nn.Conv1d(cout, cout, 3, padding=1)
        self.skip = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, time):
        h = self.conv1(torch.nn.functional.silu(self.norm1(x)))
        scale, shift = self.condition(time).chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale[:, :, None]) + shift[:, :, None]
        h = self.conv2(torch.nn.functional.silu(h))
        return (self.skip(x) + h) / math.sqrt(2)


class UNetDenoiser(nn.Module):
    """1D U-Net: 256→128→64→32 samples, with matching-resolution skips."""

    def __init__(self):
        super().__init__()
        self.register_buffer(
            "freq", torch.exp(-math.log(10_000) * torch.arange(32) / 31)
        )
        self.time = nn.Sequential(nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 64))
        self.input = nn.Conv1d(1, 16, 3, padding=1)
        self.encoders = nn.ModuleList(
            [UNetBlock(16, 16), UNetBlock(16, 32), UNetBlock(32, 64), UNetBlock(64, 64)]
        )
        self.middle = UNetBlock(64, 64)
        self.decoders = nn.ModuleList(
            [UNetBlock(128, 64), UNetBlock(96, 32), UNetBlock(48, 16)]
        )
        self.output = nn.Sequential(
            nn.GroupNorm(4, 16), nn.SiLU(), nn.Conv1d(16, 1, 3, padding=1)
        )

    def forward(self, noisy, timesteps):
        angles = timesteps.float()[:, None] * self.freq[None]
        time = self.time(torch.cat([angles.sin(), angles.cos()], -1))
        h = self.input(noisy[:, None])
        skips = []
        for i, block in enumerate(self.encoders):
            if i:
                h = torch.nn.functional.avg_pool1d(h, 2)
            h = block(h, time)
            skips.append(h)
        h = self.middle(h, time)
        for block, skip in zip(self.decoders, reversed(skips[:-1])):
            h = torch.nn.functional.interpolate(h, size=skip.shape[-1], mode="nearest")
            h = block(torch.cat([h, skip], dim=1), time)
        return self.output(h)[:, 0]


def make_model(architecture="mlp"):
    if architecture not in ("mlp", "cnn", "unet"):
        raise ValueError(f"Unknown architecture: {architecture}")
    return {"mlp": Denoiser, "cnn": ConvDenoiser, "unet": UNetDenoiser}[architecture]()
