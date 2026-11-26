"""Leo temporal CNN, adapted to Phoom's normalized waveform interface."""

import math

import torch
from torch import nn


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
