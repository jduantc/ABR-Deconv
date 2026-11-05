"""Residual network that predicts v from a noisy ABR and a diffusion step."""

import math

import torch
from torch import nn

from data import WAVEFORM_LENGTH


class Denoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Linear(WAVEFORM_LENGTH + 64, 256)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(256),
                nn.SiLU(),
                nn.Linear(256, 256),
                nn.SiLU(),
                nn.Linear(256, 256),
            )
            for _ in range(3)
        ])
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
