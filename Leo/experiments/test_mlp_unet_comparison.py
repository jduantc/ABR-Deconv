"""Check the accelerated sampler against the actual reconstruction code."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'Leo'))
sys.path.insert(0, str(ROOT/'Leo/experiments'))
from dps.reconstruct import reconstruct, stimulus_matrix
from guidance_sweep import sample
from model import Denoiser, UNetDenoiser


@pytest.mark.parametrize('factory', [Denoiser, UNetDenoiser])
def test_200_steps_match_standalone_with_unnormalized_residual(factory):
    torch.set_num_threads(1)
    torch.manual_seed(18)
    model = factory().eval().requires_grad_(False)
    pulses = np.zeros(500, np.float32)
    pulses[100:400:100] = 1
    matrix = stimulus_matrix(pulses).numpy().astype(np.float64)
    y = np.random.default_rng(34).normal(0, .7, 500).astype(np.float32)
    stats = {'A': torch.tensor(matrix.T @ matrix / len(y), dtype=torch.float32),
             'b': torch.tensor((y @ matrix / len(y))[None], dtype=torch.float32),
             'yy': torch.tensor([np.mean(y.astype(np.float64)**2)], dtype=torch.float32)}
    direct = reconstruct(model, y, pulses, guidance=6.25, seed=8)
    fast = sample(model, stats, 6.25, 1, 8, 'cpu')[0, 0]
    np.testing.assert_allclose(fast, direct, atol=3e-4, rtol=3e-4)
