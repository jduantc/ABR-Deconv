"""Check scientific equivalence and that diagnostic truth cannot steer sampling."""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import reconstruction_suite as suite
from dps.reconstruct import reconstruct, stimulus_matrix
from model import make_model


def test_sampler_matches_individual_reconstruction_and_trace_is_passive():
    torch.set_num_threads(1)
    torch.manual_seed(7)
    model = make_model("cnn").eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    pulses = np.zeros(500, np.float32)
    pulses[[90, 220]] = 1
    matrix = stimulus_matrix(pulses)
    eeg = torch.randn(500)
    obs = {
        "A": (matrix.T @ matrix / 500)[None],
        "b": (matrix.T @ eeg / 500)[None],
        "yy": eeg.square().mean()[None],
        "rows": [{"sigma": 0.5}],
        "truth": np.zeros((1, 256), np.float32),
    }
    # The older comparison sampler normalizes by sigma=0.5.
    expected = reconstruct(model, eeg, pulses, guidance=0.1 / 0.5, seed=42)
    trace = []
    actual = suite.reconstruct(model, obs, 0.1, trace=trace)
    np.testing.assert_allclose(actual[0], expected, atol=3e-5, rtol=3e-4)
    obs["truth"][:] = 100
    changed_truth = suite.reconstruct(model, obs, 0.1, trace=[])
    np.testing.assert_array_equal(changed_truth, actual)
    assert len(trace) == 200 and trace[-1]["diffusion_step"] == 200


def test_guidance_batch_matches_separate_calls():
    torch.set_num_threads(1)
    torch.manual_seed(8)
    model = make_model("cnn").eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    matrix = torch.randn(300, 256)
    eeg = torch.randn(300)
    obs = {
        "A": (matrix.T @ matrix / 300)[None],
        "b": (matrix.T @ eeg / 300)[None],
        "yy": eeg.square().mean()[None],
        "rows": [{"sigma": 0.5}],
        "truth": np.zeros((1, 256), np.float32),
    }
    batch = {
        k: (
            v * 2
            if k == "rows"
            else np.tile(v, (2, 1))
            if k == "truth"
            else v.repeat((2,) + (1,) * (v.ndim - 1))
        )
        for k, v in obs.items()
    }
    combined = suite.reconstruct(model, batch, torch.tensor([[0.1], [0.3]]), repeats=2)
    for i, guidance in enumerate((0.1, 0.3)):
        separate = suite.reconstruct(model, obs, guidance)
        np.testing.assert_allclose(combined[i], separate[0], atol=3e-5, rtol=3e-4)
