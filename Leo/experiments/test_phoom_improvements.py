"""The accelerated experiment must preserve Phoom's original DPS calculation."""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phoom_improvements as experiment
from legacy_phoom.diffusion.dps import dps


def test_gram_sampler_matches_phoom_direct_eeg_sampler():
    torch.set_num_threads(1)
    torch.manual_seed(9)
    model = experiment.Denoiser().eval().requires_grad_(False)
    obs = experiment.recordings(1, 90422, 2000)
    i = 1
    obs = dict(
        matrix=obs["matrix"],
        eeg=obs["eeg"][i : i + 1],
        truth=obs["truth"][i : i + 1],
        rows=obs["rows"][i : i + 1],
    )
    expected = dps(
        "mlp", model, 0.025, obs["eeg"], obs["matrix"], 0.2, 10**1.5, samples=2, seed=0
    )
    actual = experiment.sample(
        model,
        experiment.stats(obs, 0.025, "cpu"),
        0.025,
        np.array([10**1.5]),
        samples=2,
        seed=0,
    )
    np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-3)


def test_zero_refinement_leaves_estimate_unchanged():
    estimate = np.random.default_rng(1).normal(size=(2, 256))
    np.testing.assert_array_equal(
        experiment.refine(estimate, {}, np.eye(256), 0), estimate
    )


def test_linear_convolutions_preserve_cnn_outputs_and_gradients():
    import copy
    from fast_conv import use_linear_convolutions
    from legacy_phoom.models.cnn import ConvDenoiser

    torch.set_num_threads(1)
    torch.manual_seed(2)
    original = ConvDenoiser()
    accelerated = use_linear_convolutions(copy.deepcopy(original))
    x = torch.randn(3, 256, requires_grad=True)
    z = x.detach().clone().requires_grad_(True)
    t = torch.tensor([0, 80, 199])
    a = original(x, t)
    b = accelerated(z, t)
    torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)
    a.square().mean().backward()
    b.square().mean().backward()
    torch.testing.assert_close(x.grad, z.grad, atol=1e-6, rtol=1e-4)
    for p, q in zip(original.parameters(), accelerated.parameters()):
        torch.testing.assert_close(p.grad, q.grad, atol=1e-6, rtol=1e-4)


def test_production_refinement_has_correct_noise_scaling():
    from legacy_phoom.diffusion.refinement import measurement_refinement

    h = np.array([[1.0, 2.0], [3.0, 4.0]])
    y = np.array([[5.0, 6.0], [7.0, 8.0]])
    sigma = np.array([1.0, 2.0])
    tau = 0.1
    actual = measurement_refinement(h, y, np.eye(2), sigma, np.eye(2), tau)
    q = tau / sigma[:, None] ** 2
    np.testing.assert_allclose(actual, (h + q * y) / (1 + q))
    np.testing.assert_array_equal(
        measurement_refinement(h, y, np.eye(2), sigma, np.eye(2), 0), h
    )
