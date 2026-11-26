"""Scientific invariants: split isolation, forward timing and equivalent DPS loss."""

import sys
from pathlib import Path
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import study
from datasets import SUBJECTS, augment, pca_generate
from dps.reconstruct import stimulus_matrix, reconstruct
from model import make_model
from diffusion import cosine_schedule, clean_from_v, make_training_pair


def test_subject_split():
    a, b, c = map(set, (SUBJECTS["train"], SUBJECTS["validation"], SUBJECTS["test"]))
    assert not (a & b or a & c or b & c)
    assert a | b | c == set(range(1, 9))


def test_forward_timing():
    pulses = np.zeros(500)
    pulses[100] = 1
    h = torch.arange(256, dtype=torch.float32)
    y = stimulus_matrix(pulses) @ h
    torch.testing.assert_close(y[50:306], h)
    assert not y[:50].any() and not y[306:].any()


def test_residual_gram_value_and_gradient():
    torch.manual_seed(9)
    matrix = torch.randn(300, 256)
    y = torch.randn(300)
    h = torch.randn(256, requires_grad=True)
    direct = (matrix @ h - y).square().mean().sqrt()
    A = matrix.T @ matrix / len(y)
    b = matrix.T @ y / len(y)
    gram = (h @ A @ h - 2 * h @ b + (y * y).mean()).sqrt()
    torch.testing.assert_close(direct, gram)
    torch.testing.assert_close(
        torch.autograd.grad(direct, h)[0], torch.autograd.grad(gram, h)[0]
    )


def test_v_inverse_and_architecture_gradients():
    torch.set_num_threads(1)
    torch.manual_seed(1)
    x = torch.randn(3, 256)
    t = torch.tensor([0, 99, 199])
    noise = torch.randn_like(x)
    a = cosine_schedule()
    noisy, v = make_training_pair(x, t, noise, a)
    torch.testing.assert_close(clean_from_v(noisy, v, t, a), x)
    for arch in ["mlp", "cnn", "unet"]:
        m = make_model(arch)
        loss = (m(noisy, t) - v).square().mean()
        loss.backward()
        assert all(
            p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()
        )


def test_generators_reproducible():
    x = np.random.default_rng(0).normal(size=(30, 256)).astype(np.float32)
    for f in [augment, pca_generate]:
        a = f(x, 20, np.random.default_rng(1))
        b = f(x, 20, np.random.default_rng(1))
        assert a.shape == (20, 256) and np.isfinite(a).all()
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("noise_std", [0.25, 0.5, 2.0])
def test_batched_sampler_matches_original(noise_std):
    torch.set_num_threads(1)
    torch.manual_seed(2)
    model = make_model("mlp").eval()
    for p in model.parameters():
        p.requires_grad_(False)
    pulses = np.zeros(500, np.float32)
    pulses[[100, 250]] = 1
    matrix = stimulus_matrix(pulses)
    y = torch.randn(500)
    obs = {
        "A": (matrix.T @ matrix / 500)[None],
        "b": (matrix.T @ y / 500)[None],
        "yy": y.square().mean()[None],
        "rows": [{"sigma": noise_std}],
    }
    # The historical batched sampler divides by sigma; the standalone
    # sampler now absorbs that factor into guidance.
    expected = reconstruct(model, y, pulses, guidance=0.1 / noise_std, seed=1)
    actual = study.sample(model, obs, 0.1, seed=1)[0]
    np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-5)
