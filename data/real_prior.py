"""Priors from real responses (refs/*.npy, see real_refs.py), leaving out
the subject being tested. 13 subjects go in (sub-exp303 isn't in refs/).

RealGaussian  mean + covariance of the responses
finetune()    keep training the synthetic DDPM on them, with gain/shift augmentation
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import diffusion as dm
from sim import NK

HERE = Path(__file__).parent
REFS = HERE / "refs"


def load_refs(heldout):
    """(n, NK) responses in uV from every subject except `heldout`."""
    files = [p for p in sorted(REFS.glob("*.npy")) if not p.name.startswith(heldout)]
    return np.concatenate([np.load(p) for p in files], axis=0)


class RealGaussian:
    def __init__(self, heldout, ridge_frac=0.05):
        h = load_refs(heldout)
        self.n = len(h)
        self.mu = h.mean(0)
        C = np.cov(h.T)
        # ~400 samples for a 256x256 covariance, so add a ridge of 5% of the mean variance
        self.C = C + ridge_frac * np.trace(C) / NK * np.eye(NK)
        self.Cinv = np.linalg.inv(self.C)


def augment(h, rng):
    """Random gain, a shift of up to +-0.5 ms, and a little noise, so the
    model does not just memorise 400 waveforms."""
    n = len(h)
    gain = np.exp(rng.normal(0, 0.35, (n, 1)))
    shift = rng.integers(-5, 6, n)
    out = np.stack([np.roll(hi, s) for hi, s in zip(h, shift)]) * gain
    return out + rng.normal(0, 0.03 * h.std(), out.shape)


def finetune(heldout, steps=3000, bs=128, lr=1e-4, seed=0):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    h = load_refs(heldout)
    scale = float(h.std())
    model, _ = dm.load()
    model.train()
    ema = dm.UNet1D().to(dm.DEV)
    ema.load_state_dict(model.state_dict())
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    print(f"finetuning on {len(h)} responses (std {scale * 1e3:.1f} nV), holding out {heldout}")
    for step in range(steps):
        batch = augment(h[rng.integers(0, len(h), bs)], rng) / scale
        x0 = torch.tensor(batch, dtype=torch.float32, device=dm.DEV)[:, None, :]
        t = torch.randint(0, dm.T, (bs,), device=dm.DEV)
        a = dm.ABAR[t][:, None, None]
        eps = torch.randn_like(x0)
        loss = F.mse_loss(model(a.sqrt() * x0 + (1 - a).sqrt() * eps, t), eps)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.mul_(0.995).add_(pm, alpha=0.005)
        if step % 500 == 0 or step == steps - 1:
            print(f"step {step:5d} loss {loss.item():.4f}", flush=True)
    path = HERE / f"ckpt_real_{heldout}.pt"
    torch.save({"ema": ema.state_dict(), "scale": scale}, path)
    return path


def load_real(heldout):
    return dm.load(HERE / f"ckpt_real_{heldout}.pt")


if __name__ == "__main__":
    finetune(sys.argv[1] if len(sys.argv) > 1 else "sub-exp301")
