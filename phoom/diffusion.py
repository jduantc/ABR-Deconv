"""DDPM prior over ABRs (eps-prediction, cosine schedule, small 1-D UNet)
and DiffPIR posterior sampling. Each reverse step denoises to x0, then
solves

    (A + rho_t I) x = b + rho_t x0,    rho_t = lam * abar_t / (1 - abar_t)

i.e. the Gaussian-prior solve with mu = x0, C^-1 = rho_t I.

    python diffusion.py train     # ckpt.pt, ~10 min on a laptop GPU
"""
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sim import NK, sample_abr

DEV = "cuda" if torch.cuda.is_available() else "cpu"
CKPT = Path(__file__).parent / "ckpt.pt"
T = 1000


def cosine_abar(T=T, s=0.008):
    t = torch.arange(T + 1, dtype=torch.float64) / T
    f = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    abar = (f / f[0]).clamp(1e-5, 0.9999)
    return abar[1:].float()


ABAR = cosine_abar().to(DEV)


class ResBlock(nn.Module):
    def __init__(self, cin, cout, tdim):
        super().__init__()
        self.n1 = nn.GroupNorm(8, cin)
        self.c1 = nn.Conv1d(cin, cout, 3, padding=1)
        self.n2 = nn.GroupNorm(8, cout)
        self.c2 = nn.Conv1d(cout, cout, 3, padding=1)
        self.t = nn.Linear(tdim, cout)
        self.skip = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, temb):
        h = self.c1(F.silu(self.n1(x))) + self.t(temb)[:, :, None]
        h = self.c2(F.silu(self.n2(h)))
        return h + self.skip(x)


class Attn(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.n = nn.GroupNorm(8, c)
        self.qkv = nn.Conv1d(c, 3 * c, 1)
        self.out = nn.Conv1d(c, c, 1)

    def forward(self, x):
        q, k, v = self.qkv(self.n(x)).chunk(3, dim=1)
        a = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
        return x + self.out(a.transpose(1, 2))


class UNet1D(nn.Module):
    """Channels 32-64-128-128 over 256-128-64-32 samples, attention on the
    two coarsest levels. About 1.1M parameters."""

    def __init__(self, base=32, mult=(1, 2, 4, 4), attn_levels=(2, 3)):
        super().__init__()
        tdim = 4 * base
        self.base = base
        self.temb = nn.Sequential(nn.Linear(base, tdim), nn.SiLU(), nn.Linear(tdim, tdim))
        self.inp = nn.Conv1d(1, base, 3, padding=1)
        chs = [base * m for m in mult]
        self.down = nn.ModuleList()
        self.up = nn.ModuleList()
        cin = base
        for i, c in enumerate(chs):
            attn = Attn(c) if i in attn_levels else nn.Identity()
            self.down.append(nn.ModuleList([ResBlock(cin, c, tdim), attn]))
            cin = c
        self.mid = ResBlock(cin, cin, tdim)
        for i, c in reversed(list(enumerate(chs))):
            attn = Attn(c) if i in attn_levels else nn.Identity()
            self.up.append(nn.ModuleList([ResBlock(cin + c, c, tdim), attn]))
            cin = c
        self.out = nn.Sequential(nn.GroupNorm(8, cin), nn.SiLU(), nn.Conv1d(cin, 1, 3, padding=1))

    def forward(self, x, t):
        # sinusoidal embedding of the timestep
        half = self.base // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=x.device) / half)
        ang = t.float()[:, None] * freqs[None]
        temb = self.temb(torch.cat([ang.sin(), ang.cos()], dim=1))

        h = self.inp(x)
        skips = []
        for i, (res, attn) in enumerate(self.down):
            h = attn(res(h, temb))
            skips.append(h)
            if i < len(self.down) - 1:
                h = F.avg_pool1d(h, 2)
        h = self.mid(h, temb)
        for i, (res, attn) in enumerate(self.up):
            h = attn(res(torch.cat([h, skips.pop()], dim=1), temb))
            if i < len(self.up) - 1:
                h = F.interpolate(h, scale_factor=2, mode="nearest")
        return self.out(h)


def train(steps=12000, bs=256, lr=2e-4, n_data=200_000, seed=0):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    data = sample_abr(n_data, rng)
    scale = float(data.std())          # network works in units of this
    data = torch.tensor(data / scale, dtype=torch.float32, device=DEV)[:, None, :]
    model = UNet1D().to(DEV)
    ema = UNet1D().to(DEV)
    ema.load_state_dict(model.state_dict())
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    warmup = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 500))
    print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params, data std {scale:.4f} uV, {DEV}")
    for step in range(steps):
        x0 = data[torch.randint(0, n_data, (bs,), device=DEV)]
        t = torch.randint(0, T, (bs,), device=DEV)
        a = ABAR[t][:, None, None]
        eps = torch.randn_like(x0)
        xt = a.sqrt() * x0 + (1 - a).sqrt() * eps
        loss = F.mse_loss(model(xt, t), eps)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        warmup.step()
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.mul_(0.999).add_(pm, alpha=0.001)
        if step % 1000 == 0 or step == steps - 1:
            print(f"step {step:6d}  loss {loss.item():.4f}", flush=True)
    torch.save({"ema": ema.state_dict(), "scale": scale}, CKPT)
    return ema, scale


def load(path=CKPT):
    ck = torch.load(path, map_location=DEV)
    model = UNet1D().to(DEV)
    model.load_state_dict(ck["ema"])
    model.eval()
    return model, ck["scale"]


@torch.no_grad()
def sample_posterior(model, scale, A, b, n=32, n_steps=100, lam=1.0, zeta=1.0, seed=0):
    """n DiffPIR posterior samples given (A, b) in uV; returns (n, NK) in uV.
    A, b are rescaled to the network's units (h = scale * x). zeta = fraction
    of fresh noise when re-noising (1 = DDPM-like, 0 = DDIM)."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    At = torch.tensor(A * scale ** 2, dtype=torch.float64, device=DEV)
    bt = torch.tensor(b * scale, dtype=torch.float64, device=DEV)
    I = torch.eye(NK, dtype=torch.float64, device=DEV)
    x = torch.randn(n, 1, NK, device=DEV, generator=g)
    ts = torch.linspace(T - 1, 0, n_steps, device=DEV).round().long()
    for i, t in enumerate(ts):
        a = ABAR[t]
        eps = model(x, t.expand(n))
        x0 = (x - (1 - a).sqrt() * eps) / a.sqrt()           # denoised guess
        rho = lam * (a / (1 - a)).double()
        rhs = bt[None, :] + rho * x0[:, 0].double()
        x0 = torch.linalg.solve(At + rho * I, rhs.T).T.float()[:, None, :]   # data step
        if i + 1 < len(ts):
            a_prev = ABAR[ts[i + 1]]
            eps_hat = (x - a.sqrt() * x0) / (1 - a).sqrt()
            z = torch.randn(x.shape, device=DEV, generator=g)
            x = a_prev.sqrt() * x0 + (1 - a_prev).sqrt() * (math.sqrt(1 - zeta) * eps_hat + math.sqrt(zeta) * z)
        else:
            x = x0
    return x[:, 0].cpu().numpy() * scale


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        train()
