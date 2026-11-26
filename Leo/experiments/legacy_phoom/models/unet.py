# U-Net model implementation

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

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
