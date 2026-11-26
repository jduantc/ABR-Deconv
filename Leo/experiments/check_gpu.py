"""Fail early unless CUDA can execute training and DPS operations on this GPU."""

import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model import make_model


def main():
    print(f"PyTorch {torch.__version__}; built with CUDA {torch.version.cuda}")
    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA unavailable. Check nvidia-smi and follow experiments/GPU_SETUP.md; CPU fallback is disabled."
        )
    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GiB")
    for arch in ["mlp", "cnn", "unet"]:
        model = make_model(arch).to(device)
        x = torch.randn(4, 256, device=device)
        t = torch.arange(4, device=device)
        loss = model(x, t).square().mean()
        loss.backward()
        if not torch.isfinite(loss):
            raise RuntimeError(f"Nonfinite {arch} loss")
    A = torch.eye(256, device=device)
    torch.linalg.solve(A, torch.ones(256, device=device))
    torch.cuda.synchronize()
    print(
        "CUDA checks passed: all architectures, backward pass and DPS linear algebra."
    )


if __name__ == "__main__":
    main()
