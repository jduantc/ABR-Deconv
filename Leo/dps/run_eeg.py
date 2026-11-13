"""Reconstruct an ABR from a prepared EEG and pulse-train NPZ file."""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data import AMPLITUDE_SCALE_UV, TIME_MS
from reconstruct import load_model, reconstruct


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("input", type=Path, help="NPZ containing eeg_uv and pulses")
    parser.add_argument("--noise-std-uv", type=float, required=True)
    parser.add_argument("--guidance", type=float, default=0.1)
    parser.add_argument("--output", type=Path, default=Path("dps_eeg_result.npz"))
    args = parser.parse_args()
    torch.set_num_threads(4)

    # The model uses units of 0.1 microvolts, so convert the EEG first.
    with np.load(args.input) as data:
        eeg = data["eeg_uv"].astype(np.float32) / AMPLITUDE_SCALE_UV
        pulses = data["pulses"].astype(np.float32)
    model = load_model(args.checkpoint)
    estimate = reconstruct(model, eeg, pulses, args.noise_std_uv / AMPLITUDE_SCALE_UV,
                           guidance=args.guidance)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, abr_uv=estimate * AMPLITUDE_SCALE_UV,
                        time_ms=TIME_MS, guidance=args.guidance,
                        noise_std_uv=args.noise_std_uv)
    print(f"Saved reconstructed ABR to {args.output}")


if __name__ == "__main__":
    main()
