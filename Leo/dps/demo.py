"""Run a small reconstruction with a known synthetic ABR."""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data import AMPLITUDE_SCALE_UV, SAMPLE_RATE, TIME_MS, generate_responses
from reconstruct import load_model, reconstruct, stimulus_matrix


def main():
    torch.set_num_threads(4)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="trained model.pt from train.py")
    parser.add_argument("--guidance", type=float, default=0.1)
    parser.add_argument("--output", type=Path, default=Path("dps_demo.npz"))
    args = parser.parse_args()

    # Make a new ABR and pick one that isn't all zeros.
    responses, zeros = generate_responses(20, seed=90421)
    truth = responses[np.flatnonzero(~zeros)[0]]
    rng = np.random.default_rng(90422)
    pulses = np.zeros(SAMPLE_RATE // 2, dtype=np.float32)
    pulses[np.arange(200, len(pulses) - 300, 100)] = 1
    noise_std = 0.5  # That's 0.05 microvolts in our model.
    clean_eeg = (stimulus_matrix(pulses) @ torch.from_numpy(truth)).numpy()
    eeg = clean_eeg + rng.normal(0, noise_std, len(pulses)).astype(np.float32)

    # Now see if we can get the ABR back from the noisy EEG.
    model = load_model(args.checkpoint)
    estimate = reconstruct(model, eeg, pulses, noise_std, args.guidance)
    rmse = np.sqrt(np.mean((estimate - truth) ** 2)) * AMPLITUDE_SCALE_UV
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, truth=truth, estimate=estimate, eeg=eeg,
                        pulses=pulses, time_ms=TIME_MS, noise_std=noise_std,
                        amplitude_scale_uv=AMPLITUDE_SCALE_UV, guidance=args.guidance)
    print(f"ABR RMSE: {rmse:.5f} microvolts")
    print(f"Saved arrays to {args.output}")


if __name__ == "__main__":
    main()
