"""Create clean synthetic ABRs for diffusion training."""

import argparse
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt


SAMPLE_RATE = 10_000
WAVEFORM_LENGTH = 256
AMPLITUDE_SCALE_UV = 0.1
TIME_MS = (np.arange(WAVEFORM_LENGTH) - 50) / 10
ROOT = Path(__file__).resolve().parent


def generate_responses(count, seed):
    rng = np.random.default_rng(seed)
    time_ms = np.arange(-500, 501) / 10

    latency = rng.uniform(3, 7, (count, 1))
    width = rng.uniform(0.35, 0.9, (count, 1))
    amplitude = np.exp(rng.normal(0, 0.35, (count, 1))).clip(0.3, 2.5)
    trough_delay = rng.uniform(1.3, 3.2, (count, 1))
    trough_amplitude = rng.uniform(0.35, 0.8, (count, 1))
    trough_width = width * rng.uniform(1, 1.8, (count, 1))
    early_amplitude = rng.uniform(0, 0.3, (count, 1))
    early_latency = latency - rng.uniform(1.2, 2.4, (count, 1))

    def gaussian(center, spread):
        return np.exp(-0.5 * ((time_ms - center) / spread) ** 2)

    responses = amplitude * gaussian(latency, width)
    responses -= amplitude * trough_amplitude * gaussian(
        latency + trough_delay, trough_width
    )
    responses += amplitude * early_amplitude * gaussian(early_latency, width * 0.65)
    responses[:, time_ms < 0] = 0

    # Filter on a longer interval before keeping the central 256 samples.
    bandpass = butter(1, [80, 1500], btype="bandpass", fs=SAMPLE_RATE, output="sos")
    responses = sosfiltfilt(bandpass, responses, axis=-1)
    responses = responses[:, 450:706]

    is_zero = rng.random(count) < 0.1
    responses[is_zero] = 0
    return responses.astype(np.float32), is_zero


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20261039)
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "train.npz")
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be positive")

    responses, is_zero = generate_responses(args.count, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        responses=responses,
        is_zero=is_zero,
        time_ms=TIME_MS,
        seed=args.seed,
        sample_rate=SAMPLE_RATE,
        amplitude_scale_uV=AMPLITUDE_SCALE_UV,
    )
    print(f"Saved {len(responses)} clean responses to {args.output}")
    print(f"Zero responses: {is_zero.sum()}; multiply amplitudes by 0.1 for microvolts.")


if __name__ == "__main__":
    main()
