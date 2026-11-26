"""Make simulated EEG from an ABR, real pulse times, and recorded background."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, iirnotch, sosfilt, tf2sos


SAMPLE_RATE_HZ = 10_000
ABR_SAMPLES = 256
ABR_ONSET_INDEX = 50  # ABR sample 50 is aligned to the pulse time.
AMPLITUDE_SCALE_UV = 0.1  # one Leo model unit corresponds to 0.1 microvolts.
AMPLIFIER_GAIN = 100.0


def recording_filter_sos(sample_rate_hz: int = SAMPLE_RATE_HZ) -> np.ndarray:
    """Return the causal EEG filter: 30-2000 Hz plus 60 Hz harmonic notches."""
    sections = [
        butter(1, [30.0, 2000.0], btype="bandpass", fs=sample_rate_hz, output="sos")
    ]
    for frequency_hz in np.arange(60.0, 2000.0, 120.0):
        b, a = iirnotch(frequency_hz, frequency_hz / 5.0, fs=sample_rate_hz)
        sections.append(tf2sos(b, a))
    return np.concatenate(sections, axis=0)


def reference_channels_uv(
    samples: np.ndarray,
    resolutions_uv: np.ndarray,
    *,
    amplifier_gain: float = AMPLIFIER_GAIN,
) -> np.ndarray:
    """Calibrate [FCzA1, FCz, A2] and average the two earlobe references."""
    values = np.asarray(samples, dtype=np.float64)
    resolution = np.asarray(resolutions_uv, dtype=np.float64)
    if values.ndim < 1 or values.shape[-1] != 3:
        raise ValueError("samples must have three channels: [FCzA1, FCz, A2]")
    if resolution.shape != (3,) or not np.isfinite(resolution).all():
        raise ValueError("resolutions_uv must contain three finite header values")
    if amplifier_gain <= 0:
        raise ValueError("amplifier_gain must be positive")
    ch = values * resolution
    fcz_a1 = ch[..., 0]
    fcz_a2 = ch[..., 1] - ch[..., 2]
    return 0.5 * (fcz_a1 + fcz_a2) / amplifier_gain


def filter_recorded_eeg(
    eeg_uv_with_context: np.ndarray,
    *,
    warmup_samples: int = SAMPLE_RATE_HZ,
    sample_rate_hz: int = SAMPLE_RATE_HZ,
    center: bool = True,
) -> np.ndarray:
    """Filter a continuous EEG segment and return its post-warmup central portion.

    The input must include `warmup_samples` before the desired window so causal
    filter startup transients are outside the returned data. Time is axis 0.
    """
    eeg = np.asarray(eeg_uv_with_context, dtype=np.float64)
    if eeg.ndim != 1 or not np.isfinite(eeg).all():
        raise ValueError(
            "EEG input must be a finite one-dimensional signal in microvolts"
        )
    if not 0 <= warmup_samples < len(eeg):
        raise ValueError("warmup_samples must be smaller than the EEG segment")
    filtered = sosfilt(recording_filter_sos(sample_rate_hz), eeg)
    central = filtered[warmup_samples:].copy()
    if center:
        central -= central.mean()
    return central


def stimulus_matrix(
    pulse_train: np.ndarray,
    *,
    response_length: int = ABR_SAMPLES,
    onset_index: int = ABR_ONSET_INDEX,
) -> np.ndarray:
    """Map an ABR into EEG samples using the supplied pulse train.

    Returns a matrix with shape (EEG samples, ABR samples). Include pulse history
    in `pulse_train` before the window of interest when filtering the matrix.
    """
    pulses = np.asarray(pulse_train, dtype=np.float64)
    if pulses.ndim != 1 or pulses.size == 0 or not np.isfinite(pulses).all():
        raise ValueError("pulse_train must be a nonempty finite one-dimensional array")
    if response_length < 1 or not 0 <= onset_index < response_length:
        raise ValueError("invalid response_length or onset_index")
    eeg_time = np.arange(len(pulses))[:, None]
    abr_time = np.arange(response_length)[None, :] - onset_index
    indices = eeg_time - abr_time
    valid = (indices >= 0) & (indices < len(pulses))
    return pulses[np.clip(indices, 0, len(pulses) - 1)] * valid


def filter_forward_matrix(
    raw_matrix: np.ndarray,
    *,
    crop_samples: int = 0,
    sample_rate_hz: int = SAMPLE_RATE_HZ,
) -> np.ndarray:
    """Filter the pulse matrix before cropping its pre-window history."""
    matrix = np.asarray(raw_matrix, dtype=np.float64)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError("raw_matrix must be a finite (time, ABR samples) matrix")
    if not 0 <= crop_samples < len(matrix):
        raise ValueError("crop_samples must be smaller than the matrix time dimension")
    filtered = sosfilt(recording_filter_sos(sample_rate_hz), matrix, axis=0)
    return filtered[crop_samples:].copy()


def abr_model_units_to_uv(abr: np.ndarray) -> np.ndarray:
    """Convert Leo model amplitude units to microvolts."""
    return np.asarray(abr, dtype=np.float64) * AMPLITUDE_SCALE_UV


def make_hybrid_eeg(
    target_abr_uv: np.ndarray,
    filtered_forward_matrix: np.ndarray,
    recorded_background_uv: np.ndarray,
    *,
    background_multiplier: float = 1.0,
) -> np.ndarray:
    """Create known-truth EEG = filtered pulse response + recorded EEG background.

    `recorded_background_uv` must already have the same calibration, filtering,
    cropping, and time alignment as the forward matrix. A multiplier of 1 keeps
    its measured amplitude; other values are explicit stress-test conditions.
    """
    abr = np.asarray(target_abr_uv, dtype=np.float64)
    matrix = np.asarray(filtered_forward_matrix, dtype=np.float64)
    background = np.asarray(recorded_background_uv, dtype=np.float64)
    if abr.ndim != 1 or matrix.ndim != 2 or matrix.shape[1] != len(abr):
        raise ValueError("target_abr_uv and forward-matrix dimensions do not match")
    if background.shape != (matrix.shape[0],):
        raise ValueError(
            "background length must match the forward-matrix EEG dimension"
        )
    if background_multiplier < 0:
        raise ValueError("background_multiplier must be nonnegative")
    if not (
        np.isfinite(abr).all()
        and np.isfinite(matrix).all()
        and np.isfinite(background).all()
    ):
        raise ValueError("ABR, matrix, and background must contain only finite values")
    return matrix @ abr + background_multiplier * background


def convert_abr_to_eeg(
    abr_uv,
    *,
    data_dir,
    subject="sub-exp303",
    background_start_s=2.5,
    duration_s=0.5,
    background_multiplier=1.0,
):
    """Combine one ABR with real 4 kHz left pulses and recorded EEG background."""
    import csv
    import re
    from pathlib import Path

    import h5py

    root = Path(data_dir)
    folder = root / subject / "eeg"
    stem = f"{subject}_task-f0s2_run-01"
    header = (folder / f"{stem}_eeg.vhdr").read_text(encoding="utf-8-sig")
    cols = re.findall(r"^Ch\d+=([^,]+),[^,]*,([\d.]+),([^\r\n]+)", header, re.M)
    if (
        [c[0] for c in cols] != ["FCzA1", "FCz", "A2"]
        or any(c[2] != "µV" for c in cols)
        or "SamplingInterval=100" not in header
    ):
        raise ValueError(
            "Expected 10 kHz exp3 BrainVision channels FCzA1, FCz, A2 in µV"
        )
    resolution = np.array([float(c[1]) for c in cols])

    with (folder / f"{stem}_events.tsv").open() as f:
        events = list(csv.DictReader(f, delimiter="\t"))
    starts = [e for e in events if e["trial_type"] == "stimulus_start"]
    ends = [e for e in events if e["trial_type"] == "stimulus_end"]
    intervals = []
    for event in starts:
        candidates = [
            int(e["sample"]) - int(event["sample"])
            for e in ends
            if 9 * SAMPLE_RATE_HZ
            < int(e["sample"]) - int(event["sample"])
            < 10.5 * SAMPLE_RATE_HZ
        ]
        if candidates:
            intervals.append(min(candidates))
    if not intervals:
        raise ValueError("No matching start/end triggers for EEG clock correction")

    # Align the stimulus clock with the EEG clock.
    clock = float(np.mean(9.98 * SAMPLE_RATE_HZ / np.asarray(intervals)))
    event = next(e for e in starts if e["stimulus_type"] == "wizard")
    token = int(float(event["token"])) + 1
    stimulus_path = (
        root / "stimuli/exp3_f0s2/wizard" / f"wizard_shifted_{token:04d}.hdf5"
    )
    with h5py.File(stimulus_path) as f:
        source = f["expyfun"]
        fs_audio = float(np.asarray(source["key_fs"][()]).reshape(-1)[0])
        indices = np.asarray(source["key_pulseinds_true"]["idx_6"][()])

    count = int(round(duration_s * SAMPLE_RATE_HZ))
    guard = SAMPLE_RATE_HZ
    if count < 1 or guard + count > int(9.98 * SAMPLE_RATE_HZ):
        raise ValueError(
            "duration must fit within the selected stimulus trial after 1 second"
        )
    pulses = np.zeros(guard + count)
    indices = np.floor(indices / fs_audio / clock * SAMPLE_RATE_HZ).astype(int)
    indices = indices[(indices >= 0) & (indices < len(pulses))]
    np.add.at(pulses, indices, 1.0)

    # Filter the convolution matrix and keep the requested window.
    matrix = filter_forward_matrix(stimulus_matrix(pulses), crop_samples=guard)
    matrix -= matrix.mean(axis=0, keepdims=True)

    # Use EEG recorded before playback as the background.
    start = int(round(background_start_s * SAMPLE_RATE_HZ))
    first_playback = min(int(e["sample_corrected"]) for e in starts)
    if start < guard or start + count > first_playback - SAMPLE_RATE_HZ // 2:
        raise ValueError(
            "Background must include 1 s history and end 0.5 s before first playback"
        )
    raw = np.memmap(folder / f"{stem}_eeg.eeg", dtype="<f4", mode="r").reshape(-1, 3)
    segment = raw[start - guard : start + count]
    if len(segment) != guard + count:
        raise ValueError("Requested EEG segment exceeds recording")
    background = filter_recorded_eeg(
        reference_channels_uv(segment, resolution), warmup_samples=guard
    )

    abr = np.asarray(abr_uv, dtype=np.float64)
    if abr.shape != (ABR_SAMPLES,):
        raise ValueError("ABR must be a 256-sample waveform at 10 kHz")
    clean = matrix @ abr
    eeg = make_hybrid_eeg(
        abr, matrix, background, background_multiplier=background_multiplier
    )
    return {
        "eeg_uv": eeg,
        "clean_eeg_uv": clean,
        "background_uv": background,
        "abr_uv": abr,
        "pulses": pulses[guard:],
        "forward_matrix": matrix,
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "subject": subject,
        "background_start_s": background_start_s,
        "background_multiplier": background_multiplier,
        "clock_adjust": clock,
        "stimulus_file": str(stimulus_path),
    }


def main():
    """Example: python abr_eeg_pipeline.py abr.npy output.npz --abr-units uv"""
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="Convert a known ABR into synthetic EEG with real pulse timing and recorded background."
    )
    parser.add_argument(
        "abr", type=Path, help=".npy file containing one 256-sample ABR"
    )
    parser.add_argument(
        "output",
        type=Path,
        help="output .npz containing EEG, ABR, pulses, background and forward matrix",
    )
    parser.add_argument(
        "--abr-units",
        required=True,
        choices=("uv", "model"),
        help="input units: µV or Leo model units",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path("/mnt/sde2/cs163/data/ds008065")
    )
    parser.add_argument("--subject", default="sub-exp303")
    parser.add_argument(
        "--background-start",
        type=float,
        default=2.5,
        help="pre-playback background start time in seconds",
    )
    parser.add_argument(
        "--duration", type=float, default=0.5, help="EEG window duration in seconds"
    )
    parser.add_argument("--background-multiplier", type=float, default=1.0)
    args = parser.parse_args()
    abr = np.load(args.abr, allow_pickle=False)
    if args.abr_units == "model":
        abr = abr_model_units_to_uv(abr)
    result = convert_abr_to_eeg(
        abr,
        data_dir=args.data_dir,
        subject=args.subject,
        background_start_s=args.background_start,
        duration_s=args.duration,
        background_multiplier=args.background_multiplier,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **result)
    print(f"Saved {len(result['eeg_uv'])} EEG samples to {args.output} (µV).")


if __name__ == "__main__":
    main()
