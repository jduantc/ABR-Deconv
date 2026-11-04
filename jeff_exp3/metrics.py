"""
SNR and response-time estimation for ABR, using the same formulas
as (Polonenko, 2026), specifically the function get_snr()
in exp3_f0s2/05_analysis_f0s2_group.py of the original repo
"""

import numpy as np
from scipy.signal import butter, filtfilt

def filter_responses(w, fs, low=80.0, high=1500.0, order=1):
    """
    Arguments:
    w: already deconvolved waveform, shape [E, B, L]
    """
    b, a = butter(order, [low, high], btype="bandpass", fs=fs)
    return filtfilt(b, a, np.asarray(w), axis=-1)

def get_snr(w, fs, n_pre, dur_secs, response_ms=(0.0, 9.0), 
            baseline_ms=(-480.0, -20.0), snr_floor=1e-2):
    """
    Computes SNR and time to 0dB using the canonical Var ratio and linear
    estimation approach

    Returns:
    - RMS of noise variation
    - SNR in db
    - linearly estimated time to 0 db
    """
    w = np.asarray(w, dtype=float)
    width_ms = response_ms[1] - response_ms[0]
    width = int(round(width_ms * fs / 1000)) # in sample / array units
    response_start = n_pre + int(round(response_ms[0] * fs / 1000))
    baseline_start = n_pre + int(round(baseline_ms[0] * fs / 1000))
    baseline_stop = n_pre + int(round(baseline_ms[1] * fs / 1000))

    if width < 2 or baseline_stop - baseline_start < width:
        raise ValueError("Windows must contain enough samples for variance.")
    if not (0 <= response_start < response_start + width <= w.shape[-1]):
        raise ValueError("The response window is outside the waveform.")
    if not (0 <= baseline_start < baseline_stop <= w.shape[-1]):
        raise ValueError("The baseline window is outside the waveform.")

    # split baseline into several negative-lag windows
    n_windows = (baseline_stop - baseline_start) // width
    starts = np.round(np.linspace(
        baseline_start, baseline_stop-width, n_windows
    )).astype(int)
    noise_var = np.mean(
        [w[..., start:start+width].var(axis=-1) for start in starts], # one var per start, ear, band
        axis = 0 # average over starts, shape [E, B]
    )

    response_var = w[..., response_start:response_start + width].var(axis=-1) # shape [E, B]
    signal_var = response_var - noise_var
    ratio = signal_var / noise_var
    used_ratio = np.maximum(ratio, snr_floor)

    snr_db = 10 * np.log10(used_ratio) # still shape (E. B)
    # linearly estimates time to 0 db
    t0_seconds = np.where(
        np.isfinite(used_ratio), dur_secs / used_ratio, np.inf,
    )

    return np.sqrt(noise_var), snr_db, t0_seconds

