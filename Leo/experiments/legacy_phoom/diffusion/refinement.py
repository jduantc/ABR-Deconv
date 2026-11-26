"""Optional measurement correction for an averaged DPS reconstruction.

All amplitudes are in microvolts and covariance is in microvolts squared.
Strength must be selected on separate validation recordings; zero is a no-op.
This reuses the EEG and is not an exact posterior update.
"""

import numpy as np


def measurement_refinement(
    estimate_uv, eeg_uv, matrix, noise_std_uv, covariance_uv2, strength=0.0
):
    estimate = np.asarray(estimate_uv)
    single = estimate.ndim == 1
    estimate = np.atleast_2d(estimate)
    eeg = np.atleast_2d(eeg_uv)
    matrix = np.asarray(matrix)
    covariance = np.asarray(covariance_uv2)
    sigma = np.broadcast_to(np.asarray(noise_std_uv), (len(estimate),))
    tau = np.broadcast_to(np.asarray(strength), (len(estimate),))
    if np.any(sigma <= 0) or np.any(tau < 0):
        raise ValueError("Noise SD must be positive and strength nonnegative")
    if (
        eeg.shape != (len(estimate), len(matrix))
        or covariance.shape != (estimate.shape[1],) * 2
        or matrix.shape[1] != estimate.shape[1]
    ):
        raise ValueError("Incompatible recording, waveform or covariance shapes")
    gram = matrix.T @ matrix
    results = []
    for h, y, s, t in zip(estimate, eeg, sigma, tau):
        if t == 0:
            results.append(h.copy())
            continue
        Q = t * covariance / s**2
        results.append(
            np.linalg.solve(np.eye(len(h)) + Q @ gram, h + Q @ (matrix.T @ y))
        )
    result = np.asarray(results)
    return result[0] if single else result
