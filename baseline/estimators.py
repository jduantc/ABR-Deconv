"""Closed-form estimators.

paper_ls       per-frequency LS, sum conj(X) Y / sum |X|^2, then 30-2000 Hz
               bandpass. The core of the paper's derive_responses(). Works on
               the raw spectra, not on (A, b).
tikhonov       (A + lam I)^-1 b
GaussianPrior  h ~ N(mu, C) from example ABRs; posterior mean
               (A + C^-1)^-1 (b + C^-1 mu). This is a Wiener filter with the
               SNR curve taken from C.
"""
import numpy as np

from sim import FS, L, LAG0, NK, sample_abr

_f = np.abs(np.fft.fftfreq(L, 1 / FS))


def _bandpass(h_full, lo=30.0, hi=2000.0):
    H = np.fft.fft(h_full)
    H[(_f < lo) | (_f > hi)] = 0
    return np.fft.ifft(H).real


def paper_ls(X, Y):
    Hf = np.sum(np.conj(X) * Y, axis=0) / np.sum(np.abs(X) ** 2, axis=0)
    h_full = _bandpass(np.fft.ifft(Hf).real)
    return h_full[(np.arange(NK) + LAG0) % L]


def tikhonov(A, b, lam):
    return np.linalg.solve(A + lam * np.eye(NK), b)


class GaussianPrior:
    def __init__(self, n=20000, ridge=1e-6, rng=None):
        h = sample_abr(n, rng or np.random.default_rng(1))
        self.mu = h.mean(0)
        self.C = np.cov(h.T) + ridge * np.eye(NK)
        self.Cinv = np.linalg.inv(self.C)

    def map(self, A, b):
        """Posterior mean and covariance."""
        P = np.linalg.inv(A + self.Cinv)
        return P @ (b + self.Cinv @ self.mu), P
