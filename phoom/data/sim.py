"""Synthetic ABRs, pulse trains, EEG noise, and the (A, b) statistics.

Trial model: y = x * h + n. x is the pulse train, h the ABR (256 lags at
10 kHz, -5 to 20.5 ms), n coloured Gaussian noise. Then

    -log p(y | h) = 1/2 h^T A h - b^T h + const

and the prior-based estimators only need A and b (see sufficient_stats).
"""
import numpy as np

FS = 10_000            # EEG sample rate
TRIAL = 10.0           # seconds per trial
L = int(TRIAL * FS)
NK = 256               # kernel length
LAG0 = -50             # first lag is -5 ms, so we can check nothing shows up pre-stimulus
LAGS_MS = (np.arange(NK) + LAG0) / FS * 1e3

# Textbook click-ABR shape: waves I, III, V, the SN10 trough after V, then a
# slow middle-latency bump. (latency ms, width ms, amplitude uV)
WAVES = [
    (1.7, 0.25, 0.10),
    (3.9, 0.30, 0.12),
    (5.8, 0.45, 0.30),
    (7.5, 0.90, -0.25),
    (13.0, 2.5, 0.12),
]


def sample_abr(n, rng):
    """n random ABRs, shape (n, NK), in uV. Each one gets a common latency
    shift and gain, then per-wave jitter in latency, width and amplitude."""
    t = LAGS_MS[None, :]
    shift = rng.normal(0, 0.4, (n, 1))
    gain = np.exp(rng.normal(0, 0.35, (n, 1)))
    h = np.zeros((n, NK))
    for lat, wid, amp in WAVES:
        lat_i = lat + shift + rng.normal(0, 0.15, (n, 1)) * (1 + lat / 6)  # later waves jitter more
        wid_i = wid * np.exp(rng.normal(0, 0.2, (n, 1)))
        amp_i = amp * np.exp(rng.normal(0, 0.3, (n, 1)))
        h += amp_i * np.exp(-0.5 * ((t - lat_i) / wid_i) ** 2)
    h *= gain
    # a bit of smooth random background so the waveforms are not just 5 bumps
    H = np.fft.rfft(rng.normal(0, 1, (n, NK)), axis=1)
    f = np.fft.rfftfreq(NK, 1 / FS)
    H *= np.exp(-(f / 600) ** 2)
    h += 0.06 * np.fft.irfft(H, NK, axis=1) * gain
    # nothing before the stimulus
    h *= 0.5 * (1 + np.tanh((LAGS_MS + 0.5) / 0.4))
    return h


def pulse_train(rng, f0_mean=90.0, voiced_frac=0.55):
    """One trial of pulses. Voiced stretches of 50-400 ms with f0 wandering
    around f0_mean, separated by unvoiced gaps."""
    x = np.zeros(L)
    t = 0.0
    while t < TRIAL:
        seg = rng.uniform(0.05, 0.4)
        f0 = f0_mean * (1 + 0.08 * rng.normal())
        drift = rng.normal(0, 0.03)
        n_pulses = int(seg * f0)
        periods = 1 / (f0 * (1 + drift * np.linspace(-1, 1, n_pulses)))
        times = t + np.cumsum(periods)
        idx = np.round(times * FS).astype(int)
        idx = idx[(idx >= 100) & (idx < L - 300)]
        x[idx] = 1.0
        t += seg + rng.uniform(0.03, 0.3) * (1 - voiced_frac) / voiced_frac
    return x


# pink noise plus a white floor, roughly what the filtered EEG looks like
_fabs = np.abs(np.fft.fftfreq(L, 1 / FS))
NOISE_SHAPE = 1 / (_fabs + 2.0) + 1 / 2000.0


def noise_psd(sigma_uv):
    """Two-sided PSD, scaled so the noise has RMS sigma_uv in 1-2000 Hz.
    Units follow np.fft: E|FFT(n)|^2 = psd, and sum(psd) = L^2 * var."""
    band = (_fabs > 1) & (_fabs < 2000)
    psd = NOISE_SHAPE.copy()
    psd *= sigma_uv ** 2 * L ** 2 / psd[band].sum()
    return psd


def noise_trial(rng, psd):
    z = rng.normal(0, 1, L) + 1j * rng.normal(0, 1, L)
    # taking the real part halves the power, the sqrt(2) puts it back
    return np.fft.ifft(np.sqrt(psd / 2) * z).real * np.sqrt(2)


def kernel_fft(h):
    """FFT of h placed at lags LAG0.. in a length-L buffer."""
    full = np.zeros(L)
    full[(np.arange(NK) + LAG0) % L] = h
    return np.fft.fft(full)


def simulate(h, minutes, sigma_uv, rng):
    """Record `minutes` of EEG for a true ABR h. Returns per-trial FFTs of
    the pulse trains and the EEG."""
    n_trials = int(round(minutes * 60 / TRIAL))
    psd = noise_psd(sigma_uv)
    Hf = kernel_fft(h)
    X = np.empty((n_trials, L), complex)
    Y = np.empty((n_trials, L), complex)
    for i in range(n_trials):
        X[i] = np.fft.fft(pulse_train(rng))
        y = np.fft.ifft(X[i] * Hf).real + noise_trial(rng, psd)
        Y[i] = np.fft.fft(y)
    return X, Y


def estimate_noise_psd(Y, width_hz=20.0):
    """Noise spectrum = smoothed mean |Y|^2 (the response is ~30 dB below the noise)."""
    p = np.mean(np.abs(Y) ** 2, axis=0)
    w = int(width_hz * L / FS)
    if w % 2 == 0:
        w += 1
    box = np.zeros(L)
    box[np.arange(-(w // 2), w // 2 + 1) % L] = 1 / w
    sm = np.fft.ifft(np.fft.fft(p) * np.fft.fft(box)).real
    return np.maximum(sm, 1e-9 * sm.max())


def sufficient_stats(X, Y, psd=None):
    """A (NK x NK) and b (NK,) in uV units.

    Expand sum_f |Y - X H|^2 / N(f) with H(f) = sum_k h_k e^{-2 pi i f k}:
    the sums over f become inverse FFTs of sum|X|^2/N (whitened pulse
    autocorrelation -> Toeplitz A) and sum conj(X) Y / N (-> b)."""
    if psd is None:
        psd = estimate_noise_psd(Y)
    r = L * np.fft.ifft(np.sum(np.abs(X) ** 2, axis=0) / psd).real
    c = L * np.fft.ifft(np.sum(np.conj(X) * Y, axis=0) / psd).real
    idx = np.arange(NK)
    A = r[np.abs(idx[:, None] - idx[None, :])]
    b = c[(idx + LAG0) % L]
    return A, b


def wave_v(h):
    """Latency (ms) and amplitude (uV) of the biggest peak in 4-9 ms."""
    win = (LAGS_MS > 4) & (LAGS_MS < 9)
    i = np.argmax(np.where(win, h, -np.inf))
    return LAGS_MS[i], h[i]
