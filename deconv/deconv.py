import mne 
import numpy as np
from scipy.fftpack import fft, ifft


# constants
eeg_freq = 1e4

# how much time to retain before/after ref. point
tmin, tmax = -1, 1
n_pre, n_post = int(round(-tmin * eeg_freq)), int(round(tmax * eeg_freq))


def deconv(c, cf, y):
    """
    Deconvolves ABR signal using naive Fourier -> Inv. Fourier approach

    For K = trials, E = ears, B = bands, N = time samples per (padded) trial,
    F = fake pulse trains, C = electrode derivations
    c: (K, E, B, 1, N) is the real pulse train
    cf: (K, 1, F, 1, N) are the fake pulse trains used to estimate a common component
    y: (K, 1, 1, C, N) is the recorded EEG data

    Note that we use c to denote convolution, y observed data. This is consistent
    with inverse problem notation but may conflict with ABR notation
    """
    n = mne.filter.next_fast_len(c.shape[-1]) # calculate fast fourier length using time dimension size
    c_f = fft(c, n=n)
    cf_f = fft(cf, n=n)
    y_f = fft(y, n=n)
    y_f[..., 0] = 0 # removes constant (fft coef 0), convention for eeg inference

    tr_weights = 1./np.var(y, axis=-1, keepdims=True) # inverse variance weighting across trials
    tr_weights = tr_weights / np.sum(tr_weights, axis=0, keepdims=True)

    # deconvolution as written in preliminary paper
    # shape (E, B, C, n) 
    m = (np.conj(c_f) * y_f * tr_weights).sum(axis=0) / (c_f * np.conj(c_f)).mean(axis=(0, 1, 2), keepdims=True)[0]
    # shape (1, F, C, n)
    mf = (np.conj(cf_f) * y_f * tr_weights).sum(axis=0) / (cf_f * np.conj(cf_f)).mean(axis=(0, 1, 2), keepdims=True)[0]

    x_bands = 1e6 * ifft(m).real
    x_fake = 1e6 * ifft(mf).real

    # align properly after the division -> inv fft step, also avg over elec. derivations
    # resultant shape: (E, B, n_pre + n_post)
    x_bands = np.concatenate((x_bands[..., -n_pre:], x_bands[..., :n_post]), -1).mean(2)
    # shape: (1, F, n_pre + n_post)
    x_fake = np.concatenate((x_fake[..., -n_pre:], x_fake[..., :n_post]), -1).mean(2)
    # shape: (1, 1, n_pre + n_post)
    common_comp = x_fake.mean(1, keepdims=True)

    x_isol = x_bands - common_comp # per-band, without common comp. shape (E, B, n_pre + n_post)
    # overall, aggreagted speech response
    x_combined = x_isol.sum(1) + common_comp[:, 0, :] # shape (E, n_pre + n_post)

    return x_isol, x_combined

def deconv_wiener(c, cf, y, alpha, snr_exp=2):
    """
    Original deconvolution estimator with Wiener quadratic regularization.

    Assumed noise-to-response power ratio:
        S_n(theta) / S_h(theta) = alpha * theta**2

    theta is angular frequency in radians/sample.
    alpha >= 0 controls regularization; alpha=0 recovers deconv()
    wherever the original denominators are nonzero.

    Returns the same shapes and units as deconv().
    """
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be finite and nonnegative.")

    n = mne.filter.next_fast_len(c.shape[-1])

    c_f = fft(c, n=n)
    cf_f = fft(cf, n=n)
    y_f = fft(y, n=n)
    y_f[..., 0] = 0

    tr_weights = 1. / np.var(y, axis=-1, keepdims=True)
    tr_weights /= tr_weights.sum(axis=0, keepdims=True)

    # Signed angular frequency of each FFT bin, in radians/sample.
    theta = 2.0 * np.pi * np.fft.fftfreq(n)
    reg = alpha * np.abs(theta)**(snr_exp)

    # Same numerators and power averaging as the original estimator.
    numerator = (np.conj(c_f) * y_f * tr_weights).sum(axis=0)
    denominator = (
        c_f * np.conj(c_f)
    ).mean(axis=(0, 1, 2), keepdims=True)[0]

    numerator_c = (np.conj(cf_f) * y_f * tr_weights).sum(axis=0)
    denominator_c = (
        cf_f * np.conj(cf_f)
    ).mean(axis=(0, 1, 2), keepdims=True)[0]

    # Regularize each Fourier bin.
    m = numerator / (denominator + reg)
    mf = numerator_c / (denominator_c + reg)

    # Exactly one inverse FFT per estimate.
    x_bands = 1e6 * ifft(m).real
    x_fake = 1e6 * ifft(mf).real

    x_bands = np.concatenate(
        (x_bands[..., -n_pre:], x_bands[..., :n_post]), axis=-1
    ).mean(2)

    x_fake = np.concatenate(
        (x_fake[..., -n_pre:], x_fake[..., :n_post]), axis=-1
    ).mean(2)

    common_comp = x_fake.mean(1, keepdims=True)

    x_isol = x_bands - common_comp
    x_combined = x_isol.sum(1) + common_comp[:, 0, :]

    return x_isol, x_combined