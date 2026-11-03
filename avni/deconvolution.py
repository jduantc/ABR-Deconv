import numpy as np
import mne
from scipy.fftpack import fft, ifft

fs_eeg = int(10e3)  # EEG sampling rate (Hz)
t_min, t_max = -1, 1
n_pre, n_post = -t_min * fs_eeg, t_max * fs_eeg

# question: what is xi vs xfi and where do they come from
def deconvolution1(xi, xfi, yi):
    """
    Derive ABR responses from EEG recordings using frequency domain analysis.

    The relationship between the stimulus and EEG response is modeled as:
        Y = XH + N

    where:
        Y = Recorded EEG response
        X = Input stimulus
        H = Estimated ABR response
        N = Additive noise

    Parameters
    ----------
    xi : np.ndarray
        Stimulus signal.
    xfi : np.ndarray
        Frequency-specific stimulus signal.
    yi : np.ndarray
        Recorded EEG response signal.

    Input Dimensions
    ----------------
    (n_trials, n_ears, n_frequencies, n_channels, n_samples)

    Returns
    -------
    ws : np.ndarray
        Frequency-specific ABR responses.
        Dimensions: (n_ears, n_frequencies, n_pre + n_post)
    wsb : np.ndarray
        Reconstructed broadband ABR responses.
        Dimensions: (n_ears, n_pre + n_post)
    """

    # convert signals to frequency domain
    n_fft = mne.filter.next_fast_len(xi.shape[-1])  # next fast length for FFT
    x_fft = fft(xi, n=n_fft)
    xf_fft = fft(xfi, n=n_fft)
    y_fft = fft(yi, n=n_fft)
    y_fft[..., :1] = 0  # remove DC component / remove constant offset

    # trials weighted by inverse variance (high variance trials get lower weight)
    tw = 1. / np.var(yi, axis=-1, keepdims=True)  # samples axis for variance calculation
    tw /= tw.sum(0, keepdims=True)  # normalize

    # calculate ABRs for each frequency band: inverse FFT of X* Y / X* X
    # how EEG is relatd to the stimulus at each frequency, weighted average across trials
    numerator = (np.conj(x_fft) * y_fft * tw).sum(0)
    # stimulus power at each frequency, averaged across frequency bands, ears, and trials
    # could replace (np.conj(x_fft) * x_fft) with (np.abs(x_fft) ** 2)
    denominator = (np.conj(x_fft) * x_fft).mean(axis=(0, 1, 2))
    # 1e6 converts EEG signals from volts to microvolts
    wb = 1e6 * ifft(numerator / denominator).real
    # wb dimensions: (n_ears, n_frequencies, n_channels, n_samples)

    # wb has how much of the EEG response is attributable to the broadband stimulus
    # want the response one second before and after the stimulus
    # raw ifft is circular: results in 0, [positive], [reversed negative] time lags
    # concatenate pre- and post-stimulus segments and average over EEG channels
    wb = np.concatenate((wb[..., -n_pre:], wb[..., :n_post]), -1).mean(-2)
    # wb dimensions: (n_ears, n_frequencies, n_pre + n_post)

    # calculate common component across frequency bands associated with broadband stimulus
    # same as before for each frequncy band
    numerator_c = (np.conj(xf_fft) * y_fft * tw).sum(0)
    denominator_c = (np.conj(xf_fft) * xf_fft).mean(axis=(0, 1, 2))
    wc = 1e6 * ifft(numerator_c / denominator_c).real
    wc = np.concatenate((wc[..., -n_pre:], wc[..., :n_post]), -1).mean(-2)

    # calculate freq-specific ABRs by subtracting common component across bands
    ws = wb - wc.mean(-2, keepdims=True)  # avg across band axis

    # reconstructed broadband ABR (all frequency bands)
    wsb = ws.sum(-2) + wc.mean(-2)

    return ws, wsb


def deconvolution_wiener(xi, xfi, yi):
    # convert signals to frequency domain
    n_fft = mne.filter.next_fast_len(xi.shape[-1])  # next fast length for FFT
    x_fft = fft(xi, n=n_fft)
    xf_fft = fft(xfi, n=n_fft)
    y_fft = fft(yi, n=n_fft)
    y_fft[..., :1] = 0  # remove DC component / remove constant offset

    # estimate omega
    frequency_hz = np.fft.fftfreq(n_fft, d=1.0 / fs_eeg)
    omega = 2.0 * np.abs(frequency_hz) / fs_eeg
    alpha = 50

    tw = 1. / np.var(yi, axis=-1, keepdims=True)
    tw /= tw.sum(0, keepdims=True)

    numerator = (np.conj(x_fft) * y_fft * tw).sum(0)
    denominator = (np.abs(x_fft) ** 2).mean(axis=(0, 1, 2))
    
    power_scale = np.median(denominator[denominator > 0])
    reg = alpha * power_scale * omega**2

    wb = 1e6 * ifft(numerator / (denominator + reg)).real
    wb = np.concatenate((wb[..., -n_pre:], wb[..., :n_post]), -1).mean(-2)
    # wb dimensions: (n_ears, n_frequencies, n_pre + n_post)

    # calculate common component across frequency bands associated with broadband stimulus
    numerator_c = (np.conj(xf_fft) * y_fft * tw).sum(0)
    denominator_c = (np.abs(xf_fft) ** 2).mean(axis=(0, 1, 2))

    power_scale_c = np.median(denominator_c[denominator_c > 0])
    reg_c = alpha * power_scale_c * omega**2

    wc = 1e6 * ifft(numerator_c / (denominator_c + reg_c)).real
    wc = np.concatenate((wc[..., -n_pre:], wc[..., :n_post]), -1).mean(-2)

    # calculate freq-specific ABRs by subtracting common component across bands
    ws = wb - wc.mean(-2, keepdims=True)  # avg across band axis

    # reconstructed broadband ABR (all frequency bands)
    wsb = ws.sum(-2) + wc.mean(-2)

    return ws, wsb


