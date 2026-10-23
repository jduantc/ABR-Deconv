import numpy as np
import pandas as pd
import mne 
import os
from h5io import read_hdf5

from scipy.fftpack import fft, ifft
from scipy.signal import sosfilt, sosfilt_zi, butter, lfilter, iirnotch
import pathlib

n_subs = 1
eeg_freq = 1e4
SUBJECTS = np.arange(1, n_subs + 1)
EARS = ["left", "right"]
# how much time to retain before/after ref. point
tmin, tmax = -1, 1
n_pre, n_post = int(round(-tmin * eeg_freq)), int(round(tmax * eeg_freq))

l_freq = 30
h_freq = 2000
notches = np.arange(60, h_freq, 120)
notch_width = 5
# bandpass filter used in original code
b, a = butter(1, [l_freq, h_freq], 'bandpass', fs=eeg_freq)
b, a = [b], [a]

# add notch filters to suppress electrical interference (?)
for notch_freq in notches:
    # quality factor is def. as freq / width
    bn, an = iirnotch(notch_freq, notch_freq / notch_width, fs=eeg_freq)
    b.append(bn)
    a.append(an)
# second order sections constructed from b, a. rearranged to pass into sosfilt
sos = np.array((b, a)).transpose(1, 0, 2).reshape(-1, 6)

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

    x_isol = x_bands - common_comp # per-band, without common comp. shape (1, n_pre + n_post)
    # overall, aggreagted speech response
    x_combined = x_isol.sum(1) + common_comp[:, 0, :] # shape (E, n_pre + n_post)

    return x_isol, x_combined

def load_eeg(vhdr_path, events_path, preload=True):
    """
    Reads a BrainVision recording and event table
    returns raw (raw EEG), events (timing info of recording)
    """
    vhdr_path = pathlib.Path(vhdr_path).expanduser()
    events_path = pathlib.Path(events_path).expanduser()

    if not vhdr_path.is_file():
        raise FileNotFoundError(vhdr_path)
    if not events_path.is_file():
        raise FileNotFoundError(events_path)

    raw = mne.io.read_raw_brainvision(vhdr_path, preload=preload)
    events = pd.read_csv(events_path, delimiter='\t')

    return raw, events

def process_eeg(raw, gain=100):
    """
    Given one raw EEG recording, do data preparation and filtering
    """
    fs_eeg = float(raw.info["sfreq"])
    assert(eeg_freq == fs_eeg)

    channels = raw.ch_names
    # the data is stored in FCzA1 (forehead to left ear derivation), FCz (Forehead),
    # A2 (right ear)
    assert(channels == ['FCzA1', 'FCz', 'A2'])
    data = raw.get_data(picks=channels)
    # remove (constant) first sample amount
    data -= data[..., [0]]

    # processes to contain both derivations, normalize by amplifier gain
    eeg = np.stack([data[0], data[1] - data[2]]) / gain # shape (2, N), i.e. (channels, time samples)
    # computes initial state for sos filter
    zi = sosfilt_zi(sos) # shape (S, 2), S = # filter sections
    zi = zi[:, :, None] * eeg[:, :1].T # sets equal to first sample of each channel
    eeg, _ = sosfilt(sos, eeg, axis=-1, zi=zi.transpose(0, 2, 1)) # transpose to align with sosfilt
    return eeg, fs_eeg
