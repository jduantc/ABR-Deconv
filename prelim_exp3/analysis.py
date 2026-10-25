import numpy as np
import pandas as pd
import mne 
import os
from h5io import read_hdf5

from scipy.fftpack import fft, ifft
from scipy.signal import sosfilt, sosfilt_zi, butter, lfilter, iirnotch
import pathlib
from expyfun import binary_to_decimals

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

    x_isol = x_bands - common_comp # per-band, without common comp. shape (E, B, n_pre + n_post)
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
    event_id = {
        "New Segment/": 0,
        "Stimulus/S  1": 1,
        "Stimulus/S  2": 2,
        "Stimulus/S  4": 4,
        "Stimulus/S  8": 8,
    }

    events = mne.events_from_annotations(
        raw, event_id=event_id,
    )[0]


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

def decode_events(events, fs_eeg, n_bits_stimuli=2, n_bits_trials=10,
                  stamp_duration=9.98, tube_len_m=11*25.4e-3):
    """
    Decodes the event table from mne reading

    Returns:
    stim_events : ndarray, 
        shape (number_of_trials, 3)
        Columns: [corrected_start_sample, token_index, story_index]
    clock_adjust : float
        Audio/EEG clock correction used to map stimulus pulse times.
    """
    events = np.asarray(events)
    # Copy so the original events remain unchanged.
    events = events.astype(np.int64, copy=True)

    # Correct for sound propagation through the earphone tubing.
    tube_delay = tube_len_m / 343
    tube_delay_samples = int(round(tube_delay * fs_eeg))
    events[:, 0] += tube_delay_samples

    # Locate stimulus-start and stimulus-end markers.
    start_inds = np.flatnonzero(events[:, 2] == 1)
    starts = events[start_inds]
    ends = events[events[:, 2] == 2]

    # identifies plausible start-to-end durations.
    nominal_length = np.median(np.diff(starts[:, 0]))

    differences = ends[:, [0]] - starts[:, [0]].T
    trial_lengths = differences[
        (differences > 0) & (differences < nominal_length)
    ]

    clock_adjust = float(
        (stamp_duration * fs_eeg / trial_lengths).mean()
    )

    # Each start is followed by a fixed-length binary stamp.
    n_bits_stamp = n_bits_stimuli + 2 * n_bits_trials

    stories = np.empty(len(starts), dtype=int)
    tokens = np.empty(len(starts), dtype=int)

    for i, start in enumerate(start_inds):
        codes = events[
            start + 1:start + 1 + n_bits_stamp, 2
        ]

        if len(codes) != n_bits_stamp or not np.isin(
            codes, [4, 8]
        ).all():
            raise ValueError(
                f"Incomplete or invalid binary stamp after "
                f"start marker at sample {events[start, 0]}."
            )

        # Original encoding: trigger 4 -> bit 0, trigger 8 -> bit 1.
        bits = codes >> 3

        stories[i] = binary_to_decimals(
            bits[:n_bits_stimuli],
            n_bits_stimuli,
        )[0]

        token_start = n_bits_stimuli
        token_stop = token_start + n_bits_trials

        tokens[i] = binary_to_decimals(
            bits[token_start:token_stop],
            n_bits_trials,
        )[0]

    # Same three-column layout constructed by the original script.
    stim_events = starts.copy()
    stim_events[:, 1] = tokens
    stim_events[:, 2] = stories

    return stim_events, clock_adjust

def select_trials(stim_events, story_index, max_trials=None):
    """
    Selects one story's trials from decoded event array
    """
    trials = stim_events[
        stim_events[:, -1] == story_index
    ].copy()

    if len(trials) == 0:
        raise ValueError(f"No trials found for story index {story_index}.")

    if max_trials is not None:
        trials = trials[:max_trials]

    return trials

def extract_epochs(eeg, trials, fs_eeg, stim_dur=10.0,
                   pre_seconds=1.0, post_seconds=1.0, first_sample=0):
    """
    Extract EEG windows whose padded durations fit within the recording
    Inputs:
    eeg: ndarray, shape (C, recording_samples)
    trials: ndarray, shape (K, 3)
        Columns: corrected start sample, token index, story index.
    first_sample : int, sample number corresponding to eeg[:, 0]
    """
    n_pre = int(round(pre_seconds * fs_eeg))
    n_stim = int(round(stim_dur * fs_eeg))
    n_post = int(round(post_seconds * fs_eeg))
    n_samples = n_pre + n_stim + n_post

    # Convert event sample numbers into positions in the EEG array.
    starts = trials[:, 0].astype(np.int64) - first_sample - n_pre
    stops = starts + n_samples

    keep = (starts >= 0) & (stops <= eeg.shape[-1])

    retained_trials = trials[keep].copy()

    epochs = np.stack([
        eeg[:, start:stop]
        for start, stop in zip(starts[keep], stops[keep])
    ])

    return epochs, retained_trials

def build_pulse_trains(trials, stim_path, fs_eeg, clock_adjust, fs_stim = 48e3, 
                       stim_dur = 10.0, pre_seconds=1.0, post_seconds=1.0, n_ears=2, 
                       n_bands=5, true_key="pulseinds_true", fake_key="pulseinds_fake"):
    if not np.isfinite(clock_adjust) or clock_adjust <= 0:
        raise ValueError("clock_adjust must be finite and positive.")
    n_pre = int(round(pre_seconds * fs_eeg))
    n_stim = int(round(stim_dur * fs_eeg))
    n_post = int(round(post_seconds * fs_eeg))
    n_samples = n_pre + n_stim + n_post

    n_trials = len(trials)

    # Original ordering: band 0 left, band 0 right,
    # band 1 left, band 1 right, ...
    real = np.zeros(
        (n_trials, n_bands * n_ears, n_samples),
        dtype=float,
    )
    fake = np.zeros(
        (n_trials, n_bands, n_samples),
        dtype=float,
    )
    def to_epoch_indices(pulse_indices):
        # Audio sample indices -> EEG epoch sample indices.
        positions = (
            np.asarray(pulse_indices, dtype=float)
            / fs_stim
            / clock_adjust
            * fs_eeg
            + n_pre
        )
        # Matches the original code's integer conversion.
        return positions.astype(np.int64)

    for k, (_, token, story) in enumerate(trials):
        # load sound pulse info from stim path
        path = pathlib.Path(stim_path(int(story), int(token)))

        if not path.is_file():
            raise FileNotFoundError(path)

        data = read_hdf5(path)
        true_pulses = data[true_key]
        fake_pulses = data[fake_key]

        if len(true_pulses) != n_ears * n_bands:
            raise ValueError(f"Unexpected number of real pulse trains: {path}")

        if len(fake_pulses) != n_bands:
            raise ValueError(f"Unexpected number of sham pulse trains: {path}")

        for j, pulses in enumerate(true_pulses):
            real[k, j, to_epoch_indices(pulses)] = 1.0

        for j, pulses in enumerate(fake_pulses):
            fake[k, j, to_epoch_indices(pulses)] = 1.0

    # Separate the interleaved band/ear axis.
    c = real.reshape(
        n_trials, n_bands, n_ears, n_samples
    ).transpose(0, 2, 1, 3)

    c = c[:, :, :, None, :]
    cf = fake[:, None, :, None, :]

    return c, cf
