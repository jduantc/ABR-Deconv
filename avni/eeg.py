import numpy as np
import mne
from scipy.signal import sosfilt, sosfilt_zi, butter, iirnotch
from expyfun import binary_to_decimals

# eeg parameters
fs_eeg = int(10e3)  # EEG sampling rate (Hz)
t_min, t_max = -1, 1
n_pre, n_post = -t_min * fs_eeg, t_max * fs_eeg

# stimulus duration
stimuli = ['wizard', 'wonderland', 'drdolittle']
n_stimuli = len(stimuli)
stim_dur = 10.
n_min_speech = 35.
n_min_total = n_min_speech * n_stimuli

n_trials_speech = int(n_min_speech * 60 / stim_dur)
n_trials = int(n_trials_speech * n_stimuli)

n_bits_stimuli = int(np.ceil(np.log2(n_stimuli)))
n_bits_trials = int(np.ceil(np.log2(n_trials)))

# pre-process eeg functions - copied from their file then split
def read_eeg(files):
    if len(files) > 1:
        raw_ri = [None] * len(files)
        for ri, run in enumerate(files):
            # read the file: data in "raw_ri[ri]._data" with shape (n_channels, n_samples)
            raw_ri[ri] = mne.io.read_raw_brainvision(run, preload=True)
            # subtract DC offset so each channel starts at zero
            raw_ri[ri]._data -= raw_ri[ri]._data[..., [0]]

        # concatenate the raw data from all runs and remove bad/edge annotations
        raw = mne.concatenate_raws(raw_ri, preload=True)
        for mi in ['BAD boundary', 'EDGE boundary']:
            raw.annotations.delete(np.where(
                raw.annotations.description == mi)[0])

        # free memory
        del raw_ri

    else:
        raw = mne.io.read_raw_brainvision(files[0], preload=True)
        raw._data -= raw._data[..., [0]]

    return raw


def extract_events(raw):
    # extract experimental events from the raw EEG data
    event_id = {
        'New Segment/': 0, 
        'Stimulus/S  1': 1, 
        'Stimulus/S  2': 2,
        'Stimulus/S  4': 4, 
        'Stimulus/S  8': 8
    }
    events = mne.events_from_annotations(raw, event_id)[0]
    assert (raw.info['sfreq'] == fs_eeg)

    return events


def preprocess_eeg(raw):
    # check if recording is the expected channels
    # question: are the expected channels always in that order?
    if set(raw.ch_names) == set(['FCzA1', 'FCz', 'A2']):
        # re-reference the data to the difference of first two channels
        raw._data[1] -= raw._data[2]
        # after re-ref, drop the redundant negative one
        raw.drop_channels(['A2'])
        raw.rename_channels({'FCzA1': 'A1', 'FCz': 'A2'})  # rename remaining
        raw._data = raw._data[:2] / 100  # account for gain
    else:
        assert False

    # filter the eeg

    """
    From paper: "Modeled and measured EEG were bandpass filtered from 30-2000 Hz using a 1st 
                order causal Butterworth filter and notch filtered at odd multiples of 60 Hz
                (causal infinite impulse response, IIR) to remove electrical line noise.
    """
    l_freq = 30.
    h_freq = 2000.
    notch_freqs = np.arange(60, h_freq, 120)
    notch_width = 5
    br, ar = butter(1, [l_freq, h_freq], 'bandpass', fs=fs_eeg)
    br, ar = [br], [ar]
    for nf in notch_freqs:
        bn, an = iirnotch(nf, nf / notch_width, fs=fs_eeg)
        br.append(bn)
        ar.append(an)
    sos = np.array((br, ar)).transpose(1, 0, 2).reshape(-1, 6)  # second order sections

    zi = sosfilt_zi(sos)[:, :, None] * raw._data[:, :1].T
    raw._data, zi = sosfilt(sos, raw._data, zi=zi.transpose(0, 2, 1))
    
    return raw, zi


def create_events(events):
    # parameters
    stamp_dur = 9.98

    # correct for tubing length
    tube_len_meters = 11. * 25.4 * 1e-3
    tube_delay = tube_len_meters / 343.
    tube_delay_len = int(np.round(tube_delay * fs_eeg))
    events[:, 0] += tube_delay_len

    # get the right ends figured out (should handle missing start or end trigs)
    starts = events[events[:, -1] == 1]
    ends = events[events[:, -1] == 2]
    nom_length = np.median(np.diff(starts[:, 0]))
    all_length = ends[:, [0]] - starts[:, [0]].T
    len_trial = all_length.ravel()[(all_length.ravel() > 0) &
                                    (all_length.ravel() < nom_length)]

    # calculate adjustment for different clocks
    clock_diff = stamp_dur * fs_eeg - len_trial
    clock_adjust = (stamp_dur * fs_eeg / len_trial).mean()

    # process triggers to get event info
    start_inds = np.array(np.where(events[:, -1] == 1))[0]
    stop_inds = np.array(np.where(events[:, -1] == 2))[0]

    # stamps = [stim, tr, token]
    n_bits_stamp = np.sum([n_bits_stimuli, n_bits_trials, n_bits_trials])
    bits_epoch = [events[start + 1:start + n_bits_stamp + 1, 2] >> 3
                    for start in start_inds]
    stories = np.array([binary_to_decimals(be[
        :n_bits_stimuli], n_bits_stimuli)[0] for be in bits_epoch])
    tokens = np.array([binary_to_decimals(be[
        n_bits_stimuli:n_bits_stimuli + n_bits_trials],
        n_bits_trials)[0] for be in bits_epoch])
    trials = np.array([binary_to_decimals(be[
        -n_bits_trials:], n_bits_trials)[0] for be in bits_epoch])

    event_info = np.moveaxis(np.array([stories, trials, tokens]), 0, 1)
    stim_events = starts.copy()
    stim_events[:, -1] = stories
    stim_events[:, -2] = tokens

    return event_info, stim_events, clock_adjust
