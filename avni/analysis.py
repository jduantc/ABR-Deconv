import numpy as np
from pathlib import Path
import mne
import os
from glob import glob
from scipy.signal import sosfilt, sosfilt_zi, butter, iirnotch, lfilter
from scipy.fftpack import fft, ifft
from expyfun.io import read_hdf5, write_hdf5
from expyfun import binary_to_decimals
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from deconvolution import deconvolution1, deconvolution_wiener
from eeg import read_eeg, extract_events, preprocess_eeg, create_events


# %% parameters

# modify variables
max_trials = 100  # 1 - 210 (# of 10 second clips)
# subject = 1  # 1 - 15
deconv = deconvolution_wiener

# current directory path
src = Path(__file__).resolve().parent

# output directory
if not os.path.exists(f'{src}/responses/{deconv.__name__}'):
    os.makedirs(f'{src}/responses/{deconv.__name__}')

# directories
stimuli_path = f"{src}/data/stimuli/exp3_f0s2/{{}}/{{}}_shifted_{{:04d}}.hdf5"  # stimuli, stimuli
eeg_path = f"{src}/data/sub-exp3{{:02d}}/eeg/sub-exp3{{:02d}}_task*.vhdr"  # subject, subject
response_path = f"{src}/responses/{deconv.__name__}/sub-exp3{{:02d}}_responses.hdf5"  # subject
plot_path = f"{src}/responses/{deconv.__name__}/sub-exp3{{:02d}}_responses.jpg"  # subject

fs_stim = int(48e3)  # stimulus sampling rate (Hz)
# subjects = np.arange(1, 16)
subjects = [1]
ears = ['left', 'right']
freqs = [.5, 1, 2, 4, 8]
stimuli = ['wizard', 'wonderland', 'drdolittle']

# eeg parameters
fs_eeg = int(10e3)  # EEG sampling rate (Hz)
t_min, t_max = -1, 1
n_pre, n_post = -t_min * fs_eeg, t_max * fs_eeg

stim_dur = 10.


# %% stimulus functions
def get_stimulus_data(raw, si, stim, stim_events, clock_adjust):
    # select the start events that match the stimuli index
    events_stim = stim_events[stim_events[:, -1] == si]
    events_stim = events_stim[:max_trials]
    tok_stim = events_stim[:, -2]

    ep = mne.Epochs(raw, events_stim, tmin=t_min, tmax=stim_dur +
                    t_max - 1 / fs_eeg, preload=True, proj=None)
    y = ep._data
    n_samples = int(stim_dur * fs_eeg + n_pre + n_post)
    x = np.zeros((len(tok_stim), len(freqs) * len(ears), n_samples))
    xf = np.zeros((len(tok_stim), len(freqs), n_samples))
    for i, tok in enumerate(tok_stim):
        stim_file = stimuli_path.format(stim, stim, tok + 1)
        data = read_hdf5(stim_file)
        pinds_true = data['pulseinds_true']
        pinds_false = data['pulseinds_fake']
        for bi in range(len(freqs) * len(ears)):
            x[i, bi, ((pinds_true[bi] / float(fs_stim) / clock_adjust - t_min) *
                        float(fs_eeg)).astype(int)] = 1.
        for fi in range(len(freqs)):
            xf[i, fi, ((pinds_false[fi] / float(fs_stim) / clock_adjust - t_min) *
                        float(fs_eeg)).astype(int)] = 1.
        del data
    x = x[:y.shape[0]]
    xf = xf[:y.shape[0]]
    x = np.moveaxis(np.array([x[:, ei::2] for ei in range(len(ears))]), 0, 1)[
        ..., np.newaxis, :]  # add dim for channel
    xf = xf[:, np.newaxis, ..., np.newaxis, :]  # add dim for ear & channel
    y = y[:, np.newaxis, np.newaxis]  # add dim for ear and band

    return x, xf, y

# %% main code

if __name__ == "__main__":
    # loop through all subjects
    for sub in subjects:
        print(f'\n\n===== sub-exp3{sub:02d} =====\n\n')
        print('\nloading eeg data\n')
        files = sorted(glob(eeg_path.format(sub, sub)))  # paths to vhdr type files
        raw = read_eeg(files)
        events = extract_events(raw)
        raw, zi = preprocess_eeg(raw)
        event_info, stim_events, clock_adjust = create_events(events)

        n_reps = 2
        n_trials_stim = np.zeros(len(stimuli))
        w_mb = np.zeros((len(stimuli), len(ears), len(freqs), n_pre + n_post))
        w_sb = np.zeros((len(stimuli), len(ears), n_pre + n_post))
        w_mb_reps = np.zeros((len(stimuli), n_reps, len(ears), len(freqs), n_pre + n_post))
        w_sb_reps = np.zeros((len(stimuli), n_reps, len(ears), n_pre + n_post))

        print('\ndeconvolving\n')
        for si, stim in enumerate(stimuli):
            print(f'----- {stim}')
            # select the start events that match the stimuli index
            x, xf, y = get_stimulus_data(raw, si, stim, stim_events, clock_adjust)

            # deconvolve responses
            n_trials_stim[si] = y.shape[0]
            w_mb[si], w_sb[si] = deconv(x, xf, y)
            for ri in range(n_reps):
                w_mb_reps[si, ri], w_sb_reps[si, ri] = deconv(
                    x[ri::n_reps], xf[ri::n_reps], y[ri::n_reps])

        # %% save responses
        print('\nsaving data\n')
        write_hdf5(response_path.format(sub), dict(
            n_trials_stim=n_trials_stim,
            w_mb=w_mb,
            w_sb=w_sb,
            w_mb_reps=w_mb_reps,
            w_sb_reps=w_sb_reps,
            fs=fs_eeg,
            stimuli=stimuli,
            ears=ears,
            freqs=freqs), overwrite=True)

        # %% post-filter
        lf_post = 150.
        br, ar = butter(1, lf_post, btype='highpass', fs=fs_eeg)
        w_subtr = lfilter(br, ar, w_mb)
        w_reps = lfilter(br, ar, w_mb_reps)

        # %% plot
        colors = ['b', 'r']
        y_shift = 0.1
        ai = 0.15
        t_ms = 1e3 * np.arange(-n_pre, n_post) / float(fs_eeg)
        fig = plt.figure(figsize=(4 * len(stimuli), 8))
        for si, stim in enumerate(stimuli):
            plt.subplot(1, len(stimuli), si + 1)
            plt.grid(alpha=0.3, axis='x')
            plt.axvline(0, lw=1, color='k')
            for fi, freq in enumerate(freqs):
                for ei, ear in enumerate(ears):
                    plt.plot(t_ms, y_shift * fi + w_subtr[si, ei, fi],
                            color=colors[ei], label=ear)
                    [plt.plot(t_ms, y_shift * fi + w_reps[si, ri, ei, fi], lw=1,
                            color=colors[ei], alpha=ai) for ri in range(n_reps)]
                plt.text(0, y_shift * fi + y_shift / 6, f'{freq} kHz ',
                        ha='right', va='bottom')
            if si == 0:
                handles, labels = plt.gca().get_legend_handles_labels()
                plt.legend(handles[:2], labels[:2], loc='upper right',
                        edgecolor='none', facecolor='none')
            plt.xticks(np.arange(-40, 40, 2))
            plt.yticks(np.arange(-1.2, 1.2, y_shift))
            plt.xlim(-5, 14)
            plt.ylim(-y_shift / 1.5, y_shift * fi + y_shift / 1.5)
            plt.title(f'{stim}: f0 90 Hz')
            plt.gca().set_yticklabels([])
        fig.text(0.5, 0, 'Latency (ms)', ha='center', va='bottom')
        fig.text(0, 0.5, f'Potential ({y_shift} μV)', rotation=90,
                ha='left', va='center')
        plt.tight_layout()
        plt.savefig(plot_path.format(sub))
        plt.close('all')
