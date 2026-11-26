# Synthetic data generators borrowed from Leo's code
# Minus a couple checks and modifications that should not change the results
# Plus a wavefroms_uv helper and test_recordings that DPS is scored on for
# RMSE results

import numpy as np
import torch
from scipy.signal import butter, sosfiltfilt

SAMPLE_RATE = 10_000
WAVEFORM_LENGTH = 256
AMPLITUDE_SCALE_UV = 0.1
TIME_MS = (np.arange(WAVEFORM_LENGTH) - 50) / 10
NOISE_UV = [0.05, 0.2, 0.5]


def generate_responses(count, seed):
    rng = np.random.default_rng(seed)
    time_ms = np.arange(-500, 501) / 10

    latency = rng.uniform(3, 7, (count, 1))
    width = rng.uniform(0.35, 0.9, (count, 1))
    amplitude = np.exp(rng.normal(0, 0.35, (count, 1))).clip(0.3, 2.5)
    trough_delay = rng.uniform(1.3, 3.2, (count, 1))
    trough_amplitude = rng.uniform(0.35, 0.8, (count, 1))
    trough_width = width * rng.uniform(1, 1.8, (count, 1))
    early_amplitude = rng.uniform(0, 0.3, (count, 1))
    early_latency = latency - rng.uniform(1.2, 2.4, (count, 1))

    def gaussian(center, spread):
        return np.exp(-0.5 * ((time_ms - center) / spread) ** 2)

    responses = amplitude * gaussian(latency, width)
    responses -= amplitude * trough_amplitude * gaussian(
        latency + trough_delay, trough_width
    )
    responses += amplitude * early_amplitude * gaussian(early_latency, width * 0.65)
    responses[:, time_ms < 0] = 0

    # Filter on a longer interval before keeping the central 256 samples.
    bandpass = butter(1, [80, 1500], btype="bandpass", fs=SAMPLE_RATE, output="sos")
    responses = sosfiltfilt(bandpass, responses, axis=-1)
    responses = responses[:, 450:706]

    is_zero = rng.random(count) < 0.1
    responses[is_zero] = 0
    return responses.astype(np.float32), is_zero


def stimulus_matrix(pulses, response_length=WAVEFORM_LENGTH, onset_index=50):
    """Use the pulse timings to turn one ABR into an EEG signal."""
    pulses = torch.as_tensor(pulses, dtype=torch.float32)
    # Our ABR starts 50 samples before the pulse, so keep that offset here.
    eeg_times = torch.arange(len(pulses))[:, None]
    abr_times = torch.arange(response_length)[None, :] - onset_index
    indices = eeg_times - abr_times
    valid = (indices >= 0) & (indices < len(pulses))
    return pulses[indices.clamp(0, len(pulses) - 1)] * valid


def waveforms_uv(count, seed):
    return generate_responses(count, seed)[0].astype(np.float64) * AMPLITUDE_SCALE_UV

def test_recordings(count=12, waveform_seed=90421, noise_seed=1000):
    """Half-second recordings with a pulse every 10 ms and white noise.
    Returns the true waveforms (count, 256) in uV, the stimulus matrix (5000, 256)
    and the EEG in uV as {noise level: (count, 5000)}."""
    responses, is_zero = generate_responses(40, waveform_seed)
    truth = responses[~is_zero][:count].astype(np.float64) * AMPLITUDE_SCALE_UV
    pulses = np.zeros(SAMPLE_RATE // 2, dtype=np.float32)
    pulses[np.arange(200, len(pulses) - 300, 100)] = 1
    matrix = stimulus_matrix(pulses).numpy().astype(np.float64)
    noise = np.array([np.random.default_rng(noise_seed + i).normal(size=len(pulses)) for i in range(count)])
    eeg = {sigma: truth @ matrix.T + sigma * noise for sigma in NOISE_UV}
    return truth, matrix, eeg
