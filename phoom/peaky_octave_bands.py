"""Does peaky speech / pitch shifting change the octave-band energies?

Peaky resynthesis on 30 s of an audiobook: Praat f0 track, harmonic
amplitudes from the spectrum, then resynthesize with all harmonics in
cosine phase so they line up at each glottal pulse,

    p(t) = sum_k E(k f0'(t), t) cos(k phi'(t)),   phi' = 2 pi int f0' dt

f0' = f0, or f0 scaled to 90 Hz (Exp 3), or to 185 Hz as a stress test.
Multiband: each octave band gets its own small f0 offset, is bandpassed,
and the bands are summed. Band energies are compared on voiced samples
only, since unvoiced samples pass through unchanged.

Needs wizardofoz_01_baum.mp3 (LibriVox) next to this file.

    python peaky_octave_bands.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import parselmouth

HERE = Path(__file__).parent
FIG_DIR = HERE / "figs"
AUDIO = HERE / "wizardofoz_01_baum.mp3"

T_START, T_DUR = 60.0, 30.0
TARGET_F0 = 90.0
STRESS_F0 = 185.0            # this narrator is already ~92 Hz, so 90 Hz is barely a shift
CENTERS = np.array([500, 1000, 2000, 4000, 8000.0])
EDGES = [(fc / np.sqrt(2), fc * np.sqrt(2)) for fc in CENTERS]
F_MAX = EDGES[-1][1]         # highest harmonic we bother synthesizing
HOP = 0.005                  # envelope frame hop (s)
WIN = 0.040                  # envelope window (s)
BAND_DF0 = np.array([0.0, 0.6, 1.2, 1.8, 2.4])   # per-band f0 offsets for multiband; made up, not the paper's values


def load_excerpt():
    snd = parselmouth.Sound(str(AUDIO)).convert_to_mono()
    snd = snd.extract_part(T_START, T_START + T_DUR, preserve_times=False)
    return snd, snd.values[0].astype(float), snd.sampling_frequency


def f0_track(snd, n, fs):
    """Per-sample f0 and a voiced mask from Praat's pitch tracker."""
    pitch = snd.to_pitch(time_step=HOP, pitch_floor=60, pitch_ceiling=400)
    t_frames = pitch.xs()
    f0_frames = pitch.selected_array["frequency"]
    t = np.arange(n) / fs
    voiced = np.interp(t, t_frames, (f0_frames > 0).astype(float)) > 0.5
    good = f0_frames > 0
    f0 = np.interp(t, t_frames[good], f0_frames[good])    # interpolated through unvoiced gaps
    return f0, voiced


def harmonic_envelope(x, fs, f0):
    """E(f, t): for each frame, the peak amplitude near each harmonic of f0,
    linearly interpolated in between."""
    nwin = int(WIN * fs)
    nfft = 1 << int(np.ceil(np.log2(nwin * 4)))
    win = np.hanning(nwin)
    freqs = np.fft.rfftfreq(nfft, 1 / fs)
    hop = int(HOP * fs)
    centers = np.arange(nwin // 2, len(x) - nwin // 2, hop)
    env = np.zeros((len(centers), len(freqs)))
    for i, c in enumerate(centers):
        mag = np.abs(np.fft.rfft(x[c - nwin // 2:c + nwin // 2] * win, nfft))
        mag *= 2 / win.sum()                     # so a sinusoid of amplitude A gives a peak of A
        f0c = f0[c]
        hk = np.arange(1, int(F_MAX / f0c) + 1) * f0c
        amps = np.empty(len(hk))
        for j, fk in enumerate(hk):
            sel = (freqs > fk - f0c / 2) & (freqs <= fk + f0c / 2)
            amps[j] = mag[sel].max()
        env[i] = np.interp(freqs, np.r_[0, hk], np.r_[amps[0], amps], right=0)
    return centers, freqs, env


def synth_peaky(n, fs, f0_new, env_t, env_f, env):
    """sum_k E(k f0'(t), t) cos(k phi'(t)). No phase offsets, so every
    harmonic peaks at phi' = 0 mod 2 pi, i.e. at each new glottal pulse."""
    phi = 2 * np.pi * np.cumsum(f0_new) / fs
    frame = np.clip(np.interp(np.arange(n), env_t, np.arange(len(env_t))), 0, len(env_t) - 1)
    lo = np.floor(frame).astype(int)
    hi = np.minimum(lo + 1, len(env_t) - 1)
    w = frame - lo
    df = env_f[1]
    out = np.zeros(n)
    for k in range(1, int(F_MAX / f0_new.min()) + 1):
        fk = k * f0_new
        bin_ = np.clip(np.round(fk / df).astype(int), 0, len(env_f) - 1)
        a = (1 - w) * env[lo, bin_] + w * env[hi, bin_]
        a[fk > F_MAX] = 0
        out += a * np.cos(k * phi)
    return out, phi


def bandpass(x, fs, lo, hi):
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / fs)
    X[(f < lo) | (f >= hi)] = 0
    return np.fft.irfft(X, len(x))


def band_levels_db(x, fs, mask):
    """Energy in each octave band over the `mask` samples, in dB re the total."""
    e = np.array([np.sum(bandpass(x, fs, lo, hi)[mask] ** 2) for lo, hi in EDGES])
    return 10 * np.log10(e / e.sum())


def pulse_train(phi):
    """Impulse wherever phi crosses a multiple of 2 pi."""
    cyc = np.floor(phi / (2 * np.pi))
    pt = np.zeros(len(phi))
    pt[1:][np.diff(cyc) > 0] = 1
    return pt


def main():
    FIG_DIR.mkdir(exist_ok=True)
    snd, x, fs = load_excerpt()
    n = len(x)
    f0, voiced = f0_track(snd, n, fs)
    f0_mean = f0[voiced].mean()
    print(f"{T_START:.0f}-{T_START + T_DUR:.0f} s, fs {fs:.0f}, voiced {voiced.mean():.0%}, mean f0 {f0_mean:.1f} Hz")

    env_t, env_f, env = harmonic_envelope(x, fs, f0)
    f0_shift = f0 * TARGET_F0 / f0_mean

    peaky, _ = synth_peaky(n, fs, f0, env_t, env_f, env)
    shifted, phi_shift = synth_peaky(n, fs, f0_shift, env_t, env_f, env)
    stressed, _ = synth_peaky(n, fs, f0 * STRESS_F0 / f0_mean, env_t, env_f, env)
    multi = np.zeros(n)
    band_phis = []
    for (lo, hi), d in zip(EDGES, BAND_DF0):
        sb, phb = synth_peaky(n, fs, f0_shift + d, env_t, env_f, env)
        multi += bandpass(sb, fs, lo, hi)
        band_phis.append(phb)

    # compare on voiced samples only, and drop the edges where the envelope frames run out
    keep = voiced.copy()
    keep[:int(WIN * fs)] = False
    keep[-int(WIN * fs):] = False
    signals = {
        "original": x,
        "peaky (f0 kept)": peaky,
        f"peaky shifted to {TARGET_F0:.0f} Hz": shifted,
        f"peaky shifted to {STRESS_F0:.0f} Hz": stressed,
        "multiband peaky (90 Hz)": multi,
    }
    levels = {name: band_levels_db(s, fs, keep) for name, s in signals.items()}

    print("\nOctave-band energy on voiced samples (dB re sum of bands)")
    print(f"{'':32s}" + "".join(f"{c / 1000:>8.1f}k" for c in CENTERS))
    for name, lv in levels.items():
        print(f"{name:32s}" + "".join(f"{v:9.2f}" for v in lv))
    print("max |diff| vs original")
    for name, lv in list(levels.items())[1:]:
        print(f"  {name:30s}{np.abs(lv - levels['original']).max():6.2f} dB")

    print("\nHarmonics per octave band (bandwidth / f0):")
    for f in (f0_mean, TARGET_F0, STRESS_F0, 228.0):     # 228 Hz is the highest narrator f0 in the paper
        print(f"  f0 = {f:5.0f} Hz: " + ", ".join(f"{(hi - lo) / f:5.1f}" for lo, hi in EDGES))

    trains = np.array([pulse_train(p) for p in band_phis])
    trains -= trains.mean(axis=1, keepdims=True)
    r = np.corrcoef(trains)
    print(f"\nmultiband pulse-train correlation, largest off-diagonal: {np.abs(r[~np.eye(5, dtype=bool)]).max():.3f}")

    # band levels as bars
    fig, ax = plt.subplots(figsize=(7, 4))
    wbar = 0.8 / len(levels)
    for i, (name, lv) in enumerate(levels.items()):
        ax.bar(np.arange(5) + (i - (len(levels) - 1) / 2) * wbar, lv, wbar, label=name)
    ax.set_xticks(range(5), [f"{c / 1000:g} kHz" for c in CENTERS])
    ax.set_ylabel("band energy (dB re total)")
    ax.set_title("Octave-band energy, voiced speech")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "band_levels.png", dpi=150)

    # long-term spectra
    fig, ax = plt.subplots(figsize=(7, 4))
    f = np.fft.rfftfreq(n, 1 / fs)
    nsm = int(50 / f[1])
    smooth = np.ones(nsm) / nsm
    inband = (f >= EDGES[0][0]) & (f < F_MAX)
    for name, s in signals.items():
        p = np.convolve(np.abs(np.fft.rfft(s * keep)) ** 2, smooth, "same")
        ax.semilogx(f[1:], 10 * np.log10(p[1:] / p[inband].sum()), lw=1, label=name)
    for lo, _ in EDGES[1:]:
        ax.axvline(lo, color="k", ls=":", lw=0.7)
    ax.set_xlim(100, F_MAX)
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("power (dB re in-band total)")
    ax.set_title("Long-term spectrum (voiced); dotted = octave-band edges")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "spectra.png", dpi=150)

    # 60 ms of the shifted peaky speech, band by band, with the pulses marked
    start = int(10.0 * fs) + np.flatnonzero(keep[int(10.0 * fs):])[0]
    seg = slice(start, start + int(0.06 * fs))
    t = np.arange(seg.stop - seg.start) / fs * 1000
    pulses = np.flatnonzero(pulse_train(phi_shift)[seg])
    fig, axes = plt.subplots(6, 1, figsize=(7, 7), sharex=True)
    axes[0].plot(t, x[seg], lw=0.8, label="original")
    axes[0].plot(t, shifted[seg], lw=0.8, label="peaky 90 Hz")
    axes[0].legend(fontsize=7, loc="upper right")
    for ax, (lo, hi), c in zip(axes[1:], EDGES, CENTERS):
        ax.plot(t, bandpass(shifted, fs, lo, hi)[seg], lw=0.8, color="C1")
        ax.set_ylabel(f"{c / 1000:g}k", rotation=0, labelpad=15)
    for ax in axes:
        for p in pulses:
            ax.axvline(t[p], color="k", lw=0.5, ls=":")
        ax.set_yticks([])
    axes[-1].set_xlabel("time (ms); dotted = glottal pulses")
    fig.suptitle("Every octave band gets a click at each pulse")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "band_clicks.png", dpi=150)


if __name__ == "__main__":
    main()
