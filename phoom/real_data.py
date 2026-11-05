"""Experiment 3 (ds008065), one subject + one story: paper method vs
Gaussian prior vs diffusion prior on the same EEG and pulse trains.

All 10 pulse trains (5 bands x 2 ears) are regressed at once, so A is
2560 x 2560 block-Toeplitz and band cross-talk sits in the likelihood
rather than being subtracted with fake pulse trains. The 5/10/20 min
estimates are scored against the paper method's 35 min response.
Priors come from the other subjects (real_prior.py).

Preprocessing as in 02_analysis_f0s2_loop.py: earlobe reference, /100
amplifier gain, 30-2000 Hz Butterworth + notches at 60, 180, 300, ... Hz,
clock drift. Earphone tubing delay is already in `sample_corrected`.

    python real_data.py sub-exp301 wizard
"""
import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import torch
from scipy.signal import butter, iirnotch, lfilter, sosfilt

import diffusion as dm
import real_prior
from sim import FS, LAG0, LAGS_MS, NK, wave_v

HERE = Path(__file__).parent
DATA = HERE / "data" / "ds008065"        # symlink or copy of the OpenNeuro download
FIG = HERE / "figs"

FS_STIM = 48_000
L = 101_000                              # 10.1 s of EEG per trial (sim.py uses 10.0 s)
TRIALS_PER_MIN = 6
FREQS = [0.5, 1, 2, 4, 8]
EARS = ["left", "right"]
N_REG = len(FREQS) * len(EARS)           # regressor index is band * 2 + ear
STAMP_DUR = 9.98                         # seconds between the start and end triggers
MINUTES = [5, 10, 20]
LF_POST = 150.0                          # the paper's highpass on the final response


def preproc_sos(fs=FS, l_freq=30., h_freq=2000.):
    """Same filters as the paper's code, as second-order sections."""
    br, ar = butter(1, [l_freq, h_freq], 'bandpass', fs=fs)
    br, ar = [br], [ar]
    for nf in np.arange(60, h_freq, 120):
        bn, an = iirnotch(nf, nf / 5, fs=fs)
        br.append(bn)
        ar.append(an)
    return np.array((br, ar)).transpose(1, 0, 2).reshape(-1, 6)


def post_hp(w):
    b, a = butter(1, LF_POST, btype='highpass', fs=FS)
    return lfilter(b, a, w, axis=-1)


def read_h5(path):
    """The stimulus files use expyfun's h5io layout: lists are stored as
    groups with members idx_0, idx_1, ..."""
    def rd(obj):
        if isinstance(obj, h5py.Dataset):
            return obj[()]
        keys = list(obj.keys())
        if keys and all(k.startswith("idx_") for k in keys):
            return [rd(obj[f"idx_{i}"]) for i in range(len(keys))]
        return {k: rd(obj[k]) for k in keys}
    with h5py.File(path, "r") as f:
        d = rd(f["expyfun"])
    return {k.removeprefix("key_"): v for k, v in d.items()}


def load_subject(sub):
    """Filtered EEG in uV (runs concatenated), trial start table, and the
    stimulus/EEG clock ratio. Channels are FCzA1, FCz, A2: FCzA1 is already
    FCz re A1, FCz - A2 is the other ear, and the two get averaged."""
    eeg_dir = DATA / sub / "eeg"
    sos = preproc_sos()
    chunks = []
    events = []
    offset = 0
    for vhdr in sorted(eeg_dir.glob("*_eeg.vhdr")):
        raw = mne.io.read_raw_brainvision(vhdr, preload=True, verbose=False)
        assert raw.info["sfreq"] == FS and raw.ch_names == ["FCzA1", "FCz", "A2"]
        d = raw.get_data()
        del raw
        d -= d[:, [0]]                      # remove each channel's starting offset
        d[1] -= d[2]                        # FCz - A2
        d = d[:2] / 100.0                   # EP-PreAmp gain
        d = sosfilt(sos, d)
        chunks.append((d.mean(0) * 1e6).astype(np.float32))
        ev = pd.read_csv(str(vhdr).replace("_eeg.vhdr", "_events.tsv"), sep="\t")
        ev["sample_corrected"] += offset
        ev["sample"] += offset
        events.append(ev)
        offset += d.shape[1]
        del d
    data = np.concatenate(chunks)
    ev = pd.concat(events, ignore_index=True)
    starts = ev[ev.trial_type == "stimulus_start"].reset_index(drop=True)
    ends = ev[ev.trial_type == "stimulus_end"]
    # the stimulus computer's clock runs slightly off the EEG clock; measure it
    # from the start/end trigger spacing, same as the paper's code
    lens = ends["sample"].values[:, None] - starts["sample"].values[None, :]
    nominal = np.median(np.diff(starts["sample"].values))
    lens = lens[(lens > 0) & (lens < nominal)]
    clock_adjust = (STAMP_DUR * FS / lens).mean()
    return data, starts, clock_adjust


def pulse_trains(stim_file, clock_adjust):
    """Real pulse trains (N_REG, L) and fake ones (5, L) for one trial."""
    d = read_h5(DATA / "stimuli" / stim_file)
    x = np.zeros((N_REG, L))
    for i, p in enumerate(d["pulseinds_true"]):
        idx = (np.asarray(p) / FS_STIM / clock_adjust * FS).astype(int)
        x[i, idx[(idx >= 0) & (idx < L)]] = 1.0
    xf = np.zeros((len(FREQS), L))
    for i, p in enumerate(d["pulseinds_fake"]):
        idx = (np.asarray(p) / FS_STIM / clock_adjust * FS).astype(int)
        xf[i, idx[(idx >= 0) & (idx < L)]] = 1.0
    return x, xf


class Accum:
    """Running sums over trials, for both the paper estimator and (A, b).
    A snapshot of the sums is kept at each trial count in `checkpoints` so
    the first 5/10/20 minutes can be scored separately."""

    SUMS = ["XY", "XX", "YY", "XfY", "XX_mean", "XfXf_mean"]

    def __init__(self, checkpoints):
        self.checkpoints = set(checkpoints)
        self.n = 0
        self.var = []
        self.XY = np.zeros((N_REG, L), complex)            # sum conj(X) Y / var
        self.XX = np.zeros((N_REG, N_REG, L), complex)     # sum conj(X_i) X_j / var
        self.YY = np.zeros(L)                              # sum |Y|^2 / var
        self.XfY = np.zeros((len(FREQS), L), complex)      # same as XY, fake pulses
        self.XX_mean = np.zeros(L)                         # sum of mean_i |X_i|^2, unweighted
        self.XfXf_mean = np.zeros(L)
        self.snaps = {}

    def add(self, x, xf, y):
        X = np.fft.fft(x, axis=-1)
        Xf = np.fft.fft(xf, axis=-1)
        Y = np.fft.fft(y)
        Y[0] = 0
        v = y.var()
        self.var.append(v)
        # numerators get the 1/var trial weighting (paper) = per-trial noise scale (A, b).
        # the paper's denominator is plain |X|^2 averaged over the 10 pulse trains, no weight.
        self.XY += np.conj(X) * Y / v
        self.XX += np.einsum("il,jl->ijl", np.conj(X), X) / v
        self.YY += np.abs(Y) ** 2 / v
        self.XfY += np.conj(Xf) * Y / v
        self.XX_mean += (np.abs(X) ** 2).mean(0)
        self.XfXf_mean += (np.abs(Xf) ** 2).mean(0)
        self.n += 1
        if self.n in self.checkpoints:
            self.snapshot()

    def snapshot(self):
        snap = {k: getattr(self, k).copy() for k in self.SUMS}
        snap["var"] = list(self.var)
        self.snaps[self.n] = snap


def accumulate(y_all, trials, story, clock_adjust, checkpoints):
    """Run every trial of one story through an Accum. Always ends with a
    snapshot at the last trial, even if the recording stopped early."""
    acc = Accum(checkpoints)
    for _, tr in trials.iterrows():
        s0 = int(tr.sample_corrected)
        if s0 + L > len(y_all):
            break
        # the paper's code loads token + 1; the 'audio' column in events.tsv
        # is off by one from the files that actually exist
        fname = f"exp3_f0s2/{story}/{story}_shifted_{int(tr.token) + 1:04d}.hdf5"
        x, xf = pulse_trains(fname, clock_adjust)
        acc.add(x, xf, y_all[s0:s0 + L].astype(float))
    if acc.n not in acc.snaps:
        acc.snapshot()
    return acc


def paper_estimate(s):
    """derive_responses() from the paper's code, written on the accumulated
    sums: weighted per-frequency LS minus the mean fake-pulse response."""
    n = len(s["var"])
    wsum = np.sum(1 / np.array(s["var"]))
    wb = np.fft.ifft(s["XY"] / wsum / (s["XX_mean"] / n), axis=-1).real
    wc = np.fft.ifft(s["XfY"] / wsum / (s["XfXf_mean"] / n), axis=-1).real
    ws = wb - wc.mean(0, keepdims=True)
    return ws[:, (np.arange(NK) + LAG0) % L]


def joint_stats(s, width_hz=20.0):
    """A (N_REG*NK square) and b (N_REG*NK,) for all 10 regressors at once.
    Block (i, j) of A is the whitened cross-correlation of pulse trains i and j."""
    n = len(s["var"])
    # noise spectrum from the EEG itself, 20 Hz boxcar, same as sim.estimate_noise_psd
    p = s["YY"] / n
    w = int(width_hz * L / FS)
    if w % 2 == 0:
        w += 1
    box = np.zeros(L)
    box[np.arange(-(w // 2), w // 2 + 1) % L] = 1 / w
    N = np.maximum(np.fft.ifft(np.fft.fft(p) * np.fft.fft(box)).real, 1e-9 * p.max())
    idx = np.arange(NK)
    lagdiff = (idx[:, None] - idx[None, :]) % L
    A = np.zeros((N_REG, NK, N_REG, NK))
    for i in range(N_REG):
        for j in range(N_REG):
            r = L * np.fft.ifft(s["XX"][i, j] / N).real
            A[i, :, j, :] = r[lagdiff]
    c = L * np.fft.ifft(s["XY"] / N, axis=-1).real
    b = c[:, (idx + LAG0) % L]
    return A.reshape(N_REG * NK, N_REG * NK), b.reshape(-1)


def gauss_joint(A, b, prior):
    """Same prior N(mu, C) on each of the 10 responses, solved jointly."""
    Cinv = np.kron(np.eye(N_REG), prior.Cinv)
    mu = np.tile(prior.mu, N_REG)
    return np.linalg.solve(A + Cinv, b + Cinv @ mu).reshape(N_REG, NK)


@torch.no_grad()
def diffusion_joint(model, scale, A, b, n=16, n_steps=100, lam=1.0, zeta=1.0, seed=0):
    """diffusion.sample_posterior, but the denoiser runs on each of the 10
    responses separately and the data step solves all of them together."""
    dev = dm.DEV
    g = torch.Generator(device=dev).manual_seed(seed)
    At = torch.tensor(A * scale ** 2, dtype=torch.float64, device=dev)
    bt = torch.tensor(b * scale, dtype=torch.float64, device=dev)
    I = torch.eye(N_REG * NK, dtype=torch.float64, device=dev)
    x = torch.randn(n, N_REG, NK, device=dev, generator=g)
    ts = torch.linspace(dm.T - 1, 0, n_steps, device=dev).round().long()
    for i, t in enumerate(ts):
        a = dm.ABAR[t]
        eps = model(x.reshape(n * N_REG, 1, NK), t.expand(n * N_REG)).reshape(n, N_REG, NK)
        x0 = (x - (1 - a).sqrt() * eps) / a.sqrt()
        rho = lam * (a / (1 - a)).double()
        rhs = bt[None, :] + rho * x0.reshape(n, -1).double()
        x0 = torch.linalg.solve(At + rho * I, rhs.T).T.float().reshape(n, N_REG, NK)
        if i + 1 < len(ts):
            a_prev = dm.ABAR[ts[i + 1]]
            eps_hat = (x - a.sqrt() * x0) / (1 - a).sqrt()
            z = torch.randn(x.shape, device=dev, generator=g)
            x = a_prev.sqrt() * x0 + (1 - a_prev).sqrt() * (np.sqrt(1 - zeta) * eps_hat + np.sqrt(zeta) * z)
        else:
            x = x0
    return x.cpu().numpy() * scale


WIN = (LAGS_MS >= 0) & (LAGS_MS <= 15)


def score(est, ref):
    """Correlation and RMSE against the reference over 0-15 ms, per regressor."""
    corr = [np.corrcoef(e[WIN], r[WIN])[0, 1] for e, r in zip(est, ref)]
    rmse = [np.sqrt(np.mean((e[WIN] - r[WIN]) ** 2)) for e, r in zip(est, ref)]
    return np.array(corr), np.array(rmse)


def main(sub="sub-exp301", story="wizard"):
    FIG.mkdir(exist_ok=True)
    y_all, starts, clock_adjust = load_subject(sub)
    trials = starts[starts.stimulus_type == story].reset_index(drop=True)
    print(f"{sub} {story}: {len(trials)} trials, clock_adjust {clock_adjust:.6f}, EEG rms {y_all.std():.1f} uV")
    acc = accumulate(y_all, trials, story, clock_adjust, [m * TRIALS_PER_MIN for m in MINUTES])
    print(f"accumulated {acc.n} trials")

    n_full = max(acc.snaps)
    full_min = n_full / TRIALS_PER_MIN
    ref = post_hp(paper_estimate(acc.snaps[n_full]))
    model, scale = real_prior.load_real(sub)
    prior = real_prior.RealGaussian(sub)
    print(f"priors built from {prior.n} responses of the other subjects")

    ests = {}
    results = {}
    for n in sorted(acc.snaps):
        snap = acc.snaps[n]
        A, b = joint_stats(snap)
        samples = diffusion_joint(model, scale, A, b)
        ests[n] = dict(paper=post_hp(paper_estimate(snap)),
                       gauss=post_hp(gauss_joint(A, b, prior)),
                       diffusion=post_hp(samples.mean(0)),
                       diff_sd=post_hp(samples).std(0))
        if n < n_full:
            results[n / TRIALS_PER_MIN] = {k: score(ests[n][k], ref) for k in ["paper", "gauss", "diffusion"]}
        print(f"{n / TRIALS_PER_MIN:5.1f} min done", flush=True)

    lines = [f"{sub}, {story}: short recordings vs the paper method's {full_min:.0f}-min response, "
             f"mean over 10 regressors (5 bands x 2 ears), 0-15 ms, after the 150 Hz highpass\n"]
    lines.append(f"{'':15s}" + "".join(f"{m:>8.0f}min" for m in results))
    for k in ["paper", "gauss", "diffusion"]:
        lines.append(f"corr {k:10s}" + "".join(f"{results[m][k][0].mean():11.3f}" for m in results))
    for k in ["paper", "gauss", "diffusion"]:
        lines.append(f"rmse {k:10s}" + "".join(f"{results[m][k][1].mean():11.4f}" for m in results))
    lines.append("\nper-band correlation at 5 min (left | right):")
    for fi, f in enumerate(FREQS):
        row = f"{f:4.1f} kHz "
        for k in ["paper", "gauss", "diffusion"]:
            c = results[min(results)][k][0]
            row += f"  {k}: {c[fi * 2]:+.2f}|{c[fi * 2 + 1]:+.2f}"
        lines.append(row)
    lines.append("\nreference wave V (lag ms, amp uV), left ear: " + ", ".join(
        f"{f}k ({wave_v(ref[fi * 2])[0]:.1f}, {wave_v(ref[fi * 2])[1]:.3f})" for fi, f in enumerate(FREQS)))
    txt = "\n".join(lines)
    print(txt)
    (FIG / f"real_{sub}_{story}.txt").write_text(txt)

    # 5-minute estimates against the reference, every band and ear
    n5 = min(acc.snaps)
    short_min = n5 / TRIALS_PER_MIN
    fig, axes = plt.subplots(2, len(FREQS), figsize=(18, 6), sharex=True)
    for fi, f in enumerate(FREQS):
        for ei in range(2):
            ax = axes[ei, fi]
            r = fi * 2 + ei
            e = ests[n5]
            ax.plot(LAGS_MS, ref[r], "k", lw=1.5, label=f"paper, {full_min:.0f} min (reference)")
            ax.plot(LAGS_MS, e["paper"][r], "C2", lw=0.8, label=f"paper, {short_min:.0f} min")
            ax.plot(LAGS_MS, e["gauss"][r], "C0", lw=1, label=f"Gaussian MAP, {short_min:.0f} min")
            ax.plot(LAGS_MS, e["diffusion"][r], "C3", lw=1, label=f"diffusion, {short_min:.0f} min")
            ax.fill_between(LAGS_MS, e["diffusion"][r] - 2 * e["diff_sd"][r],
                            e["diffusion"][r] + 2 * e["diff_sd"][r], color="C3", alpha=0.15)
            ax.set_title(f"{f} kHz, {EARS[ei]}", fontsize=9)
            ax.axvline(0, color="gray", lw=0.5, ls=":")
            ax.set_xlim(-5, 15)
    axes[0, 0].legend(fontsize=7)
    axes[1, 0].set_xlabel("lag (ms)")
    axes[0, 0].set_ylabel("uV")
    fig.suptitle(f"{sub}, {story}: {short_min:.0f} min estimates vs the {full_min:.0f} min reference")
    fig.tight_layout()
    fig.savefig(FIG / f"real_{sub}_{story}.png", dpi=130)


if __name__ == "__main__":
    main(*sys.argv[1:])
