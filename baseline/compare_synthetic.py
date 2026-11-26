"""Paper LS vs Tikhonov vs Gaussian prior vs diffusion prior on simulated EEG.

Per seed: random true ABR, 20 min of simulated EEG, estimates from the first
1/2/5/10/20 min. Reports RMSE, correlation, wave V errors, and +-2 sd coverage.

    python compare_synthetic.py      # a few minutes on a GPU, writes figs/
"""
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import diffusion as dm
from estimators import GaussianPrior, paper_ls, tikhonov
from sim import LAGS_MS, NK, TRIAL, sample_abr, simulate, sufficient_stats, wave_v

FIG = Path(__file__).parent / "figs"
MINUTES = [1, 2, 5, 10, 20]
SIGMA_UV = 12.0
SEEDS = 6
N_POST = 32
TIK_GRID = [0.03, 0.1, 0.3, 1, 3, 10]    # times trace(A)/NK; the best one is picked knowing the truth
METHODS = ["paper LS", "Tikhonov (oracle lam)", "Gaussian MAP / Wiener", "diffusion (post. mean)"]


def metrics(est, truth):
    lat_t, amp_t = wave_v(truth)
    lat_e, amp_e = wave_v(est)
    return dict(rmse=np.sqrt(np.mean((est - truth) ** 2)),
                corr=np.corrcoef(est, truth)[0, 1],
                lat_err=lat_e - lat_t, amp_err=amp_e - amp_t)


def run_all(h, X, Y, prior, model, scale):
    A, b = sufficient_stats(X, Y)
    out = {"paper LS": paper_ls(X, Y)}
    tiks = [tikhonov(A, b, g * np.trace(A) / NK) for g in TIK_GRID]
    out["Tikhonov (oracle lam)"] = min(tiks, key=lambda e: np.mean((e - h) ** 2))
    out["Gaussian MAP / Wiener"], P = prior.map(A, b)
    samples = dm.sample_posterior(model, scale, A, b, n=N_POST)
    out["diffusion (post. mean)"] = samples.mean(0)
    gauss_sd = np.sqrt(np.diag(P))
    cover = (np.mean(np.abs(samples.mean(0) - h) < 2 * samples.std(0)),
             np.mean(np.abs(out["Gaussian MAP / Wiener"] - h) < 2 * gauss_sd))
    return out, samples, cover


def main():
    FIG.mkdir(exist_ok=True)
    model, scale = dm.load()
    prior = GaussianPrior()
    results = {m: {k: [] for k in MINUTES} for m in METHODS}
    coverage = {m: [] for m in MINUTES}
    examples = {}
    t0 = time.time()
    for seed in range(SEEDS):
        rng = np.random.default_rng(100 + seed)
        h = sample_abr(1, rng)[0]
        X, Y = simulate(h, max(MINUTES), SIGMA_UV, rng)
        for minutes in MINUTES:
            n = int(minutes * 60 / TRIAL)   # shorter recordings are prefixes of the long one
            ests, samples, cover = run_all(h, X[:n], Y[:n], prior, model, scale)
            for name, e in ests.items():
                results[name][minutes].append(metrics(e, h))
            coverage[minutes].append(cover)
            if seed == 0:
                examples[minutes] = (h, ests, samples)
        print(f"seed {seed} done, {time.time() - t0:.0f}s", flush=True)

    lines = [f"noise {SIGMA_UV} uV RMS, {SEEDS} seeds, mean over seeds\n"]
    for key, label in [("rmse", "RMSE (uV)"), ("corr", "corr"),
                       ("lat_err", "|wave V latency err| (ms)"), ("amp_err", "wave V amp err (uV)")]:
        lines.append(f"\n{label}")
        lines.append(f"{'':26s}" + "".join(f"{m:>8d}min" for m in MINUTES))
        for name in METHODS:
            row = []
            for m in MINUTES:
                v = np.array([r[key] for r in results[name][m]])
                row.append(np.mean(np.abs(v)) if key == "lat_err" else np.mean(v))
            lines.append(f"{name:26s}" + "".join(f"{v:11.3f}" for v in row))
    lines.append("\nfraction of lags where the truth is inside the +-2 sd band (should be ~0.95)")
    lines.append(f"{'':26s}" + "".join(f"{m:>8d}min" for m in MINUTES))
    for j, name in enumerate(["diffusion samples", "Gaussian posterior"]):
        lines.append(f"{name:26s}" + "".join(f"{np.mean([c[j] for c in coverage[m]]):11.3f}" for m in MINUTES))
    txt = "\n".join(lines)
    print(txt)
    (FIG / "results.txt").write_text(txt)

    # example reconstructions, one row per method, one column per duration
    fig, axes = plt.subplots(len(METHODS), len(MINUTES), figsize=(16, 9), sharex=True, sharey=True)
    for j, minutes in enumerate(MINUTES):
        h, ests, samples = examples[minutes]
        for i, name in enumerate(METHODS):
            ax = axes[i, j]
            ax.plot(LAGS_MS, h, "k", lw=1.2, label="true ABR")
            color = "C3" if name.startswith("diffusion") else "C0"
            if name.startswith("diffusion"):
                ax.plot(LAGS_MS, samples[:8].T, color=color, lw=0.4, alpha=0.5)
            ax.plot(LAGS_MS, ests[name], color=color, lw=1.2, label=name)
            ax.set_title(f"{name}, {minutes} min" if i == 0 else name, fontsize=9)
            ax.axvline(0, color="gray", lw=0.5, ls=":")
            if j == 0:
                ax.set_ylabel("uV")
    axes[0, 0].set_ylim(-0.6, 0.8)
    axes[-1, 0].set_xlabel("lag (ms)")
    fig.suptitle(f"Deconvolved ABR vs recording time (noise {SIGMA_UV} uV RMS, 90 Hz peaky speech)")
    fig.tight_layout()
    fig.savefig(FIG / "reconstructions.png", dpi=130)

    # metrics vs minutes
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for name in METHODS:
        for ax, key in zip(axes, ["rmse", "corr", "amp_err"]):
            vals = [np.mean([r[key] for r in results[name][m]]) for m in MINUTES]
            ax.plot(MINUTES, vals, "o-", label=name)
    for ax, lab in zip(axes, ["RMSE (uV)", "correlation with truth", "wave V amplitude error (uV)"]):
        ax.set_xscale("log")
        ax.set_xticks(MINUTES, [str(m) for m in MINUTES])
        ax.set_xlabel("recording time (min)")
        ax.set_title(lab)
        ax.grid(alpha=0.3)
    axes[2].axhline(0, color="k", lw=0.5)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "metrics_vs_minutes.png", dpi=130)


if __name__ == "__main__":
    main()
