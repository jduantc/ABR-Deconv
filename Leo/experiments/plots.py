"""Presentation figures with shared axes, units, seeds and explicit limitations."""

import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

NAMES = {
    "analytic": "Analytic",
    "earndb_templates": "EARNDB templates",
    "earndb_pca": "EARNDB fitted PCA",
}
COLORS = {"analytic": "#2364aa", "earndb_templates": "#d95f02", "earndb_pca": "#168568"}
STYLES = {"mlp": "o-", "cnn": "s--", "unet": "^-."}


def plot_run_loss(run):
    """Refresh a small, readable loss chart during training, every 1,000 updates."""
    frame = pd.read_csv(run / "losses.csv")
    config = json.loads((run / "settings.json").read_text())
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(frame.step, frame.train_v_mse, label="Training (interval mean)", alpha=0.8)
    ax.plot(frame.step, frame.validation_v_mse, label="Fixed validation (EMA)")
    for step in config["checkpoints"]:
        if step <= frame.step.max():
            ax.axvline(step, lw=0.6, ls=":", color="gray")
    ax.set(
        title=f"{NAMES[config['pipeline']]} / {config['architecture'].upper()} / seed {config['seed']}",
        xlabel="Training updates",
        ylabel="v-prediction MSE",
    )
    ax.set_yscale("log")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    tmp = run / "losses.tmp.png"
    fig.savefig(tmp, dpi=150)
    tmp.replace(run / "losses.png")
    plt.close(fig)


def finish(fig, output, name, note):
    fig.text(0.01, 0.01, textwrap.fill(note, width=160), fontsize=8, color="#444444")
    fig.tight_layout(rect=(0, 0.09, 1, 0.94))
    fig.savefig(output / f"{name}.png", dpi=180)
    fig.savefig(output / f"{name}.pdf")
    plt.close(fig)


def plot_all(root):
    root = Path(root)
    output = root / "figures"
    output.mkdir(exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.2,
            "pdf.fonttype": 42,
        }
    )
    config = json.loads((root / "study_config.json").read_text())
    data = pd.read_csv(root / "test_metrics.csv")
    validation = pd.read_csv(root / "validation_guidance.csv")
    baseline = pd.read_csv(root / "baseline_metrics.csv")
    steps = sorted(data.step.unique())
    seeds = sorted(data.seed.unique())
    architectures = config.get("architectures", ["mlp", "cnn"])
    note = f"{len(seeds)} training seeds; fixed paired observations; guidance selected on validation only. EARNDB uses synthetic amplitude normalization."
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, domain in zip(axes, ["analytic", "earndb"]):
        for (pipe, arch), group in data[data.domain == domain].groupby(
            ["pipeline", "architecture"]
        ):
            g = group.groupby(["step", "seed"]).rmse_uv.mean().unstack()
            x = g.index.to_numpy()
            m = g.mean(1).to_numpy()
            ax.plot(
                x,
                m,
                STYLES[arch],
                color=COLORS[pipe],
                label=f"{NAMES[pipe]} / {arch.upper()}",
            )
            ax.fill_between(x, g.min(1), g.max(1), alpha=0.10, color=COLORS[pipe])
        ax.axhline(
            baseline[baseline.domain == domain].rmse_uv.mean(),
            color="#555555",
            ls=":",
            label="Fixed ridge baseline",
        )
        ax.set(
            title="Analytic test responses"
            if domain == "analytic"
            else "Held-out EARNDB subjects N7–N8",
            xlabel="Training updates",
            ylabel="Reconstruction RMSE (synthetic µV)",
            xticks=steps,
        )
        ax.set_xscale("log")
        ax.set_xticks(
            steps, labels=[f"{s // 1000}k" if s >= 1000 else str(s) for s in steps]
        )
    axes[0].legend(fontsize=7)
    fig.suptitle(
        "Does longer training improve reconstruction? Lower is better", fontsize=14
    )
    finish(
        fig,
        output,
        "01_training_duration",
        note + " Shading: min–max across training seeds (not a confidence interval).",
    )

    fig, axes = plt.subplots(
        3,
        len(architectures),
        figsize=(6 * len(architectures), 10),
        sharex=True,
        squeeze=False,
    )
    for row, pipe in enumerate(NAMES):
        for col, arch in enumerate(architectures):
            ax = axes[row, col]
            for seed in seeds:
                frame = pd.read_csv(
                    root / "models" / f"{pipe}_{arch}_{seed}" / "losses.csv"
                )
                ax.plot(
                    frame.step, frame.validation_v_mse, label=f"Validation, seed {seed}"
                )
                ax.plot(
                    frame.step,
                    frame.train_v_mse,
                    ls=":",
                    alpha=0.45,
                    label=f"Training, seed {seed}",
                )
            for step in steps:
                ax.axvline(step, color="gray", lw=0.6, ls="--")
            ax.set(title=f"{NAMES[pipe]} / {arch.upper()}", ylabel="v-prediction MSE")
            ax.set_yscale("log")
    axes[0, 0].legend(fontsize=7)
    for ax in axes[-1]:
        ax.set_xlabel("Training updates")
    fig.suptitle(
        "Training and fixed validation loss at multiple checkpoints", fontsize=14
    )
    finish(
        fig,
        output,
        "02_learning_curves",
        note
        + " Validation loss uses the same fixed waveforms, timesteps and noise for all runs.",
    )

    for domain in ["analytic", "earndb"]:
        fig, axes = plt.subplots(
            3,
            len(architectures),
            figsize=(6 * len(architectures), 9),
            sharex=True,
            sharey=True,
            squeeze=False,
        )
        for row, pipe in enumerate(NAMES):
            for col, arch in enumerate(architectures):
                ax = axes[row, col]
                group = validation[
                    (validation.pipeline == pipe)
                    & (validation.architecture == arch)
                    & (validation.domain == domain)
                ]
                for step, g in group.groupby("step"):
                    curve = g.groupby("guidance").rmse_uv.mean()
                    ax.plot(curve.index, curve.values, "o-", label=f"{step:,} updates")
                ax.set(
                    title=f"{NAMES[pipe]} / {arch.upper()}",
                    ylabel="Validation RMSE (synthetic µV)",
                )
                ax.set_xscale("symlog", linthresh=0.03)
        axes[0, 0].legend(fontsize=8)
        for ax in axes[-1]:
            ax.set_xlabel("DPS guidance strength")
        fig.suptitle(
            f"Guidance sweep on {domain} validation responses — lower is better",
            fontsize=14,
        )
        finish(
            fig,
            output,
            f"03_guidance_{domain}",
            note + " Zero guidance is an unconditional prior sample.",
        )

    predictions = np.load(root / "predictions.npz")
    rows = pd.read_csv(root / "test_observations.csv")
    time_ms = (np.arange(256) - 50) / 10
    for arch in architectures:
        for page, start in enumerate(range(0, len(steps), 4), 1):
            page_steps = steps[start : start + 4]
            fig, axes = plt.subplots(
                2,
                len(page_steps),
                figsize=(max(8, 3.6 * len(page_steps)), 7),
                sharex=True,
                sharey="row",
                squeeze=False,
            )
            for row, domain in enumerate(["analytic", "earndb"]):
                # Fixed first case, selected by identity before looking at errors.
                i = int(
                    rows.index[
                        (rows.domain == domain)
                        & (rows.noise == "white")
                        & (rows.sigma == 0.5)
                    ][0]
                )
                for col, step in enumerate(page_steps):
                    ax = axes[row, col]
                    ax.plot(
                        time_ms,
                        predictions["truth"][i] * 0.1,
                        color="black",
                        lw=2,
                        label="Known simulated target"
                        if domain == "analytic"
                        else "Normalized measured reference",
                    )
                    ax.plot(
                        time_ms,
                        predictions["ridge"][i] * 0.1,
                        color="gray",
                        ls=":",
                        label="Ridge",
                    )
                    for pipe in NAMES:
                        ax.plot(
                            time_ms,
                            predictions[f"{pipe}_{arch}_{step}"][i] * 0.1,
                            color=COLORS[pipe],
                            label=NAMES[pipe],
                        )
                    ax.set(
                        title=f"{domain} • {step:,} updates",
                        xlabel="Lag (ms)",
                        ylabel="Synthetic amplitude (µV)",
                    )
                    ax.axvline(0, color="gray", lw=0.5)
            axes[0, 0].legend(fontsize=7)
            fig.suptitle(
                f"Paired waveform comparisons — {arch.upper()} — page {page}",
                fontsize=14,
            )
            finish(
                fig,
                output,
                f"04_waveforms_{arch}_{page}",
                note
                + f" Fixed first white-noise/low-noise case per domain; training seed {seeds[0]}.",
            )

    final = data[data.step == max(steps)]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    conditions = [("white", 0.5), ("white", 2.0), ("colored", 0.5), ("colored", 2.0)]
    for ax, domain in zip(axes, ["analytic", "earndb"]):
        for (pipe, arch), g in final[final.domain == domain].groupby(
            ["pipeline", "architecture"]
        ):
            m = g.groupby(["noise", "sigma"]).rmse_uv.mean()
            ax.plot(
                range(4),
                [m[c] for c in conditions],
                STYLES[arch],
                color=COLORS[pipe],
                label=f"{NAMES[pipe]} / {arch.upper()}",
            )
        ax.set(
            title=f"{domain} test targets",
            xticks=range(4),
            xticklabels=[
                "White\n0.05 µV",
                "White\n0.20 µV",
                "Colored\n0.05 µV",
                "Colored\n0.20 µV",
            ],
            ylabel="RMSE (synthetic µV)",
            xlabel="Added EEG noise: type and RMS",
        )
    axes[0].legend(fontsize=7)
    fig.suptitle(f"Noise robustness after {max(steps):,} updates", fontsize=14)
    finish(
        fig,
        output,
        "05_noise_robustness",
        note
        + " Colored noise deliberately tests mismatch to the white-noise DPS likelihood.",
    )

    summary = (
        data.groupby(["pipeline", "architecture", "step", "domain"])
        .agg(
            rmse_uv=("rmse_uv", "mean"),
            correlation=("correlation", "mean"),
            peak_latency_error_ms=("peak_latency_error_ms", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(root / "summary.csv", index=False)
    data[data.domain == "earndb"].groupby(
        ["subject", "pipeline", "architecture", "step"]
    ).rmse_uv.mean().to_csv(root / "per_subject.csv")
    report = [
        "# ABR training comparison",
        "",
        "## Design",
        "",
        f"Three pipelines × {len(architectures)} architectures × {len(seeds)} seeds; checkpoints: {steps} updates.",
        f"Each pipeline has {config['count']:,} generated training waveforms. Batch size: {config['batch_size']}; seeds: {seeds}. Models use the same v objective, optimizer, batch size and EMA rate; parameter counts differ and are recorded in each run's settings.json.",
        "Subject split before any fitting or augmentation: N1–N5 train, N6 validation, N7–N8 test. Both repetitions, every frequency and level for a subject stay together.",
        "Guidance is selected by pooled mean RMSE on fixed analytic + EARNDB validation observations, separately for each checkpoint/model/seed. Test targets, EEG and sampler seeds are identical across comparisons. No test-based tuning.",
        "Analytic data use separate generator seeds. EARNDB data use averaged tone-burst references; every waveform is resampled 48→10 kHz and normalized to a declared synthetic 0.04 µV RMS. Absolute EARNDB calibration is unresolved, so these results measure synthetic shape recovery, not calibrated clinical performance.",
        "EARNDB templates use training-subject waveforms with gain and ±0.3 ms shift augmentation. Fitted PCA learns 16 modes from training subjects only. High-level SPL≥70 dB subset is fixed in advance. The final 5 ms of each periodic epoch supplies the plotted negative lags.",
        "All observations simulate 0.2 seconds of irregular pulses; white/colored noise at synthetic 0.05/0.20 µV RMS. The original 200-step DPS sampler is used. Colored noise is a deliberate likelihood mismatch.",
        "",
        "## Results",
        "",
        "| Pipeline | Model | Updates | Test domain | RMSE (synthetic µV) | Correlation |",
        "|---|---|---:|---|---:|---:|",
    ]
    for r in summary.itertuples():
        report.append(
            f"| {NAMES[r.pipeline]} | {r.architecture.upper()} | {r.step:,} | {r.domain} | {r.rmse_uv:.4f} | {r.correlation:.3f} |"
        )
    report += [
        "",
        "## Presentation guide",
        "",
        "1. 01_training_duration: compare short, original-length and longer training. Shading is training-seed range, not a confidence interval.",
        "2. 02_learning_curves: show whether validation improvement saturates and how training loss differs.",
        "3. 03_guidance_*: show the complete validation guidance sweep at each checkpoint.",
        "4. 04_waveforms_*: show the same prespecified examples across checkpoints and data pipelines.",
        "5. 05_noise_robustness: compare both noise levels and white/colored noise.",
        "",
        "## Limits",
        "",
        "Only two test subjects and two training seeds by default: descriptive evidence, not population-level significance. Correlation is undefined for null waveforms and excluded from its mean. Peak error is the location of the maximum in 3–10 ms, not a validated wave-V detector. EARNDB labels retain measurement noise. Tone-burst morphology and synthetic pulse convolution do not validate recovery of real speech ABRs. The small CNN and MLP are not parameter- or wall-time-matched. Baseline ridge is fixed (0.01 × mean Gram diagonal), not an optimized classical estimator.",
        "",
        "## Source and citation",
        "",
        "PhysioNet EARNDB v1.0.0, https://physionet.org/content/earndb/1.0.0/ (ODC Attribution license). Cite Silva & Epstein (2010), Estimating loudness growth from tone-burst evoked responses, JASA 127:3629–3642, doi:10.1121/1.3397457. File hashes and split membership are saved in data/manifest.json.",
    ]
    (root / "REPORT.md").write_text("\n".join(report) + "\n")
    with PdfPages(root / "comparison_slides.pdf") as deck:
        fig = plt.figure(figsize=(16, 9))
        fig.patch.set_facecolor("white")
        fig.text(
            0.07, 0.79, "ABR reconstruction experiments", fontsize=30, weight="bold"
        )
        fig.text(
            0.07,
            0.66,
            "Training duration • DPS guidance • synthetic-data pipeline • "
            + " / ".join(a.upper() for a in architectures),
            fontsize=19,
        )
        design = (
            f"{config['count']:,} training waveforms per pipeline; {len(seeds)} training seeds\n"
            f"Checkpoints: {', '.join(f'{s:,}' for s in steps)} updates\n"
            "Subject split: N1–N5 training / N6 validation / N7–N8 testing\n"
            "Guidance chosen on validation; fixed paired test observations\n"
            "EARNDB morphology uses declared synthetic amplitudes; calibration unresolved"
        )
        fig.text(0.07, 0.35, design, fontsize=19, linespacing=1.7)
        fig.text(
            0.07,
            0.07,
            "EARNDB v1.0.0 — Silva & Epstein (2010), JASA. Synthetic benchmark; not clinical validation.",
            fontsize=12,
        )
        deck.savefig(fig)
        plt.close(fig)
        for path in sorted(output.glob("*.png")):
            fig = plt.figure(figsize=(16, 9))
            ax = fig.add_axes([0, 0, 1, 1])
            ax.imshow(plt.imread(path))
            ax.axis("off")
            deck.savefig(fig)
            plt.close(fig)
        fig = plt.figure(figsize=(16, 9))
        fig.patch.set_facecolor("white")
        fig.text(
            0.07, 0.85, "How to interpret these comparisons", fontsize=28, weight="bold"
        )
        lines = [
            "Lower held-out RMSE indicates better recovery in this simulation.",
            "Report each test domain separately; inspect both held-out participants.",
            "Longer training can overfit: compare all prespecified checkpoints.",
            "Guidance is selected within the tested grid, not a proven global optimum.",
            "The CNN is much smaller; this comparison is not parameter- or time-matched.",
            "Two test subjects and two seeds do not establish population significance.",
            "EARNDB averages retain noise; synthetic shape recovery is not real-EEG validation.",
        ]
        for i, line in enumerate(lines):
            fig.text(0.07, 0.72 - i * 0.085, "• " + line, fontsize=18)
        deck.savefig(fig)
        plt.close(fig)
    print(f"Saved presentation PNG/PDF figures and report to {root}", flush=True)
