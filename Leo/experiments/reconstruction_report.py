"""Consistent publication-style figures and one self-contained HTML report."""

import base64
import csv
import html
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from study import save_rows

COLORS = {"analytic": "#2463A0", "earndb_templates": "#C46835", "ridge": "#64748B"}
LABELS = {"analytic": "Analytic prior", "earndb_templates": "EARNDB template prior"}


def read(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def produce(output):
    output = Path(output)
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "axes.labelcolor": "#334155",
            "axes.edgecolor": "#CBD5E1",
            "text.color": "#0F172A",
            "xtick.color": "#475569",
            "ytick.color": "#475569",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.grid": True,
            "grid.color": "#E2E8F0",
            "grid.linewidth": 0.6,
        }
    )
    models = sorted(p for p in output.glob("*_cnn_*") if (p / "test.csv").exists())
    blocks = []

    def save(fig, name, title, caption):
        fig.set_layout_engine("constrained")
        for ext in ("png", "pdf", "svg"):
            fig.savefig(figures / f"{name}.{ext}", dpi=220)
        plt.close(fig)
        payload = base64.b64encode((figures / f"{name}.png").read_bytes()).decode()
        blocks.append(
            f'<section><h2>{title}</h2><img src="data:image/png;base64,{payload}" alt="{html.escape(title)}"><p>{caption}</p></section>'
        )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for m in models:
        pipeline = m.name.rsplit("_cnn_", 1)[0]
        rows = read(output / "models" / m.name / "training.csv")
        x = [int(r["update"]) for r in rows]
        for ax, key in zip(axes, ("train_v_mse", "validation_v_mse")):
            ax.plot(
                x,
                [float(r[key]) for r in rows],
                color=COLORS[pipeline],
                alpha=0.8,
                marker="o",
                ms=3,
                label=f"{LABELS[pipeline]}, seed {m.name.rsplit('_', 1)[1]}",
            )
            ax.set_xlabel("Optimizer updates")
            ax.set_ylabel("v-prediction MSE (model units²)")
    axes[0].set_title("Training objective")
    axes[1].set_title("Held-out denoising objective")
    axes[1].legend(fontsize=8, frameon=False)
    save(
        fig,
        "01_training",
        "1 · Training convergence",
        "Each line is an independent training seed. This objective measures v prediction; reconstruction is evaluated separately below.",
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, pipeline in zip(axes, COLORS):
        for m in [p for p in models if p.name.startswith(pipeline + "_cnn")]:
            rows = read(m / "validation.csv")
            checkpoints = sorted(set(int(r["checkpoint"]) for r in rows))
            scores = [
                min(float(r["rmse_uv"]) for r in rows if int(r["checkpoint"]) == c)
                for c in checkpoints
            ]
            ax.plot(
                checkpoints,
                scores,
                marker="o",
                color=COLORS[pipeline],
                alpha=0.8,
                linestyle="-" if m.name.endswith("_17") else "--",
                label="Seed " + m.name.rsplit("_", 1)[1],
            )
        ax.set(
            title=LABELS[pipeline],
            xlabel="Training checkpoint (updates)",
            ylabel="Validation ABR RMSE (µV)",
        )
        ax.legend(frameon=False)
    save(
        fig,
        "02_checkpoint",
        "2 · Reconstruction-based checkpoint selection",
        "At each checkpoint, guidance is chosen by validation RMSE. The minimum over checkpoints determines the selected model. No test score enters this selection.",
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, pipeline in zip(axes, COLORS):
        for m in [p for p in models if p.name.startswith(pipeline + "_cnn")]:
            chosen = json.loads((m / "selection.json").read_text())
            rows = [
                r
                for r in read(m / "validation.csv")
                if int(r["checkpoint"]) == chosen["checkpoint"]
            ]
            ax.plot(
                [float(r["guidance"]) for r in rows],
                [float(r["rmse_uv"]) for r in rows],
                marker="o",
                color=COLORS[pipeline],
                alpha=0.8,
                linestyle="-" if m.name.endswith("_17") else "--",
                label=f"Seed {m.name.rsplit('_', 1)[1]}, update {chosen['checkpoint']}",
            )
        ax.set(
            title=LABELS[pipeline],
            xlabel="Guidance strength",
            ylabel="Validation ABR RMSE (µV)",
        )
        ax.legend(frameon=False, fontsize=8)
    save(
        fig,
        "03_guidance",
        "3 · Guidance strength",
        "Validation reconstruction error at the selected checkpoint. A minimum at the edge of the tested grid means the optimum has not been bracketed.",
    )

    baseline = read(output / "ridge_test.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    summaries = []
    for ax, domain in zip(axes, ("analytic", "earndb")):
        categories = [
            ("white", 0.5),
            ("white", 2.0),
            ("colored", 0.5),
            ("colored", 2.0),
        ]
        for pipeline in ("analytic", "earndb_templates", "ridge"):
            series = []
            if pipeline == "ridge":
                groups = [baseline]
            else:
                groups = [
                    read(m / "test.csv")
                    for m in models
                    if m.name.startswith(pipeline + "_cnn")
                ]
            for noise, sigma in categories:
                vals = [
                    np.mean(
                        [
                            float(r["rmse_uv"])
                            for r in group
                            if r["domain"] == domain
                            and r["noise"] == noise
                            and float(r["sigma"]) == sigma
                        ]
                    )
                    for group in groups
                ]
                series.append(vals)
            means = [np.mean(v) for v in series]
            ax.plot(
                range(4),
                means,
                marker="o",
                color=COLORS[pipeline],
                label=LABELS.get(pipeline, "Tuned ridge"),
            )
            if len(groups) > 1:
                ax.fill_between(
                    range(4),
                    [min(v) for v in series],
                    [max(v) for v in series],
                    color=COLORS[pipeline],
                    alpha=0.12,
                )
        ax.set_xticks(
            range(4),
            [
                "White\n0.05 µV",
                "White\n0.20 µV",
                "Colored\n0.05 µV",
                "Colored\n0.20 µV",
            ],
        )
        ax.set(
            title="Analytic references"
            if domain == "analytic"
            else "Held-out EARNDB references",
            ylabel="Test ABR RMSE (µV)",
            xlabel="Measurement noise type and SD",
        )
    axes[1].legend(frameon=False, fontsize=8)
    save(
        fig,
        "04_heldout",
        "4 · Held-out reconstruction and noise robustness",
        "Models use validation-selected checkpoints and guidance. Curves average the training and sampling seeds; shading is the range of training-seed means, not a confidence interval. Ridge regularization is selected on validation cases.",
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for pipeline in ("analytic", "earndb_templates"):
        all_traces = []
        for m in [p for p in models if p.name.startswith(pipeline + "_cnn")]:
            rows = read(m / "diffusion_trace.csv")
            for seed in (42, 43, 44):
                all_traces.append([r for r in rows if int(r["sampling_seed"]) == seed])
        for ax, key in zip(axes, ("clean_estimate_rmse_uv", "normalized_eeg_residual")):
            y = np.array([[float(r[key]) for r in rows] for rows in all_traces])
            ax.plot(
                range(1, 201), y.mean(0), color=COLORS[pipeline], label=LABELS[pipeline]
            )
            ax.fill_between(
                range(1, 201), y.min(0), y.max(0), color=COLORS[pipeline], alpha=0.12
            )
            ax.set_xlabel("Reverse-diffusion iteration")
    axes[0].set(
        title="Estimated clean ABR versus reference",
        ylabel="Diagnostic clean-estimate RMSE (µV)",
    )
    axes[1].set(
        title="Predicted EEG versus measurement",
        ylabel="Noise-normalized EEG residual RMS",
    )
    axes[1].legend(frameon=False, fontsize=8)
    save(
        fig,
        "05_diffusion",
        "5 · Inside the reconstruction loop",
        "Reference ABRs are used only to record diagnostic error; they never enter sampling gradients. The clean estimate can fluctuate during stochastic sampling. These curves are not a guarantee of monotonic convergence or calibrated uncertainty.",
    )

    refs = np.load(output / "references.npz")
    testrows = read(output / "test_observations.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    times = (np.arange(256) - 50) / 10
    for ax, domain in zip(axes, ("analytic", "earndb")):
        idx = next(
            i
            for i, r in enumerate(testrows)
            if r["domain"] == domain
            and r["noise"] == "colored"
            and float(r["sigma"]) == 2.0
        )
        ax.plot(
            times,
            refs["truth"][idx] * 0.1,
            color="#0F172A",
            lw=2,
            label="Reference ABR",
        )
        ax.plot(
            times,
            refs["ridge"][idx] * 0.1,
            color=COLORS["ridge"],
            linestyle="--",
            label="Tuned ridge",
        )
        for pipeline in ("analytic", "earndb_templates"):
            pred = (
                np.concatenate(
                    [
                        np.load(m / "predictions.npz")["predictions"][:, idx, :]
                        for m in models
                        if m.name.startswith(pipeline + "_cnn")
                    ]
                )
                * 0.1
            )
            ax.plot(times, pred.mean(0), color=COLORS[pipeline], label=LABELS[pipeline])
            ax.fill_between(
                times, pred.min(0), pred.max(0), color=COLORS[pipeline], alpha=0.12
            )
        ax.set(
            title=f"{domain.upper()} · {testrows[idx]['record']}",
            xlabel="Time relative to stimulus (ms)",
            ylabel="ABR amplitude (µV)",
        )
    axes[1].legend(frameon=False, fontsize=8)
    save(
        fig,
        "06_waveforms",
        "6 · Reconstructed waveform examples",
        "Examples are the first test waveform in each domain under high colored noise, selected by fixed index rather than quality. Colored lines are the mean across training and sampling seeds; shading spans all individual reconstructions and is not a calibrated credible interval. Quantitative RMSE tables average individual reconstruction errors rather than scoring this ensemble mean.",
    )

    fig, ax = plt.subplots(figsize=(11, 4))
    ridge_validation = read(output / "ridge_validation.csv")
    ax.semilogx(
        [float(r["coefficient"]) for r in ridge_validation],
        [float(r["rmse_uv"]) for r in ridge_validation],
        color=COLORS["ridge"],
        marker="o",
    )
    ax.set(
        xlabel="Relative ridge penalty",
        ylabel="Validation ABR RMSE (µV)",
        title="Regularization chosen before test evaluation",
    )
    save(
        fig,
        "07_ridge",
        "7 · Baseline regularization",
        "The ridge coefficient is selected by the minimum on this validation grid. The test set is evaluated once using that selected coefficient.",
    )

    for m in models:
        rows = read(m / "test.csv")
        selected = json.loads((m / "selection.json").read_text())
        summaries.append(
            {
                "model": m.name,
                "checkpoint": selected["checkpoint"],
                "guidance": selected["guidance"],
                "validation_reconstruction_rmse_uv": selected["rmse_uv"],
                "test_reconstruction_rmse_uv": float(
                    np.mean([float(r["rmse_uv"]) for r in rows])
                ),
            }
        )
    save_rows(output / "summary.csv", summaries)
    ridge_score = float(np.mean([float(r["rmse_uv"]) for r in baseline]))
    bodyrows = "".join(
        f"<tr><td>{s['model']}</td><td>{s['checkpoint']}</td><td>{s['guidance']}</td><td>{s['validation_reconstruction_rmse_uv']:.5f}</td><td>{s['test_reconstruction_rmse_uv']:.5f}</td></tr>"
        for s in summaries
    )
    best = min(summaries, key=lambda s: s["test_reconstruction_rmse_uv"])
    findings = f"Observed test RMSE ranges from {min(s['test_reconstruction_rmse_uv'] for s in summaries):.5f} to {max(s['test_reconstruction_rmse_uv'] for s in summaries):.5f} µV. Tuned ridge reaches {ridge_score:.5f} µV (relative penalty {float(baseline[0]['coefficient']):g}, selected on validation). The lowest observed test value belongs to {best['model']}; this is descriptive and was not used to select or tune a model."
    key_findings = []
    for pipeline in ("analytic", "earndb_templates"):
        scores = [
            r["test_reconstruction_rmse_uv"]
            for r in summaries
            if r["model"].startswith(pipeline + "_cnn")
        ]
        mean = float(np.mean(scores))
        relative = (mean / ridge_score - 1) * 100
        key_findings.append(
            f"{LABELS[pipeline]}: mean held-out reconstruction RMSE {mean:.5f} µV, {abs(relative):.1f}% {'higher' if relative > 0 else 'lower'} than the tuned ridge baseline in this pilot."
        )
    for domain in ("analytic", "earndb"):
        domain_scores = {}
        for pipeline in ("analytic", "earndb_templates"):
            values = [
                float(r["rmse_uv"])
                for m in models
                if m.name.startswith(pipeline + "_cnn")
                for r in read(m / "test.csv")
                if r["domain"] == domain
            ]
            domain_scores[pipeline] = float(np.mean(values))
        key_findings.append(
            f"On {domain.upper()} reference targets: analytic prior {domain_scores['analytic']:.5f} µV; template prior {domain_scores['earndb_templates']:.5f} µV. The domain-specific comparison is more informative than the pooled score alone."
        )
    boundary = sum(r["guidance"] == 3.0 for r in summaries)
    key_findings.append(
        f"{boundary} of {len(summaries)} configurations selected the strongest tested guidance (3.0). For these configurations, the guidance optimum is not bracketed by this grid."
    )
    subject_rows = []
    for label, rows in [(m.name, read(m / "test.csv")) for m in models] + [
        ("ridge", baseline)
    ]:
        for subject in (7, 8):
            values = [float(r["rmse_uv"]) for r in rows if int(r["subject"]) == subject]
            subject_rows.append(
                dict(model=label, subject=subject, rmse_uv=float(np.mean(values)))
            )
    save_rows(output / "per_subject.csv", subject_rows)
    subject_table = "".join(
        f"<tr><td>{r['model']}</td><td>N{r['subject']}</td><td>{r['rmse_uv']:.5f}</td></tr>"
        for r in subject_rows
    )
    finding_items = "".join(f"<li>{html.escape(x)}</li>" for x in key_findings)
    config = json.loads((output / "config.json").read_text())
    environment_path = output / "environment.json"
    environment = (
        json.loads(environment_path.read_text()) if environment_path.exists() else {}
    )
    report = f"""<!doctype html><html lang="en"><meta charset="utf-8"><title>ABR reconstruction experiments</title>
<style>body{{max-width:1120px;margin:56px auto;padding:0 28px;font:16px/1.65 system-ui;color:#243247;background:#f8fafc}}h1{{font-size:38px;line-height:1.15;color:#0f172a}}h2{{font-size:24px;color:#0f172a}}section,.intro{{background:white;padding:28px;margin:24px 0;border:1px solid #e2e8f0;border-radius:12px}}img{{width:100%;height:auto}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:10px;text-align:left;border-bottom:1px solid #e2e8f0}}pre{{white-space:pre-wrap;font-size:13px}}a{{color:#2463a0}}.eyebrow{{letter-spacing:.15em;text-transform:uppercase;color:#64748b;font-size:12px}}</style>
<div class="eyebrow">CS163 · Caltech · Reconstruction study</div><h1>Recovering ABRs from simulated noisy EEG</h1><p>Evaluated configurations: <strong>{len(models)} / 4</strong> · Two priors, two training seeds</p>
<div class="intro"><h2>Question and design</h2><p>Can a trained diffusion prior recover a reference ABR from a known pulse train and simulated noisy recording? The primary outcome is mean per-observation waveform reconstruction RMSE, not v-prediction error. For each reconstruction: RMSE = sqrt(mean((estimated ABR − reference ABR)²)) × 0.1 µV per model unit.</p>
<p>New CNN models are trained independently on analytic waveforms and EARNDB template augmentations, with two training seeds. Each uses 60,000 available training waveforms, batch size 32, AdamW, EMA, and 1,000 optimizer updates. Checkpoints at 200, 400, 800 and 1,000 are assessed with guidance 0, 0.3, 1 and 3 on validation cases. The checkpoint/guidance pair with minimum validation RMSE is frozen before test evaluation. Three sampling seeds assess stochastic sensitivity. A six-value ridge grid is tuned on the same validation observations.</p>
<p>The architecture is held fixed: a 7,745-parameter temporal CNN with 16 channels and six residual convolution blocks at dilations 1, 2, 4, 8, 16 and 32. Training predicts v using mean squared error under a 200-step cosine diffusion schedule. AdamW uses learning rate 2×10⁻⁴ and weight decay 10⁻⁴, gradients are clipped at norm 1, and EMA uses rate 0.005. Examples are sampled with replacement: 1,000 updates at batch size 32 means 32,000 waveform presentations, not 1,000 epochs.</p>
<p>Analytic ABRs vary peak timing, width and amplitude, include a trough and optional early component, and are filtered at 80–1,500 Hz; 10% are null waveforms. EARNDB ABR averages are restricted to SPL≥70 dB and ABR channels identified by nV headers. Digital waveforms have DC removed using the epoch tail, are resampled from 48 to 10 kHz, and are normalized to synthetic 0.04 µV RMS. The periodic epoch tail supplies the negative-time portion of the window; it is not measured pre-onset EEG. Template augmentation uses log-normal gain (log SD 0.2) and shifts from −0.3 to +0.3 ms with zero padding. No test subject contributes to these training templates.</p>
<p>EARNDB subjects N1–N5 train the prior, N6 provides validation references, and N7–N8 provide test references. Four reference targets per domain are used in validation and eight per domain in testing (four references each from N7 and N8); each has white/colored noise at synthetic 0.05/0.20 µV standard deviation, giving 32 validation observations and 64 test observations. Analytic targets use independent split seeds. All methods share paired recordings. Observation duration is 0.2 seconds. RMSE is computed over 256 waveform samples, then averaged over observations and sampling seeds. Repeated noisy observations and seeds are not independent subjects.</p>
<p>Forward model: EEG = stimulus matrix × reference ABR + noise. Diffusion sampling uses 200 reverse steps and an EEG-residual gradient. The clean reference is excluded from that gradient. Trace RMSE measures the estimated clean ABR at each step; final test RMSE measures the returned reconstruction.</p>
<p><a href="https://forge.coreweave.com/wandb/abr-deconv/abr-deconv-reconstruction">CoreWeave Forge: training, validation reconstruction and sampling traces</a></p></div>
<section><h2>Results</h2><p>{findings}</p><table><tr><th>Training configuration</th><th>Selected update</th><th>Guidance</th><th>Validation RMSE (µV)</th><th>Test RMSE (µV)</th></tr>{bodyrows}</table><h2>Key findings</h2><ul>{finding_items}</ul><h2>Individual held-out participants</h2><table><tr><th>Configuration</th><th>Subject</th><th>ABR RMSE (µV)</th></tr>{subject_table}</table></section>
{"".join(blocks)}
<section><h2>Interpretation and limits</h2><p>This is a small reconstruction-focused pilot. Only one EARNDB participant supplies validation targets and two supply a balanced set of test targets; the number of waveforms does not establish population-level certainty. Template references are measured averages with residual noise and are normalized to synthetic 0.04 µV RMS. Their physical calibration is unresolved. Results measure recovery of those references in simulated EEG, not performance on real continuous EEG. No probability intervals or significance claims are made.</p><p>The early training budget tests checkpoint selection, not maximum attainable performance. Guidance-grid boundary minima should motivate a separately specified validation sweep. A lower denoising loss need not imply lower reconstruction RMSE, and a lower EEG residual need not imply a more accurate ABR. The high-noise waveform examples illustrate that lower average RMSE can coexist with attenuated response peaks. RMSE alone does not establish faithful peak amplitude or timing; the raw test tables also provide waveform correlation and a descriptive 3–10 ms peak-timing error, which is not a validated clinical wave-V annotation. Future improvements should be selected on validation data; repeated test inspection can bias subsequent comparisons.</p></section>
<section><h2>Reproduction and provenance</h2><p>EARNDB v1.0.0: <a href="https://physionet.org/content/earndb/1.0.0/">PhysioNet dataset</a>, ODC Attribution license. Silva &amp; Epstein (2010), Estimating loudness growth from tone-burst evoked responses, JASA 127:3629–3642, <a href="https://doi.org/10.1121/1.3397457">doi:10.1121/1.3397457</a>. Downloaded record hashes are preserved in the data manifest.</p><p>Run <code>.venv/bin/python experiments/reconstruction_suite.py</code> from Leo. The complete settings are below; source hashes are saved in source_sha256.json. The matching execution sources are preserved in execution_source.zip; the cleaned source tree additionally includes formatting and reporting updates. Raw tables, predictions and checkpoint files remain in the run folder. Separate PNG, PDF and SVG figures are exported under figures/. This report embeds every plot and is readable as a single standalone file.</p><pre>{html.escape(json.dumps(config, indent=2))}</pre><h3>Execution environment</h3><pre>{html.escape(json.dumps(environment, indent=2))}</pre></section></html>"""
    (output / "REPORT.html").write_text(report)


def upload_report(output):
    """Publish figures and the self-contained report to the existing Forge project."""
    import wandb

    output = Path(output)
    with wandb.init(
        entity="abr-deconv",
        project="abr-deconv-reconstruction",
        group=output.name,
        name="reconstruction-report",
        job_type="report",
    ) as run:
        run.log(
            {
                f"figures/{p.stem}": wandb.Image(str(p))
                for p in sorted((output / "figures").glob("*.png"))
            }
        )
        artifact = wandb.Artifact("reconstruction-focus-report", type="evaluation")
        for p in [
            output / "REPORT.html",
            output / "summary.csv",
            output / "per_subject.csv",
            output / "config.json",
        ]:
            artifact.add_file(str(p))
        for name in (
            "execution_source.zip",
            "source_sha256.json",
            "environment.json",
            "data_manifest.json",
        ):
            path = output / name
            if path.exists():
                artifact.add_file(str(path))
        for path in sorted((output / "models").glob("*/*.pt")):
            artifact.add_file(
                str(path),
                name="checkpoints/" + str(path.relative_to(output / "models")),
            )
        run.log_artifact(artifact)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("output", type=Path)
    produce(p.parse_args().output)
