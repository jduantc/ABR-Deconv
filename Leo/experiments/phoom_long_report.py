"""Standalone convergence report for the longer MLP comparison."""

import base64
import csv
import html
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import wandb

LABELS = {
    "mlp_phoom_baseline": "Final EMA 0.001",
    "mlp_fast_ema_final": "Final EMA 0.005",
    "mlp_selected_fixed": "Validation-selected checkpoint/EMA",
    "mlp_tuned": "Selected + tuned guidance",
    "mlp_tuned_refined": "Selected + guidance + correction",
}
COLORS = {
    "mlp_phoom_baseline": "#64748B",
    "mlp_fast_ema_final": "#2463A0",
    "mlp_tuned_refined": "#C46835",
}


def read(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def report(output):
    output = Path(output)
    config = json.loads((output / "protocol.json").read_text())
    choices = json.loads((output / "selection.json").read_text())
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "axes.grid": True,
            "grid.color": "#E2E8F0",
            "axes.edgecolor": "#CBD5E1",
            "text.color": "#0F172A",
            "figure.facecolor": "white",
        }
    )
    blocks = []

    def save(fig, name, title, caption):
        fig.set_layout_engine("constrained")
        for ext in ("png", "pdf", "svg"):
            fig.savefig(figures / f"{name}.{ext}", dpi=220, facecolor="white")
        plt.close(fig)
        encoded = base64.b64encode((figures / f"{name}.png").read_bytes()).decode()
        blocks.append(
            f'<section><h2>{title}</h2><img src="data:image/png;base64,{encoded}" alt="{title}"><p>{caption}</p></section>'
        )

    data = read(output / "mlp/training.csv")
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(
        [int(r["update"]) for r in data],
        [float(r["v_mse"]) for r in data],
        color="#2463A0",
    )
    ax.set(
        xlabel="Optimizer updates",
        ylabel="Training v-prediction MSE",
        title="Longer training of the same MLP",
    )
    save(
        fig,
        "01_training",
        "1 · Training objective",
        "Training loss is a denoising objective. Reconstruction error is assessed separately below.",
    )
    data = read(output / "mlp/checkpoint_validation.csv")
    fig, ax = plt.subplots(figsize=(11, 4))
    for rate, color in [(0.001, "#64748B"), (0.005, "#2463A0")]:
        selected = [r for r in data if float(r["ema_rate"]) == rate]
        ax.plot(
            [int(r["checkpoint"]) for r in selected],
            [float(r["rmse_uv"]) for r in selected],
            marker="o",
            color=color,
            label=f"EMA {rate}",
        )
    ax.set(
        xlabel="Training checkpoint (updates)",
        ylabel="Validation reconstruction RMSE (µV)",
        title="Does the EMA advantage persist as training continues?",
    )
    ax.legend(frameon=False)
    save(
        fig,
        "02_validation",
        "2 · Reconstruction convergence",
        "Each point averages eight DPS draws on separate validation recordings. Checkpoint and EMA selection use this grid; it is not a test curve.",
    )
    data = read(output / "mlp/guidance_validation.csv")
    fig, ax = plt.subplots(figsize=(11, 4))
    for sigma, color in [(0.05, "#2463A0"), (0.2, "#C46835"), (0.5, "#73549A")]:
        selected = [r for r in data if float(r["noise_uv"]) == sigma]
        ax.semilogx(
            [float(r["guidance"]) for r in selected],
            [float(r["rmse_uv"]) for r in selected],
            marker="o",
            color=color,
            label=f"Noise SD {sigma:g} µV",
        )
    ax.set(
        xlabel="Guidance strength",
        ylabel="Validation reconstruction RMSE (µV)",
        title="Guidance after checkpoint selection",
    )
    ax.legend(frameon=False)
    save(
        fig,
        "03_guidance",
        "3 · Guidance tuning",
        "Guidance is selected per known noise level using validation cases only. A boundary minimum does not identify a global optimum.",
    )
    rows = read(output / "test_metrics.csv")
    primary = [r for r in rows if int(r["samples"]) == 32]
    summary = []
    methods = list(dict.fromkeys(r["method"] for r in primary))
    for split in ("phoom_benchmark", "fresh_holdout"):
        for method in methods:
            for sigma in (0.05, 0.2, 0.5):
                values = [
                    float(r["rmse_uv"])
                    for r in primary
                    if r["split"] == split
                    and r["method"] == method
                    and float(r["noise_uv"]) == sigma
                ]
                summary.append(
                    dict(
                        split=split,
                        method=method,
                        noise_uv=sigma,
                        rmse_uv=float(np.mean(values)),
                    )
                )
    with (output / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, split in zip(axes, ("phoom_benchmark", "fresh_holdout")):
        for method, color in COLORS.items():
            vals = [
                next(
                    r["rmse_uv"]
                    for r in summary
                    if r["split"] == split
                    and r["method"] == method
                    and r["noise_uv"] == s
                )
                for s in (0.05, 0.2, 0.5)
            ]
            ax.plot(range(3), vals, marker="o", color=color, label=LABELS[method])
        ax.set_xticks(range(3), ["0.05", "0.20", "0.50"])
        ax.set(
            title=split.replace("_", " ").title(),
            xlabel="Measurement noise SD (µV)",
            ylabel="32-sample mean ABR RMSE (µV)",
        )
    axes[1].legend(frameon=False, fontsize=8)
    save(
        fig,
        "04_heldout",
        "4 · Final paired test comparison",
        "Final slow and fast EMA use the same trained trajectory, guidance and 32-sample averaging. The selected/refined pipeline is selected on validation data. No test score chooses a setting.",
    )
    examples = np.load(output / "examples.npz")
    times = (np.arange(256) - 50) / 10
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, split in zip(axes, ("phoom_benchmark", "fresh_holdout")):
        truth = np.load(output / f"{split}_truth.npz")["truth"]
        idx = 24
        ax.plot(times, truth[idx], color="#0F172A", lw=2, label="Reference ABR")
        for method, color in COLORS.items():
            ax.plot(
                times,
                examples[split + "_" + method][idx],
                color=color,
                label=LABELS[method],
            )
        ax.set(
            title=split.replace("_", " ").title() + " · high noise",
            xlabel="Time relative to stimulus (ms)",
            ylabel="ABR amplitude (µV)",
        )
    axes[1].legend(frameon=False, fontsize=8)
    save(
        fig,
        "05_waveforms",
        "5 · Fixed-index waveform examples",
        "These are the first reference waveform at 0.50 µV noise SD in each set, chosen by index rather than quality. Waveform fidelity should be inspected alongside RMSE.",
    )
    table = []
    findings = []
    for split in ("phoom_benchmark", "fresh_holdout"):
        baseline = np.mean(
            [
                r["rmse_uv"]
                for r in summary
                if r["split"] == split and r["method"] == "mlp_phoom_baseline"
            ]
        )
        for method in methods:
            values = [
                r["rmse_uv"]
                for r in summary
                if r["split"] == split and r["method"] == method
            ]
            mean = float(np.mean(values))
            gain = 100 * (1 - mean / baseline)
            table.append(
                f"<tr><td>{split}</td><td>{LABELS[method]}</td><td>{mean:.6f}</td><td>{gain:+.1f}%</td></tr>"
            )
            if method in ("mlp_fast_ema_final", "mlp_tuned_refined"):
                findings.append(
                    f"{split}: {LABELS[method]} RMSE {mean:.6f} µV versus final slow-EMA baseline {baseline:.6f} µV ({gain:+.1f}% reduction)."
                )
    project = config["project"]
    doc = f"""<!doctype html><html lang="en"><meta charset="utf-8"><title>Longer MLP reconstruction training</title><style>body{{max-width:1150px;margin:48px auto;padding:0 26px;background:#f8fafc;color:#243247;font:16px/1.65 system-ui}}h1{{font-size:38px;line-height:1.15}}section{{background:white;padding:26px;border:1px solid #e2e8f0;border-radius:12px;margin:24px 0}}img{{width:100%}}table{{width:100%;border-collapse:collapse;font-size:13px}}td,th{{padding:9px;text-align:left;border-bottom:1px solid #e2e8f0}}pre{{white-space:pre-wrap;font-size:12px}}</style>
<h1>Does the reconstruction improvement persist with longer training?</h1><section><h2>Methodology</h2><p>The same 544,768-parameter MLP is trained from seed 0 for {config["steps"]:,} optimizer updates with batch size 256 on 20,000 analytic waveforms. Two EMA shadows, 0.001 and 0.005, follow the same parameter trajectory. Starting from the same seed reproduces the short-budget recipe; prior checkpoints lacked raw optimizer state, so this is a fresh full trajectory, not a claimed optimizer continuation.</p><p>AdamW uses learning rate 2×10⁻⁴, a 500-update warmup, zero weight decay and gradient clipping at norm 1. Waveforms use Phoom’s generator and training-standard-deviation normalization. Reconstruction uses the same 200-step DPS algorithm, with numerically verified Gram-statistic acceleration. The primary outcome is mean per-waveform ABR reconstruction RMSE in µV after averaging 32 stochastic draws, distinct from training v-MSE.</p><p>Checkpoints are assessed on six new validation reference waveforms with eight-draw means. Validation selects checkpoint/EMA, guidance per noise level and covariance-regularized measurement correction. The original benchmark is a previously observed reference, not a new test. The fresh holdout uses new waveform/noise seeds {config["fresh_waveform_seed"]} and {config["fresh_noise_seed"]}, disjoint from the earlier pilot. Each test set has 12 nonzero reference ABRs and half-second recordings with regular pulses every 10 ms and white noise at SD 0.05, 0.20 and 0.50 µV. Settings are frozen before scoring. All methods share measurements and Monte Carlo seeds.</p><p>The correction uses covariance estimated only from training waveforms and includes strength zero as a candidate. It is an empirical reuse of the measurements, not an exact posterior update. There is one training seed; this synthetic study does not establish performance on real EEG. An EMA benefit in a short run need not persist after convergence. {config["steps"]:,} updates are longer than the pilot but do not establish equivalence to Phoom’s million-update models.</p><p><a href="https://forge.coreweave.com/wandb/abr-deconv/{project}">CoreWeave Forge project</a></p></section><section><h2>Results</h2><ul>{"".join("<li>" + html.escape(x) + "</li>" for x in findings)}</ul><table><tr><th>Split</th><th>Method</th><th>Mean reconstruction RMSE (µV)</th><th>Reduction vs final slow EMA</th></tr>{"".join(table)}</table></section>{"".join(blocks)}<section><h2>Frozen selections</h2><pre>{html.escape(json.dumps(choices, indent=2))}</pre><h2>Protocol</h2><pre>{html.escape(json.dumps(config, indent=2))}</pre><p>Raw tables, samples, checkpoints, source hashes and optimizer/RNG resume state are retained beside this standalone report. All figures are exported separately as PNG, PDF and SVG.</p></section></html>"""
    (output / "REPORT.html").write_text(doc)
    with wandb.init(
        entity="abr-deconv",
        project=project,
        group=output.name,
        name="long-training held-out results",
        job_type="evaluation",
    ) as run:
        columns = list(summary[0])
        run.log(
            {
                "reconstruction_results": wandb.Table(
                    columns=columns, data=[[r[k] for k in columns] for r in summary]
                )
            }
        )
        run.log(
            {
                f"figures/{p.stem}": wandb.Image(str(p))
                for p in sorted(figures.glob("*.png"))
            }
        )
        for split in ("phoom_benchmark", "fresh_holdout"):
            for method in methods:
                run.summary[f"{split}/{method}/mean_rmse_uv"] = float(
                    np.mean(
                        [
                            r["rmse_uv"]
                            for r in summary
                            if r["split"] == split and r["method"] == method
                        ]
                    )
                )
        artifact = wandb.Artifact("phoom-long-training-report", type="evaluation")
        for p in output.glob("*"):
            if p.is_file() and p.suffix in (".html", ".csv", ".json"):
                artifact.add_file(str(p))
        for p in output.glob("mlp/step_*.pt"):
            artifact.add_file(str(p), name="checkpoints/" + p.name)
        run.log_artifact(artifact)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    report(parser.parse_args().output)
