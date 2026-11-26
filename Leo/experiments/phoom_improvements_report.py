"""Render the matched-budget benchmark and publish all results to Forge."""

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

PROJECT = "abr-deconv-phoom-improvements"
COLORS = {
    "mlp_phoom_baseline": "#64748B",
    "mlp_tuned_refined": "#2463A0",
    "cnn_tuned_refined": "#C46835",
}
LABELS = {
    "mlp_phoom_baseline": "Phoom-style MLP baseline",
    "mlp_tuned_refined": "MLP + validation settings/refinement",
    "cnn_tuned_refined": "CNN + validation settings/refinement",
}


def read(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def report(output):
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
            "axes.grid": True,
            "grid.color": "#E2E8F0",
            "grid.linewidth": 0.6,
            "axes.edgecolor": "#CBD5E1",
            "text.color": "#0F172A",
            "axes.labelcolor": "#334155",
            "figure.facecolor": "white",
        }
    )
    rows = read(output / "test_metrics.csv")
    primary = [r for r in rows if int(r["samples"]) == 32]
    config = json.loads((output / "protocol.json").read_text())
    choices = json.loads((output / "selection.json").read_text())
    methods = list(dict.fromkeys(r["method"] for r in primary))
    blocks = []
    summary = []
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

    def save(fig, name, title, caption):
        fig.set_layout_engine("constrained")
        for ext in ("png", "pdf", "svg"):
            fig.savefig(figures / f"{name}.{ext}", dpi=220, facecolor="white")
        plt.close(fig)
        encoded = base64.b64encode((figures / f"{name}.png").read_bytes()).decode()
        blocks.append(
            f'<section><h2>{title}</h2><img alt="{title}" src="data:image/png;base64,{encoded}"><p>{caption}</p></section>'
        )

    fig, ax = plt.subplots(figsize=(11, 4))
    for architecture, color in [("mlp", "#2463A0"), ("cnn", "#C46835")]:
        data = read(output / architecture / "training.csv")
        ax.plot(
            [int(r["update"]) for r in data],
            [float(r["v_mse"]) for r in data],
            color=color,
            label=architecture.upper(),
        )
    ax.set(
        xlabel="Optimizer updates",
        ylabel="Training v-prediction MSE",
        title="Same batch size, data and optimizer; different architectures",
    )
    ax.legend(frameon=False)
    save(
        fig,
        "01_training",
        "1 · Matched-budget training",
        "Both networks use 20,000 analytic waveforms, batch size 256 and the same update budget. This comparison matches example presentations, not parameter count, FLOPs or training time.",
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, architecture, color in zip(axes, ("mlp", "cnn"), ("#2463A0", "#C46835")):
        data = read(output / architecture / "checkpoint_validation.csv")
        for rate, style in [(0.001, "--"), (0.005, "-")]:
            selected = [r for r in data if float(r["ema_rate"]) == rate]
            ax.plot(
                [int(r["checkpoint"]) for r in selected],
                [float(r["rmse_uv"]) for r in selected],
                marker="o",
                linestyle=style,
                color=color,
                label=f"EMA {rate}",
            )
        ax.set(
            title=architecture.upper(),
            xlabel="Training checkpoint (updates)",
            ylabel="Validation reconstruction RMSE (µV)",
        )
        ax.legend(frameon=False)
    save(
        fig,
        "02_checkpoint_ema",
        "2 · Checkpoint and EMA selection",
        "Error is measured after averaging eight DPS samples, using the baseline guidance values. EMA and checkpoint are selected jointly on validation recordings, before guidance tuning or test scoring.",
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, architecture, color in zip(axes, ("mlp", "cnn"), ("#2463A0", "#C46835")):
        data = read(output / architecture / "guidance_validation.csv")
        for sigma, style in [(0.05, "-"), (0.2, "--"), (0.5, ":")]:
            selected = [r for r in data if float(r["noise_uv"]) == sigma]
            ax.semilogx(
                [float(r["guidance"]) for r in selected],
                [float(r["rmse_uv"]) for r in selected],
                marker="o",
                linestyle=style,
                color=color,
                label=f"Noise SD {sigma:g} µV",
            )
        ax.set(
            title=architecture.upper(),
            xlabel="Guidance strength",
            ylabel="Validation reconstruction RMSE (µV)",
        )
        ax.legend(frameon=False, fontsize=8)
    save(
        fig,
        "03_guidance",
        "3 · Guidance selected by measurement noise",
        "The guidance grid is fixed in advance and selection is separate for each known noise level. A minimum at an edge of the grid does not establish an optimum.",
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, split in zip(axes, ("phoom_benchmark", "fresh_holdout")):
        for method, color in COLORS.items():
            values = [
                next(
                    r["rmse_uv"]
                    for r in summary
                    if r["split"] == split
                    and r["method"] == method
                    and r["noise_uv"] == s
                )
                for s in (0.05, 0.2, 0.5)
            ]
            ax.plot(range(3), values, marker="o", color=color, label=LABELS[method])
        ax.set_xticks(range(3), ["0.05", "0.20", "0.50"])
        ax.set(
            title=split.replace("_", " ").title(),
            xlabel="Measurement noise SD (µV)",
            ylabel="32-sample mean ABR RMSE (µV)",
        )
    axes[1].legend(frameon=False, fontsize=7)
    save(
        fig,
        "04_test_rmse",
        "4 · Held-out reconstruction accuracy",
        "All methods use the same 32 stochastic draws per recording and the same simulated EEG. The fresh holdout uses waveform/noise seeds that were not used for validation or Phoom’s benchmark. Lines show means over 12 paired reference waveforms per noise level; no confidence interval or population-level claim is implied.",
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, split in zip(axes, ("phoom_benchmark", "fresh_holdout")):
        for method, color in [
            ("mlp_phoom_baseline", "#64748B"),
            ("mlp_tuned", "#2463A0"),
            ("cnn_tuned", "#C46835"),
        ]:
            values = [
                np.mean(
                    [
                        float(r["rmse_uv"])
                        for r in rows
                        if r["split"] == split
                        and r["method"] == method
                        and int(r["samples"]) == n
                    ]
                )
                for n in (1, 4, 8, 16, 32)
            ]
            ax.plot(
                [1, 4, 8, 16, 32],
                values,
                marker="o",
                color=color,
                label=method.replace("_", " "),
            )
        ax.set(
            title=split.replace("_", " ").title(),
            xlabel="Draws averaged before scoring",
            ylabel="Reconstruction RMSE (µV)",
        )
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 4, 8, 16, 32], [1, 4, 8, 16, 32])
    axes[1].legend(frameon=False, fontsize=8)
    save(
        fig,
        "05_averaging",
        "5 · The contribution of sample averaging",
        "Each point scores a prefix of the same 32 draws; this adds no extra sampling runs. The one-draw point and the mean-of-32 point are different estimators. Covariance refinement is omitted here to isolate averaging.",
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    split = "fresh_holdout"
    baseline = {
        (float(r["noise_uv"]), int(r["record"])): float(r["rmse_uv"])
        for r in primary
        if r["split"] == split and r["method"] == "mlp_phoom_baseline"
    }
    for ax, method in zip(axes, ("mlp_tuned_refined", "cnn_tuned_refined")):
        groups = []
        for sigma in (0.05, 0.2, 0.5):
            groups.append(
                [
                    float(r["rmse_uv"]) - baseline[(sigma, int(r["record"]))]
                    for r in primary
                    if r["split"] == split
                    and r["method"] == method
                    and float(r["noise_uv"]) == sigma
                ]
            )
        ax.boxplot(groups, tick_labels=["0.05", "0.20", "0.50"], showmeans=True)
        ax.axhline(0, color="#64748B", linestyle="--")
        ax.set(
            title=LABELS[method],
            xlabel="Noise SD (µV)",
            ylabel="Paired RMSE difference versus baseline (µV)",
        )
    save(
        fig,
        "06_paired_differences",
        "6 · Individual held-out cases",
        "Negative differences favor the candidate. Boxes describe variation across the 12 paired reference waveforms, not independent participants or a confidence interval. The same reference appears at three noise levels.",
    )

    predictions = np.load(output / "examples.npz")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    time_ms = (np.arange(256) - 50) / 10
    for ax, split in zip(axes, ("phoom_benchmark", "fresh_holdout")):
        truth = np.load(output / f"{split}_truth.npz")["truth"]
        idx = 24
        ax.plot(time_ms, truth[idx], color="#0F172A", lw=2, label="Reference ABR")
        for method, color in COLORS.items():
            ax.plot(
                time_ms,
                predictions[split + "_" + method][idx],
                color=color,
                label=LABELS[method],
            )
        ax.set(
            title=split.replace("_", " ").title() + " · high-noise example",
            xlabel="Time relative to stimulus (ms)",
            ylabel="ABR amplitude (µV)",
        )
    axes[1].legend(frameon=False, fontsize=7)
    save(
        fig,
        "07_waveforms",
        "7 · Reconstructed waveform examples",
        "The first waveform at 0.50 µV noise SD is selected by fixed index in each test set, independent of performance. All displayed reconstructions are means of 32 samples, followed by the candidate’s validation-selected correction where applicable.",
    )

    findings = []
    table = []
    for split in ("phoom_benchmark", "fresh_holdout"):
        base = np.mean(
            [
                r["rmse_uv"]
                for r in summary
                if r["split"] == split and r["method"] == "mlp_phoom_baseline"
            ]
        )
        for method in methods:
            values = [
                next(
                    r["rmse_uv"]
                    for r in summary
                    if r["split"] == split
                    and r["method"] == method
                    and r["noise_uv"] == sigma
                )
                for sigma in (0.05, 0.2, 0.5)
            ]
            mean = float(np.mean(values))
            gain = 100 * (1 - mean / base)
            table.append(
                f"<tr><td>{split}</td><td>{method}</td>"
                + "".join(f"<td>{v:.6f}</td>" for v in values)
                + f"<td>{mean:.6f}</td><td>{gain:+.1f}%</td></tr>"
            )
        for method in ("mlp_tuned_refined", "cnn_tuned_refined"):
            mean = np.mean(
                [
                    r["rmse_uv"]
                    for r in summary
                    if r["split"] == split and r["method"] == method
                ]
            )
            gain = 100 * (1 - mean / base)
            findings.append(
                f"{split}: {method} mean RMSE {mean:.6f} µV versus baseline {base:.6f} µV ({gain:+.1f}% reduction)."
            )
    reference_path = output / "published_reference.json"
    published = (
        json.loads(reference_path.read_text()) if reference_path.exists() else {}
    )
    doc = f"""<!doctype html><html lang="en"><meta charset="utf-8"><title>Phoom reconstruction improvements</title>
<style>body{{max-width:1200px;margin:48px auto;padding:0 26px;background:#f8fafc;color:#243247;font:16px/1.65 system-ui}}h1{{font-size:38px;line-height:1.15;color:#0f172a}}h2{{color:#0f172a}}section{{background:white;border:1px solid #e2e8f0;border-radius:12px;padding:26px;margin:24px 0}}img{{width:100%;height:auto}}table{{border-collapse:collapse;font-size:12px;width:100%}}td,th{{padding:8px;border-bottom:1px solid #e2e8f0;text-align:left}}pre{{white-space:pre-wrap;font-size:12px}}a{{color:#2463a0}}</style>
<h1>Reconstruction improvements to Phoom’s DPS recipe</h1><p>Matched-budget pilot · validation-only selection · independent holdout</p>
<section><h2>Protocol and primary outcome</h2><p>The primary metric is mean per-waveform RMSE in µV between a known analytic ABR and the reconstructed mean of 32 stochastic DPS draws. Training v-MSE is a separate diagnostic. Both architectures use exactly Phoom’s synthetic waveform generator, 20,000 training waveforms (seed 5000), dataset-standard-deviation normalization, batch size 256, AdamW at 2×10⁻⁴, zero weight decay, 500-update warmup, and gradient clipping at norm 1. The MLP is Phoom’s 544,768-parameter model; the CNN is Leo’s 7,745-parameter dilated residual network. The single training seed is 0. CNN convolutions use an equivalent explicit-window/GEMM implementation to avoid slow deterministic cuDNN kernels on the GTX 1080; forward outputs and input/parameter gradients are checked against the original convolutions. This is a runtime optimization, not a change in architecture or an RMSE intervention. Equal updates and presentations do not mean equal FLOPs.</p>
<p>EMA 0.001 is Phoom’s baseline; EMA 0.005 comes from Leo’s experiments. Both shadows follow the same trained weights, so this ablation requires no extra training. At updates 1,000, 2,000 and {config["steps"]}, validation reconstruction error selects a checkpoint and EMA using eight-draw means. A fixed guidance grid then selects one value per known noise SD. A covariance-regularized correction is tuned on validation recordings, with strength zero explicitly included. The correction solves (I + τ C MᵀM/σ²)h = h_DPS + τ C Mᵀy/σ² using covariance C estimated from training ABRs. It reuses measurements and is an empirically tuned refinement, not an exact posterior update or calibrated uncertainty method.</p>
<p>Validation: six nonzero reference ABRs (waveform seed 90422, noise seed 2000). Original benchmark: 12 nonzero reference ABRs (90421, 1000). Fresh holdout: 12 nonzero references (90423, 3000). All observations are half-second recordings with regular 10 ms pulse intervals and white Gaussian noise at SD 0.05, 0.20 and 0.50 µV. All choices are frozen before either test set is scored. Every method shares recordings and Monte Carlo seeds. The original baseline uses the last checkpoint, EMA 0.001 and Phoom’s fixed MLP guidance [10, 31.62, 100]. The selected_fixed variants isolate checkpoint/EMA selection; tuned adds guidance selection; tuned_refined adds the correction.</p>
<p>The Gram-statistic implementation accelerates the EEG residual while preserving the same mathematical loss and gradients; a numerical test compares its returned samples with Phoom’s original direct EEG implementation. The code makes no claim that these simulations establish performance on real EEG.</p><p><a href="https://forge.coreweave.com/wandb/abr-deconv/{PROJECT}">CoreWeave Forge project</a></p></section>
<section><h2>Results</h2><ul>{"".join("<li>" + html.escape(x) + "</li>" for x in findings)}</ul><table><tr><th>Split</th><th>Method</th><th>0.05 µV noise</th><th>0.20 µV noise</th><th>0.50 µV noise</th><th>Mean RMSE</th><th>Reduction vs baseline</th></tr>{"".join(table)}</table></section>
{"".join(blocks)}
<section><h2>Interpretation and limitations</h2><p>This controlled comparison concerns newly trained, matched-budget models. Phoom’s hosted runs include models trained for millions of updates; their checkpoints were unavailable in the repository and Forge artifacts when this experiment began. Logged metrics from those runs are reference observations, not a controlled head-to-head result. A reduction versus the new baseline does not establish superiority over those mature models. The published benchmark has already been inspected by previous work, so the fresh holdout is the stronger check on transfer.</p><p>One training seed and 12 held-out synthetic reference waveforms per set are a pilot, not a population study. Validation selection uses eight draws while final evaluation uses 32; this sampling-budget difference can alter rankings. All waveforms come from the same synthetic generator, which limits physiological generalization. Guidance-grid boundary selections do not identify a global optimum. Averaging and refinement can suppress peaks; RMSE should be interpreted alongside waveform shape and timing. All failures and regressions remain in the tables.</p></section>
<section><h2>Frozen selections</h2><pre>{html.escape(json.dumps(choices, indent=2))}</pre><h2>Reproduction</h2><p>From phoom/, run <code>python ../Leo/experiments/phoom_improvements.py</code> in the configured environment. Raw per-record errors, sample arrays, training histories, validation grids and checkpoints are retained. Seven figures are exported as PNG, PDF and SVG; this HTML embeds all plots for standalone viewing.</p><pre>{html.escape(json.dumps(config, indent=2))}</pre><h2>Hosted reference runs (not a paired comparison)</h2><pre>{html.escape(json.dumps(published, indent=2))}</pre></section></html>"""
    (output / "REPORT.html").write_text(doc)
    with wandb.init(
        entity="abr-deconv",
        project=PROJECT,
        group=output.name,
        name="heldout-results",
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
        for r in summary:
            run.log(
                {
                    f"{r['split']}/{r['method']}/rmse_{round(r['noise_uv'] * 1000)}nV": r[
                        "rmse_uv"
                    ]
                }
            )
        run.log(
            {
                f"figures/{p.stem}": wandb.Image(str(p))
                for p in sorted(figures.glob("*.png"))
            }
        )
        artifact = wandb.Artifact("phoom-improvements-report", type="evaluation")
        for p in output.glob("*"):
            if p.is_file() and p.suffix in (".html", ".csv", ".json"):
                artifact.add_file(str(p))
        for p in output.glob("*/*.pt"):
            artifact.add_file(str(p), name="checkpoints/" + str(p.relative_to(output)))
        run.log_artifact(artifact)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    report(parser.parse_args().output)
