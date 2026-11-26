"""Publish fixed-protocol long-training reconstruction comparisons."""

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
    "ema_0.001": "EMA 0.001",
    "ema_0.005": "EMA 0.005",
    "ema_0.001_refined": "EMA 0.001 + correction",
    "ema_0.005_refined": "EMA 0.005 + correction",
}
COLORS = {
    "ema_0.001": "#64748B",
    "ema_0.005": "#2463A0",
    "ema_0.001_refined": "#3F8A71",
    "ema_0.005_refined": "#C46835",
}


def read(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def render(output, interim=False):
    output = Path(output)
    config = json.loads((output / "protocol.json").read_text())
    validation = read(output / "validation.csv")
    split = "interim_1m" if interim else "final_fresh"
    datasets = [split] if interim else [split, "phoom_benchmark"]
    figures = output / ("interim_figures" if interim else "figures")
    figures.mkdir(exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#E2E8F0",
            "axes.titleweight": "bold",
            "text.color": "#0F172A",
            "figure.facecolor": "white",
        }
    )
    blocks = []

    def save(fig, name, title, caption):
        fig.set_layout_engine("constrained")
        for ext in ("png", "pdf", "svg"):
            fig.savefig(figures / f"{name}.{ext}", dpi=200, facecolor="white")
        plt.close(fig)
        encoded = base64.b64encode((figures / f"{name}.png").read_bytes()).decode()
        blocks.append(
            f'<section><h2>{title}</h2><img alt="{title}" src="data:image/png;base64,{encoded}"><p>{caption}</p></section>'
        )

    fig, ax = plt.subplots(figsize=(11, 4))
    for rate, color in [(0.001, "#64748B"), (0.005, "#2463A0")]:
        rows = sorted(
            [r for r in validation if float(r["ema_rate"]) == rate],
            key=lambda r: int(r["update"]),
        )
        ax.plot(
            [int(r["update"]) for r in rows],
            [float(r["rmse_uv"]) for r in rows],
            marker="o",
            color=color,
            label=f"EMA {rate}",
        )
    ax.set(
        xscale="log",
        xlabel="Optimizer updates",
        ylabel="Validation ABR RMSE (µV)",
        title="Same model trajectory, two EMA rates",
    )
    ax.legend(frameon=False)
    save(
        fig,
        "01_validation",
        "1 · Reconstruction versus training duration",
        "All points use the same 24 validation waveforms, fixed guidance and eight-sample averages. Validation is separate from the tests. The horizontal axis is logarithmic.",
    )
    test = {name: read(output / f"{name}_metrics.csv") for name in datasets}
    fig, axes = plt.subplots(
        1, len(datasets), figsize=(11, 4), sharey=True, squeeze=False
    )
    for ax, name in zip(axes[0], datasets):
        for method, color in COLORS.items():
            values = [
                np.mean(
                    [
                        float(r["rmse_uv"])
                        for r in test[name]
                        if r["method"] == method and float(r["noise_uv"]) == s
                    ]
                )
                for s in (0.05, 0.2, 0.5)
            ]
            ax.plot(range(3), values, marker="o", color=color, label=LABELS[method])
        ax.set_xticks(range(3), ["0.05", "0.20", "0.50"])
        ax.set(
            title=name.replace("_", " ").title(),
            xlabel="Noise SD (µV)",
            ylabel="32-sample mean ABR RMSE (µV)",
        )
    axes[0][-1].legend(frameon=False, fontsize=7)
    save(
        fig,
        "02_test",
        "2 · Held-out reconstruction RMSE",
        "All four methods use identical recordings and stochastic draws. Correction strength 0.1 was fixed from the previous pilot, before these test sets were generated.",
    )
    summary = []
    for name in datasets:
        for method in COLORS:
            vals = [float(r["rmse_uv"]) for r in test[name] if r["method"] == method]
            summary.append(
                dict(split=name, method=method, mean_rmse_uv=float(np.mean(vals)))
            )
    summary_file = output / ("interim_summary.csv" if interim else "summary.csv")
    with summary_file.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(summary)
    body = "".join(
        f"<tr><td>{r['split']}</td><td>{LABELS[r['method']]}</td><td>{r['mean_rmse_uv']:.6f}</td></tr>"
        for r in summary
    )
    horizon = 1000000 if interim else config["steps"]
    content = f"""<!doctype html><html lang="en"><meta charset="utf-8"><title>Matched-budget ABR reconstruction</title><style>body{{max-width:1150px;margin:48px auto;padding:0 26px;background:#f8fafc;color:#243247;font:16px/1.65 system-ui}}h1{{font-size:38px;line-height:1.15}}section{{background:white;padding:26px;border:1px solid #e2e8f0;border-radius:12px;margin:24px 0}}img{{width:100%}}table{{width:100%;border-collapse:collapse}}td,th{{padding:9px;text-align:left;border-bottom:1px solid #e2e8f0}}pre{{white-space:pre-wrap;font-size:12px}}</style><h1>ABR reconstruction after {horizon:,} updates</h1><section><h2>Methodology</h2><p>This continues the same MLP training trajectory from its saved 50k raw weights, AdamW optimizer state and sampling-generator state. Both EMA rates follow identical optimizer updates. The validation set has 24 newly generated nonzero analytic ABRs, three white-noise levels, and eight stochastic reconstructions per case. Test scores average 32 reconstructions first, then measure waveform RMSE in µV. Guidance is Phoom’s fixed per-noise setting; optional covariance-regularized correction strength 0.1 was chosen in the earlier pilot. No setting is selected from the test scores.</p><p>The fresh {"interim" if interim else "final"} set contains 24 new waveforms generated with seeds disjoint from training, validation, and the earlier pilot. {"The benchmark uses Phoom’s previously observed 12-waveform set; it is included for historical context." if not interim else "This interim set is separate from the final holdout."} A single training seed and synthetic EEG limit generalization. Results compare a reproduced Phoom-style control; Phoom’s original trained checkpoint was unavailable.</p></section><section><h2>Results</h2><table><tr><th>Split</th><th>Method</th><th>Mean ABR RMSE (µV)</th></tr>{body}</table></section>{"".join(blocks)}<section><h2>Protocol</h2><pre>{html.escape(json.dumps(config, indent=2))}</pre></section></html>"""
    path = output / ("INTERIM_1M.html" if interim else "REPORT.html")
    path.write_text(content)
    return path, summary_file, figures


def report_interim(output, run):
    output = Path(output)
    report_path, summary, figures = render(output, True)
    run.log(
        {
            f"interim_figures/{p.stem}": wandb.Image(str(p))
            for p in sorted(figures.glob("*.png"))
        }
    )
    artifact = wandb.Artifact("phoom-matched-interim-1m", type="evaluation")
    for p in (
        report_path,
        summary,
        output / "interim_1m_metrics.csv",
        output / "protocol.json",
    ):
        artifact.add_file(str(p))
    run.log_artifact(artifact)


def report(output, project):
    output = Path(output)
    report_path, summary, figures = render(output, False)
    with wandb.init(
        entity="abr-deconv",
        project=project,
        group=output.name,
        name="10m final comparison",
        job_type="evaluation",
    ) as run:
        run.log(
            {
                f"figures/{p.stem}": wandb.Image(str(p))
                for p in sorted(figures.glob("*.png"))
            }
        )
        artifact = wandb.Artifact("phoom-matched-10m-report", type="evaluation")
        for p in (
            report_path,
            summary,
            output / "validation.csv",
            output / "protocol.json",
            output / "final_fresh_metrics.csv",
            output / "phoom_benchmark_metrics.csv",
        ):
            artifact.add_file(str(p))
        for p in output.glob("step_10000000_ema_*.pt"):
            artifact.add_file(str(p), name="checkpoints/" + p.name)
        run.log_artifact(artifact)
