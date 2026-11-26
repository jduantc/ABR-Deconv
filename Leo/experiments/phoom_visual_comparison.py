"""Create accessible, self-contained figures for the Phoom-style comparisons."""

import base64
import csv
import html
import json
from pathlib import Path
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import wandb

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "Leo/results/phoom_improvements"
LONG = ROOT / "Leo/runs/phoom_long_50k"
OUT = ROOT / "Leo/results/phoom_visual_comparison"
BLUE = "#215D9C"
TEAL = "#007F78"
ORANGE = "#C66A32"
GRAY = "#738398"
INK = "#17283F"
GRID = "#DCE5EC"


def read(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    figures = OUT / "figures"
    figures.mkdir(exist_ok=True)
    data = OUT / "data"
    data.mkdir(exist_ok=True)
    for source, name in [
        (BASE / "summary.csv", "short_summary.csv"),
        (BASE / "test_metrics.csv", "short_test_metrics.csv"),
        (LONG / "summary.csv", "long_summary.csv"),
        (LONG / "mlp/checkpoint_validation.csv", "long_validation.csv"),
    ]:
        if source.exists():
            shutil.copy2(source, data / name)
        if not (data / name).exists():
            raise FileNotFoundError(f"Missing source data: {source}")
    short = read(data / "short_summary.csv")
    long = read(data / "long_summary.csv")
    raw = read(data / "short_test_metrics.csv")
    validation = read(data / "long_validation.csv")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#B9C9D4",
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": "#536579",
            "ytick.color": "#536579",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "grid.color": GRID,
            "axes.titleweight": "bold",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    blocks = []
    manifests = []

    def export(fig, name, title, explanation):
        fig.set_layout_engine("constrained")
        for ext in ("png", "pdf", "svg"):
            fig.savefig(figures / f"{name}.{ext}", dpi=220, facecolor="white")
        plt.close(fig)
        encoded = base64.b64encode((figures / f"{name}.png").read_bytes()).decode()
        blocks.append(
            f'<section><h2>{html.escape(title)}</h2><img src="data:image/png;base64,{encoded}" alt="{html.escape(title)}"><p>{explanation}</p></section>'
        )
        manifests.append({"figure": name, "title": title, "explanation": explanation})

    def value(rows, split, method, sigma=None):
        matches = [
            float(r["rmse_uv"]) * 1000
            for r in rows
            if r["split"] == split
            and r["method"] == method
            and (sigma is None or float(r["noise_uv"]) == sigma)
        ]
        if not matches:
            raise ValueError((split, method, sigma))
        return float(np.mean(matches))

    def style(ax):
        ax.set_axisbelow(True)
        ax.grid(axis="y", color=GRID, lw=0.8)

    def annotate(ax, bars):
        for b in bars:
            ax.annotate(
                f"{b.get_height():.2f}",
                (b.get_x() + b.get_width() / 2, b.get_height()),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                fontsize=10,
                fontweight="bold",
                color=INK,
            )

    # Both sets belong to the same 4k experiment; means equally weight the three noise levels.
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(2)
    width = 0.32
    base = [
        value(short, s, "mlp_phoom_baseline")
        for s in ("phoom_benchmark", "fresh_holdout")
    ]
    new = [
        value(short, s, "mlp_tuned_refined")
        for s in ("phoom_benchmark", "fresh_holdout")
    ]
    bars1 = ax.bar(x - width / 2, base, width, color=GRAY, label="Phoom-style baseline")
    bars2 = ax.bar(
        x + width / 2, new, width, color=TEAL, label="Faster EMA + correction"
    )
    annotate(ax, bars1)
    annotate(ax, bars2)
    for i, (b, n) in enumerate(zip(base, new)):
        ax.text(
            i,
            max(b, n) + 0.68,
            f"{100 * (1 - n / b):.1f}% lower",
            ha="center",
            color=TEAL,
            fontweight="bold",
            fontsize=12,
        )
    ax.set_xticks(
        x, ["Phoom benchmark\n12 reference ABRs", "Fresh holdout\n12 reference ABRs"]
    )
    ax.set(
        ylabel="Mean reconstruction RMSE (nV)",
        title="4,000 updates · improvement on both test sets",
        ylim=(0, max(base) * 1.29),
    )
    style(ax)
    ax.legend(frameon=False, ncol=2, loc="upper left")
    export(
        fig,
        "01_before_after",
        "The matched-budget result",
        "Mean per-ABR RMSE, averaged equally across 0.05, 0.20 and 0.50 µV noise levels. Each ABR estimate is the mean of 32 samples. Both models trained for 4,000 updates using the same MLP trajectory. Lower is better.",
    )

    fig, ax = plt.subplots(figsize=(11, 5))
    methods = ["mlp_phoom_baseline", "mlp_selected_fixed", "mlp_tuned_refined"]
    vals = [value(short, "fresh_holdout", m) for m in methods]
    bars = ax.bar(np.arange(3), vals, color=[GRAY, BLUE, TEAL], width=0.58)
    annotate(ax, bars)
    for j in (1, 2):
        ax.annotate(
            f"−{vals[j - 1] - vals[j]:.2f} nV",
            ((j - 0.5), max(vals[j - 1], vals[j]) + 0.4),
            ha="center",
            color=BLUE if j == 1 else TEAL,
            fontweight="bold",
        )
    ax.set_xticks(
        range(3),
        [
            "Baseline\nEMA 0.001",
            "Change EMA\nto 0.005",
            "Add EEG correction\nstrength 0.1",
        ],
    )
    ax.set(
        ylabel="Mean reconstruction RMSE (nV)",
        title="Fresh holdout · where the reduction came from",
        ylim=(0, max(vals) * 1.27),
    )
    style(ax)
    export(
        fig,
        "02_components",
        "What changed",
        "The faster EMA and then the measurement correction are shown in sequence. The EMA shadows followed the same training updates. Guidance tuning selected the original MLP guidance values, so it made no additional change to this result.",
    )

    fig, ax = plt.subplots(figsize=(11, 5))
    sigmas = (0.05, 0.2, 0.5)
    x = np.arange(3)
    width = 0.32
    b = [value(short, "fresh_holdout", "mlp_phoom_baseline", s) for s in sigmas]
    n = [value(short, "fresh_holdout", "mlp_tuned_refined", s) for s in sigmas]
    a = ax.bar(x - width / 2, b, width, color=GRAY, label="Phoom-style baseline")
    c = ax.bar(x + width / 2, n, width, color=TEAL, label="Improved MLP")
    annotate(ax, a)
    annotate(ax, c)
    for i, (u, v) in enumerate(zip(b, n)):
        ax.text(
            i,
            max(u, v) + 0.7,
            f"{100 * (1 - v / u):.1f}% lower",
            ha="center",
            color=TEAL,
            fontweight="bold",
        )
    ax.set_xticks(x, ["0.05 µV", "0.20 µV", "0.50 µV"])
    ax.set(
        xlabel="White-noise standard deviation in the EEG",
        ylabel="Mean reconstruction RMSE (nV)",
        title="Fresh holdout · improvement by noise level",
        ylim=(0, max(b) * 1.22),
    )
    style(ax)
    ax.legend(frameon=False, ncol=2, loc="upper left")
    export(
        fig,
        "03_noise_levels",
        "Across noise levels",
        "The same 12 reference ABRs and corresponding simulated recordings are used for both methods at each noise level. The noise is white and its standard deviation is assumed known.",
    )

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.8), sharey=True)
    for ax, sigma in zip(axes, sigmas):
        baseline = {
            int(r["record"]): float(r["rmse_uv"]) * 1000
            for r in raw
            if r["split"] == "fresh_holdout"
            and r["method"] == "mlp_phoom_baseline"
            and int(r["samples"]) == 32
            and float(r["noise_uv"]) == sigma
        }
        improved = {
            int(r["record"]): float(r["rmse_uv"]) * 1000
            for r in raw
            if r["split"] == "fresh_holdout"
            and r["method"] == "mlp_tuned_refined"
            and int(r["samples"]) == 32
            and float(r["noise_uv"]) == sigma
        }
        assert baseline.keys() == improved.keys() and len(baseline) == 12
        wins = 0
        for i in sorted(baseline):
            better = improved[i] < baseline[i]
            wins += better
            ax.plot(
                [0, 1],
                [baseline[i], improved[i]],
                color=TEAL if better else ORANGE,
                alpha=0.56,
                lw=1.5,
                marker="o",
                ms=3,
            )
        ax.plot(
            [0, 1],
            [np.mean(list(baseline.values())), np.mean(list(improved.values()))],
            color=INK,
            lw=2.8,
            marker="o",
            ms=6,
            label="Mean",
        )
        ax.set_xticks([0, 1], ["Baseline", "Improved"])
        ax.set(
            title=f"{sigma:g} µV noise\n{wins} of 12 lower",
            ylabel="ABR RMSE (nV)" if ax is axes[0] else "",
        )
        style(ax)
    export(
        fig,
        "04_paired_cases",
        "Individual paired reconstructions",
        "Each thin line is one reference ABR: teal means its error decreased and orange means it increased. The dark line connects means. The same 12 waveforms appear in all panels; they are not 36 independent participants.",
    )

    fig, ax = plt.subplots(figsize=(11, 5))
    for rate, color in ((0.001, GRAY), (0.005, BLUE)):
        rows = sorted(
            [r for r in validation if float(r["ema_rate"]) == rate],
            key=lambda r: int(r["checkpoint"]),
        )
        ax.plot(
            [int(r["checkpoint"]) / 1000 for r in rows],
            [float(r["rmse_uv"]) * 1000 for r in rows],
            marker="o",
            lw=2.3,
            color=color,
            label=f"EMA {rate:g}",
        )
    ax.set(
        xlabel="Training updates (thousands)",
        ylabel="Validation reconstruction RMSE (nV)",
        title="Longer training · the EMA advantage changes",
    )
    style(ax)
    ax.legend(frameon=False)
    export(
        fig,
        "05_longer_training",
        "What happened by 50,000 updates",
        "This is validation RMSE, using eight-sample averages on six validation waveforms; it is not the 32-sample test RMSE shown above. The two curves follow the same model trajectory. One training seed and a small validation set do not establish population-level behavior.",
    )

    fig, ax = plt.subplots(figsize=(11, 5))
    methods = [
        "mlp_phoom_baseline",
        "mlp_fast_ema_final",
        "mlp_selected_fixed",
        "mlp_tuned_refined",
    ]
    vals = [value(long, "fresh_holdout", m) for m in methods]
    bars = ax.bar(range(4), vals, color=[GRAY, BLUE, "#7A9EB5", TEAL], width=0.62)
    annotate(ax, bars)
    ax.set_xticks(
        range(4),
        [
            "50k updates\nEMA 0.001",
            "50k updates\nEMA 0.005",
            "Validation-selected\n4k + EMA 0.005",
            "Selected 4k\n+ correction",
        ],
    )
    ax.set(
        ylabel="Mean reconstruction RMSE (nV)",
        title="50k follow-up · a separate fresh holdout",
        ylim=(0, max(vals) * 1.2),
    )
    style(ax)
    export(
        fig,
        "06_50k_test",
        "The 50k follow-up",
        "This fresh holdout differs from the 4k experiment’s holdout, so its values should not be compared across those figures as if they used identical recordings. The 4k checkpoint and EMA were selected on validation data; the correction strength was also selected before test scoring.",
    )

    truth = np.load(BASE / "fresh_holdout_truth.npz")["truth"]
    examples = np.load(BASE / "examples.npz")
    times = (np.arange(256) - 50) / 10
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, sigma, idx in zip(axes, (0.2, 0.5), (12, 24)):
        ax.plot(
            times, truth[idx] * 1000, color=INK, lw=2.6, label="Known reference ABR"
        )
        ax.plot(
            times,
            examples["fresh_holdout_mlp_phoom_baseline"][idx] * 1000,
            color=GRAY,
            lw=1.8,
            label="Phoom-style baseline",
        )
        ax.plot(
            times,
            examples["fresh_holdout_mlp_tuned_refined"][idx] * 1000,
            color=TEAL,
            lw=1.8,
            label="Improved MLP",
        )
        ax.axvline(0, color="#A5B4C2", lw=0.8, ls=":")
        ax.set(
            title=f"{sigma:g} µV noise",
            xlabel="Time relative to stimulus (ms)",
            ylabel="ABR amplitude (nV)" if ax is axes[0] else "",
        )
        style(ax)
    axes[1].legend(frameon=False, fontsize=8, loc="upper right")
    export(
        fig,
        "07_waveforms",
        "What two reconstructions look like",
        "The first held-out reference waveform is shown at two noise levels, selected by fixed index rather than quality. The same reference and EEG are supplied to each method. These examples do not summarize all cases.",
    )

    (OUT / "figure_manifest.json").write_text(json.dumps(manifests, indent=2))
    document = f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>ABR reconstruction: what improved?</title><style>body{{max-width:1120px;margin:50px auto;padding:0 24px;background:#f2f6f8;color:#17283f;font:16px/1.65 system-ui}}h1{{font-size:clamp(2.2rem,5vw,3.4rem);line-height:1.1;letter-spacing:-.03em}}h2{{color:#17283f}}section,.intro{{background:white;border:1px solid #dce5ec;border-radius:14px;padding:26px;margin:22px 0}}img{{display:block;width:100%;height:auto}}p{{max-width:92ch}}.tag{{color:#007f78;text-transform:uppercase;letter-spacing:.14em;font-size:.78rem;font-weight:700}}a{{color:#215d9c}}@media(max-width:650px){{body{{margin:24px auto;padding:0 12px}}section,.intro{{padding:18px}}}}</style><div class="tag">CS163 · reconstruction experiments</div><h1>What improved in ABR reconstruction?</h1><div class="intro"><p>The figures show the completed 4,000-update and 50,000-update Phoom-style MLP experiments. We compare reconstruction RMSE against known reference ABRs, after averaging 32 stochastic reconstructions. Lower RMSE means the reconstructed waveform is closer to its reference.</p><p>At 4,000 updates, faster EMA plus a validation-selected measurement correction reduced mean RMSE by {100 * (1 - new[1] / base[1]):.1f}% on a fresh synthetic holdout. The longer follow-up shows why checkpoint and EMA choices need continued testing.</p><p><strong>Scope:</strong> one training seed; 12 synthetic reference ABRs per test set; half-second EEG simulations with white noise. The Phoom-style baseline is our matched-budget reproduction, not Phoom’s unavailable million-update checkpoint. The ongoing 10-million-update run is not included in these completed results.</p><p><a href="https://forge.coreweave.com/wandb/abr-deconv/abr-deconv-phoom-improvements">Open the experiment runs in CoreWeave Forge</a></p></div>{"".join(blocks)}<section><h2>Method and data</h2><p>Both methods share a trained MLP, simulated pulse timings, EEG recordings, and sampling seeds. The baseline uses EMA 0.001 and Phoom’s fixed guidance. The improved pipeline selects EMA/checkpoint and a covariance-regularized EEG correction using validation recordings. The correction reuses the EEG; it is an empirical refinement, not an exact posterior update. Test data are used only for the displayed evaluations.</p><p>The 4k and 50k follow-ups use different fresh holdout seeds. Validation points average eight reconstructions; final test points average 32 before calculating RMSE. Comparisons with Phoom’s actual million-update models require his weights and a shared evaluation. PNG, PDF and SVG exports and the source tables accompany this self-contained page.</p></section></html>"""
    (OUT / "COMPARISON.html").write_text(document, encoding="utf-8")
    with wandb.init(
        entity="abr-deconv",
        project="abr-deconv-phoom-improvements",
        name="visual comparison",
        job_type="visualization",
        settings=wandb.Settings(disable_code=True),
    ) as run:
        run.log(
            {
                f"figures/{p.stem}": wandb.Image(str(p))
                for p in sorted(figures.glob("*.png"))
            }
        )
        artifact = wandb.Artifact("phoom-visual-comparison", type="evaluation")
        artifact.add_file(str(OUT / "COMPARISON.html"))
        artifact.add_file(str(OUT / "figure_manifest.json"))
        for p in data.glob("*.csv"):
            artifact.add_file(str(p), name="data/" + p.name)
        for p in figures.glob("*.png"):
            artifact.add_file(str(p), name="figures/" + p.name)
        artifact.add_file(str(Path(__file__)), name="phoom_visual_comparison.py")
        logged = run.log_artifact(artifact)
        logged.wait()
        print("Forge artifact:", logged.name, logged.state, flush=True)
    print("Saved", len(blocks), "figures and the standalone visual comparison to", OUT)


if __name__ == "__main__":
    main()
