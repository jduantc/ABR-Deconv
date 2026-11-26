"""Run subject-separated data, training-duration, architecture and guidance study."""

import argparse
import copy
import csv
import json
import platform
import os
import sys
import time
import concurrent.futures
import multiprocessing
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from data import AMPLITUDE_SCALE_UV, generate_responses
from diffusion import cosine_schedule, make_training_pair, clean_from_v
from model import make_model
from dps.reconstruct import stimulus_matrix
from datasets import PIPELINES, SUBJECTS, prepare


def save_rows(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_status(args, phase):
    lines = [
        "# Experiment status",
        "",
        f"Phase: **{phase}**",
        "",
        f"Updated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "| Pipeline | Model | Seed | Latest update | Loss chart |",
        "|---|---|---:|---:|---|",
    ]
    for pipeline in PIPELINES:
        for architecture in args.architectures:
            for seed in args.seeds:
                run = args.output / "models" / f"{pipeline}_{architecture}_{seed}"
                step = 0
                if (run / "losses.csv").exists():
                    with (run / "losses.csv").open() as f:
                        rows = list(csv.DictReader(f))
                    if rows:
                        step = rows[-1]["step"]
                link = (
                    f"[loss curve](models/{run.name}/losses.png)"
                    if (run / "losses.png").exists()
                    else "Pending"
                )
                lines.append(
                    f"| {pipeline} | {architecture.upper()} | {seed} | {step} | {link} |"
                )
    temp = args.output / "LIVE_STATUS.tmp"
    temp.write_text("\n".join(lines) + "\n")
    temp.replace(args.output / "LIVE_STATUS.md")


def targets(data, split, n):
    with np.load(data / "references.npz") as f:
        mask = np.isin(f["subjects"], SUBJECTS[split])
        references = f["responses"][mask]
        subjects = f["subjects"][mask]
        names = f["records"][mask]
    seed = 92011 if split == "validation" else 73009
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(references), min(n, len(references)), replace=False)
    analytic, _ = generate_responses(n, seed)
    # Keep the generator's null examples; undefined null metrics are masked below.
    return [
        ("analytic", x, 0, f"analytic-{split}-{i}") for i, x in enumerate(analytic)
    ] + [("earndb", references[i], int(subjects[i]), str(names[i])) for i in chosen]


def observations(data, split, n):
    """Fixed paired observations reused for every model/checkpoint/guidance."""
    rng = np.random.default_rng(20003 if split == "validation" else 40009)
    rows, aa, bb, yy, truths = [], [], [], [], []
    for domain, h, subject, record in targets(data, split, n):
        pulses = np.zeros(2000, np.float32)
        positions = np.cumsum(rng.integers(75, 150, 30))
        pulses[positions[positions < 1700] + 100] = 1
        matrix = stimulus_matrix(pulses).numpy()
        A = matrix.T @ matrix / len(pulses)
        for noise in ("white", "colored"):
            z = rng.normal(size=len(pulses))
            if noise == "colored":
                f = np.fft.rfftfreq(len(z))
                z = np.fft.irfft(np.fft.rfft(z) / np.sqrt(f + 0.005), n=len(z))
            z = z / np.std(z)
            for sigma in (0.5, 2.0):
                y = matrix @ h + sigma * z
                aa.append(A)
                bb.append(matrix.T @ y / len(y))
                yy.append(y @ y / len(y))
                truths.append(h)
                rows.append(
                    dict(
                        domain=domain,
                        subject=subject,
                        record=record,
                        noise=noise,
                        sigma=sigma,
                        observation_seed=split,
                    )
                )
    return {
        "A": torch.tensor(np.array(aa), dtype=torch.float32),
        "b": torch.tensor(np.array(bb), dtype=torch.float32),
        "yy": torch.tensor(yy, dtype=torch.float32),
        "truth": np.array(truths),
        "rows": rows,
    }


def sample(model, obs, guidance, seed=42):
    """Batched original 200-step DPS with exact residual sufficient statistics."""
    model.eval()
    device = next(model.parameters()).device
    A, b, yy = (obs[key].to(device) for key in ("A", "b", "yy"))
    a_bar = cosine_schedule()
    B = len(obs["rows"])
    g = torch.Generator(device=device).manual_seed(seed)
    current = torch.randn(B, 256, generator=g, device=device)
    a_bar = a_bar.to(device)
    sigma = torch.tensor([r["sigma"] for r in obs["rows"]], device=device)
    for t in range(len(a_bar) - 1, -1, -1):
        current = current.detach().requires_grad_(True)
        step = torch.full((B,), t, dtype=torch.long, device=device)
        clean = clean_from_v(current, model(current, step), step, a_bar)
        quad = torch.einsum("bi,bij,bj->b", clean, A, clean)
        residual = (quad - 2 * (clean * b).sum(1) + yy).clamp_min(1e-9)
        loss = residual.sqrt() / sigma
        gradient = torch.autograd.grad(loss.sum(), current)[0]
        a = a_bar[t]
        previous_a = a_bar[t - 1] if t else torch.tensor(1.0, device=device)
        alpha = a / previous_a
        beta = 1 - alpha
        mean = previous_a.sqrt() * beta / (1 - a) * clean
        mean += alpha.sqrt() * (1 - previous_a) / (1 - a) * current
        variance = beta * (1 - previous_a) / (1 - a)
        noise = torch.randn(current.shape, generator=g, device=device) if t else 0
        current = (mean + variance.sqrt() * noise - guidance * gradient).detach()
    result = current.cpu().numpy()
    if not np.isfinite(result).all():
        raise RuntimeError(f"Nonfinite DPS output at guidance {guidance}")
    return result


def metrics(estimate, truth):
    error = (estimate - truth) * AMPLITUDE_SCALE_UV
    a = estimate - estimate.mean(-1, keepdims=True)
    b = truth - truth.mean(-1, keepdims=True)
    denom = np.sqrt((a * a).sum(-1) * (b * b).sum(-1))
    corr = np.divide(
        (a * b).sum(-1), denom, out=np.full(len(a), np.nan), where=denom > 1e-12
    )
    # Peak timing is undefined for nulls. EARNDB peak is a descriptive 3–10 ms maximum.
    win = slice(80, 151)
    latency = np.abs(estimate[:, win].argmax(1) - truth[:, win].argmax(1)) / 10
    latency[np.ptp(truth, axis=1) < 1e-8] = np.nan
    return np.sqrt((error * error).mean(-1)), corr, latency


def train_one(args, pipeline, architecture, seed):
    run = args.output / "models" / f"{pipeline}_{architecture}_{seed}"
    run.mkdir(parents=True, exist_ok=True)
    final = run / f"step_{max(args.checkpoints)}.pt"
    if final.exists() and all(
        (run / f"step_{s}.pt").exists() for s in args.checkpoints
    ):
        return run
    torch.manual_seed(seed)
    x = torch.tensor(
        np.load(args.output / "data" / f"{pipeline}.npz")["responses"],
        device=args.device,
    )
    model = make_model(architecture).to(args.device)
    ema = copy.deepcopy(model).eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    schedule = cosine_schedule().to(args.device)
    val = torch.tensor(
        np.stack([t[1] for t in targets(args.output / "data", "validation", 64)]),
        device=args.device,
    )
    gen = torch.Generator(device=args.device).manual_seed(173)
    vt = torch.randint(200, (len(val),), generator=gen, device=args.device)
    vn = torch.randn(val.shape, generator=gen, device=args.device)
    vi, vtarget = make_training_pair(val, vt, vn, schedule)
    config = dict(
        pipeline=pipeline,
        architecture=architecture,
        seed=seed,
        batch_size=args.batch_size,
        learning_rate=2e-4,
        weight_decay=1e-4,
        ema_rate=0.005,
        train_count=len(x),
        checkpoints=args.checkpoints,
        parameters=sum(p.numel() for p in model.parameters()),
    )
    (run / "settings.json").write_text(json.dumps(config, indent=2))
    rows = []
    start = time.monotonic()
    recent = []
    for step in range(1, max(args.checkpoints) + 1):
        clean = x[torch.randint(len(x), (args.batch_size,), device=args.device)]
        ts = torch.randint(200, (len(clean),), device=args.device)
        noisy, target = make_training_pair(clean, ts, torch.randn_like(clean), schedule)
        loss = (model(noisy, ts) - target).square().mean()
        if not torch.isfinite(loss).item():
            raise RuntimeError(f"Nonfinite training loss: {run.name} at step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        with torch.no_grad():
            for pe, p in zip(ema.parameters(), model.parameters()):
                pe.lerp_(p, 0.005)
        recent.append(loss.item())
        if step % 200 == 0 or step in args.checkpoints:
            with torch.no_grad():
                vloss = (ema(vi, vt) - vtarget).square().mean().item()
            rows.append(
                dict(
                    step=step,
                    train_v_mse=np.mean(recent),
                    validation_v_mse=vloss,
                    seconds=time.monotonic() - start,
                )
            )
            save_rows(run / "losses.csv", rows)
            recent = []
        if step % 1000 == 0 or step == max(args.checkpoints):
            from plots import plot_run_loss

            plot_run_loss(run)
            # Separate CPU workers share this file; only the single GPU worker updates it.
            if args.device == "cuda":
                write_status(args, "Training")
        if step in args.checkpoints:
            torch.save(
                {
                    "model": ema.state_dict(),
                    "architecture": architecture,
                    "step": step,
                    "config": config,
                },
                run / f"step_{step}.pt",
            )
            print(
                f"{run.name} step={step} val={vloss:.5f} elapsed={time.monotonic() - start:.0f}s",
                flush=True,
            )
    return run


def training_job(job):
    args, pipeline, architecture, seed = job
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    return train_one(args, pipeline, architecture, seed)


_OBS_CACHE = {}


def evaluation_job(job):
    args, pipeline, architecture, seed, step = job
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    shard = args.output / "evaluation" / f"{pipeline}_{architecture}_{seed}_{step}"
    shard.mkdir(parents=True, exist_ok=True)
    if (shard / "prediction.npz").exists():
        return shard
    key = (str(args.output), args.validation_cases, args.test_cases)
    if key not in _OBS_CACHE:
        _OBS_CACHE[key] = (
            observations(args.output / "data", "validation", args.validation_cases),
            observations(args.output / "data", "test", args.test_cases),
        )
    val, test = _OBS_CACHE[key]
    run = args.output / "models" / f"{pipeline}_{architecture}_{seed}"
    model = make_model(architecture).to(args.device)
    model.load_state_dict(
        torch.load(
            run / f"step_{step}.pt", weights_only=True, map_location=args.device
        )["model"]
    )
    for p in model.parameters():
        p.requires_grad_(False)
    scores, validation_rows = [], []
    for guidance in args.guidances:
        prediction = sample(model, val, guidance)
        rmse, _, _ = metrics(prediction, val["truth"])
        scores.append(float(rmse.mean()))
        for domain in ("analytic", "earndb"):
            idx = [i for i, r in enumerate(val["rows"]) if r["domain"] == domain]
            validation_rows.append(
                dict(
                    pipeline=pipeline,
                    architecture=architecture,
                    seed=seed,
                    step=step,
                    guidance=guidance,
                    domain=domain,
                    rmse_uv=rmse[idx].mean(),
                )
            )
    selected = args.guidances[int(np.argmin(scores))]
    prediction = sample(model, test, selected)
    rmse, corr, latency = metrics(prediction, test["truth"])
    test_rows = [
        dict(
            **row,
            pipeline=pipeline,
            architecture=architecture,
            seed=seed,
            step=step,
            guidance=selected,
            rmse_uv=rmse[i],
            correlation=corr[i],
            peak_latency_error_ms=latency[i],
        )
        for i, row in enumerate(test["rows"])
    ]
    save_rows(shard / "validation.csv", validation_rows)
    save_rows(shard / "test.csv", test_rows)
    np.savez_compressed(shard / "prediction.npz", prediction=prediction)
    print(
        f"Evaluated {run.name} step={step} guidance={selected} RMSE={rmse.mean():.5f}uV",
        flush=True,
    )
    return shard


def evaluate(args):
    data = args.output / "data"
    val = observations(data, "validation", args.validation_cases)
    test = observations(data, "test", args.test_cases)
    save_rows(args.output / "validation_observations.csv", val["rows"])
    save_rows(args.output / "test_observations.csv", test["rows"])
    validation_rows, test_rows, examples = [], [], {}
    jobs = [
        (args, pipe, arch, seed, step)
        for pipe in PIPELINES
        for arch in args.architectures
        for seed in args.seeds
        for step in args.checkpoints
    ]
    if args.device == "cpu" and args.workers > 1:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            shards = list(pool.map(evaluation_job, jobs))
    else:
        shards = [evaluation_job(job) for job in jobs]
    for job, shard in zip(jobs, shards):
        _, pipe, arch, seed, step = job
        for path, rows in [
            (shard / "validation.csv", validation_rows),
            (shard / "test.csv", test_rows),
        ]:
            with path.open() as f:
                rows.extend(csv.DictReader(f))
        if seed == args.seeds[0]:
            examples[f"{pipe}_{arch}_{step}"] = np.load(shard / "prediction.npz")[
                "prediction"
            ]
    save_rows(args.output / "validation_guidance.csv", validation_rows)
    save_rows(args.output / "test_metrics.csv", test_rows)
    # Fixed ridge baseline: tuning is never done using the test set.
    A, b = test["A"], test["b"]
    ridge = 0.01 * A.diagonal(dim1=1, dim2=2).mean(1)
    estimate = torch.linalg.solve(A + ridge[:, None, None] * torch.eye(256), b).numpy()
    rmse, corr, latency = metrics(estimate, test["truth"])
    save_rows(
        args.output / "baseline_metrics.csv",
        [
            dict(
                **row,
                rmse_uv=rmse[i],
                correlation=corr[i],
                peak_latency_error_ms=latency[i],
            )
            for i, row in enumerate(test["rows"])
        ],
    )
    np.savez_compressed(
        args.output / "predictions.npz", truth=test["truth"], ridge=estimate, **examples
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=HERE.parent / "runs" / "comparison")
    p.add_argument(
        "--phase", choices=["all", "data", "train", "evaluate", "plot"], default="all"
    )
    p.add_argument("--count", type=int, default=60000)
    p.add_argument(
        "--checkpoints", type=int, nargs="+", default=[1000, 4000, 8000, 12000, 16000]
    )
    p.add_argument(
        "--architectures",
        choices=["mlp", "cnn", "unet"],
        nargs="+",
        default=["mlp", "cnn", "unet"],
    )
    p.add_argument("--seeds", type=int, nargs="+", default=[17, 29])
    p.add_argument(
        "--guidances", type=float, nargs="+", default=[0, 0.03, 0.1, 0.3, 1.0]
    )
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--validation-cases", type=int, default=12)
    p.add_argument("--test-cases", type=int, default=24)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=3,
        help="CPU training processes; CUDA always uses one",
    )
    args = p.parse_args()
    if (
        min(
            args.count,
            args.batch_size,
            args.validation_cases,
            args.test_cases,
            *args.checkpoints,
        )
        < 1
    ):
        p.error("counts must be positive")
    if min(args.guidances) < 0 or not np.isfinite(args.guidances).all():
        p.error("guidance must be finite and nonnegative")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    config = {
        k: str(v) if isinstance(v, Path) else v
        for k, v in vars(args).items()
        if k not in ("phase", "workers")
    }
    config.update(
        torch=torch.__version__, numpy=np.__version__, python=platform.python_version()
    )
    config_file = args.output / "study_config.json"
    if config_file.exists() and json.loads(config_file.read_text()) != config:
        raise ValueError(
            "Existing output uses a different configuration; choose a new output folder"
        )
    config_file.write_text(json.dumps(config, indent=2))
    write_status(args, args.phase)
    if args.phase in ("all", "data"):
        prepare(args.output / "data", args.count)
    if args.phase in ("all", "train"):
        jobs = [
            (args, pipeline, architecture, seed)
            for pipeline in PIPELINES
            for architecture in args.architectures
            for seed in args.seeds
        ]
        if args.device == "cpu" and args.workers > 1:
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=args.workers,
                mp_context=multiprocessing.get_context("spawn"),
            ) as pool:
                list(pool.map(training_job, jobs))
        else:
            for job in jobs:
                training_job(job)
    if args.phase in ("all", "evaluate"):
        evaluate(args)
    if args.phase in ("all", "plot"):
        from plots import plot_all

        plot_all(args.output)
    write_status(args, "Complete" if args.phase == "all" else f"{args.phase} complete")


if __name__ == "__main__":
    main()
