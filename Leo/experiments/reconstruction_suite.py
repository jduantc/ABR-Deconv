"""Reconstruction-focused ABR experiments with validation-only model selection.

Run from Leo/: .venv/bin/python experiments/reconstruction_suite.py
"""

import argparse
import copy
import csv
import hashlib
import json
import os
import platform
from pathlib import Path
import sys
import time
import uuid

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
import wandb

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from data import AMPLITUDE_SCALE_UV
from diffusion import clean_from_v, cosine_schedule, make_training_pair
from model import make_model
from study import metrics, observations, save_rows, targets

CHECKPOINTS = (200, 400, 800, 1000)
GUIDANCES = (0.0, 0.3, 1.0, 3.0)
PIPELINES = ("analytic", "earndb_templates")
SEEDS = (17, 29)
SAMPLE_SEEDS = (42, 43, 44)


def reconstruct(model, obs, guidance, seed=42, trace=None, repeats=1):
    """Historical 200-step DPS with noise-normalized batched Gram residuals.

    Reference RMSE is a passive diagnostic; truth never enters the gradient.
    """
    device = next(model.parameters()).device
    A, b, yy = (obs[k].to(device) for k in ("A", "b", "yy"))
    sigma = torch.tensor([r["sigma"] for r in obs["rows"]], device=device)
    schedule = cosine_schedule().to(device)
    generator = torch.Generator(device=device).manual_seed(seed)
    current = torch.randn(
        len(b) // repeats, 256, generator=generator, device=device
    ).repeat(repeats, 1)
    truth = torch.tensor(obs["truth"], device=device) if trace is not None else None
    for t in reversed(range(len(schedule))):
        current = current.detach().requires_grad_(True)
        step = torch.full((len(b),), t, device=device, dtype=torch.long)
        clean = clean_from_v(current, model(current, step), step, schedule)
        residual2 = (
            (clean * torch.bmm(A, clean.unsqueeze(-1)).squeeze(-1)).sum(1)
            - 2 * (clean * b).sum(1)
            + yy
        ).clamp_min(1e-9)
        loss = residual2.sqrt() / sigma
        gradient = torch.autograd.grad(loss.sum(), current)[0]
        a = schedule[t]
        previous_a = schedule[t - 1] if t else torch.tensor(1.0, device=device)
        alpha = a / previous_a
        beta = 1 - alpha
        mean = previous_a.sqrt() * beta / (1 - a) * clean
        mean += alpha.sqrt() * (1 - previous_a) / (1 - a) * current
        variance = beta * (1 - previous_a) / (1 - a)
        noise = (
            torch.randn(
                (len(b) // repeats, 256), generator=generator, device=device
            ).repeat(repeats, 1)
            if t
            else 0
        )
        current = (mean + variance.sqrt() * noise - guidance * gradient).detach()
        if trace is not None:
            trace.append(
                {
                    "diffusion_step": 200 - t,
                    "clean_estimate_rmse_uv": float(
                        (
                            (clean.detach() - truth).square().mean(1).sqrt()
                            * AMPLITUDE_SCALE_UV
                        ).mean()
                    ),
                    "normalized_eeg_residual": float(loss.detach().mean()),
                }
            )
    result = current.cpu().numpy()
    if not np.isfinite(result).all():
        raise RuntimeError("Nonfinite reconstruction")
    return result


def ridge(obs, coefficient):
    A, b = obs["A"], obs["b"]
    penalty = coefficient * A.diagonal(dim1=1, dim2=2).mean(1)
    return torch.linalg.solve(A + penalty[:, None, None] * torch.eye(256), b).numpy()


def fit(output, data, pipeline, seed, run, device):
    folder = output / "models" / f"{pipeline}_cnn_{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "step_1000.pt").exists():
        return folder
    torch.manual_seed(seed)
    x = torch.tensor(np.load(data / f"{pipeline}.npz")["responses"], device=device)
    model = make_model("cnn").to(device)
    ema = copy.deepcopy(model).eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    schedule = cosine_schedule().to(device)
    validation = torch.tensor(
        np.stack([r[1] for r in targets(data, "validation", 64)]), device=device
    )
    generator = torch.Generator(device=device).manual_seed(173)
    vt = torch.randint(200, (len(validation),), generator=generator, device=device)
    vn = torch.randn(validation.shape, generator=generator, device=device)
    vi, target_v = make_training_pair(validation, vt, vn, schedule)
    rows, recent = [], []
    start = time.monotonic()
    for update in range(1, 1001):
        clean = x[torch.randint(len(x), (32,), device=device)]
        t = torch.randint(200, (32,), device=device)
        noisy, target = make_training_pair(clean, t, torch.randn_like(clean), schedule)
        loss = (model(noisy, t) - target).square().mean()
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite training loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        with torch.no_grad():
            for pe, p in zip(ema.parameters(), model.parameters()):
                pe.lerp_(p, 0.005)
        recent.append(loss.item())
        if update % 200 == 0:
            with torch.no_grad():
                vl = (ema(vi, vt) - target_v).square().mean().item()
            row = dict(
                update=update,
                train_v_mse=float(np.mean(recent)),
                validation_v_mse=vl,
                seconds=time.monotonic() - start,
            )
            rows.append(row)
            save_rows(folder / "training.csv", rows)
            run.log(
                {
                    "train/update": update,
                    "train/v_mse": row["train_v_mse"],
                    "train/validation_v_mse": vl,
                }
            )
            recent = []
        if update in CHECKPOINTS:
            torch.save(
                {"model": ema.state_dict(), "architecture": "cnn", "step": update},
                folder / f"step_{update}.pt",
            )
            print(f"{pipeline} seed={seed} trained update={update}", flush=True)
    return folder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("runs/reconstruction_focus")
    )
    parser.add_argument("--data", type=Path, default=Path("runs/comparison_16k/data"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    output, data = args.output.resolve(), args.data.resolve()
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    config = dict(
        pipelines=PIPELINES,
        train_seeds=SEEDS,
        checkpoints=CHECKPOINTS,
        guidances=GUIDANCES,
        sampling_seeds=SAMPLE_SEEDS,
        validation_targets_per_domain=4,
        test_targets_per_domain=8,
        batch_size=32,
        learning_rate=0.0002,
        ema=0.005,
        device=args.device,
        data=str(data),
        torch=torch.__version__,
    )
    config = json.loads(json.dumps(config))
    config_path = output / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Output configuration differs; use a new --output directory")
    config_path.write_text(json.dumps(config, indent=2))
    environment = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0) if args.device == "cuda" else "CPU",
    }
    (output / "environment.json").write_text(json.dumps(environment, indent=2))
    (output / "source_sha256.json").write_text(
        json.dumps(
            {
                str(p.relative_to(HERE.parent)): hashlib.sha256(
                    p.read_bytes()
                ).hexdigest()
                for p in [
                    Path(__file__),
                    HERE / "study.py",
                    HERE.parent / "model.py",
                    HERE.parent / "diffusion.py",
                ]
            },
            indent=2,
        )
    )
    (output / "STATUS.txt").write_text("RUNNING\n")
    val = observations(data, "validation", 4)
    # Select references by subject before scoring, independent of model quality.
    full_test = observations(data, "test", 32)
    indices = []
    for domain, subject, count in (
        ("analytic", 0, 8),
        ("earndb", 7, 4),
        ("earndb", 8, 4),
    ):
        records = list(
            dict.fromkeys(
                r["record"]
                for r in full_test["rows"]
                if r["domain"] == domain and r["subject"] == subject
            )
        )[:count]
        if len(records) != count:
            raise ValueError(f"Insufficient references for {domain}, subject {subject}")
        indices.extend(
            i
            for i, r in enumerate(full_test["rows"])
            if r["domain"] == domain
            and r["subject"] == subject
            and r["record"] in records
        )
    test = {
        k: ([v[i] for i in indices] if k == "rows" else v[indices])
        for k, v in full_test.items()
    }
    save_rows(output / "validation_observations.csv", val["rows"])
    save_rows(output / "test_observations.csv", test["rows"])
    ridge_scores = [
        (c, float(metrics(ridge(val, c), val["truth"])[0].mean()))
        for c in (1e-4, 1e-3, 0.01, 0.1, 1.0, 10.0)
    ]
    ridge_c = min(ridge_scores, key=lambda x: x[1])[0]
    save_rows(
        output / "ridge_validation.csv",
        [dict(coefficient=c, rmse_uv=s) for c, s in ridge_scores],
    )
    ridge_pred = ridge(test, ridge_c)
    np.savez_compressed(
        output / "references.npz", truth=test["truth"], ridge=ridge_pred
    )
    rmse, corr, latency = metrics(ridge_pred, test["truth"])
    save_rows(
        output / "ridge_test.csv",
        [
            dict(
                **r,
                rmse_uv=float(rmse[i]),
                correlation=float(corr[i]),
                peak_error_ms=float(latency[i]),
                coefficient=ridge_c,
            )
            for i, r in enumerate(test["rows"])
        ],
    )
    for pipeline in PIPELINES:
        for seed in SEEDS:
            name = f"{pipeline}_cnn_{seed}"
            if (output / name / "COMPLETE").exists():
                continue
            id_path = output / "wandb_run_ids.json"
            run_ids = json.loads(id_path.read_text()) if id_path.exists() else {}
            if name not in run_ids:
                run_ids[name] = uuid.uuid4().hex[:8]
                id_path.write_text(json.dumps(run_ids, indent=2))
            run = wandb.init(
                id=run_ids[name],
                resume="allow",
                entity="abr-deconv",
                project="abr-deconv-reconstruction",
                group=output.name,
                name=name,
                config={**config, "pipeline": pipeline, "seed": seed},
                settings=wandb.Settings(disable_code=True),
            )
            run.define_metric("train/*", step_metric="train/update")
            run.define_metric("sampling/*", step_metric="sampling/step")
            folder = fit(output, data, pipeline, seed, run, args.device)
            dest = output / name
            dest.mkdir(exist_ok=True)
            validation_rows = []
            if (dest / "validation.csv").exists():
                with (dest / "validation.csv").open() as f:
                    validation_rows = [
                        dict(
                            checkpoint=int(r["checkpoint"]),
                            guidance=float(r["guidance"]),
                            rmse_uv=float(r["rmse_uv"]),
                        )
                        for r in csv.DictReader(f)
                    ]
            for checkpoint in CHECKPOINTS:
                model = make_model("cnn").to(args.device).eval()
                model.load_state_dict(
                    torch.load(
                        folder / f"step_{checkpoint}.pt",
                        weights_only=True,
                        map_location=args.device,
                    )["model"]
                )
                for p in model.parameters():
                    p.requires_grad_(False)
                missing = [
                    g
                    for g in GUIDANCES
                    if not any(
                        r["checkpoint"] == checkpoint and r["guidance"] == g
                        for r in validation_rows
                    )
                ]
                if not missing:
                    continue
                batch = {
                    k: (
                        v * len(missing)
                        if k == "rows"
                        else np.tile(v, (len(missing), 1))
                        if k == "truth"
                        else v.repeat((len(missing),) + (1,) * (v.ndim - 1))
                    )
                    for k, v in val.items()
                }
                strengths = torch.tensor(missing, device=args.device).repeat_interleave(
                    len(val["rows"])
                )[:, None]
                predictions = reconstruct(model, batch, strengths, repeats=len(missing))
                for guidance, prediction in zip(
                    missing, np.split(predictions, len(missing))
                ):
                    score = float(metrics(prediction, val["truth"])[0].mean())
                    validation_rows.append(
                        dict(checkpoint=checkpoint, guidance=guidance, rmse_uv=score)
                    )
                    run.log(
                        {
                            "validation/checkpoint": checkpoint,
                            "validation/guidance": guidance,
                            "validation/reconstruction_rmse_uv": score,
                        }
                    )
                    save_rows(dest / "validation.csv", validation_rows)
                    print(
                        f"{name} checkpoint={checkpoint} guidance={guidance}: validation RMSE={score:.5f}",
                        flush=True,
                    )
            selected = min(validation_rows, key=lambda r: r["rmse_uv"])
            model.load_state_dict(
                torch.load(
                    folder / f"step_{selected['checkpoint']}.pt",
                    weights_only=True,
                    map_location=args.device,
                )["model"]
            )
            test_rows, traces, predictions = [], [], []
            for sample_seed in SAMPLE_SEEDS:
                trace = []
                prediction = reconstruct(
                    model, test, selected["guidance"], sample_seed, trace
                )
                predictions.append(prediction)
                rmse, corr, latency = metrics(prediction, test["truth"])
                test_rows.extend(
                    dict(
                        **r,
                        sampling_seed=sample_seed,
                        rmse_uv=float(rmse[i]),
                        correlation=float(corr[i]),
                        peak_error_ms=float(latency[i]),
                    )
                    for i, r in enumerate(test["rows"])
                )
                traces.extend(dict(**r, sampling_seed=sample_seed) for r in trace)
                for row in trace:
                    run.log(
                        {
                            "sampling/seed": sample_seed,
                            "sampling/step": row["diffusion_step"],
                            "sampling/clean_estimate_rmse_uv": row[
                                "clean_estimate_rmse_uv"
                            ],
                            "sampling/normalized_eeg_residual": row[
                                "normalized_eeg_residual"
                            ],
                        }
                    )
            save_rows(dest / "test.csv", test_rows)
            save_rows(dest / "diffusion_trace.csv", traces)
            np.savez_compressed(
                dest / "predictions.npz", predictions=np.array(predictions)
            )
            (dest / "selection.json").write_text(json.dumps(selected, indent=2))
            run.summary["test/reconstruction_rmse_uv"] = float(
                np.mean([r["rmse_uv"] for r in test_rows])
            )
            run.summary["test/ridge_rmse_uv"] = float(
                metrics(ridge_pred, test["truth"])[0].mean()
            )
            run.summary["selected_checkpoint"] = selected["checkpoint"]
            run.summary["selected_guidance"] = selected["guidance"]
            run.finish()
            (dest / "COMPLETE").touch()
    from reconstruction_report import produce, upload_report

    produce(output)
    upload_report(output)
    (output / "STATUS.txt").write_text("COMPLETE\n")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        folder = (
            Path(sys.argv[sys.argv.index("--output") + 1])
            if "--output" in sys.argv
            else Path("runs/reconstruction_focus")
        )
        if folder.exists():
            (folder / "STATUS.txt").write_text("FAILED — see process log\n")
        raise
