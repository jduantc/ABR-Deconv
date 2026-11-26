"""Continue the controlled MLP trajectory to Phoom's 10-million-update budget.

The starting 50k checkpoint contains raw weights, both EMA shadows, optimizer,
and sampling RNG state. Uses independent validation, interim and final holdouts.
"""

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import argparse
import copy
import csv
import json
from pathlib import Path
import time

import numpy as np
import torch
import wandb

from phoom_improvements import (
    BASE_GUIDANCE,
    EMA_RATES,
    NOISE_UV,
    ROOT,
    model_for,
    recordings,
    refine,
    rmse,
    sample,
    stats,
    waveforms_uv,
    cosine_schedule,
    make_training_pair,
)

CHECKPOINTS = (50000, 100000, 250000, 500000, 1000000, 2000000, 5000000, 10000000)
PROJECT = "abr-deconv-phoom-matched-training"


def save_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def restore(state, device):
    model = model_for("mlp", device)
    model.load_state_dict(state["model"])
    emas = {r: copy.deepcopy(model).eval().requires_grad_(False) for r in EMA_RATES}
    for rate in EMA_RATES:
        emas[rate].load_state_dict(state["emas"][str(rate)])
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.0)
    optimizer.load_state_dict(state["optimizer"])
    generator = torch.Generator(device=device)
    generator.set_state(state["generator"].cpu())
    return model, emas, optimizer, generator


def update_once(model, emas, optimizer, generator, x, schedule):
    clean = x[torch.randint(len(x), (256,), generator=generator, device=x.device)]
    u = torch.rand(256, generator=generator, device=x.device)
    noise = torch.randn(clean.shape, generator=generator, device=x.device)
    t = (u * 200).long().clamp(max=199)
    noisy, target = make_training_pair(clean, t, noise, schedule)
    loss = (model(noisy, t) - target).square().mean()
    if not torch.isfinite(loss):
        raise RuntimeError("Nonfinite training loss")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    with torch.no_grad():
        for rate, ema in emas.items():
            for pe, parameter in zip(ema.parameters(), model.parameters()):
                pe.lerp_(parameter, rate)
    return float(loss.detach())


def save_state(path, model, emas, optimizer, generator, update, training_rows):
    state = {
        "model": model.state_dict(),
        "emas": {str(r): m.state_dict() for r, m in emas.items()},
        "optimizer": optimizer.state_dict(),
        "generator": generator.get_state(),
        "update": update,
        "training_rows": training_rows,
    }
    temp = path.with_suffix(".tmp")
    torch.save(state, temp)
    temp.replace(path)


def test_horizon(
    output, run, emas, scale, covariance, horizon, split, wave_seed, noise_seed, count
):
    """All settings fixed before test; compare 32-sample means case by case."""
    obs = recordings(count, wave_seed, noise_seed)
    if len(obs["truth"]) != 3 * count:
        raise RuntimeError("Insufficient non-null test waveforms")
    device = next(emas[EMA_RATES[0]].parameters()).device
    sufficient = stats(obs, scale, device)
    strengths = np.repeat(BASE_GUIDANCE, count)
    rows = []
    predictions = {}
    truth = obs["truth"]
    for rate in EMA_RATES:
        draws = sample(emas[rate], sufficient, scale, strengths, samples=32, seed=7000)
        mean = draws.mean(1)
        for corrected in (False, True):
            estimate = refine(mean, obs, covariance, 0.1) if corrected else mean
            key = f"ema_{rate}" + ("_refined" if corrected else "")
            predictions[key] = estimate
            errors = rmse(estimate, truth)
            rows.extend(
                dict(
                    horizon=horizon,
                    split=split,
                    method=key,
                    **r,
                    rmse_uv=float(errors[i]),
                )
                for i, r in enumerate(obs["rows"])
            )
            for sigma in NOISE_UV:
                vals = [
                    float(errors[i])
                    for i, r in enumerate(obs["rows"])
                    if r["noise_uv"] == sigma
                ]
                run.log(
                    {
                        "test/update": horizon,
                        f"test/{split}/{key}/rmse_{round(sigma * 1000)}nV": float(
                            np.mean(vals)
                        ),
                    }
                )
    path = output / f"{split}_metrics.csv"
    save_csv(path, rows)
    np.savez_compressed(output / f"{split}_predictions.npz", truth=truth, **predictions)
    print(
        "TEST",
        horizon,
        split,
        {
            m: float(np.mean([r["rmse_uv"] for r in rows if r["method"] == m]))
            for m in predictions
        },
        flush=True,
    )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "Leo/runs/phoom_matched_10m"
    )
    parser.add_argument(
        "--start",
        type=Path,
        default=ROOT / "Leo/runs/phoom_long_50k/mlp/last_training.pt",
    )
    parser.add_argument("--steps", type=int, default=10000000)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    checkpoints = sorted(
        set([c for c in CHECKPOINTS if c <= args.steps] + [args.steps])
    )
    protocol = dict(
        steps=args.steps,
        checkpoints=checkpoints,
        train_seed=5000,
        initial_seed=0,
        training_count=20000,
        batch_size=256,
        ema_rates=list(EMA_RATES),
        guidance=BASE_GUIDANCE.tolist(),
        refinement=0.1,
        validation_waveform_seed=90426,
        validation_noise_seed=5001,
        validation_count=24,
        validation_draws=8,
        interim_waveform_seed=90440,
        interim_noise_seed=5501,
        final_waveform_seed=90441,
        final_noise_seed=5502,
        test_count=24,
        test_draws=32,
        note="Fixed EMA/guidance/correction comparisons; no test-driven hyperparameter selection.",
    )
    protocol_file = output / "protocol.json"
    if protocol_file.exists() and json.loads(protocol_file.read_text()) != protocol:
        raise ValueError("Output protocol differs; choose a new output directory")
    protocol_file.write_text(json.dumps(protocol, indent=2))
    (output / "STATUS.txt").write_text("RUNNING\n")
    state_path = output / "last_training.pt"
    source = state_path if state_path.exists() else args.start.resolve()
    if not source.exists():
        raise FileNotFoundError(f"Missing prior optimizer checkpoint: {source}")
    state = torch.load(source, map_location=args.device, weights_only=False)
    first = int(state["update"])
    if first < 50000:
        raise ValueError("The 50k trajectory must finish before continuation")
    x_uv = waveforms_uv(20000, 5000)
    scale = float(x_uv.std())
    covariance = np.cov(x_uv.T)
    x = torch.tensor(x_uv / scale, dtype=torch.float32, device=args.device)
    model, emas, optimizer, generator = restore(state, args.device)
    schedule = cosine_schedule().to(args.device)
    val = recordings(
        24, protocol["validation_waveform_seed"], protocol["validation_noise_seed"]
    )
    if len(val["truth"]) != 72:
        raise RuntimeError("Insufficient non-null validation waveforms")
    val_stats = stats(val, scale, args.device)
    guidance = np.repeat(BASE_GUIDANCE, 24)
    validation_rows = [
        dict(
            update=int(r["update"]),
            ema_rate=float(r["ema_rate"]),
            rmse_uv=float(r["rmse_uv"]),
        )
        for r in read_csv(output / "validation.csv")
    ]
    training_rows = state.get("training_rows", [])
    run_id_file = output / "wandb_run_id.txt"
    run_id = (
        run_id_file.read_text().strip()
        if run_id_file.exists()
        else wandb.util.generate_id()
        if hasattr(wandb.util, "generate_id")
        else os.urandom(8).hex()
    )
    run_id_file.write_text(run_id)
    run = wandb.init(
        entity="abr-deconv",
        project=PROJECT,
        id=run_id,
        resume="allow",
        name="MLP · 10 million updates",
        config=protocol,
        settings=wandb.Settings(disable_code=True),
    )
    run.define_metric("train/*", step_metric="train/update")
    run.define_metric("validation/*", step_metric="validation/update")
    run.define_metric("test/*", step_metric="test/update")

    def assess(update, evaluated_emas=None):
        evaluated_emas = emas if evaluated_emas is None else evaluated_emas
        values = {}
        for rate in EMA_RATES:
            existing = [
                r
                for r in validation_rows
                if r["update"] == update and r["ema_rate"] == rate
            ]
            if existing:
                score = existing[0]["rmse_uv"]
            else:
                prediction = sample(
                    evaluated_emas[rate], val_stats, scale, guidance, samples=8
                ).mean(1)
                score = float(rmse(prediction, val["truth"]).mean())
                validation_rows.append(
                    dict(update=update, ema_rate=rate, rmse_uv=score)
                )
                save_csv(output / "validation.csv", validation_rows)
            values[f"validation/rmse_ema_{str(rate).replace('.', '_')}_uv"] = score
        run.log({"validation/update": update, **values})
        print("VALIDATION", update, values, flush=True)

    old_files = {
        r: args.start.resolve().parent / f"step_4000_ema_{r}.pt" for r in EMA_RATES
    }
    if first >= 50000 and all(p.exists() for p in old_files.values()):
        old_emas = {
            r: model_for("mlp", args.device).eval().requires_grad_(False)
            for r in EMA_RATES
        }
        for r in EMA_RATES:
            old_emas[r].load_state_dict(
                torch.load(old_files[r], map_location=args.device, weights_only=True)[
                    "state"
                ]
            )
        assess(4000, old_emas)
        del old_emas
    assess(first)
    recent = []
    start = time.monotonic()
    for update in range(first + 1, args.steps + 1):
        loss = update_once(model, emas, optimizer, generator, x, schedule)
        recent.append(loss)
        if update % 2000 == 0:
            row = dict(
                update=update,
                v_mse=float(np.mean(recent)),
                seconds=time.monotonic() - start,
            )
            training_rows.append(row)
            save_csv(output / "training.csv", training_rows)
            run.log({"train/update": update, "train/v_mse": row["v_mse"]})
            recent = []
            print("TRAIN", update, "loss", row["v_mse"], flush=True)
        if update in checkpoints:
            for rate, ema in emas.items():
                torch.save(
                    {
                        "state": ema.state_dict(),
                        "scale": scale,
                        "step": update,
                        "ema_rate": rate,
                    },
                    output / f"step_{update}_ema_{rate}.pt",
                )
            assess(update)
            save_state(
                state_path, model, emas, optimizer, generator, update, training_rows
            )
            if update == 1000000:
                test_horizon(
                    output,
                    run,
                    emas,
                    scale,
                    covariance,
                    update,
                    "interim_1m",
                    protocol["interim_waveform_seed"],
                    protocol["interim_noise_seed"],
                    protocol["test_count"],
                )
                from phoom_matched_report import report_interim

                report_interim(output, run)

    if (
        args.steps >= 1000000
        and not (output / "interim_1m_metrics.csv").exists()
        and first >= 1000000
    ):
        print(
            "No intermediate 1m evaluation was saved; it cannot be reconstructed from the later model.",
            flush=True,
        )
    test_horizon(
        output,
        run,
        emas,
        scale,
        covariance,
        args.steps,
        "final_fresh",
        protocol["final_waveform_seed"],
        protocol["final_noise_seed"],
        protocol["test_count"],
    )
    test_horizon(
        output,
        run,
        emas,
        scale,
        covariance,
        args.steps,
        "phoom_benchmark",
        90421,
        1000,
        12,
    )
    run.finish()
    from phoom_matched_report import report

    report(output, PROJECT)
    (output / "STATUS.txt").write_text("COMPLETE\n")


if __name__ == "__main__":
    main()
