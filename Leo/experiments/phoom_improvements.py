"""Matched-budget Phoom-style MLP vs Leo CNN, selected by reconstruction RMSE.

Run from the repository root: python Leo/experiments/phoom_improvements.py
"""

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import wandb

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from legacy_phoom.data.synthetic import NOISE_UV, test_recordings, waveforms_uv
from legacy_phoom.diffusion.schedule import (
    cosine_schedule,
    make_training_pair,
    clean_from_v,
)
from legacy_phoom.models.mlp import Denoiser
from legacy_phoom.models.cnn import ConvDenoiser
from legacy_phoom.diffusion.refinement import measurement_refinement
from fast_conv import use_linear_convolutions

PROJECT = "abr-deconv-phoom-improvements"
EMA_RATES = (0.001, 0.005)
GUIDANCE_GRID = (3.0, 10.0, 10**1.5, 100.0, 10**2.5)
BASE_GUIDANCE = np.array([10.0, 10**1.5, 100.0])


def save_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def recordings(count, waveform_seed, noise_seed):
    truth, matrix, eeg = test_recordings(count, waveform_seed, noise_seed)
    rows = [
        dict(record=i, noise_uv=sigma) for sigma in NOISE_UV for i in range(len(truth))
    ]
    return dict(
        truth=np.tile(truth, (3, 1)),
        matrix=matrix,
        eeg=np.concatenate([eeg[s] for s in NOISE_UV]),
        rows=rows,
    )


def stats(obs, scale, device):
    matrix, eeg = obs["matrix"], obs["eeg"]
    n = len(matrix)
    return dict(
        A=torch.tensor(
            scale**2 * matrix.T @ matrix / n, dtype=torch.float32, device=device
        ),
        b=torch.tensor(scale * eeg @ matrix / n, dtype=torch.float32, device=device),
        yy=torch.tensor(np.mean(eeg**2, axis=1), dtype=torch.float32, device=device),
        sigma=torch.tensor([r["noise_uv"] for r in obs["rows"]], device=device),
    )


def sample(model, sufficient, scale, guidance, samples=32, seed=0):
    """Phoom's 200-step MLP DPS, accelerated with exact residual statistics."""
    device = next(model.parameters()).device
    A = sufficient["A"]
    b = sufficient["b"].repeat_interleave(samples, dim=0)
    yy = sufficient["yy"].repeat_interleave(samples)
    sigma = sufficient["sigma"].repeat_interleave(samples)
    strengths = torch.as_tensor(
        guidance, device=device, dtype=torch.float32
    ).repeat_interleave(samples)[:, None]
    schedule = cosine_schedule().to(device)
    generator = torch.Generator(device=device).manual_seed(seed)
    current = torch.randn(len(b), 256, generator=generator, device=device)
    for t in reversed(range(200)):
        current = current.detach().requires_grad_(True)
        step = torch.full((len(b),), t, dtype=torch.long, device=device)
        clean = clean_from_v(current, model(current, step), step, schedule)
        residual2 = (
            (clean @ A * clean).sum(1) - 2 * (clean * b).sum(1) + yy
        ).clamp_min(1e-12)
        loss = residual2.sqrt() / sigma
        gradient = torch.autograd.grad(loss.sum(), current)[0]
        a = schedule[t]
        previous = schedule[t - 1] if t else torch.ones((), device=device)
        alpha = a / previous
        beta = 1 - alpha
        mean = previous.sqrt() * beta / (1 - a) * clean.detach()
        mean += alpha.sqrt() * (1 - previous) / (1 - a) * current.detach()
        variance = beta * (1 - previous) / (1 - a)
        noise = (
            torch.randn(current.shape, generator=generator, device=device) if t else 0
        )
        current = mean + variance.sqrt() * noise - strengths * gradient
    result = (current.detach() * scale).view(-1, samples, 256).cpu().numpy()
    if not np.isfinite(result).all():
        raise RuntimeError("Nonfinite DPS output")
    return result


def rmse(prediction, truth):
    return np.sqrt(np.mean((prediction - truth) ** 2, axis=-1))


def model_for(name, device):
    return (
        Denoiser() if name == "mlp" else use_linear_convolutions(ConvDenoiser())
    ).to(device)


def refine(estimate, obs, covariance, strength):
    if strength == 0:
        return estimate.copy()
    return measurement_refinement(
        estimate,
        obs["eeg"],
        obs["matrix"],
        [r["noise_uv"] for r in obs["rows"]],
        covariance,
        strength,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=ROOT / "Leo/runs/phoom_improvements")
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--architectures", nargs="+", choices=["mlp", "cnn"], default=["mlp", "cnn"]
    )
    p.add_argument("--checkpoints", nargs="+", type=int)
    p.add_argument("--project", default=PROJECT)
    p.add_argument("--validation-waveform-seed", type=int, default=90422)
    p.add_argument("--validation-noise-seed", type=int, default=2000)
    p.add_argument("--fresh-waveform-seed", type=int, default=90423)
    p.add_argument("--fresh-noise-seed", type=int, default=3000)
    args = p.parse_args()
    if args.steps <= 2000:
        raise ValueError(
            "Use --steps greater than 2000 for the three-checkpoint comparison"
        )
    checkpoints = sorted(set(args.checkpoints or [1000, 2000, args.steps]))
    if checkpoints[-1] != args.steps or checkpoints[0] < 1:
        raise ValueError("Checkpoints must be positive and end at --steps")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    config = dict(
        steps=args.steps,
        batch=256,
        train_count=20000,
        train_waveform_seed=5000,
        seed=0,
        checkpoints=checkpoints,
        architectures=args.architectures,
        project=args.project,
        validation_waveform_seed=args.validation_waveform_seed,
        validation_noise_seed=args.validation_noise_seed,
        fresh_waveform_seed=args.fresh_waveform_seed,
        fresh_noise_seed=args.fresh_noise_seed,
        ema_rates=EMA_RATES,
        validation_waveforms=6,
        validation_samples=8,
        test_waveforms=12,
        test_samples=32,
        guidance_grid=GUIDANCE_GRID,
        noise_uv=NOISE_UV,
        device=args.device,
        selection="validation only; test RMSE never used to choose settings",
    )
    config = json.loads(json.dumps(config))
    protocol_path = out / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != config:
        raise ValueError(
            "Existing output has a different protocol; choose a new --output"
        )
    protocol_path.write_text(json.dumps(config, indent=2))
    (out / "STATUS.txt").write_text("RUNNING\n")
    hashes = {
        str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest()
        for f in [
            Path(__file__),
            ROOT / "Leo/experiments/legacy_phoom/diffusion/dps.py",
            ROOT / "Leo/experiments/legacy_phoom/diffusion/train.py",
            ROOT / "Leo/experiments/legacy_phoom/data/synthetic.py",
            ROOT / "Leo/experiments/legacy_phoom/models/cnn.py",
        ]
    }
    (out / "source_sha256.json").write_text(json.dumps(hashes, indent=2))
    train_uv = waveforms_uv(20000, 5000)
    scale = float(train_uv.std())
    covariance = np.cov(train_uv.T)
    np.savez_compressed(
        out / "prior_statistics.npz",
        mean=train_uv.mean(0),
        covariance=covariance,
        scale=scale,
    )
    x = torch.tensor(train_uv / scale, dtype=torch.float32, device=args.device)
    val = recordings(6, args.validation_waveform_seed, args.validation_noise_seed)
    val_stats = stats(val, scale, args.device)
    baseline_strengths = np.repeat(BASE_GUIDANCE, 6)
    selection_path = out / "selection.json"
    selections = (
        json.loads(selection_path.read_text()) if selection_path.exists() else {}
    )
    for saved in selections.values():
        for key in ("guidance", "refinement"):
            saved[key] = {float(k): v for k, v in saved[key].items()}
    for architecture in args.architectures:
        if architecture in selections:
            continue
        dest = out / architecture
        dest.mkdir(exist_ok=True)
        run = wandb.init(
            entity="abr-deconv",
            project=args.project,
            group=out.name,
            name=architecture,
            config={**config, "architecture": architecture, "scale_uv": scale},
            settings=wandb.Settings(disable_code=True),
        )
        run.define_metric("train/*", step_metric="train/update")
        torch.manual_seed(0)
        model = model_for(architecture, args.device)
        emas = {
            rate: copy.deepcopy(model).eval().requires_grad_(False)
            for rate in EMA_RATES
        }
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.0)
        generator = torch.Generator(device=args.device).manual_seed(1000)
        schedule = cosine_schedule().to(args.device)
        training_rows = []
        validation_rows = []
        recent = []
        start = time.monotonic()
        first = 1
        resume_file = dest / "last_training.pt"
        if resume_file.exists():
            state = torch.load(
                resume_file, weights_only=False, map_location=args.device
            )
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            generator.set_state(state["generator"].cpu())
            for rate in EMA_RATES:
                emas[rate].load_state_dict(state["emas"][str(rate)])
            first = state["update"] + 1
            training_rows = state["training_rows"]
            validation_rows = state["validation_rows"]
        for update in range(first, args.steps + 1):
            optimizer.param_groups[0]["lr"] = 2e-4 * min(1.0, update / 500)
            clean = x[
                torch.randint(len(x), (256,), generator=generator, device=args.device)
            ]
            u = torch.rand(256, generator=generator, device=args.device)
            noise = torch.randn(clean.shape, generator=generator, device=args.device)
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
            recent.append(loss.item())
            if update % 200 == 0:
                row = dict(
                    update=update,
                    v_mse=float(np.mean(recent)),
                    seconds=time.monotonic() - start,
                )
                training_rows.append(row)
                save_csv(dest / "training.csv", training_rows)
                run.log({"train/update": update, "train/v_mse": row["v_mse"]})
                recent = []
                print(architecture, "trained", update, flush=True)
            if update in config["checkpoints"]:
                for rate, ema in emas.items():
                    torch.save(
                        {
                            "state": ema.state_dict(),
                            "scale": scale,
                            "step": update,
                            "ema_rate": rate,
                        },
                        dest / f"step_{update}_ema_{rate}.pt",
                    )
                    pred = sample(
                        ema, val_stats, scale, baseline_strengths, samples=8
                    ).mean(1)
                    score = float(rmse(pred, val["truth"]).mean())
                    validation_rows.append(
                        dict(checkpoint=update, ema_rate=rate, rmse_uv=score)
                    )
                    save_csv(dest / "checkpoint_validation.csv", validation_rows)
                    run.log(
                        {
                            "validation/checkpoint": update,
                            "validation/ema": rate,
                            "validation/reconstruction_rmse_uv": score,
                        }
                    )
                    print(architecture, "validation", update, rate, score, flush=True)
                state = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "generator": generator.get_state(),
                    "emas": {str(r): m.state_dict() for r, m in emas.items()},
                    "update": update,
                    "training_rows": training_rows,
                    "validation_rows": validation_rows,
                }
                temporary = dest / "last_training.tmp"
                torch.save(state, temporary)
                temporary.replace(resume_file)
        selected = min(validation_rows, key=lambda r: r["rmse_uv"])
        selected_model = (
            model_for(architecture, args.device).eval().requires_grad_(False)
        )
        selected_model.load_state_dict(
            torch.load(
                dest / f"step_{selected['checkpoint']}_ema_{selected['ema_rate']}.pt",
                weights_only=True,
            )["state"]
        )
        guidance_rows = []
        validation_predictions = {}
        for guidance in GUIDANCE_GRID:
            draws = sample(
                selected_model,
                val_stats,
                scale,
                np.full(len(val["rows"]), guidance),
                samples=8,
            )
            validation_predictions[guidance] = draws.mean(1)
            error = rmse(draws.mean(1), val["truth"])
            for sigma in NOISE_UV:
                idx = [i for i, r in enumerate(val["rows"]) if r["noise_uv"] == sigma]
                guidance_rows.append(
                    dict(
                        guidance=guidance,
                        noise_uv=sigma,
                        rmse_uv=float(error[idx].mean()),
                    )
                )
        save_csv(dest / "guidance_validation.csv", guidance_rows)
        strengths = {
            sigma: min(
                (r for r in guidance_rows if r["noise_uv"] == sigma),
                key=lambda r: r["rmse_uv"],
            )["guidance"]
            for sigma in NOISE_UV
        }
        selected_prediction = np.stack(
            [
                validation_predictions[strengths[r["noise_uv"]]][i]
                for i, r in enumerate(val["rows"])
            ]
        )
        refinement_rows = []
        for strength in (0.0, 0.01, 0.1, 1.0, 10.0):
            estimate = refine(selected_prediction, val, covariance, strength)
            error = rmse(estimate, val["truth"])
            for sigma in NOISE_UV:
                mask = [r["noise_uv"] == sigma for r in val["rows"]]
                refinement_rows.append(
                    dict(
                        strength=strength,
                        noise_uv=sigma,
                        rmse_uv=float(error[mask].mean()),
                    )
                )
        save_csv(dest / "refinement_validation.csv", refinement_rows)
        refinements = {
            sigma: min(
                (r for r in refinement_rows if r["noise_uv"] == sigma),
                key=lambda r: r["rmse_uv"],
            )["strength"]
            for sigma in NOISE_UV
        }
        selections[architecture] = dict(
            **selected, guidance=strengths, refinement=refinements
        )
        (out / "selection.json").write_text(json.dumps(selections, indent=2))
        run.summary["validation/selected_rmse_uv"] = selected["rmse_uv"]
        run.finish()
    # Freeze all choices before constructing or scoring either test set.
    (out / "SELECTION_FROZEN").touch()
    results = []
    examples = {}
    for split, wave_seed, noise_seed in [
        ("phoom_benchmark", 90421, 1000),
        ("fresh_holdout", args.fresh_waveform_seed, args.fresh_noise_seed),
    ]:
        obs = recordings(12, wave_seed, noise_seed)
        sufficient = stats(obs, scale, args.device)
        np.savez_compressed(out / f"{split}_truth.npz", truth=obs["truth"])
        for architecture in args.architectures:
            dest = out / architecture
            sel = selections[architecture]
            recipes = [
                ("selected_fixed", sel["checkpoint"], sel["ema_rate"], BASE_GUIDANCE),
                (
                    "tuned",
                    sel["checkpoint"],
                    sel["ema_rate"],
                    [sel["guidance"][s] for s in NOISE_UV],
                ),
            ]
            if architecture == "mlp":
                recipes.insert(0, ("phoom_baseline", args.steps, 0.001, BASE_GUIDANCE))
                if args.architectures == ["mlp"]:
                    recipes.insert(
                        1, ("fast_ema_final", args.steps, 0.005, BASE_GUIDANCE)
                    )
            for label, checkpoint, ema_rate, strengths in recipes:
                model = (
                    model_for(architecture, args.device).eval().requires_grad_(False)
                )
                model.load_state_dict(
                    torch.load(
                        dest / f"step_{checkpoint}_ema_{ema_rate}.pt", weights_only=True
                    )["state"]
                )
                all_draws = []
                for i in range(0, len(obs["rows"]), len(obs["rows"])):
                    chunk = {
                        k: (v if k == "A" else v[i : i + len(obs["rows"])])
                        for k, v in sufficient.items()
                    }
                    g = np.repeat(np.asarray(strengths), 12)[i : i + len(obs["rows"])]
                    all_draws.append(
                        sample(model, chunk, scale, g, samples=32, seed=7000 + i)
                    )
                draws = np.concatenate(all_draws)
                estimate = draws.mean(1)
                method = architecture + "_" + label
                examples[split + "_" + method] = estimate
                np.savez_compressed(out / f"{split}_{method}_draws.npz", draws=draws)
                for n in (1, 4, 8, 16, 32):
                    errors = rmse(draws[:, :n].mean(1), obs["truth"])
                    results.extend(
                        dict(
                            split=split,
                            method=method,
                            samples=n,
                            **r,
                            rmse_uv=float(errors[i]),
                        )
                        for i, r in enumerate(obs["rows"])
                    )
                if label == "tuned":
                    refined = np.stack(
                        [
                            refine(
                                estimate[i : i + 1],
                                dict(
                                    matrix=obs["matrix"],
                                    eeg=obs["eeg"][i : i + 1],
                                    rows=[r],
                                ),
                                covariance,
                                sel["refinement"][r["noise_uv"]],
                            )[0]
                            for i, r in enumerate(obs["rows"])
                        ]
                    )
                    errors = rmse(refined, obs["truth"])
                    method = architecture + "_tuned_refined"
                    examples[split + "_" + method] = refined
                    results.extend(
                        dict(
                            split=split,
                            method=method,
                            samples=32,
                            **r,
                            rmse_uv=float(errors[i]),
                        )
                        for i, r in enumerate(obs["rows"])
                    )
                save_csv(out / "test_metrics.csv", results)
                print(
                    "TEST",
                    split,
                    architecture,
                    label,
                    float(rmse(estimate, obs["truth"]).mean()),
                    flush=True,
                )
    np.savez_compressed(out / "examples.npz", **examples)
    if args.architectures == ["mlp"]:
        from phoom_long_report import report
    else:
        from phoom_improvements_report import report
    report(out)
    (out / "STATUS.txt").write_text("COMPLETE\n")


if __name__ == "__main__":
    main()
