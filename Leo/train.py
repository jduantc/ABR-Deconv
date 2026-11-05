"""Train one diffusion denoiser on the saved synthetic ABRs."""

import argparse
import copy
import csv
import json
from pathlib import Path

import numpy as np
import torch

from data import AMPLITUDE_SCALE_UV, ROOT, WAVEFORM_LENGTH, generate_responses
from diffusion import DIFFUSION_STEPS, cosine_schedule, make_training_pair
from model import Denoiser


def train(args):
    device = torch.device(args.device)
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)

    with np.load(args.data) as saved:
        training = np.array(saved["responses"], dtype=np.float32)
        data_seed = int(saved["seed"])
    if training.ndim != 2 or training.shape[1] != WAVEFORM_LENGTH:
        raise ValueError("Training data must have shape (number of responses, 256).")
    if len(training) == 0 or not np.isfinite(training).all():
        raise ValueError("Training data must be nonempty and finite.")

    output = args.output or ROOT / "runs" / str(args.seed)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "model.pt").exists() or (output / "losses.csv").exists():
        raise FileExistsError(f"Choose a new --output folder; {output} already has a run.")

    settings = {
        "seed": args.seed,
        "data_seed": data_seed,
        "training_count": len(training),
        "validation_seed": args.seed + 1000,
        "validation_count": args.validation_size,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "diffusion_steps": DIFFUSION_STEPS,
        "learning_rate": 2e-4,
        "weight_decay": 1e-4,
        "ema_update_rate": 0.005,
        "prediction": "v",
        "amplitude_scale_uV": AMPLITUDE_SCALE_UV,
        "device": str(device),
        "data_file": str(args.data.resolve()),
    }
    (output / "settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")

    clean_training = torch.tensor(training, device=device)
    model = Denoiser().to(device)
    ema_model = copy.deepcopy(model).eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    alpha_bar = cosine_schedule().to(device)

    # Keep validation waveforms, noise, and steps fixed to compare checkpoints.
    validation, _ = generate_responses(args.validation_size, args.seed + 1000)
    validation = torch.tensor(validation, device=device)
    generator = torch.Generator(device=device).manual_seed(args.seed + 2000)
    validation_steps = torch.randint(
        DIFFUSION_STEPS, (len(validation),), device=device, generator=generator
    )
    validation_noise = torch.randn(validation.shape, device=device, generator=generator)
    validation_input, validation_target = make_training_pair(
        validation, validation_steps, validation_noise, alpha_bar
    )

    best_loss = float("inf")
    recent_losses = []
    print(f"Training on {device}: {len(training)} waveforms, {args.steps} updates.", flush=True)
    with (output / "losses.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["step", "training_v_mse", "validation_v_mse"])
        writer.writeheader()
        for step in range(1, args.steps + 1):
            indices = torch.randint(len(clean_training), (args.batch_size,), device=device)
            clean = clean_training[indices]
            timesteps = torch.randint(DIFFUSION_STEPS, (len(clean),), device=device)
            noise = torch.randn_like(clean)
            noisy, target = make_training_pair(clean, timesteps, noise, alpha_bar)

            prediction = model(noisy, timesteps)
            loss = (prediction - target).square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()

            # EMA weights provide a smoother copy for validation and saving.
            with torch.no_grad():
                for ema_parameter, parameter in zip(ema_model.parameters(), model.parameters()):
                    ema_parameter.lerp_(parameter, 0.005)
            recent_losses.append(loss.item())

            if step % args.log_every == 0 or step == args.steps:
                with torch.no_grad():
                    predicted = ema_model(validation_input, validation_steps)
                    validation_loss = (predicted - validation_target).square().mean().item()
                row = {
                    "step": step,
                    "training_v_mse": sum(recent_losses) / len(recent_losses),
                    "validation_v_mse": validation_loss,
                }
                writer.writerow(row)
                file.flush()
                print(
                    f"Step {step}: train={row['training_v_mse']:.5f}, "
                    f"validation={validation_loss:.5f}", flush=True
                )
                recent_losses.clear()
                if validation_loss < best_loss:
                    best_loss = validation_loss
                    torch.save({
                        "model": ema_model.state_dict(),
                        "step": step,
                        "seed": args.seed,
                        "validation_v_mse": validation_loss,
                        "config": settings,
                    }, output / "model.pt")

    print(f"Saved best EMA model and training log in {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "train.npz")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=20261029)
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--validation-size", type=int, default=512)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--device", choices=["cpu", "cuda"],
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.validation_size, args.log_every) < 1:
        parser.error("Step counts, batch size, and validation size must be positive.")
    if not args.data.exists():
        parser.error(f"Training data not found: {args.data}. Run python data.py first.")
    train(args)


if __name__ == "__main__":
    main()
