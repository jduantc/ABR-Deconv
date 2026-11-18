"""Train a denoiser for a fixed number of steps and log to wandb.

    python -m diffusion.train mlp --steps 2000000 --check-every 50000
    python -m diffusion.train unet --steps 400000 --check-every 10000
    python -m diffusion.train mlp --steps 10000000 --check-every 50000 --resume
"""

import argparse
import copy
import os
from pathlib import Path

import numpy as np
import torch
import wandb

from data.synthetic import waveforms_uv
from diffusion.dps import GUIDANCE, test_errors
from diffusion.schedule import cosine_schedule, predict_clean, schedule, training_loss
from models.mlp import Denoiser
from models.unet import UNet1D

SEED = 0
BATCH, LEARNING_RATE, WARMUP, EMA_RATE = 256, 2e-4, 500, 0.001
SHARED = 199 


def linear_error(train):
    eigenvalues = np.clip(np.linalg.eigvalsh(np.cov(train.T)), 0, None)
    a = cosine_schedule()[:SHARED].double().numpy()
    return np.array([np.mean(eigenvalues * s / (eigenvalues + s)) for s in (1 - a) / a])


def noisy_set(waveforms, seed):
    device = waveforms.device
    clean = waveforms.repeat(8, 1)
    level = torch.arange(len(clean), device=device) % SHARED
    a = cosine_schedule().to(device)[level][:, None]
    generator = torch.Generator(device=device).manual_seed(seed)
    noise = torch.randn(clean.shape, device=device, generator=generator)
    return clean, level, a.sqrt() * clean + (1 - a).sqrt() * noise


@torch.no_grad()
def validation_loss(network, model, noisy, linear, alpha_bar, levels):
    clean, level, inputs = noisy
    total = torch.zeros(SHARED, dtype=torch.float64, device=clean.device)
    for i in range(0, len(clean), 2000):
        rows = slice(i, i + 2000)
        estimate = predict_clean(network, model, inputs[rows], levels[level[rows]], alpha_bar)
        total.index_add_(0, level[rows], (estimate - clean[rows]).double().square().mean(1))
    error = (total / torch.bincount(level, minlength=SHARED)).cpu().numpy()
    return float(np.exp(np.mean(np.log(error / linear))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("network", choices=["mlp", "unet"])
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--check-every", type=int, required=True)
    parser.add_argument("--resume", action="store_true", help="continue from last.pt in the same wandb run")
    args = parser.parse_args()
    assert args.check_every % args.eval_every == 0
    network = args.network
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output = Path("runs") / network
    output.mkdir(parents=True, exist_ok=True)

    train_uv = waveforms_uv(20_000, 5000 + SEED)
    val_uv = waveforms_uv(2_000, 6000 + SEED)
    scale = float(train_uv.std())
    train = torch.tensor(train_uv / scale, dtype=torch.float32, device=device)
    val = torch.tensor(val_uv / scale, dtype=torch.float32, device=device)

    torch.manual_seed(SEED)
    model = (UNet1D() if network == "unet" else Denoiser()).to(device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.0)
    alpha_bar, levels = schedule(network)
    alpha_bar, levels = alpha_bar.to(device), levels.to(device)
    linear = linear_error(train_uv / scale)
    val_set = noisy_set(val, SEED + 2000)
    seen_set = noisy_set(train[:len(val)], SEED + 3000)
    batches = torch.Generator(device=device).manual_seed(SEED + 1000)

    first = 1
    if args.resume:
        last = torch.load(output / "last.pt", map_location=device)
        model.load_state_dict(last["net"])
        ema.load_state_dict(last["ema"])
        optimizer.load_state_dict(last["opt"])
        batches.set_state(last["draw"].cpu())
        first = last["step"] + 1
        run = wandb.init(entity="abr-deconv", project="diffusion-prior", id=last["wandb_id"], resume="must")
        run.config.update({"steps": args.steps}, allow_val_change=True)
    else:
        config = {"network": network, "steps": args.steps, "batch": BATCH, "lr": LEARNING_RATE, "warmup_steps": WARMUP,
                  "ema_rate": EMA_RATE, "n_train": len(train), "n_val": len(val), "train_std_uV": scale, "seed": SEED,
                  "parameters": sum(p.numel() for p in model.parameters()), "dps_guidance": GUIDANCE[network]}
        run = wandb.init(entity="abr-deconv", project="diffusion-prior", group="no early stopping",
                         name=f"{network} {args.steps} steps", config=config)

    loss_name = "train/eps_loss" if network == "unet" else "train/v_loss"
    running = torch.zeros((), device=device)
    for step in range(first, args.steps + 1):
        for group in optimizer.param_groups:
            group["lr"] = LEARNING_RATE * min(1.0, step / WARMUP)
        clean = train[torch.randint(len(train), (BATCH,), device=device, generator=batches)]
        u = torch.rand(BATCH, device=device, generator=batches)
        noise = torch.randn(clean.shape, device=device, generator=batches)
        timesteps = (u * len(alpha_bar)).long().clamp(max=len(alpha_bar) - 1)
        loss = training_loss(network, model, clean, timesteps, noise, alpha_bar)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        with torch.no_grad():
            for ema_parameter, parameter in zip(ema.parameters(), model.parameters()):
                ema_parameter.lerp_(parameter, EMA_RATE)
            running += loss.detach()
        if step % args.eval_every:
            continue

        log = {loss_name: running.item() / args.eval_every,
               "val/loss": validation_loss(network, ema, val_set, linear, alpha_bar, levels),
               "val/loss_on_training_waveforms": validation_loss(network, ema, seen_set, linear, alpha_bar, levels)}
        running.zero_()
        if step % args.check_every == 0:
            for noise_uv, error in test_errors(network, ema, scale).items():
                log[f"dps/rmse_{round(noise_uv * 1000)}nV_noise"] = error
            torch.save({"state": ema.state_dict(), "scale": scale, "recipe": network, "step": step},
                       output / f"step_{step}.pt")
            torch.save({"net": model.state_dict(), "ema": ema.state_dict(), "opt": optimizer.state_dict(), "step": step,
                        "draw": batches.get_state(), "scale": scale, "wandb_id": run.id}, output / "last.tmp")
            os.replace(output / "last.tmp", output / "last.pt")
            print(f"{network} step {step}  " + "  ".join(f"{key} {value:.4g}" for key, value in log.items()), flush=True)
        run.log(log, step=step)
    run.finish()


if __name__ == "__main__":
    main()
