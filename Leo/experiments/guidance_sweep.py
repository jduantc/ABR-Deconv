"""Two-stage unnormalized-guidance sweep for a frozen MLP checkpoint.

Run from the repository root. Results are saved incrementally; a test set is
constructed only after the validation choice has been frozen.
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("WANDB_BASE_URL", "https://api.forge.coreweave.com")
os.environ.setdefault("WANDB_INIT_TIMEOUT", "60")

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Leo"))
from legacy_phoom.data.synthetic import test_recordings, NOISE_UV
from legacy_phoom.models.mlp import Denoiser
from diffusion import cosine_schedule, clean_from_v

COARSE = [0., .1, .3, 1., 2., 3., 4., 6., 10., 20.]


def write_csv(path, rows):
    if rows:
        with path.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def load_model(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = Denoiser().to(device).eval().requires_grad_(False)
    model.load_state_dict(checkpoint['state'])
    return model, float(checkpoint['scale']), checkpoint


def make_observations(count, wave_seed, noise_seed, scale, device):
    truth, matrix, eeg = test_recordings(count, wave_seed, noise_seed)
    if len(truth) != count:
        raise RuntimeError(f'Expected {count} nonzero waveforms, got {len(truth)}')
    observations = np.concatenate([eeg[s] for s in NOISE_UV]) / scale
    # Compute the sufficient statistics in float64, then use float32 on the GPU.
    A = matrix.T @ matrix / len(matrix)
    b = observations @ matrix / len(matrix)
    yy = np.mean(observations ** 2, axis=1)
    return {
        'A': torch.tensor(A, dtype=torch.float32, device=device),
        'b': torch.tensor(b, dtype=torch.float32, device=device),
        'yy': torch.tensor(yy, dtype=torch.float32, device=device),
        'truth': np.tile(truth, (len(NOISE_UV), 1)),
        'matrix': matrix, 'eeg': eeg,
        'sigmas': np.repeat(NOISE_UV, count), 'count': count,
    }


def sample(model, stats, guidance, samples, seed, device):
    """Same update as Leo/dps/reconstruct.py, batched via residual statistics.

    EEG and output are in the model's waveform units. No noise-SD division.
    """
    A = stats['A']
    b = stats['b'].repeat_interleave(samples, 0)
    yy = stats['yy'].repeat_interleave(samples)
    schedule = cosine_schedule().to(device)
    gen = torch.Generator(device=device).manual_seed(seed)
    current = torch.randn((len(b), 256), generator=gen, device=device)
    for t in range(199, -1, -1):
        current = current.detach().requires_grad_(True)
        step = torch.full((len(b),), t, dtype=torch.long, device=device)
        clean = clean_from_v(current, model(current, step), step, schedule)
        residual_sq = ((clean @ A * clean).sum(1) - 2 * (clean * b).sum(1) + yy).clamp_min(1e-12)
        gradient = torch.autograd.grad(residual_sq.sqrt().sum(), current)[0]
        a = schedule[t]
        prev = schedule[t - 1] if t else torch.ones((), device=device)
        alpha = a / prev
        beta = 1 - alpha
        mean = prev.sqrt() * beta / (1-a) * clean.detach()
        mean += alpha.sqrt() * (1-prev) / (1-a) * current.detach()
        variance = beta * (1-prev) / (1-a)
        noise = torch.randn(current.shape, generator=gen, device=device) if t else 0
        current = mean + variance.sqrt() * noise - guidance * gradient
    result = current.detach().view(-1, samples, 256).cpu().numpy()
    if not np.isfinite(result).all():
        raise RuntimeError(f'Non-finite reconstruction for guidance {guidance}')
    return result


def errors(draws, truth, scale):
    pred = draws.mean(axis=1) * scale
    return np.sqrt(np.mean((pred - truth) ** 2, axis=1)), pred


def verify_equivalence(checkpoint):
    """Compare a full 200-step trajectory against the user's direct function."""
    from dps.reconstruct import reconstruct, stimulus_matrix
    torch.set_num_threads(1)
    model, scale, _ = load_model(checkpoint, 'cpu')
    pulses = np.zeros(500, np.float32)
    pulses[100:400:100] = 1
    A = stimulus_matrix(pulses).numpy().astype(np.float64)
    rng = np.random.default_rng(713)
    y = rng.normal(0, .5, len(pulses)).astype(np.float32)
    stats = {'A': torch.tensor(A.T @ A / len(y), dtype=torch.float32),
             'b': torch.tensor((y @ A / len(y))[None], dtype=torch.float32),
             'yy': torch.tensor([np.mean(y.astype(np.float64)**2)], dtype=torch.float32)}
    reference = reconstruct(model, y, pulses, guidance=1., seed=8)
    actual = sample(model, stats, 1., 1, 8, 'cpu')[0, 0]
    delta = float(np.max(np.abs(reference - actual)))
    if not np.allclose(reference, actual, atol=2e-4, rtol=2e-4):
        raise AssertionError(f'Batched sampler does not match standalone: max abs error {delta}')
    return delta


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, default=ROOT/'Leo/results/guidance_sweep')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--checkpoint', type=Path, default=ROOT/'Leo/runs/phoom_matched_10m/step_1000000_ema_0.005.pt')
    args = p.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out/'protocol.json').exists():
        raise FileExistsError('Output already exists; choose a new --output')
    started = time.monotonic()
    torch.set_num_threads(1)
    verification = verify_equivalence(args.checkpoint)
    model, scale, checkpoint = load_model(args.checkpoint, args.device)
    protocol = {'checkpoint': str(args.checkpoint), 'checkpoint_sha256': hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        'model_updates': int(checkpoint['step']), 'ema_rate': checkpoint['ema_rate'], 'scale_uv': scale,
        'coarse_grid': COARSE, 'validation_waveform_seed': 93231, 'validation_noise_seed': 700001,
        'test_waveform_seed': 93232, 'test_noise_seed': 700201,
        'validation_count': 32, 'test_count': 32, 'validation_samples': 8, 'test_samples': 32,
        'sampling_seed': 7193, 'noise_uv': NOISE_UV, 'device': args.device,
        'selection': 'single global guidance minimizing mean validation per-waveform RMSE across all 3 noise levels',
        'loss': 'RMS of EEG residual in model units; no noise standard deviation normalization',
        'sampler_max_abs_difference_cpu': verification}
    (out/'protocol.json').write_text(json.dumps(protocol, indent=2))
    (out/'STATUS.txt').write_text('RUNNING\n')
    run = None
    try:
        import wandb
        run = wandb.init(entity='abr-deconv', project='abr-deconv-guidance-sweep', name='unnormalized guidance · 1m MLP', config=protocol, settings=wandb.Settings(disable_code=True))
        print('FORGE', run.url, flush=True)
    except Exception as exc:
        print('FORGE_UNAVAILABLE', type(exc).__name__, str(exc), flush=True)
    validation = make_observations(32, 93231, 700001, scale, args.device)
    rows, records = [], []
    def evaluate(g, stage):
        t0 = time.monotonic()
        draws = sample(model, validation, g, 8, 7193, args.device)
        err, _ = errors(draws, validation['truth'], scale)
        row = {'stage': stage, 'guidance': g, 'mean_rmse_uv': float(err.mean()),
               **{f'rmse_{s}_uv': float(err[validation['sigmas']==s].mean()) for s in NOISE_UV},
               'seconds': time.monotonic()-t0}
        rows.append(row)
        records.extend({'stage': stage, 'guidance': g, 'noise_uv': float(s), 'record': i % 32, 'rmse_uv': float(e)}
                       for i, (s, e) in enumerate(zip(validation['sigmas'], err)))
        write_csv(out/'validation_grid.csv', rows)
        write_csv(out/'validation_per_record.csv', records)
        if run: run.log({'validation/guidance': g, 'validation/mean_rmse_uv': row['mean_rmse_uv'],
                         **{f'validation/rmse_{s}_uv': row[f'rmse_{s}_uv'] for s in NOISE_UV}})
        print('VALIDATION', json.dumps(row), flush=True)
    for g in COARSE:
        evaluate(g, 'coarse')
    best = min(rows, key=lambda r:r['mean_rmse_uv'])
    ix = COARSE.index(best['guidance'])
    lo = COARSE[max(0, ix-1)]
    hi = COARSE[min(len(COARSE)-1, ix+1)]
    if ix == len(COARSE)-1:
        hi *= 2
    fine = sorted(set(round(x, 6) for x in np.linspace(lo, hi, 9)) - set(COARSE))
    for g in fine:
        evaluate(g, 'fine')
    best = min(rows, key=lambda r:r['mean_rmse_uv'])
    selected = best['guidance']
    selection = {'guidance': selected, 'validation_mean_rmse_uv': best['mean_rmse_uv'],
                 'fine_grid': fine, 'boundary_warning': selected in (0., max(COARSE + fine))}
    (out/'selection.json').write_text(json.dumps(selection, indent=2))
    (out/'SELECTION_FROZEN').write_text('Selection fixed before test set construction.\n')
    print('SELECTION', json.dumps(selection), flush=True)
    test = make_observations(32, 93232, 700201, scale, args.device)
    test_rows, sample_rows, predictions = [], [], {'truth': test['truth']}
    for label, g in [('unguided', 0.), ('original_default', .1), ('selected', selected)]:
        draws = sample(model, test, g, 32, 8293, args.device)
        err, pred = errors(draws, test['truth'], scale)
        for count in (1, 4, 8, 16, 32):
            sample_err, _ = errors(draws[:, :count], test['truth'], scale)
            for i, (s, e) in enumerate(zip(test['sigmas'], sample_err)):
                sample_rows.append({'method':label, 'guidance':g, 'samples':count,
                                    'noise_uv':float(s), 'record':i % 32, 'rmse_uv':float(e)})
        write_csv(out/'test_samples_per_record.csv', sample_rows)
        predictions[label] = pred
        for i, (s, e) in enumerate(zip(test['sigmas'], err)):
            test_rows.append({'method':label, 'guidance':g, 'noise_uv':float(s), 'record':i % 32, 'rmse_uv':float(e)})
        write_csv(out/'test_per_record.csv', test_rows)
        print('TEST', label, g, float(err.mean()), flush=True)
        if run: run.log({f'test/{label}_mean_rmse_uv':float(err.mean())})
    np.savez_compressed(out/'test_predictions.npz', **predictions)
    # Exploratory checkpoint sensitivity; selection remains frozen.
    sensitivity = []
    checkpoints = [(4000, ROOT/'Leo/runs/phoom_long_50k/mlp/step_4000_ema_0.005.pt'),
                   (500000, ROOT/'Leo/runs/phoom_matched_10m/step_500000_ema_0.005.pt')]
    for update, path in checkpoints:
        if not path.exists():
            print('SENSITIVITY_CHECKPOINT_MISSING', path, flush=True)
            continue
        earlier, earlier_scale, _ = load_model(path, args.device)
        earlier_obs = make_observations(32, 93231, 700001, earlier_scale, args.device)
        for g in (3., 4., 5.5, 6.25, 7., 8.5, 10.):
            draws = sample(earlier, earlier_obs, g, 8, 7193, args.device)
            err, _ = errors(draws, earlier_obs['truth'], earlier_scale)
            row = {'checkpoint_updates': update, 'guidance': g, 'mean_rmse_uv': float(err.mean()),
                   **{f'rmse_{s}_uv':float(err[earlier_obs['sigmas']==s].mean()) for s in NOISE_UV}}
            sensitivity.append(row)
            write_csv(out/'checkpoint_sensitivity.csv', sensitivity)
            print('CHECKPOINT_SENSITIVITY', json.dumps(row), flush=True)
            if run:
                run.log({'sensitivity/checkpoint':update, 'sensitivity/guidance':g,
                         'sensitivity/rmse_uv':row['mean_rmse_uv']})
    (out/'runtime.json').write_text(json.dumps({'seconds':time.monotonic()-started}, indent=2))
    from guidance_sweep_report import render
    report, figures = render(out)
    if run:
        for name in figures:
            run.log({f'figures/{name.stem}': wandb.Image(str(name))})
        for name in (report, out/'protocol.json', out/'selection.json',
                     out/'validation_grid.csv', out/'test_per_record.csv'):
            run.save(str(name), base_path=str(out), policy='now')
        if (out/'checkpoint_sensitivity.csv').exists():
            run.save(str(out/'checkpoint_sensitivity.csv'), base_path=str(out), policy='now')
        run.save(str(out/'test_samples_per_record.csv'), base_path=str(out), policy='now')
        run.summary['selected_guidance'] = selected
        run.summary['report'] = str(report)
        run.finish()
    (out/'STATUS.txt').write_text('COMPLETE\n')
    print('COMPLETE', report, flush=True)


if __name__ == '__main__':
    main()
