# Reconstruction improvements to Phoom's code

Run the controlled study from `phoom/`, using the shared experiment environment:

```bash
python ../Leo/experiments/phoom_improvements.py
```

The runner uses separate validation and test waveform/noise seeds, matches
training updates and batch size, scores means of 32 reconstructions for every
method, and logs to CoreWeave Forge project `abr-deconv-phoom-improvements`.
Checkpoint, EMA, guidance and measurement-correction settings are selected on
validation recordings. The default output is `Leo/runs/phoom_improvements`.
Use a new `--output` directory when changing the protocol. One training seed is
used; results are a synthetic pilot, not a claim about real EEG or a controlled
comparison with Phoom's million-update models whose weights were unavailable.

The completed single-file report is `Leo/results/phoom_improvements/REPORT.html`.
Raw tables and seven separate PNG/PDF/SVG figures are included beside it.

The small CNN did not beat the MLP in this experiment. The useful changes were
faster EMA and a validation-selected covariance-regularized measurement correction.
The correction reuses the EEG; it is an empirical refinement, not a calibrated
posterior update.

## Optional settings in Phoom's trainer

The original default EMA remains 0.001. To try the pilot's 0.005 setting:

```bash
python -m diffusion.train mlp --steps 4000 --check-every 1000 --ema-rate 0.005
```

`--resume` inherits a saved EMA rate if no flag is supplied. Legacy checkpoints
without the field inherit 0.001. Explicitly changing the flag on resume starts a
new EMA regime and is recorded in the run configuration.

## Optional correction after DPS sampling

All amplitudes here are in microvolts. Estimate covariance from training
waveforms only, then select `strength` on separate validation recordings:

```python
from diffusion.refinement import measurement_refinement

estimate = draws.mean(axis=1)  # recordings × 256
corrected = measurement_refinement(
    estimate, eeg_uv, stimulus_matrix, noise_std_uv,
    covariance_uv2=training_covariance,
    strength=validation_selected_strength,
)
```

A strength of zero returns the input estimate. The pilot selected 0.1 for the
MLP at all three tested noise levels; that value is not a universal default.

## Checks

```bash
python -m pytest ../Leo/experiments/test_phoom_improvements.py -q
```

Checks cover sampler equivalence with Phoom's direct EEG implementation,
equivalence of CNN outputs/gradients after convolution acceleration, zero
correction, and physical noise scaling in the correction.

## Longer training follow-up

The 4,000-update experiment screens changes; it does not establish convergence.
The follow-up uses one MLP trajectory with both EMA shadows through 50,000
updates. It evaluates checkpoints at 4k, 10k, 20k, 35k and 50k, then compares
both final EMA settings and the validation-selected reconstruction pipeline.
New validation and holdout seeds keep this follow-up separate from the pilot.

From `phoom/`:

```bash
python ../Leo/experiments/phoom_improvements.py \
  --steps 50000 --architectures mlp \
  --checkpoints 4000 10000 20000 35000 50000 \
  --project abr-deconv-phoom-long-training \
  --validation-waveform-seed 90424 --validation-noise-seed 4000 \
  --fresh-waveform-seed 90425 --fresh-noise-seed 4500 \
  --output ../Leo/runs/phoom_long_50k
```

The runner saves full model, EMA, optimizer and sampling-generator state at each
checkpoint. Rerunning the same command resumes from that checkpoint. At the end,
it automatically creates a standalone report, exports five figures in PNG/PDF/SVG,
and publishes the results and checkpoints to the named Forge project.
