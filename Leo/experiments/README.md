# Training and DPS comparison

Run from `Leo/` using Python 3.10+:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r experiments/requirements.txt
python experiments/study.py
```

PyTorch automatically selects CUDA when available. Install a CUDA-enabled PyTorch
build appropriate to your NVIDIA driver using https://pytorch.org/get-started/locally/.
Use `--device cuda --workers 1` to require CUDA; otherwise CPU training uses three
independent processes with one thread each. No driver installation is performed.

Default design:

* Three pipelines, each with 60,000 waveforms: analytic ABRs; EARNDB template
  augmentation; and a 16-mode Gaussian PCA generator fitted to EARNDB training subjects.
* Existing residual MLP (544,768 parameters), dilated temporal CNN (7,745), and compact 1D U-Net (197,313).
  These models are not parameter-matched. The U-Net has four temporal scales with skip connections and timestep conditioning.
* Two initialization/training seeds (17, 29), batch size 32, AdamW at 2e-4,
  weight decay 1e-4, EMA rate 0.005. Same settings and examples across architectures.
* Save EMA checkpoints at 1,000, 4,000, 8,000, 12,000 and 16,000 optimizer updates. At batch size
  32 the maximum is 512,000 sampled-example presentations; update counts are not epochs. The old training script defaults to batch size 256: equal update
  counts do **not** reproduce its compute or number of example presentations.
* Five guidance strengths: 0, 0.03, 0.1, 0.3, 1.0. Every checkpoint/seed selects its
  guidance using validation RMSE pooled equally over both target domains.
* Two target domains, white and colored noise, and synthetic noise RMS 0.05 and
  0.20 µV. All observations are 0.2 seconds; this experiment does not vary
  recording duration or establish results for minutes of real EEG.

## Splits and data provenance

EARNDB v1.0.0 averaged records are downloaded once (about 9 MB), with source-file
SHA-256 hashes saved in `data/manifest.json`. ABR channels are identified by `nV`
header units so OAE-only records are excluded. The SPL≥70 dB subset is specified
before training. Both repetitions, all levels and both stimulus frequencies of
each subject stay together: **N1–N5 train, N6 validation, N7–N8 test**.
Augmentation and PCA see training subjects only. Random analytic validation/test
targets use independent seeds. Validation/test observation CSVs record every
subject, source file, noise type and level.

The published WFDB header gains produce implausibly large physical values with
the standard reader. This code therefore uses **digital waveform morphology**, DC
removal, antialiased 48→10 kHz resampling, and per-waveform normalization to a
declared **synthetic 0.04 µV RMS**. It does not repair or claim physical calibration.
The original headers are preserved. The end of the periodic 41.7 ms epoch supplies
the negative 5 ms of the 256-sample window. Templates are measured averages with
residual noise, not clean physiological ground truth. These choices must appear
on slides describing EARNDB results.

Source: https://physionet.org/content/earndb/1.0.0/ — ODC Attribution license.
Cite Silva & Epstein (2010), *Estimating loudness growth from tone-burst evoked
responses*, JASA 127:3629–3642, doi:10.1121/1.3397457.

## Outputs and presentation

`Leo/runs/comparison/` contains configuration, hashes, checkpoints, losses, raw
validation/test metrics, predictions, per-subject summaries, `REPORT.md`, and
PNG/PDF figures under `figures/`:

1. Training duration versus held-out reconstruction RMSE.
2. Training/validation learning curves, with saved checkpoints marked.
3. Full validation guidance curves at every checkpoint, separately by domain.
4. The same prespecified waveform example at each checkpoint, for each model.
5. Noise robustness at the final checkpoint.

Shading shows the range across training seeds; it is not a confidence interval.
Only two participants are held out. Avoid population-level significance claims.
`per_subject.csv` exposes the individual participant results. Null-target
correlations/peak latencies are undefined and stored as NaN. Peak timing is the
maximum in 3–10 ms, not an independently validated wave-V annotation. The fixed
ridge baseline is a reference, not an optimized classical competitor.

Run phases separately to inspect data and losses before evaluation:

```bash
python experiments/study.py --phase data
python experiments/study.py --phase train
python experiments/study.py --phase evaluate
python experiments/study.py --phase plot
python -m pytest experiments/test_study.py -q
```

Configuration must match when reusing an output directory. A completed training
run is skipped; incomplete runs restart from their seed rather than resuming
optimizer state. Use a new `--output` for a different configuration. To move a
completed CPU run to GPU evaluation, use a new output/configuration and copy the
data/models deliberately; never mix training results silently.

For a larger GPU run, use e.g. `--seeds 17 29 43 --batch-size 256 --output
runs/comparison_gpu`. The run automatically records all arguments and versions.
`dps/reconstruct.py` loads both new architectures and older MLP checkpoints.

Live loss charts are refreshed every 1,000 updates in each model folder as `losses.png`. Loss values are logged every 200 updates in `losses.csv`. `LIVE_STATUS.md` links to the charts. The full design is 18 training runs and 90 saved checkpoints.

Current server run: `runs/comparison_16k`, launched with `experiments/run_initial_comparison.sh`. The initial comparison stops at 16k for all setups; 32k/64k extensions are reserved for explicitly selected configurations.

## W&B convergence dashboard

The separate monitor reads saved CSVs and completed evaluation shards. It does
not restart or modify the study. Sign in once with `.venv/bin/wandb login`, then
run:

```bash
.venv/bin/python experiments/wandb_monitor.py --output runs/comparison_16k
```

The default destination is entity `abr-deconv`, project `abr-deconv-comparison`;
use `--entity` and `--project` to override. It creates one W&B run per model and
seed, backfills saved history, and follows new reconstruction evaluations. Plot
`train/v_mse` and `train/validation_v_mse` against `train/update`; plot
`reconstruction/test_rmse_uv` and domain-specific RMSE against
`reconstruction/checkpoint`. Guidance validation curves and selected guidance
are also logged. These reconstruction points are checkpoint-level results, not
per-diffusion-step curves. The monitor stops when `JOB_STATUS.txt` says COMPLETE
or FAILED; use `--once` to import available results and exit. State is stored in
`wandb_monitor_state.json` so a restarted monitor skips points already recorded.
EARNDB references have synthetic amplitude normalization; reconstruction RMSE is
not a calibrated real-EEG measurement. Avoid selecting models using test RMSE.
