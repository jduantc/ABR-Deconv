# Synthetic ABR training

Synthetic ABR generation and diffusion denoiser training. The files are arranged as follows:

```text
data.py           Create, filter, and save clean ABR waveforms.
model.py          Neural network that predicts the diffusion target, v.
diffusion.py      Noise schedule, noisy inputs, v targets, and clean estimates.
train.py          Network training and validation. The saved weights have the lowest validation loss.
requirements.txt  Dependencies for the scripts.
.gitignore        Git exclusions: generated data, training outputs, and temporary files.

data/             Synthetic waveforms saved as .npz files.
runs/             One folder for each training run.
  model.pt        Model weights.
  losses.csv      Training and validation losses.
  settings.json   Settings for that run.
__pycache__/      Python puts its temporary files here automatically.
```

## Training comparisons

See [experiments/README.md](experiments/README.md) for subject-separated EARNDB
pipelines, MLP/CNN comparisons, training checkpoints, guidance sweeps and presentation figures.

The original trainer also accepts `--architecture cnn --checkpoints 1000 8000 16000`
when `--steps 16000` is supplied. The comparison runner uses its own fixed splits.

## Reconstruction-focused follow-up

The primary outcome is waveform RMSE between the reconstructed ABR and the
reference ABR used to simulate the EEG. Training v-prediction MSE and the EEG
residual used for guidance are separate diagnostics.

```bash
python -m venv .venv
.venv/bin/pip install -r experiments/requirements.txt
.venv/bin/wandb login
# Prepare data only; this does not launch the original long training study.
.venv/bin/python experiments/study.py --phase data --output runs/reconstruction_data
.venv/bin/python experiments/reconstruction_suite.py --data runs/reconstruction_data/data
```

The follow-up trains CNNs with analytic and EARNDB template priors using two
training seeds, saves four early checkpoints, selects checkpoint/guidance by
validation reconstruction RMSE, and tests three sampling seeds against a
validation-tuned ridge baseline. The synthetic observation design contains
32 validation and 64 test cases, with test references balanced across subjects N7 and N8. Reference-based diagnostics do not influence
the reconstruction gradients.

Training and evaluation are tracked in [CoreWeave Forge](https://forge.coreweave.com/wandb/abr-deconv/abr-deconv-reconstruction).
The consolidated report is `results/reconstruction_focus/REPORT.html`; open it
in a browser. All figures are also exported separately as PNG, PDF and SVG.
The run writes raw tables, predictions, source hashes and checkpoints under
`runs/reconstruction_focus/`. Checkpoints and raw downloaded records are excluded
from Git. Credentials are supplied through W&B login, never through source files.

```bash
.venv/bin/python -m pytest experiments/test_study.py experiments/test_reconstruction.py -q
.venv/bin/python experiments/reconstruction_report.py runs/reconstruction_focus
```

The Forge report run also publishes an evaluation artifact containing the standalone
report, figures as run media, provenance, and the 16 saved EMA checkpoints. The Git
results bundle retains small prediction arrays and raw tables so figures can be
regenerated without training.

The standalone `dps/reconstruct.py` uses RMS EEG residual without noise-SD
normalization. Pass `guidance` by keyword; `dps/run_eeg.py` no longer requires
`--noise-std-uv`. Guidance may need retuning. To reproduce a previous update
for a fixed noise SD, divide the old guidance by that noise SD in model units.
The historical batched experiment implementations retain their original loss.

## Realistic EEG simulation and comparisons

`abr_eeg_pipeline.py` combines a known ABR with recorded pulse timing and EEG
background. It requires NumPy, SciPy, h5py, and an exp3 recording from ds008065.
Run `python Leo/abr_eeg_pipeline.py --help` for the input and output options.
The output is a single-band simulation with a known target ABR.

The experiment scripts and tests are in `experiments/`. The folder
`experiments/legacy_phoom/` preserves the Phoom-derived code used by the
historical Leo studies, including versions not present in the current root
modules. Existing reports describe the settings used at the time of each run.
`results/mlp_unet_eeg_comparison/REPORT.html` compares the old simulated EEG
with the newer recorded-background simulation. The experiments require the
original dataset and trained checkpoints; these are not included in Git.
