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
