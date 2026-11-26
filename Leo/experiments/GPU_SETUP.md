# Enable the RTX 3050 Ti and run the comparison

Detected on this machine: NVIDIA RTX 3050 Ti Mobile (GA107M), Debian 13,
kernel 6.12.117, using `nouveau`. No NVIDIA CUDA driver or `/dev/nvidia*`
is currently available. Driver setup needs root access and a reboot.
The experiment scripts do not change your system drivers or reboot the machine.

## 1. Install Debian's NVIDIA driver as root

An installer for this machine is included. Run it in your own terminal so the
administrator password stays local:

```bash
cd ~/Seafile/Caltech/Year4/CS163/ABR-Deconv/Leo
bash experiments/install_nvidia_driver.sh
```

It backs up APT sources, enables Debian's non-free components, installs the
proprietary DKMS driver and matching headers, and checks that a module was built.
It does not reboot. If your account cannot use sudo, run `su -` and then
`bash /home/leki/Seafile/Caltech/Year4/CS163/ABR-Deconv/Leo/experiments/install_nvidia_driver.sh`.
The equivalent manual steps follow.

Open a terminal and enter the **root password**:

```bash
su -
nano /etc/apt/sources.list
```

The active Debian lines currently have `main non-free-firmware`. Add `contrib`
and `non-free` to each active line, preserving its URL and suite. For example:

```text
deb http://deb.debian.org/debian/ trixie main contrib non-free non-free-firmware
deb http://security.debian.org/debian-security trixie-security main contrib non-free non-free-firmware
deb http://deb.debian.org/debian/ trixie-updates main contrib non-free non-free-firmware
```

Save the file. Still in the root shell:

```bash
apt update
apt install "linux-headers-$(uname -r)" nvidia-kernel-dkms nvidia-driver firmware-misc-nonfree
```

If the installation reports a DKMS build error, resolve that before rebooting;
do not assume the driver installed successfully. Save your work and reboot from
the desktop. After logging back in, run:

```bash
nvidia-smi
```

It should list the RTX 3050 Ti. If not, inspect `dkms status` and
`journalctl -k -b | grep -Ei 'nvidia|nouveau|secure'`; follow Debian's driver
instructions for any Secure Boot key-enrollment requirement. The normal NVIDIA
installer handles nouveau configuration; do not manually unload your display
driver during the running desktop session.

Sources: [Debian NVIDIA driver guide](https://wiki.debian.org/NvidiaGraphicsDrivers/),
[Debian 13 nvidia-driver package](https://packages.debian.org/trixie/nvidia-driver).

## 2. Create the CUDA Python environment

As `leki`, not root:

```bash
cd ~/Seafile/Caltech/Year4/CS163/ABR-Deconv/Leo
bash experiments/setup_gpu_env.sh
```

This installs the official PyTorch 2.6 CUDA 12.4 build into `Leo/.venv` and tests
all three architectures and a backward pass on the GPU. The CUDA wheel includes the
runtime; a separate CUDA toolkit installation is not required for this project.
The check fails visibly when CUDA cannot execute, rather than switching to CPU.
Installer reference: [official PyTorch previous versions](https://pytorch.org/get-started/previous-versions/).

## 3. Run all experiments and produce slides

```bash
bash experiments/run_gpu.sh
```

Default: 3 data pipelines × 3 models × 2 seeds, with 1k/4k/8k/12k/16k/32k/64k checkpoints,
batch size 32 and five validation guidance strengths. This keeps the comparison
manageable on a 4 GB mobile GPU. For a third seed use a separate output folder:

```bash
bash experiments/run_gpu.sh --seeds 17 29 43 --output runs/comparison_gpu_3seeds
```

The script reuses the downloaded source records, creates fresh training data and
models, and writes final artifacts to `Leo/runs/comparison_gpu/`:

* `comparison_slides.pdf` — presentation deck with experimental design and limits.
* `figures/` — PNG/PDF training-duration, learning-curve, guidance, waveform and
  noise-robustness comparisons.
* `REPORT.md`, `summary.csv`, `per_subject.csv` — results and presentation notes.
* Configuration, source hashes, raw metrics, predictions and EMA checkpoints.

The earlier CPU work is preserved in `Leo/runs/cpu_pilot/` and explicitly labeled
partial. It is not a completed comparison and will not be mixed into GPU results.

The EARNDB results use **synthetic amplitude normalization** because its physical
calibration remains unresolved. See [README.md](README.md) before presenting them.
