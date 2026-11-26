# Server deployment

SSH alias: `Doma` (configured locally). Server hostname: `pop-os`.
GPU: GTX 1080, 8 GB; NVIDIA driver 580.82.09.

Working directory:

```text
/home/leki/Caltech/Year 4/CS163/ABR-Deconv-GPU-Study/Leo
```

Only project source and the EARNDB source cache are copied. The server has its own
private `.venv` using PyTorch 2.6 / CUDA 12.4, which retains support for the GTX
1080's Pascal architecture. No system drivers or global Python packages are changed.

The full study is 18 training runs (3 pipelines × 3 architectures × 2 seeds), each
ending at 16,000 updates. Five checkpoints per run give 90 saved EMA checkpoints:
1k, 4k, 8k, 12k and 16k. Longer runs require an explicit selection after validation. Validation selects guidance from the same five
strengths. Train/validation/test subject membership is fixed before augmentation.

The detached server job continues after the SSH connection closes. To inspect it:

```bash
ssh Doma 'tail -30 "$HOME/Caltech/Year 4/CS163/ABR-Deconv-GPU-Study/Leo/server_job.log"'
ssh Doma 'cat "$HOME/Caltech/Year 4/CS163/ABR-Deconv-GPU-Study/Leo/runs/comparison_16k/JOB_STATUS.txt"'
```

Results are under `Leo/runs/comparison_16k/` on the server:

* `LIVE_STATUS.md`: progress table with links to each model's live loss graph.
* `models/*/losses.png`: refreshed every 1,000 updates; CSV values every 200.
* `console.log`, `environment.txt`, `source_sha256.txt`: execution provenance.
* `REPORT.md`, `summary.csv`, `per_subject.csv`: generated after evaluation.
* `figures/` and `comparison_slides.pdf`: final presentation material.

`JOB_STATUS.txt` says `COMPLETE` only after the whole pipeline succeeds, or `FAILED`
if a check or stage exits with an error. The server launcher validates CUDA and
runs the six scientific tests before beginning training.

The blanket 64k job was stopped at the user’s request. Its artifacts remain under `runs/comparison_gpu/`; matching checkpoints through 16k were reused in the capped study and documented in `reused_checkpoints.json`.
