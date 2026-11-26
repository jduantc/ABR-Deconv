"""Versioned EARNDB averages, subject-separated augmentation and fitted PCA."""

import concurrent.futures
import hashlib
import json
import re
import urllib.request
from pathlib import Path

import numpy as np
import wfdb
from scipy.signal import resample_poly

from data import generate_responses

URL = "https://physionet.org/files/earndb/1.0.0/average/"
SUBJECTS = {"train": [1, 2, 3, 4, 5], "validation": [6], "test": [7, 8]}
PIPELINES = ("analytic", "earndb_templates", "earndb_pca")


def download(cache):
    cache.mkdir(parents=True, exist_ok=True)
    index = cache / "records.json"
    if index.exists():
        names = json.loads(index.read_text())
    else:
        listing = urllib.request.urlopen(URL, timeout=60).read().decode()
        # Include every averaged record, including single-channel _x records.
        names = sorted(set(re.findall(r'href="(N\d+_evoked_ave[^"/]+\.hea)"', listing)))
    if not names:
        raise RuntimeError("EARNDB listing contained no averaged records")

    def get(name):
        for suffix in (".hea", ".dat"):
            filename = Path(name).with_suffix(suffix).name
            path = cache / filename
            if not path.exists():
                content = urllib.request.urlopen(URL + filename, timeout=60).read()
                path.write_bytes(content)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(get, names))
    index.write_text(json.dumps(names, indent=2))
    return names


def load_references(cache):
    names = download(cache)
    waves, subjects, records = [], [], []
    excluded = []
    for name in names:
        record = wfdb.rdrecord(str(cache / name.removesuffix(".hea")), physical=False)
        abr = [i for i, unit in enumerate(record.units) if unit.lower() == "nv"]
        # Prespecified high-level subset; do not use test quality scores to select records.
        level = int(re.search(r"_ave(\d+)_", name)[1])
        if not abr or level < 70:
            excluded.append(name)
            continue
        # Published gains yield implausible physical scales in the standard WFDB
        # reader. Use digital morphology only; never assert calibrated EARNDB uV.
        x = record.d_signal[:, abr[0]].astype(np.float64)
        # Last 10 ms is used only to remove DC; no label/quality tuning on test.
        x = x - x[-480:].mean()
        x = resample_poly(x, 5, 24)  # 48 kHz -> 10 kHz
        # Periodic epoch tail supplies -5..0 ms, not measured pre-onset EEG.
        x = np.concatenate([x[-50:], x[:206]])
        if x.shape != (256,) or not np.isfinite(x).all():
            raise ValueError(f"Invalid waveform: {name}")
        rms = np.sqrt(np.mean(x * x))
        if rms <= 0:
            raise ValueError(f"Constant EARNDB reference: {name}")
        # A declared synthetic amplitude of 0.04 uV RMS, not a recovered calibration.
        waves.append(0.4 * x / rms)
        subjects.append(int(re.match(r"N(\d+)", name)[1]))
        records.append(name)
    return np.asarray(waves, dtype=np.float32), np.array(subjects), records, excluded


def augment(templates, count, rng):
    source = rng.integers(len(templates), size=count)
    gain = np.exp(rng.normal(0, 0.2, (count, 1)))
    shifts = rng.integers(-3, 4, count)
    # Interpolation pads with zeros rather than wrapping a response across time.
    t = np.arange(256)
    out = (
        np.stack(
            [
                np.interp(t - s, t, templates[i], left=0, right=0)
                for i, s in zip(source, shifts)
            ]
        )
        * gain
    )
    return out.astype(np.float32)


def pca_generate(training, count, rng):
    mu = training.mean(0)
    _, s, vt = np.linalg.svd(training - mu, full_matrices=False)
    rank = min(16, len(s))
    weights = rng.normal(size=(count, rank)).clip(-2.5, 2.5)
    out = mu + (weights * s[:rank] / np.sqrt(len(training) - 1)) @ vt[:rank]
    return out.astype(np.float32)


def prepare(output, count=60000):
    output.mkdir(parents=True, exist_ok=True)
    waves, subjects, records, excluded = load_references(output / "earndb_cache")
    owners = {}
    for wave, subject in zip(waves, subjects):
        digest = hashlib.sha256(wave.tobytes()).hexdigest()
        if digest in owners and owners[digest] != subject:
            raise ValueError(
                "Duplicate normalized reference occurs in different subjects"
            )
        owners[digest] = subject
    np.savez_compressed(
        output / "references.npz", responses=waves, subjects=subjects, records=records
    )
    train = waves[np.isin(subjects, SUBJECTS["train"])]
    files = {}
    for i, pipeline in enumerate(PIPELINES):
        seed = 18200 + i
        rng = np.random.default_rng(seed)
        if pipeline == "analytic":
            # Chunking bounds the longer-interval filter's peak memory.
            x = np.concatenate(
                [
                    generate_responses(min(2000, count - j), seed + j)[0]
                    for j in range(0, count, 2000)
                ]
            )
        elif pipeline == "earndb_templates":
            x = augment(train, count, rng)
        else:
            x = pca_generate(train, count, rng)
        path = output / f"{pipeline}.npz"
        np.savez_compressed(path, responses=x, seed=seed)
        files[pipeline] = {
            "count": len(x),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    manifest = {
        "source": URL,
        "source_version": "1.0.0",
        "subject_split": SUBJECTS,
        "records": records,
        "excluded_no_ABR_or_SPL_below_70": excluded,
        "datasets": files,
        "split_records": {
            split: [r for r, s in zip(records, subjects) if int(s) in ids]
            for split, ids in SUBJECTS.items()
        },
        "source_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((output / "earndb_cache").glob("*"))
        },
        "units": "model unit = 0.1 microvolt",
        "seed": 18200,
        "reference_note": "Measured averages, SPL>=70; digital morphology normalized to synthetic 0.04 uV RMS. Physical EARNDB calibration unresolved. Not noise-free ground truth.",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(
        f"Prepared {len(waves)} EARNDB ABR averages; {len(train)} from training subjects",
        flush=True,
    )
    return manifest
