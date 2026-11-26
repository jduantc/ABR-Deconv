#!/usr/bin/env python3
"""Backfill and follow study CSVs in W&B without changing the running study."""

import argparse
import csv
import hashlib
import json
import os
import time
from pathlib import Path

import wandb


def read_csv(path):
    try:
        with path.open(newline="") as f:
            return list(csv.DictReader(f))
    except (FileNotFoundError, OSError, csv.Error):
        return []


def save_state(path, state):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(state, indent=2))
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/comparison_16k"))
    parser.add_argument(
        "--entity", default=os.environ.get("WANDB_ENTITY", "abr-deconv")
    )
    parser.add_argument(
        "--project", default=os.environ.get("WANDB_PROJECT", "abr-deconv-comparison")
    )
    parser.add_argument("--interval", type=float, default=30)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--model", help="Limit to one model directory name")
    parser.add_argument(
        "--state", type=Path, help="Monitor state file; defaults inside output"
    )
    args = parser.parse_args()
    output = args.output.resolve()
    state_path = args.state or output / "wandb_monitor_state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    study_path = output / "study_config.json"
    config = json.loads(study_path.read_text()) if study_path.exists() else {}
    runs = {}
    os.environ.setdefault("WANDB_DISABLE_CODE", "true")
    try:
        while True:
            for model_dir in sorted((output / "models").glob("*")):
                if not model_dir.is_dir() or (
                    args.model and model_dir.name != args.model
                ):
                    continue
                name = model_dir.name
                settings_path = model_dir / "settings.json"
                if not settings_path.exists():
                    continue
                try:
                    settings = json.loads(settings_path.read_text())
                except (OSError, json.JSONDecodeError):
                    continue
                if name not in runs:
                    run_id = hashlib.sha256(
                        f"{args.entity}/{args.project}/{output}/{name}".encode()
                    ).hexdigest()[:16]
                    run = wandb.init(
                        entity=args.entity,
                        project=args.project,
                        id=run_id,
                        resume="allow",
                        name=name,
                        group=output.name,
                        config={"study": config, **settings},
                        job_type="comparison",
                        reinit="create_new",
                    )
                    run.define_metric("train/update")
                    run.define_metric("train/*", step_metric="train/update")
                    run.define_metric("reconstruction/checkpoint")
                    run.define_metric(
                        "reconstruction/*", step_metric="reconstruction/checkpoint"
                    )
                    runs[name] = run
                run = runs[name]
                entry = state.setdefault(
                    name, {"last_training_update": 0, "evaluated_checkpoints": []}
                )
                for row in read_csv(model_dir / "losses.csv"):
                    try:
                        step = int(row["step"])
                        if step <= entry["last_training_update"]:
                            continue
                        run.log(
                            {
                                "train/update": step,
                                "train/v_mse": float(row["train_v_mse"]),
                                "train/validation_v_mse": float(
                                    row["validation_v_mse"]
                                ),
                                "train/elapsed_seconds": float(row["seconds"]),
                            }
                        )
                        entry["last_training_update"] = step
                    except (KeyError, TypeError, ValueError):
                        continue
                eval_dir = output / "evaluation"
                for shard in sorted(
                    eval_dir.glob(name + "_*"),
                    key=lambda p: (
                        int(p.name.rsplit("_", 1)[-1])
                        if p.name.rsplit("_", 1)[-1].isdigit()
                        else -1
                    ),
                ):
                    try:
                        step = int(shard.name.rsplit("_", 1)[-1])
                    except ValueError:
                        continue
                    if (
                        step in entry["evaluated_checkpoints"]
                        or not (shard / "prediction.npz").exists()
                    ):
                        continue
                    test = read_csv(shard / "test.csv")
                    val = read_csv(shard / "validation.csv")
                    if not test or not val:
                        continue
                    try:
                        values = {
                            "reconstruction/checkpoint": step,
                            "reconstruction/test_rmse_uv": sum(
                                float(r["rmse_uv"]) for r in test
                            )
                            / len(test),
                            "reconstruction/selected_guidance": float(
                                test[0]["guidance"]
                            ),
                        }
                        for domain in ("analytic", "earndb"):
                            selected = [
                                float(r["rmse_uv"])
                                for r in test
                                if r["domain"] == domain
                            ]
                            if selected:
                                values[f"reconstruction/test_rmse_uv_{domain}"] = sum(
                                    selected
                                ) / len(selected)
                        for row in val:
                            key = f"reconstruction/validation_rmse_uv_{row['domain']}_guidance_{row['guidance']}"
                            values[key] = float(row["rmse_uv"])
                        run.log(values)
                        entry["evaluated_checkpoints"].append(step)
                    except (KeyError, TypeError, ValueError):
                        continue
                save_state(state_path, state)
            status_file = output / "JOB_STATUS.txt"
            status = status_file.read_text().strip() if status_file.exists() else ""
            if args.once or status == "COMPLETE" or status.startswith("FAILED"):
                break
            time.sleep(args.interval)
    finally:
        for run in runs.values():
            run.finish()


if __name__ == "__main__":
    main()
