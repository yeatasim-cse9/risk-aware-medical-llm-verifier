"""Synthetic CPU checkpoint pilot. No medical data, models, or external dependencies.

Run as a standalone Kaggle script or import from unit tests. The GitHub build
injects its exact commit into CODE_COMMIT before submission to Kaggle.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from datetime import datetime, timezone

CODE_COMMIT = "__GIT_COMMIT__"
EXPERIMENT_ID = "cpu-pilot-automation-001"
DATASET_VERSION = "synthetic-v1"
MODEL_VERSION = "sha256-demo-v1"
SEED = 42
TOTAL_TASKS = 100
CHUNK_SIZE = 40
CHECKPOINT_NAME = "cpu-pilot-automation.json"


def canonical(data: dict) -> bytes:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def digest(data: dict) -> str:
    return hashlib.sha256(canonical(data)).hexdigest()


def result_for(task_id: int) -> str:
    return hashlib.sha256(f"{SEED}:{task_id}".encode("ascii")).hexdigest()


def new_state() -> dict:
    return {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "dataset_version": DATASET_VERSION,
        "model_version": MODEL_VERSION,
        "git_commit": CODE_COMMIT,
        "seed": SEED,
        "total_tasks": TOTAL_TASKS,
        "completed_task_ids": [],
        "results": {},
        "status": "in_progress",
    }


def validate(state: dict) -> None:
    if not isinstance(state, dict):
        raise ValueError("Checkpoint must be an object")
    fixed = new_state()
    for key in ("schema_version", "experiment_id", "dataset_version",
                "model_version", "git_commit", "seed", "total_tasks"):
        if type(state.get(key)) is not type(fixed[key]) or state[key] != fixed[key]:
            raise ValueError(f"Checkpoint mismatch: {key}")
    ids = state.get("completed_task_ids")
    results = state.get("results")
    if not isinstance(ids, list) or not isinstance(results, dict):
        raise ValueError("Invalid checkpoint task structure")
    if any(type(i) is not int or not 0 <= i < TOTAL_TASKS for i in ids):
        raise ValueError("Out-of-range task ID")
    if ids != sorted(set(ids)):
        raise ValueError("Duplicate or unsorted task IDs")
    if set(results) != {str(i) for i in ids}:
        raise ValueError("Results and completed task IDs differ")
    for i in ids:
        if results[str(i)] != result_for(i):
            raise ValueError(f"Task result checksum failed: {i}")
    if state.get("status") not in ("in_progress", "completed"):
        raise ValueError("Invalid status")
    if state["status"] == "completed" and len(ids) != TOTAL_TASKS:
        raise ValueError("Completed status without all tasks")
    stored_digest = state.get("integrity_sha256")
    payload = {k: v for k, v in state.items() if k != "integrity_sha256"}
    if not isinstance(stored_digest, str) or stored_digest != digest(payload):
        raise ValueError("Checkpoint integrity checksum mismatch")


def atomic_save(state: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {k: v for k, v in state.items() if k != "integrity_sha256"}
    payload["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    payload["integrity_sha256"] = digest(payload)
    validate(payload)
    temp = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=".checkpoint-", suffix=".tmp",
            dir=path.parent, delete=False
        ) as fp:
            temp = Path(fp.name)
            fp.write(json.dumps(payload, sort_keys=True, indent=2).encode("utf-8"))
            fp.flush()
            os.fsync(fp.fileno())
        os.replace(temp, path)
    finally:
        if temp is not None and temp.exists():
            temp.unlink()


def locate_checkpoint(input_dir: Path, output_file: Path) -> Path | None:
    if output_file.is_file():
        return output_file
    if not input_dir.exists():
        return None
    matches = sorted(input_dir.rglob(CHECKPOINT_NAME))
    if len(matches) > 1:
        raise ValueError("Ambiguous input checkpoints: choose one source")
    return matches[0] if matches else None


def run(output_dir: Path, input_dir: Path, max_new: int = CHUNK_SIZE) -> dict:
    if type(max_new) is not int or not 0 <= max_new <= TOTAL_TASKS:
        raise ValueError("Invalid chunk size")
    output_file = output_dir / "checkpoints" / CHECKPOINT_NAME
    source = locate_checkpoint(input_dir, output_file)
    if source is None:
        state = new_state()
        print("Starting a new synthetic run")
    else:
        state = json.loads(source.read_text(encoding="utf-8-sig"))
        validate(state)
        print(f"Resume checkpoint: {source}")
    old = len(state["completed_task_ids"])
    done = set(state["completed_task_ids"])
    for task_id in range(TOTAL_TASKS):
        if task_id in done or len(done) - old >= max_new:
            continue
        state["results"][str(task_id)] = result_for(task_id)
        done.add(task_id)
        state["completed_task_ids"] = sorted(done)
        state["status"] = "completed" if len(done) == TOTAL_TASKS else "in_progress"
        # Each synthetic task is cheap. Save on every task to test recovery.
        atomic_save(state, output_file)
    if not output_file.is_file():
        atomic_save(state, output_file)
    verified = json.loads(output_file.read_text(encoding="utf-8"))
    validate(verified)
    newly = len(verified["completed_task_ids"]) - old
    print(f"Experiment: {EXPERIMENT_ID}")
    print(f"Git commit: {CODE_COMMIT}")
    print(f"Previously completed: {old}")
    print(f"Newly processed: {newly}")
    print(f"Total completed: {len(verified['completed_task_ids'])}/{TOTAL_TASKS}")
    print(f"Status: {verified['status']}")
    print(f"Checkpoint: {output_file}")
    print("CHECKPOINT PILOT OK")
    return verified


if __name__ == "__main__":
    run(
        Path(os.getenv("PILOT_OUTPUT_DIR", "/kaggle/working")),
        Path(os.getenv("PILOT_INPUT_DIR", "/kaggle/input")),
    )
