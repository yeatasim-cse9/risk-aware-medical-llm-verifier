"""Fail-closed, read-only Kaggle synthetic-pilot restart decision.

Never submits Kaggle jobs. Only a fully verified checkpoint can produce a
"ready_for_resume" decision, which requires a separate explicit controller.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from verify_checkpoint import verify

EXPECTED_COMMIT = "1a90191ae2734a74807eb53fa7d5b27a4e44c78f"
EXPECTED_ID = "cpu-pilot-automation-001"
EXPECTED_TOTAL = 100
RUNNING_STATES = {"queued", "pending", "starting", "initializing", "running"}
FAILED_STATES = {"failed", "error", "cancelled", "cancelacknowledged"}


def decide(remote_status: str, state: dict | None, latest_slot: str) -> tuple[str, str]:
    """Return (decision, bounded reason), with no cloud mutations."""
    if latest_slot not in {"a", "b"}:
        return ("pause", "INVALID_SLOT")
    if remote_status in RUNNING_STATES:
        return ("wait", "REMOTE_SESSION_ACTIVE")
    if remote_status in FAILED_STATES:
        return ("pause", "REMOTE_SESSION_FAILED")
    if remote_status != "complete":
        return ("pause", "UNKNOWN_OR_MISSING_REMOTE_STATE")
    if not isinstance(state, dict):
        return ("pause", "CHECKPOINT_MISSING_OR_UNVERIFIED")

    if (
        state.get("experiment_id") != EXPECTED_ID
        or state.get("git_commit") != EXPECTED_COMMIT
        or type(state.get("total_tasks")) is not int
        or state["total_tasks"] != EXPECTED_TOTAL
    ):
        return ("pause", "CHECKPOINT_METADATA_MISMATCH")

    ids = state.get("completed_task_ids")
    if not isinstance(ids, list) or not ids:
        return ("pause", "NO_COMPLETED_TASKS")
    if ids != list(range(len(ids))) or len(ids) > EXPECTED_TOTAL:
        return ("pause", "INVALID_OR_DUPLICATE_TASK_IDS")

    if len(ids) == EXPECTED_TOTAL:
        if state.get("status") == "completed":
            return ("stop", "ALL_TASKS_COMPLETE_NO_RESTART")
        return ("pause", "FINAL_STATUS_INCONSISTENT")

    if state.get("status") != "in_progress":
        return ("pause", "PARTIAL_STATUS_INCONSISTENT")

    next_slot = "b" if latest_slot == "a" else "a"
    # This does not authorize a launch; a subsequent controller must check
    # quota, concurrency, source-dataset version, version availability, etc.
    return ("ready_for_review", f"CHECKPOINT_VALID_NEXT_SLOT_{next_slot.upper()}")


def decide_from_checkpoint(directory: Path, remote_status: str, slot: str) -> tuple[str, str]:
    # Verify SHA256, per-task hashes, data schema, and source SHA. If the
    # checkpoint fails validation, stop the job, never trigger a restart.
    state = verify(directory, expected_commit=EXPECTED_COMMIT)
    return decide(remote_status, state, slot)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--remote-status", choices=["complete"], required=True)
    parser.add_argument("--slot", choices=["a", "b"], required=True)
    args = parser.parse_args()

    try:
        decision, reason = decide_from_checkpoint(args.directory, args.remote_status, args.slot)
    except Exception:
        print("SAFE_MONITOR: PAUSE CHECKPOINT_VERIFICATION_FAILED")
        return 1

    print(f"SAFE_MONITOR: {decision.upper()} {reason}")
    if decision == "pause":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
