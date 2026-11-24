"""Step 10.4A: fail-closed two-slot Kaggle CPU resume preflight.

READ-ONLY. It never calls 'kaggle kernels push' or modifies a Kaggle run.
The quota signal is intentionally UNKNOWN unless independently attested;
a dry-run cannot prove remaining free-tier CPU/GPU usage.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess

from kaggle_status import check
from safe_monitor_decision import decide, EXPECTED_COMMIT
from verify_checkpoint import verify

OWNER = "yeatasimbillah"
KERNELS = {
    "a": f"{OWNER}/medical-llm-cpu-auto-a",
    "b": f"{OWNER}/medical-llm-cpu-auto-b",
}
ACTIVE = {"running", "queued", "pending", "starting", "initializing"}
FAILED = {"failed", "error", "cancelled", "cancelacknowledged"}
KNOWN = ACTIVE | FAILED | {"complete", "missing"}
QUOTA_SIGNALS = {"unknown", "insufficient", "operator_attested"}


def assess(
    remote: dict[str, str],
    checkpoints: dict[str, dict],
    quota_signal: str = "unknown",
) -> tuple[str, str]:
    """Assess previously verified checkpoints, without granting permission to submit.

    "ready_for_manual_review" is not permission for automatic execution.
    """
    if set(remote) != {"a", "b"} or set(checkpoints) - {"a", "b"}:
        return ("pause", "INVALID_REMOTE_SLOT_SET")
    if quota_signal not in QUOTA_SIGNALS:
        return ("pause", "INVALID_QUOTA_SIGNAL")
    if any(status not in KNOWN for status in remote.values()):
        return ("pause", "UNKNOWN_REMOTE_STATUS")
    if any(status in FAILED for status in remote.values()):
        return ("pause", "REMOTE_FAILURE_MANUAL_REVIEW")
    if any(status in ACTIVE for status in remote.values()):
        return ("wait", "REMOTE_JOB_ACTIVE_NO_NEW_SUBMISSION")

    completed = [slot for slot in KERNELS if remote[slot] == "complete"]
    if not completed:
        return ("pause", "NO_COMPLETED_SESSION_OR_TRUSTED_CHECKPOINT")
    if any(slot not in checkpoints for slot in completed):
        return ("pause", "CHECKPOINT_MISSING")
    if any(slot in checkpoints for slot in KERNELS if remote[slot] == "missing"):
        return ("pause", "CHECKPOINT_FOR_MISSING_KERNEL")

    counts = {}
    for slot in completed:
        state = checkpoints[slot]
        decision, _reason = decide("complete", state, slot)
        if decision not in {"stop", "ready_for_review"}:
            return ("pause", "CHECKPOINT_METADATA_OR_PROGRESS_INVALID")
        counts[slot] = len(state["completed_task_ids"])

    if len(completed) == 2 and counts["a"] == counts["b"]:
        return ("pause", "AMBIGUOUS_LATEST_CHECKPOINT")

    latest = max(completed, key=lambda slot: counts[slot])
    highest = counts[latest]
    if highest == 100:
        return ("stop", "ALL_100_TASKS_COMPLETE_NO_RESTART")

    target = "b" if latest == "a" else "a"
    if remote[target] not in {"missing", "complete"}:
        return ("pause", "TARGET_SLOT_UNAVAILABLE")
    # This is a conservative local gate; an operator attestation is not an
    # actual Kaggle API quota check, and never itself launches a notebook.
    if quota_signal == "unknown":
        return ("pause", "QUOTA_NOT_INDEPENDENTLY_VERIFIED")
    if quota_signal == "insufficient":
        return ("pause", "QUOTA_INSUFFICIENT")
    return ("ready_for_manual_review", f"LATEST_{latest.upper()}_{highest}_NEXT_{target.upper()}")


def status_and_checkpoints(root: Path) -> tuple[dict[str, str], dict[str, dict]]:
    remote = {slot: check(ref, "any") for slot, ref in KERNELS.items()}
    if any(status in ACTIVE | FAILED or status not in KNOWN for status in remote.values()):
        # Do not download outputs of active/failed/unknown notebooks.
        return remote, {}
    states = {}
    for slot, kernel in KERNELS.items():
        if remote[slot] != "complete":
            continue
        target = root / slot
        target.mkdir(parents=True, exist_ok=True)
        proc = subprocess.run(
            ["kaggle", "kernels", "output", kernel, "-p", str(target)],
            capture_output=True, text=True, timeout=180,
        )
        if proc.returncode != 0:
            # Kaggle CLI output could contain remote data; never echo it.
            raise RuntimeError("KAGGLE_CHECKPOINT_DOWNLOAD_FAILED")
        states[slot] = verify(target, expected_commit=EXPECTED_COMMIT)
    return remote, states


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--quota-signal", choices=sorted(QUOTA_SIGNALS), default="unknown",
        help="Synthetic-policy input; not an API-measured free-tier quota.",
    )
    parser.add_argument("--work-dir", type=Path, default=Path("build/resume-preflight"))
    args = parser.parse_args()
    try:
        remote, checkpoints = status_and_checkpoints(args.work_dir)
        decision, reason = assess(remote, checkpoints, args.quota_signal)
    except Exception:
        # Prevent credential-bearing URLs or API responses from leaking into logs.
        print("AUTO_RESUME_PREFLIGHT: PAUSE REMOTE_OR_CHECKPOINT_VERIFICATION_FAILED")
        return 1
    print(f"AUTO_RESUME_PREFLIGHT: {decision.upper()} {reason}")
    # Explicitly no automated launch path in this step.
    if decision == "ready_for_manual_review":
        print("AUTO_RESUME_PREFLIGHT: MANUAL_APPROVAL_REQUIRED_NO_AUTO_SUBMISSION")
    if decision == "pause":
        # A conservative guard blocks proceeding rather than claiming success.
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
