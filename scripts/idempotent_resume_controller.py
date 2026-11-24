"""Step 10.4C: idempotent Kaggle resume intent and conservative recovery.

READ-ONLY CLI: this script NEVER submits Kaggle kernels. All transitions are
pure functions for offline testing; the currently completed synthetic pilot
must remain stopped. A future launch integration must durably reserve intents
with a compare-and-swap (CAS) write BEFORE any external submission.

There is no exactly-once Kaggle submit guarantee. Ambiguous submit results,
failed jobs, missing checkpoints, unexpected competing operations, and stale
reservations require manual reconciliation, never blind retries.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any, Callable

from auto_resume_preflight import status_and_checkpoints, assess
from safe_monitor_decision import EXPECTED_COMMIT, EXPECTED_ID, EXPECTED_TOTAL

SCOPE = "synthetic_cpu_pilot_only"
MAX_TASKS = EXPECTED_TOTAL
PHASES = {"idle", "reserved", "submitted", "needs_review", "completed"}
REVIEW_REASONS = {
    "AMBIGUOUS_KAGGLE_SUBMISSION",
    "REMOTE_SESSION_FAILED",
    "TARGET_CHECKPOINT_NOT_ADVANCING",
    "TARGET_REMOTE_STATUS_UNKNOWN",
    "RESERVATION_RECOVERY_REQUIRES_REVIEW",
}
RUN_ID = re.compile(r"[1-9][0-9]{0,19}\Z")


class ResumeSafetyError(ValueError):
    """Sanitized, non-sensitive error code only."""


def make_state(count: int, slot: str, phase: str = "idle") -> dict:
    """Create the minimal trusted synthetic-only control record."""
    if type(count) is not int or count not in {40, 80, 100}:
        raise ResumeSafetyError("INVALID_CONFIRMED_PROGRESS")
    if slot not in {"a", "b"}:
        raise ResumeSafetyError("INVALID_CONFIRMED_SLOT")
    if (count == 100) != (phase == "completed"):
        raise ResumeSafetyError("INCONSISTENT_COMPLETION_PHASE")
    if phase not in {"idle", "completed"}:
        raise ResumeSafetyError("INITIAL_PHASE_NOT_ALLOWED")
    return {
        "schema_version": 1,
        "scope": SCOPE,
        "experiment_id": EXPECTED_ID,
        "source_commit": EXPECTED_COMMIT,
        "phase": phase,
        "confirmed_completed_tasks": count,
        "confirmed_slot": slot,
        "intent": None,
        "review_reason": None,
    }


def intent_key(source_count: int, source_slot: str, target_slot: str) -> str:
    if (source_count, source_slot, target_slot) not in {
        (40, "a", "b"), (80, "b", "a"),
    }:
        raise ResumeSafetyError("INVALID_RESUME_EDGE")
    return f"{EXPECTED_COMMIT}:cpu-pilot:{source_slot}:{source_count}:{target_slot}"


def validate_state(record: Any) -> dict:
    if not isinstance(record, dict) or set(record) != {
        "schema_version", "scope", "experiment_id", "source_commit",
        "phase", "confirmed_completed_tasks", "confirmed_slot", "intent",
        "review_reason",
    }:
        raise ResumeSafetyError("CONTROL_LEDGER_SCHEMA_INVALID")
    if (
        type(record["schema_version"]) is not int
        or record["schema_version"] != 1
        or record["scope"] != SCOPE
        or record["experiment_id"] != EXPECTED_ID
        or record["source_commit"] != EXPECTED_COMMIT
    ):
        raise ResumeSafetyError("CONTROL_LEDGER_PROVENANCE_MISMATCH")
    count = record["confirmed_completed_tasks"]
    slot = record["confirmed_slot"]
    phase = record["phase"]
    if type(count) is not int or count not in {40, 80, 100}:
        raise ResumeSafetyError("CONTROL_LEDGER_PROGRESS_INVALID")
    if slot not in {"a", "b"} or phase not in PHASES:
        raise ResumeSafetyError("CONTROL_LEDGER_PHASE_INVALID")
    intent = record["intent"]
    reason = record["review_reason"]
    if count == 100:
        if phase != "completed" or intent is not None or reason is not None:
            raise ResumeSafetyError("CONTROL_LEDGER_COMPLETED_CONFLICT")
    elif phase == "idle":
        if intent is not None or reason is not None:
            raise ResumeSafetyError("CONTROL_LEDGER_IDLE_CONFLICT")
    else:
        if not isinstance(intent, dict) or set(intent) != {
            "key", "source_slot", "source_count", "target_slot",
            "reserved_by_run", "submission_attempted",
        }:
            raise ResumeSafetyError("CONTROL_LEDGER_INTENT_INVALID")
        src = intent["source_slot"]
        dst = intent["target_slot"]
        n = intent["source_count"]
        if (
            src != slot or type(n) is not int or n != count
            or intent["key"] != intent_key(n, src, dst)
        ):
            raise ResumeSafetyError("CONTROL_LEDGER_INTENT_PROVENANCE_INVALID")
        run = intent["reserved_by_run"]
        if not isinstance(run, str) or not RUN_ID.fullmatch(run):
            raise ResumeSafetyError("CONTROL_LEDGER_RUN_ID_INVALID")
        if type(intent["submission_attempted"]) is not bool:
            raise ResumeSafetyError("CONTROL_LEDGER_ATTEMPT_FLAG_INVALID")
        if phase == "reserved" and intent["submission_attempted"]:
            raise ResumeSafetyError("RESERVED_BUT_SUBMISSION_MARKED_ATTEMPTED")
        if phase == "submitted" and not intent["submission_attempted"]:
            raise ResumeSafetyError("SUBMITTED_WITHOUT_ATTEMPT_MARKER")
        if phase == "needs_review" and reason not in REVIEW_REASONS:
            raise ResumeSafetyError("CONTROL_LEDGER_REVIEW_REASON_INVALID")
        if phase != "needs_review" and reason is not None:
            raise ResumeSafetyError("CONTROL_LEDGER_UNEXPECTED_REVIEW_REASON")
    return record


def reserve_once(record: dict, source_slot: str, source_count: int,
                 target_slot: str, run_id: str) -> tuple[str, dict]:
    """Pure reservation decision; caller MUST CAS-persist before submitting."""
    validate_state(record)
    key = intent_key(source_count, source_slot, target_slot)
    if not isinstance(run_id, str) or not RUN_ID.fullmatch(run_id):
        raise ResumeSafetyError("UNTRUSTED_RUN_IDENTIFIER")
    if record["phase"] == "completed":
        return "completed_no_submit", record
    if record["intent"] is not None:
        # A repeated request is never authorization to resubmit.
        if record["intent"]["key"] == key:
            return "duplicate_reservation_blocked", record
        return "different_reservation_blocked", record
    if record["phase"] != "idle":
        return "manual_review_required", record
    if record["confirmed_slot"] != source_slot or record["confirmed_completed_tasks"] != source_count:
        return "stale_or_regressed_checkpoint_blocked", record
    proposed = dict(record)
    proposed["phase"] = "reserved"
    proposed["intent"] = {
        "key": key, "source_slot": source_slot,
        "source_count": source_count, "target_slot": target_slot,
        "reserved_by_run": run_id,
        "submission_attempted": False,
    }
    validate_state(proposed)
    return "reservation_proposed_not_dispatched", proposed


def mark_before_external_submission(record: dict, run_id: str) -> dict:
    """Mark attempted BEFORE making an irreversible Kaggle push.

    This prevents an automatic second push after a crash. Without Kaggle's
    own idempotency token, manual remote reconciliation is still necessary.
    """
    validate_state(record)
    if record["phase"] != "reserved" or record["intent"]["reserved_by_run"] != run_id:
        raise ResumeSafetyError("SUBMISSION_CLAIM_NOT_OWNED")
    out = dict(record)
    out["phase"] = "submitted"
    out["intent"] = {**record["intent"], "submission_attempted": True}
    validate_state(out)
    return out


def require_review(record: dict, reason: str) -> dict:
    """Fail closed; never erase an unresolved reservation or retry."""
    validate_state(record)
    if reason not in REVIEW_REASONS:
        raise ResumeSafetyError("UNKNOWN_RECOVERY_REASON")
    if record["phase"] not in {"reserved", "submitted", "needs_review"}:
        raise ResumeSafetyError("NO_PENDING_SUBMISSION_TO_REVIEW")
    out = {**record, "phase": "needs_review", "review_reason": reason}
    validate_state(out)
    return out


def reconcile_observed(record: dict, remote: dict[str, str],
                       verified_checkpoints: dict[str, dict]) -> tuple[str, dict]:
    """Require verified current Kaggle evidence. Never infer success from a push response."""
    validate_state(record)
    if set(remote) != {"a", "b"}:
        return "pause_remote_state_unavailable", record

    base, _reason = assess(remote, verified_checkpoints, quota_signal="unknown")
    if base == "wait":
        return "wait_active_kaggle_job", record
    if base == "pause" and any(v in {"error", "failed", "cancelled", "cancelacknowledged"}
                               for v in remote.values()):
        if record["intent"] is not None:
            return "needs_manual_failure_review", require_review(record, "REMOTE_SESSION_FAILED")
        return "pause_failed_remote_job", record

    if record["phase"] == "completed":
        if base == "stop":
            # Confirm that the finished record agrees with the latest verified output.
            valid = verified_checkpoints.get(record["confirmed_slot"], {})
            if (remote[record["confirmed_slot"]] == "complete"
                    and len(valid.get("completed_task_ids", [])) == MAX_TASKS):
                return "stop_completed_no_duplicate", record
        return "pause_ledger_remote_disagreement", record

    if record["intent"] is None:
        # This is a conservative audit-only controller. Unknown CPU quota
        # does not authorize creating a new reservation.
        return ("pause_cpu_quota_unknown" if base == "pause"
                else "pause_unclaimed_resume"), record

    if record["phase"] == "needs_review":
        return "manual_review_required", record
    if record["phase"] == "reserved":
        # A reserved but unattempted submission is not self-retrying.
        return "reservation_held_no_auto_submit", record

    target = record["intent"]["target_slot"]
    if remote.get(target) != "complete":
        return "submission_unknown_manual_reconciliation", require_review(
            record, "TARGET_REMOTE_STATUS_UNKNOWN"
        )
    target_state = verified_checkpoints.get(target)
    if not isinstance(target_state, dict):
        return "checkpoint_missing_manual_review", require_review(
            record, "TARGET_REMOTE_STATUS_UNKNOWN"
        )
    # Assess has already verified metadata and status for all completed slots.
    if base == "pause" and _reason != "QUOTA_NOT_INDEPENDENTLY_VERIFIED":
        # Unknown CPU quota blocks NEW launches, not verification of a
        # previously attempted run that has a valid saved checkpoint.
        return "checkpoint_invalid_manual_review", require_review(
            record, "TARGET_REMOTE_STATUS_UNKNOWN"
        )
    count = len(target_state["completed_task_ids"])
    old = record["intent"]["source_count"]
    if count <= old:
        return "progress_not_advancing_manual_review", require_review(
            record, "TARGET_CHECKPOINT_NOT_ADVANCING"
        )
    if count == MAX_TASKS:
        return "reconciled_completed", make_state(MAX_TASKS, target, "completed")
    if (old, target, count) != (40, "b", 80):
        return "unexpected_resume_progress_manual_review", require_review(
            record, "TARGET_CHECKPOINT_NOT_ADVANCING"
        )
    return "reconciled_advanced", make_state(count, target, "idle")


def proposed_compare_and_swap(
    load_current: Callable[[], tuple[dict, str]],
    save_if_sha: Callable[[str, dict], bool],
    source_slot: str, source_count: int, target_slot: str, run_id: str,
) -> tuple[str, dict]:
    """CAS reservation protocol for a future dispatch-capable integration.

    A conflict never retries automatically. The returned proposal is not
    submission authorization. Crucially, a target GitHub branch must enforce
    isolated GitHub Actions concurrency around every external sender.
    """
    current, sha = load_current()
    status, proposed = reserve_once(
        current, source_slot, source_count, target_slot, run_id
    )
    if status != "reservation_proposed_not_dispatched":
        return status, current
    if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise ResumeSafetyError("UNTRUSTED_GITHUB_LEDGER_SHA")
    if not save_if_sha(sha, proposed):
        # GitHub Contents API 409/422 conflicts must fail closed.
        return "concurrent_claim_conflict_no_submit", current
    return "durably_reserved_no_submit", proposed


def live_inspect(ledger_path: Path, work_dir: Path) -> tuple[str, str]:
    """Uses Kaggle CLI status/output read-only. No GitHub/Kaggle mutations."""
    record = validate_state(json.loads(ledger_path.read_text(encoding="utf-8")))
    remote, checkpoints = status_and_checkpoints(work_dir)
    outcome, state = reconcile_observed(record, remote, checkpoints)
    if outcome == "stop_completed_no_duplicate" and state["phase"] == "completed":
        return "STOP", "ALREADY_COMPLETE_DUPLICATE_SUBMISSION_BLOCKED"
    if outcome.startswith("wait"):
        return "WAIT", "REMOTE_JOB_ACTIVE"
    return "PAUSE", "MANUAL_RECONCILIATION_OR_CPU_QUOTA_CHECK_REQUIRED"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ledger", type=Path,
        default=Path("monitoring/synthetic-resume-control.json")
    )
    parser.add_argument("--work-dir", type=Path, default=Path("build/duplicate-resume"))
    args = parser.parse_args()
    try:
        result, reason = live_inspect(args.ledger, args.work_dir)
    except Exception:
        # No exception text: Kaggle errors sometimes embed private URL details.
        result, reason = "PAUSE", "REMOTE_OR_CONTROL_LEDGER_VALIDATION_FAILED"
    print(f"IDEMPOTENT_RESUME_CONTROLLER: {result} {reason}")
    print("IDEMPOTENT_RESUME_CONTROLLER: NO_REMOTE_SUBMISSION")
    return 0 if result in {"STOP", "WAIT"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
