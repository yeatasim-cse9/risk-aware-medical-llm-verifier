"""Step 10.4B: CPU-only Kaggle quota/cooldown guard (READ-ONLY).

Safe before introducing any automatic submit path. Current Kaggle CLI quota
reporting concerns GPU/TPU accelerators; it is not a reliable signal of
remaining CPU availability. Unknown quota always blocks partial-run resume.

No Kaggle kernels push, state mutation, Telegram message, or patient data.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import re
from pathlib import Path

from auto_resume_preflight import status_and_checkpoints, assess

PILOT_SCOPE = "synthetic-cpu-pilot-only"
COOLDOWN_HOURS = 6
MAX_ATTEMPTS_24H = 2
MAX_CONSECUTIVE_FAILURES = 2
MAX_ATTESTATION_AGE_MINUTES = 30
ISO_UTC = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z\Z")
ATTEMPT_ID = re.compile(r"ghrun-[1-9]\d{0,18}\Z")


class GuardError(ValueError):
    """Never embed user identifiers or remote response bodies in errors."""


def parse_utc(value: object) -> datetime:
    if not isinstance(value, str) or not ISO_UTC.fullmatch(value):
        raise GuardError("INVALID_UTC_TIMESTAMP")
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        raise GuardError("INVALID_UTC_TIMESTAMP") from None


def check_history(history: object, now: datetime) -> list[tuple[datetime, str]]:
    """Require an explicit *complete*, ordered operator record, never infer no past runs."""
    if not isinstance(history, dict):
        raise GuardError("ATTEMPT_HISTORY_UNAVAILABLE")
    if (
        history.get("schema_version") != 1
        or history.get("scope") != PILOT_SCOPE
        or history.get("source") != "operator_record"
        or history.get("complete") is not True
    ):
        raise GuardError("ATTEMPT_HISTORY_UNTRUSTED")
    attempts = history.get("attempts")
    if not isinstance(attempts, list) or len(attempts) > 1000:
        raise GuardError("ATTEMPT_HISTORY_INVALID")
    seen: set[str] = set()
    ordered: list[tuple[datetime, str]] = []
    for attempt in attempts:
        if not isinstance(attempt, dict):
            raise GuardError("ATTEMPT_HISTORY_INVALID")
        ident = attempt.get("id")
        if not isinstance(ident, str) or not ATTEMPT_ID.fullmatch(ident) or ident in seen:
            raise GuardError("ATTEMPT_HISTORY_INVALID")
        seen.add(ident)
        outcome = attempt.get("outcome")
        if outcome not in {"completed", "failed"}:
            # A submission still active or with unknown outcome is unsafe.
            raise GuardError("ATTEMPT_OUTCOME_UNCONFIRMED")
        stamp = parse_utc(attempt.get("submitted_at_utc"))
        if stamp > now:
            raise GuardError("ATTEMPT_TIMESTAMP_IN_FUTURE")
        if ordered and stamp <= ordered[-1][0]:
            raise GuardError("ATTEMPT_HISTORY_UNSORTED")
        ordered.append((stamp, outcome))
    return ordered


def evaluate(
    remote: dict[str, str],
    checkpoints: dict[str, dict],
    now: datetime,
    *,
    quota_signal: str = "unknown",
    quota_attested_at_utc: str | None = None,
    history: object = None,
) -> tuple[str, str]:
    """Return safe decision. There is intentionally NO auto-submit decision."""
    if not isinstance(now, datetime) or now.tzinfo is None or now.utcoffset() != timedelta(0):
        return ("pause", "INVALID_CURRENT_UTC_TIME")
    base, reason = assess(remote, checkpoints, quota_signal)
    if base != "ready_for_manual_review":
        # STOP (100/100) wins over missing quota history. Active job means WAIT.
        # Unknown quota or remote failures already PAUSE before any retry.
        return base, reason

    # This path requires explicit operator attestation and is still MANUAL ONLY.
    if quota_signal != "operator_attested" or quota_attested_at_utc is None:
        return ("pause", "QUOTA_ATTESTATION_REQUIRED")
    try:
        observed = parse_utc(quota_attested_at_utc)
    except GuardError:
        return ("pause", "QUOTA_ATTESTATION_TIMESTAMP_INVALID")
    age = now - observed
    if age < timedelta(0) or age > timedelta(minutes=MAX_ATTESTATION_AGE_MINUTES):
        return ("pause", "QUOTA_ATTESTATION_STALE_OR_FUTURE")

    try:
        attempts = check_history(history, now)
    except GuardError as exc:
        return ("pause", str(exc))

    consecutive_failures = 0
    for _, outcome in reversed(attempts):
        if outcome != "failed":
            break
        consecutive_failures += 1
    if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
        return ("pause", "REPEATED_FAILURES_REQUIRE_MANUAL_DIAGNOSIS")

    last = attempts[-1][0] if attempts else None
    if last is not None:
        cooldown = timedelta(hours=COOLDOWN_HOURS * (2 ** consecutive_failures))
        if now - last < cooldown:
            return ("pause", "COOLDOWN_OR_FAILURE_BACKOFF_ACTIVE")

    cutoff = now - timedelta(hours=24)
    recent = sum(1 for time, _ in attempts if time > cutoff)
    if recent >= MAX_ATTEMPTS_24H:
        return ("pause", "ROLLING_24H_ATTEMPT_LIMIT_REACHED")

    # An operator's attestation is not an API-measured CPU quota. Approval to
    # review does not confer permission to submit, and no submit code exists.
    return ("ready_for_manual_review", "COOLDOWN_AND_LOCAL_POLICY_CHECKS_PASSED")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path, default=Path("build/quota-cooldown-preflight"))
    args = parser.parse_args()
    try:
        remote, states = status_and_checkpoints(args.work_dir)
        decision, reason = evaluate(
            remote, states, datetime.now(timezone.utc),
            quota_signal="unknown", quota_attested_at_utc=None, history=None,
        )
    except Exception:
        # Deliberately do not print Kaggle HTTP errors, tokens, or remote data.
        print("QUOTA_COOLDOWN_GUARD: PAUSE REMOTE_OR_CHECKPOINT_ERROR")
        return 1
    print(f"QUOTA_COOLDOWN_GUARD: {decision.upper()} {reason}")
    print("QUOTA_COOLDOWN_GUARD: NO_KAGGLE_SUBMISSION")
    return 1 if decision == "pause" else 0


if __name__ == "__main__":
    raise SystemExit(main())
