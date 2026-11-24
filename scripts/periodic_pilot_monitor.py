"""Scheduled synthetic Kaggle progress monitoring with persistent deduplication.

Strictly limited to the verified CPU pilot. No Kaggle submission, dataset
reads, medical content, or user-supplied notification messages.
The public ledger contains synthetic counters only; tokens and chat IDs
never appear in files or logs.

"Exactly once" across Telegram + GitHub cannot be guaranteed; if Telegram
accepts a message and GitHub state persistence fails, the next check can
retry the message. Normally successful state updates suppress duplicates.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
import re
import sys
import urllib.error
import urllib.parse
import urllib.request

from notify_synthetic_pilot_status import (
    PILOT_ID, SOURCE_COMMIT, TOTAL, build_status,
)
from safe_monitor_decision import decide
from telegram_notification_test import send_message, NotificationError
from verify_checkpoint import verify

REPO = "yeatasim-cse9/risk-aware-medical-llm-verifier"
LEDGER_PATH = "monitoring/synthetic-pilot-telegram-ledger.json"
LEDGER_SCOPE = "synthetic_cpu_pilot_only"
SLOT = "a"
BRANCH = "main"


class MonitorError(Exception):
    """Safe status code that never includes credentials or chat identifiers."""


def progress_key(count: int, status: str) -> str:
    if type(count) is not int or not 0 <= count <= TOTAL:
        raise MonitorError("INVALID_TASK_COUNT")
    if status != ("completed" if count == TOTAL else "in_progress"):
        raise MonitorError("INCONSISTENT_PROGRESS_STATUS")
    return f"{SOURCE_COMMIT}:{SLOT}:{count}:{status}"


def ledger_with_progress(count: int, status: str) -> dict:
    return {
        "schema_version": 1,
        "scope": LEDGER_SCOPE,
        "experiment_id": PILOT_ID,
        "last_notified": {
            "slot": SLOT,
            "count": count,
            "status": status,
            "key": progress_key(count, status),
        },
    }


def validate_ledger(data: object) -> dict:
    if not isinstance(data, dict):
        raise MonitorError("INVALID_LEDGER")
    if (
        data.get("schema_version") != 1
        or data.get("scope") != LEDGER_SCOPE
        or data.get("experiment_id") != PILOT_ID
    ):
        raise MonitorError("LEDGER_IDENTITY_MISMATCH")
    entry = data.get("last_notified")
    if not isinstance(entry, dict) or entry.get("slot") != SLOT:
        raise MonitorError("LEDGER_ENTRY_INVALID")
    count, status = entry.get("count"), entry.get("status")
    if entry.get("key") != progress_key(count, status):
        raise MonitorError("LEDGER_FINGERPRINT_INVALID")
    return data


def request_github(token: str, method: str, payload: object | None = None) -> dict:
    if not token:
        raise MonitorError("GITHUB_TOKEN_MISSING")
    base = f"https://api.github.com/repos/{REPO}/contents/{LEDGER_PATH}"
    url = base + ("?ref=" + BRANCH if method == "GET" else "")
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=body, method=method,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "medical-llm-synthetic-monitor",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            decoded = json.load(resp)
    except urllib.error.HTTPError as exc:
        reasons = {
            401: "GITHUB_AUTH_FAILED",
            403: "GITHUB_LEDGER_PERMISSION_DENIED",
            404: "GITHUB_LEDGER_MISSING",
            409: "GITHUB_LEDGER_WRITE_CONFLICT",
            422: "GITHUB_LEDGER_REJECTED_UPDATE",
            429: "GITHUB_RATE_LIMIT",
        }
        raise MonitorError(reasons.get(exc.code, "GITHUB_API_HTTP_FAILURE")) from None
    except (urllib.error.URLError, TimeoutError, OSError):
        raise MonitorError("GITHUB_API_UNAVAILABLE") from None
    except (ValueError, TypeError):
        raise MonitorError("GITHUB_API_UNEXPECTED_JSON") from None
    if not isinstance(decoded, dict):
        raise MonitorError("GITHUB_API_INVALID_RESPONSE")
    return decoded


def read_ledger(token: str) -> tuple[dict, str]:
    result = request_github(token, "GET")
    sha = result.get("sha")
    if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise MonitorError("GITHUB_LEDGER_BLOB_SHA_INVALID")
    if result.get("encoding") != "base64":
        raise MonitorError("GITHUB_LEDGER_ENCODING_INVALID")
    encoded = result.get("content")
    if not isinstance(encoded, str):
        raise MonitorError("GITHUB_LEDGER_CONTENT_MISSING")
    try:
        data = json.loads(base64.b64decode(encoded, validate=False).decode("utf-8"))
    except (ValueError, TypeError, UnicodeError):
        raise MonitorError("GITHUB_LEDGER_PARSE_FAILED") from None
    return validate_ledger(data), sha


def write_ledger(token: str, previous_sha: str, data: dict) -> None:
    validate_ledger(data)
    encoded = base64.b64encode(
        (json.dumps(data, indent=2, sort_keys=True) + "\n").encode("utf-8")
    ).decode("ascii")
    response = request_github(token, "PUT", {
        "message": "chore(monitor): record verified synthetic Telegram progress",
        "content": encoded,
        "sha": previous_sha,
        "branch": BRANCH,
    })
    if not isinstance(response.get("commit"), dict):
        raise MonitorError("GITHUB_LEDGER_UPDATE_UNCONFIRMED")


def schedule_guard(token: str, event: str) -> bool:
    ledger, _ = read_ledger(token)
    entry = ledger["last_notified"]
    if event == "schedule" and entry["count"] == TOTAL and entry["status"] == "completed":
        # The immutable synthetic pilot is already finished. Skip even
        # Kaggle API polling on regular schedule to conserve free-tier limits.
        print("SCHEDULED_MONITOR: TERMINAL_ALREADY_REPORTED_SKIP_KAGGLE")
        return False
    print("SCHEDULED_MONITOR: POLL_ALLOWED")
    return True


def process_verified_checkpoint(directory: Path, github_token: str,
                                telegram_token: str, chat_id: str) -> str:
    # Validate actual persisted Kaggle results, not job success alone.
    state = verify(directory, expected_commit=SOURCE_COMMIT)
    message = build_status(state)
    decision, _reason = decide("complete", state, SLOT)
    if decision not in {"stop", "ready_for_review"}:
        raise MonitorError("UNSAFE_CHECKPOINT_DECISION")
    count = len(state["completed_task_ids"])
    status = state["status"]
    key = progress_key(count, status)

    ledger, sha = read_ledger(github_token)
    last = ledger["last_notified"]
    if key == last["key"]:
        print("SCHEDULED_MONITOR: DUPLICATE_PROGRESS_SKIPPED")
        if decision == "stop":
            print("SCHEDULED_MONITOR: EXPERIMENT_COMPLETE_NO_RESTART")
        return "duplicate"

    if count <= last["count"]:
        # All legitimate progress in this immutable synthetic pilot is
        # strictly increasing. Fail closed on regression or inconsistency.
        raise MonitorError("PROGRESS_REGRESSION_OR_REPLAY_DETECTED")
    if not telegram_token or not chat_id:
        raise MonitorError("TELEGRAM_SECRETS_MISSING")

    # Delivery before ledger save gives at-least-once, not exactly-once,
    # semantics. A failed delivery never marks a message as notified.
    send_message(telegram_token, chat_id, message)
    write_ledger(github_token, sha, ledger_with_progress(count, status))
    print("SCHEDULED_MONITOR: NEW_VERIFIED_PROGRESS_NOTIFIED_AND_PERSISTED")
    if decision == "stop":
        print("SCHEDULED_MONITOR: EXPERIMENT_COMPLETE_NO_RESTART")
    return "notified"


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="operation", required=True)
    sub.add_parser("guard")
    proc = sub.add_parser("process")
    proc.add_argument("checkpoint_dir", type=Path)
    args = parser.parse_args()

    try:
        token = os.environ.get("GH_LEDGER_TOKEN", "")
        if args.operation == "guard":
            event = os.environ.get("GITHUB_EVENT_NAME", "")
            if event not in {"push", "schedule", "workflow_dispatch"}:
                raise MonitorError("UNAUTHORIZED_WORKFLOW_EVENT")
            poll = schedule_guard(token, event)
            output = os.environ.get("GITHUB_OUTPUT")
            if not output:
                raise MonitorError("GITHUB_OUTPUT_MISSING")
            with open(output, "a", encoding="utf-8") as handle:
                handle.write("poll=" + ("true" if poll else "false") + "\n")
        else:
            process_verified_checkpoint(
                args.checkpoint_dir, token,
                os.environ.get("TELEGRAM_BOT_TOKEN", ""),
                os.environ.get("TELEGRAM_CHAT_ID", "").strip(),
            )
        return 0
    except (MonitorError, NotificationError) as exc:
        print(f"SCHEDULED_MONITOR_ERROR: {exc}", file=sys.stderr)
        return 1
    except Exception:
        # Avoid printing an unknown exception with a token-bearing URL.
        print("SCHEDULED_MONITOR_ERROR: UNEXPECTED_FAILURE", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
