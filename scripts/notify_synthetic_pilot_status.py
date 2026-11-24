"""Send status from a verified synthetic Kaggle checkpoint to Telegram.

This is a smoke test of GitHub -> Kaggle -> GitHub -> Telegram.
Never connect it to restricted clinical data or unsanitized predictions.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from telegram_notification_test import NotificationError, send_message
from verify_checkpoint import verify

PILOT_ID = "cpu-pilot-automation-001"
SOURCE_COMMIT = "1a90191ae2734a74807eb53fa7d5b27a4e44c78f"
TOTAL = 100


def build_status(state: dict) -> str:
    if state.get("experiment_id") != PILOT_ID:
        raise ValueError("Unexpected experiment ID")
    if state.get("git_commit") != SOURCE_COMMIT:
        raise ValueError("Unrecognized source commit")
    if state.get("total_tasks") != TOTAL:
        raise ValueError("Unexpected task count")
    ids = state.get("completed_task_ids")
    if not isinstance(ids, list) or ids != list(range(len(ids))):
        raise ValueError("Unexpected task ID set")
    completed = len(ids)
    if completed > TOTAL:
        raise ValueError("Too many completed tasks")
    status = state.get("status")
    if status != ("completed" if completed == TOTAL else "in_progress"):
        raise ValueError("Checkpoint status inconsistent")
    return (
        "✅ Medical LLM Research Automation — SYNTHETIC CPU PILOT\n"
        f"Experiment ID: {PILOT_ID}\n"
        f"Progress: {completed}/{TOTAL} ({completed * 100 // TOTAL}%)\n"
        f"Checkpoint status: {status}\n"
        "Kaggle notebook: medical-llm-cpu-auto-a\n"
        f"Verified code commit: {SOURCE_COMMIT[:12]}\n"
        "This is synthetic test data; no medical records were processed."
    )


def notify_verified_checkpoint(directory: Path) -> None:
    # Verify SHA256, all task digests, metadata and immutable source commit.
    state = verify(directory, expected_commit=SOURCE_COMMIT)
    message = build_status(state)
    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        raise NotificationError("MISSING_TELEGRAM_SECRETS")
    send_message(token, chat_id, message)
    print("VERIFIED KAGGLE CHECKPOINT STATUS DELIVERED TO TELEGRAM")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    try:
        notify_verified_checkpoint(args.directory)
    except NotificationError as exc:
        print(f"STATUS_DELIVERY_FAILED: {exc}")
        return 1
    except Exception:
        # Never print exceptions that might contain bot token URLs.
        print("STATUS_DELIVERY_FAILED: CHECKPOINT_INVALID_OR_UNKNOWN_ERROR")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
