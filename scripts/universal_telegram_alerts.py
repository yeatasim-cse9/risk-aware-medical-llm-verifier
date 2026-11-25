"""Step 10.5F — Telegram success/failure alerts for allowlisted research jobs.

Cloud workflow_run -> sanitized GitHub status notification.
Scheduled poll -> only registered, synthetic Kaggle jobs whose terminal event
has not been recorded. Actual Kaggle completion requires verified saved output.

Existing completed notebooks are seeded as HISTORICAL_SUPPRESSED: no backfill.
No code in this module launches Kaggle, trains a model, or handles patient data.
Telegram delivery plus GitHub-ledger write is not atomic (at-least-once only).
"""
from __future__ import annotations

import argparse
import base64
import copy
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import urllib.error
import urllib.request

from kaggle_status import ACTIVE, TERMINAL, check
from telegram_notification_test import NotificationError, send_message
from verify_checkpoint import verify as verify_cpu
from verify_gpu_smoke import verify as verify_cuda
from verify_tiny_gpu_inference import verify as verify_tiny
from verify_medhallu_audit import verify as verify_metadata
from cpu_telegram_e2e import verify as verify_cpu_e2e

REPO = "yeatasim-cse9/risk-aware-medical-llm-verifier"
LEDGER_PATH = "monitoring/universal-telegram-alert-ledger.json"
BRANCH = "main"
SCOPE = "approved_synthetic_infrastructure_only"
CPU_SHA = "1a90191ae2734a74807eb53fa7d5b27a4e44c78f"
GPU_SHA = "d92b5d6054b6f3802d146e5a1662127e61223993"
TINY_SHA = "184a1270aa6aaef8bc0e82c142e3285d0ea82084"
MAX_HISTORY = 500

# To monitor a NEW notebook, explicitly review the license, privacy, source
# provenance and checkpoint verifier; then update this allowlist AND ledger.
# Never auto-discover or ingest arbitrary Kaggle notebooks.
WATCHED = {
    "cpu-a": {
        "kernel": "yeatasimbillah/medical-llm-cpu-auto-a",
        "kind": "cpu_a", "commit": CPU_SHA,
        "label": "Synthetic CPU pilot A",
    },
    "cpu-b": {
        "kernel": "yeatasimbillah/medical-llm-cpu-auto-b",
        "kind": "cpu_b", "commit": CPU_SHA,
        "label": "Synthetic CPU pilot B",
    },
    "gpu-smoke": {
        "kernel": "yeatasimbillah/medical-llm-gpu-smoke",
        "kind": "cuda_smoke", "commit": GPU_SHA,
        "label": "Private CUDA smoke",
    },
    "gpu-tiny": {
        "kernel": "yeatasimbillah/medical-llm-gpu-inference-pilot",
        "kind": "tiny_inference", "commit": TINY_SHA,
        "label": "Untrained GPU Transformer pilot",
    },
    "cpu-telegram-e2e-v1": {
        "kernel": "yeatasimbillah/medical-llm-telegram-cpu-e2e-v1",
        "kind": "cpu_telegram_e2e", "commit": "fixed-protocol-v1",
        "label": "Private synthetic CPU Telegram e2e smoke v1",
    },
    "medhallu-metadata": {
        "kernel": "yeatasimbillah/medical-llm-medhallu-metadata-audit",
        "kind": "medhallu_metadata",
        "commit": "9fcfaad9ec13c4fd524e14ae51d85092eca6fe46",
        "label": "MedHallu public metadata audit (not E1–E7)",
    },
}

WORKFLOWS = {
    "Research Repository CI",
    "Kaggle CPU Pilot Controller",
    "Kaggle CPU Telegram End-to-End Smoke",
    "Kaggle Private GPU Smoke Controller",
    "Kaggle GPU Tiny Transformer Inference Controller",
    "Verify Submitted Private Kaggle GPU Smoke",
    "Verify Submitted Tiny GPU Inference and Checkpoint",
    "Verify Submitted Private Kaggle GPU Smoke",
    "Step 10.5E Read Only End-to-End Synthetic Infrastructure Audit",
    "Pretrained Medical Model License Audit - No Downloads",
    "MiniLM Pinned Model Metadata and GPU Feasibility - No Weights",
    "Scheduled Synthetic Kaggle Telegram Monitor",
    "MedHallu Metadata Audit Controller",
    "MedHallu Private Native Row Audit Controller",
    "MedHallu E1 Candidate Private Split Audit Controller",
    "Kaggle MedHallu Read-only Failure Diagnostic",
    "E1 MedHallu Baseline Experiment",
    "E2 MedVAL Retrieval-Free Judge",
    "E3 MedVAL Evidence NLI",
    "E4 MedVAL Hybrid Verifier",
    "E5 BioASQ External Generalization",
    "E6 RAG Verification And Revision",
    "E7 Component Ablations",
}
GH_TERMINAL = {"success", "failure", "cancelled", "timed_out"}
GH_EVENTS = {"push", "workflow_dispatch", "schedule"}


class AlertError(ValueError):
    """Safe diagnostic code; exceptions never include remote message bodies."""


def validate_ledger(data: object) -> dict:
    if not isinstance(data, dict) or set(data) != {
        "schema_version", "scope", "bootstrap_sent", "kaggle_events",
        "github_events",
    }:
        raise AlertError("INVALID_ALERT_LEDGER_SCHEMA")
    if type(data["schema_version"]) is not int or data["schema_version"] != 1:
        raise AlertError("INVALID_ALERT_LEDGER_VERSION")
    if data["scope"] != SCOPE or type(data["bootstrap_sent"]) is not bool:
        raise AlertError("INVALID_ALERT_LEDGER_SCOPE")
    states = data["kaggle_events"]
    if not isinstance(states, dict) or set(states) != set(WATCHED):
        raise AlertError("INVALID_ALERT_LEDGER_KERNEL_SET")
    for job, entry in states.items():
        if not isinstance(entry, dict) or set(entry) != {"last_event", "delivery"}:
            raise AlertError("INVALID_ALERT_LEDGER_KERNEL_ENTRY")
        event = entry["last_event"]
        if event is not None and event not in {
            f"{job}:complete",
            f"{job}:failed",
            f"{job}:verification_failed",
            f"{job}:started",
        }:
            raise AlertError("INVALID_ALERT_LEDGER_EVENT_KEY")
        if entry["delivery"] not in {
            "none", "historical_suppressed", "sent",
        }:
            raise AlertError("INVALID_ALERT_LEDGER_DELIVERY_STATE")
        if (event is None) != (entry["delivery"] == "none"):
            raise AlertError("INVALID_ALERT_LEDGER_EVENT_DELIVERY")
        if entry["delivery"] == "historical_suppressed" and event != f"{job}:complete":
            raise AlertError("INVALID_HISTORICAL_COMPLETION_SEED")
    history = data["github_events"]
    if not isinstance(history, list) or len(history) > MAX_HISTORY:
        raise AlertError("INVALID_GITHUB_ALERT_HISTORY")
    if len(set(history)) != len(history):
        raise AlertError("DUPLICATE_GITHUB_HISTORY_KEYS")
    if any(
        not isinstance(v, str)
        or re.fullmatch(r"github:[1-9][0-9]{0,19}:[1-9][0-9]{0,5}:(started|success|failure|cancelled|timed_out)", v) is None
        for v in history
    ):
        raise AlertError("INVALID_GITHUB_ALERT_KEY")
    return data


def github_content_api(token: str, method: str, payload: dict | None = None) -> dict:
    if not token:
        raise AlertError("GITHUB_READ_WRITE_TOKEN_MISSING")
    if method not in {"GET", "PUT"}:
        raise AlertError("UNSUPPORTED_GITHUB_ACTION")
    url = f"https://api.github.com/repos/{REPO}/contents/{LEDGER_PATH}"
    if method == "GET":
        url += "?ref=" + BRANCH
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url, method=method, data=body,
        headers={
            "Authorization": "Bearer " + token,
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "medical-llm-synthetic-telegram-alerts",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            result = json.load(response)
    except urllib.error.HTTPError as exc:
        if exc.code in {409, 422}:
            raise AlertError("LEDGER_WRITE_CONFLICT_OR_REJECTION") from None
        if exc.code in {401, 403}:
            raise AlertError("GITHUB_LEDGER_ACCESS_DENIED") from None
        raise AlertError("GITHUB_LEDGER_REQUEST_FAILED") from None
    except (urllib.error.URLError, OSError, TimeoutError):
        raise AlertError("GITHUB_LEDGER_NETWORK_FAILURE") from None
    except (ValueError, UnicodeError):
        raise AlertError("GITHUB_LEDGER_UNEXPECTED_RESPONSE") from None
    if not isinstance(result, dict):
        raise AlertError("GITHUB_LEDGER_INVALID_RESPONSE")
    return result


def read_ledger(token: str) -> tuple[dict, str]:
    result = github_content_api(token, "GET")
    sha, raw = result.get("sha"), result.get("content")
    if (not isinstance(sha, str) or re.fullmatch(r"[0-9a-f]{40}", sha) is None
            or result.get("encoding") != "base64"
            or not isinstance(raw, str)):
        raise AlertError("GITHUB_LEDGER_UNTRUSTED_BLOB")
    try:
        data = json.loads(base64.b64decode(raw, validate=False).decode("utf-8"))
    except (ValueError, UnicodeError, TypeError):
        raise AlertError("GITHUB_LEDGER_DECODE_ERROR") from None
    return validate_ledger(data), sha


def write_ledger(token: str, old_sha: str, data: dict) -> None:
    validate_ledger(data)
    if not isinstance(old_sha, str) or re.fullmatch(r"[0-9a-f]{40}", old_sha) is None:
        raise AlertError("GITHUB_LEDGER_UNTRUSTED_WRITE_SHA")
    body = base64.b64encode(
        (json.dumps(data, sort_keys=True, indent=2) + "\n").encode("utf-8")
    ).decode("ascii")
    response = github_content_api(token, "PUT", {
        "message": "chore(alerts): remember sanitized synthetic status delivery",
        "content": body, "sha": old_sha, "branch": BRANCH,
    })
    if not isinstance(response.get("commit"), dict):
        raise AlertError("GITHUB_LEDGER_WRITE_UNCONFIRMED")


def bootstrap(github_token: str, bot_token: str, chat_id: str) -> str:
    """One-time informative delivery to verify new alert integration."""
    data, sha = read_ledger(github_token)
    if data["bootstrap_sent"]:
        return "bootstrap_already_recorded"
    message = (
        "🔔 Medical LLM research alerts enabled\n"
        "GitHub workflow completion/failure + registered synthetic Kaggle "
        "completion/failure monitoring configured.\n"
        "No medical data and no new GPU job."
    )
    send_message(bot_token, chat_id, message)
    updated = copy.deepcopy(data)
    updated["bootstrap_sent"] = True
    write_ledger(github_token, sha, updated)
    return "bootstrap_sent"


def github_alert_payload(event: object) -> tuple[str, str] | None:
    """Only GitHub-signed workflow_run event data, never untrusted job logs."""
    if not isinstance(event, dict) or event.get("action") not in {"in_progress", "completed"}:
        return None
    run = event.get("workflow_run")
    if not isinstance(run, dict):
        return None
    if (run.get("name") not in WORKFLOWS
            or run.get("head_branch") != "main"
            or run.get("event") not in GH_EVENTS
            or not isinstance(run.get("head_repository"), dict)
            or run["head_repository"].get("full_name") != REPO):
        return None

    commit = run.get("head_commit")
    if isinstance(commit, dict) and isinstance(commit.get("message"), str):
        if commit["message"].startswith("chore(alerts):"):
            return None

    action = event["action"]
    conclusion = run.get("conclusion") if action == "completed" else "started"
    if conclusion not in GH_TERMINAL | {"started"}:
        return None
    number, attempt = run.get("id"), run.get("run_attempt")
    if (type(number) is not int or not 1 <= number <= 10**20 - 1
            or type(attempt) is not int or not 1 <= attempt <= 999999):
        return None

    name = run["name"]
    # Continuous CI successes from every push would create noise and can
    # feed back through ledger-write commits; CI failures still get alerts.
    if (
        (name == "Research Repository CI" and run["event"] == "push"
         and conclusion in {"started", "success"})
        or (name == "Scheduled Synthetic Kaggle Telegram Monitor"
            and run["event"] == "schedule"
            and conclusion in {"started", "success"})
    ):
        return None

    if conclusion == "started":
        message = (
            f"▶️ GitHub workflow STARTED: {name}\n"
            f"Run #{number} (attempt {attempt}).\n"
            "Controller or CI start only; this is NOT Kaggle execution or a scientific result."
        )
    elif conclusion == "success":
        message = (
            f"✅ GitHub workflow finished: {name}\n"
            f"Run #{number} (attempt {attempt}).\n"
            "This confirms GitHub status, not Kaggle runtime completion."
        )
    else:
        message = (
            f"⚠️ GitHub workflow {conclusion}: {name}\n"
            f"Run #{number} (attempt {attempt}). Check GitHub Actions logs.\n"
            "No private or clinical data included."
        )
    key = f"github:{number}:{attempt}:{conclusion}"
    if len(message) > 500:
        raise AlertError("GITHUB_ALERT_MESSAGE_EXCEEDS_SAFE_LIMIT")
    return key, message


def handle_github_event(github_token: str, bot_token: str, chat_id: str,
                        event: object) -> str:
    item = github_alert_payload(event)
    if item is None:
        return "unregistered_or_nonactionable_workflow_skipped"
    key, message = item
    ledger, _ = read_ledger(github_token)
    if key in ledger["github_events"]:
        return "github_run_duplicate_skipped"
    # Send once; only ledger persistence is retried on competing events.
    send_message(bot_token, chat_id, message)
    for attempt in range(8):
        fresh, sha = read_ledger(github_token)
        if key in fresh["github_events"]:
            return "github_event_already_recorded_after_delivery"
        updated = copy.deepcopy(fresh)
        updated["github_events"] = (updated["github_events"] + [key])[-MAX_HISTORY:]
        try:
            write_ledger(github_token, sha, updated)
            return "github_event_sent"
        except AlertError as exc:
            if str(exc) != "LEDGER_WRITE_CONFLICT_OR_REJECTION" or attempt == 7:
                raise
    raise AlertError("GITHUB_LEDGER_RETRIES_EXHAUSTED")


def check_checkpoint(kind: str, folder: Path, commit: str) -> str:
    """A complete Kaggle status ALONE does not imply successful computation."""
    if kind in {"cpu_a", "cpu_b"}:
        state = verify_cpu(folder, expected_commit=commit)
        n = len(state["completed_task_ids"])
        if (kind == "cpu_a" and (n, state["status"]) != (100, "completed")):
            raise AlertError("CPU_A_CHECKPOINT_NOT_FINAL")
        if (kind == "cpu_b" and (n, state["status"]) != (80, "in_progress")):
            raise AlertError("CPU_B_SESSION_PROGRESS_MISMATCH")
        return f"Saved verified checkpoint {n}/100 (synthetic CPU session)."
    if kind == "cpu_telegram_e2e":
        verify_cpu_e2e(folder)
        return "Independent no-data CPU synthetic output verified (5/5)."
    if kind == "cuda_smoke":
        state = verify_cuda(folder, expected_commit=commit)
        if state["device"] != "cuda:0":
            raise AlertError("GPU_SMOKE_DEVICE_MISMATCH")
        return "Saved CUDA smoke checkpoint verified."
    if kind == "medhallu_metadata":
        verified = verify_metadata(folder, expected_commit=commit)
        if verified.get("scientific_experiments_executed") is not False:
            raise AlertError("METADATA_MISLABELED_AS_EXPERIMENT")
        return "MedHallu public metadata manifest independently verified; no E1–E7 result."
    if kind == "tiny_inference":
        state = verify_tiny(folder, expected_commit=commit)
        if state["device"] != "cuda:0" or state["model_pretrained"] is not False:
            raise AlertError("TINY_MODEL_CHECKPOINT_MISMATCH")
        return "Untrained Transformer CUDA checkpoint verified."
    raise AlertError("UNREGISTERED_CHECKPOINT_KIND")


def poll_kaggle(github_token: str, bot_token: str, chat_id: str,
                work_dir: Path) -> str:
    ledger, _ = read_ledger(github_token)
    pending = [
        job for job in WATCHED
        if ledger["kaggle_events"][job]["last_event"] != f"{job}:complete"
    ]
    if not pending:
        # Existing synthetic jobs already terminal: no wasteful Kaggle calls.
        return "all_registered_historical_jobs_already_complete"

    count = 0
    for job in pending:
        info = WATCHED[job]
        # A failed/unknown status or output never changes a completion claim.
        state = check(info["kernel"], "any")
        if state == "missing":
            continue
        if state in ACTIVE:
            key = f"{job}:started"
            fresh, sha = read_ledger(github_token)
            prior = fresh["kaggle_events"][job]["last_event"]
            if prior is not None:
                continue
            message = (
                f"▶️ Kaggle notebook STARTED/QUEUED (observed): {info['label']}\n"
                "Actual work may still be queued; no E1–E7 result claimed."
            )
            send_message(bot_token, chat_id, message)
            updated = copy.deepcopy(fresh)
            updated["kaggle_events"][job] = {"last_event": key, "delivery": "sent"}
            write_ledger(github_token, sha, updated)
            count += 1
            continue
        if state not in TERMINAL:
            raise AlertError("KAGGLE_REMOTE_STATE_UNRECOGNIZED")
        if state == "complete":
            dest = work_dir / job
            dest.mkdir(parents=True, exist_ok=True)
            proc = subprocess.run(
                ["kaggle", "kernels", "output", info["kernel"], "-p", str(dest)],
                capture_output=True, text=True, timeout=180,
            )
            if proc.returncode != 0:
                state = "verification_failed"
                summary = "saved output unavailable; manual review required."
            else:
                try:
                    summary = check_checkpoint(info["kind"], dest, info["commit"])
                except Exception:
                    state = "verification_failed"
                    summary = "saved checkpoint failed independent verification."
        else:
            state, summary = "failed", "Kaggle execution failed or was cancelled."

        key = f"{job}:{state}"
        fresh, sha = read_ledger(github_token)
        if fresh["kaggle_events"][job]["last_event"] == key:
            continue
        if state == "complete":
            message = f"✅ Kaggle job completed: {info['label']}\n{summary}\nNo clinical experiment results implied."
        else:
            message = f"⚠️ Kaggle job requires attention: {info['label']}\n{summary}\nCheck Kaggle privately; no results claimed."
        send_message(bot_token, chat_id, message)
        updated = copy.deepcopy(fresh)
        updated["kaggle_events"][job] = {
            "last_event": key, "delivery": "sent",
        }
        write_ledger(github_token, sha, updated)
        count += 1
    return f"kaggle_new_alerts_{count}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("bootstrap", "github", "poll"), required=True)
    parser.add_argument("--work-dir", type=Path, default=Path("build/telegram-universal"))
    args = parser.parse_args()
    try:
        github_token = os.environ.get("GH_LEDGER_TOKEN", "")
        bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
        chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
        if args.mode == "bootstrap":
            result = bootstrap(github_token, bot_token, chat_id)
        elif args.mode == "github":
            path = os.environ.get("GITHUB_EVENT_PATH", "")
            if not path:
                raise AlertError("GITHUB_EVENT_PAYLOAD_MISSING")
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
            result = handle_github_event(github_token, bot_token, chat_id, payload)
        else:
            result = poll_kaggle(github_token, bot_token, chat_id, args.work_dir)
    except (AlertError, NotificationError) as exc:
        # No raw Telegram API URLs, messages, tokens or patient text.
        print("UNIVERSAL_ALERTS: FAILED " + str(exc), file=sys.stderr)
        return 1
    except Exception:
        print("UNIVERSAL_ALERTS: FAILED SAFE_UNEXPECTED_ERROR", file=sys.stderr)
        return 1
    print("UNIVERSAL_ALERTS: " + result.upper())
    print("UNIVERSAL_ALERTS: NO_KAGGLE_SUBMISSION")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
