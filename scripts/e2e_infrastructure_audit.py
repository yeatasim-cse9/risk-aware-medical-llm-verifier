"""Step 10.5E: read-only integration audit of four historical synthetic notebooks.

Checks CPU A/B, private GPU CUDA smoke, private toy Transformer inference,
frozen model-card metadata, and existing synthetic Telegram/completion ledgers.

NEVER submits notebooks, spends GPU time, sends Telegram, fetches medical data
or downloads pretrained model weights. It does NOT replay actual notifications
or prove a scheduled trigger; it only cross-checks persisted evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from audit_pretrained_models import validate as validate_model_audit
from auto_resume_preflight import assess
from idempotent_resume_controller import validate_state, reconcile_observed
from kaggle_status import check
from notify_synthetic_pilot_status import build_status
from periodic_pilot_monitor import validate_ledger, progress_key
from safe_monitor_decision import EXPECTED_COMMIT, EXPECTED_ID, EXPECTED_TOTAL
from verify_checkpoint import verify as verify_cpu
from verify_gpu_smoke import verify as verify_cuda_smoke
from verify_tiny_gpu_inference import verify as verify_tiny_model

REPO_ROOT = Path(__file__).resolve().parents[1]
SLOTS = {
    "cpu_a": "yeatasimbillah/medical-llm-cpu-auto-a",
    "cpu_b": "yeatasimbillah/medical-llm-cpu-auto-b",
    "gpu_smoke": "yeatasimbillah/medical-llm-gpu-smoke",
    "gpu_inference": "yeatasimbillah/medical-llm-gpu-inference-pilot",
}
FROZEN_SHAS = {
    "cpu_a": EXPECTED_COMMIT,
    "cpu_b": EXPECTED_COMMIT,
    "gpu_smoke": "d92b5d6054b6f3802d146e5a1662127e61223993",
    "gpu_inference": "184a1270aa6aaef8bc0e82c142e3285d0ea82084",
}
MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REV = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
WEIGHT_METADATA_SHA = "53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db"


class IntegrationError(ValueError):
    """Use generic, non-sensitive errors for all cloud and provenance failures."""


def load_json(path: Path) -> object:
    data = path.read_bytes()
    if len(data) > 200_000:
        raise IntegrationError("STATIC_AUDIT_RECORD_TOO_LARGE")
    return json.loads(data.decode("utf-8-sig"))


def static_contract(root: Path) -> tuple[dict, dict, dict]:
    """Require previously completed control ledgers and still-closed model gates."""
    telegram = validate_ledger(load_json(
        root / "monitoring/synthetic-pilot-telegram-ledger.json"
    ))
    control = validate_state(load_json(
        root / "monitoring/synthetic-resume-control.json"
    ))
    pin = load_json(root / "configs/model-pins/all-minilm-l6-v2.json")
    validate_model_audit(load_json(root / "configs/pretrained-model-audit.json"))

    if not isinstance(pin, dict) or (
        pin.get("schema_version") != 1
        or pin.get("model_id") != MODEL
        or pin.get("model_repo_revision_sha") != MODEL_REV
        or pin.get("hub_model_safetensors_lfs_sha256_metadata") != WEIGHT_METADATA_SHA
        or pin.get("model_card_license_claim") != "apache-2.0"
        or pin.get("weight_sha256_independently_checked") is not None
        or pin.get("tokenizer_sha256_independently_checked") is not None
        or pin.get("max_cuda_memory_allocated_bytes_measured") is not None
        or pin.get("model_weights_downloaded") is not False
        or pin.get("model_execution_authorized") is not False
        or pin.get("patient_or_medval_data_transfer_authorized") is not False
        or pin.get("kaggle_accelerator_fit_proven") is not False
    ):
        raise IntegrationError("PINNED_MODEL_PERMISSION_OR_PROVENANCE_CONFLICT")

    last = telegram["last_notified"]
    if (
        telegram["experiment_id"] != EXPECTED_ID
        or last["count"] != EXPECTED_TOTAL
        or last["slot"] != "a"
        or last["status"] != "completed"
        or last["key"] != progress_key(EXPECTED_TOTAL, "completed")
        or control["phase"] != "completed"
        or control["confirmed_completed_tasks"] != EXPECTED_TOTAL
        or control["confirmed_slot"] != "a"
        or control["source_commit"] != EXPECTED_COMMIT
        or control["intent"] is not None
    ):
        raise IntegrationError("SYNTHETIC_COMPLETION_OR_DEDUP_LEDGER_CONFLICT")
    return telegram, control, pin


def audit_saved_outputs(root: Path, dirs: dict[str, Path],
                        statuses: dict[str, str]) -> dict:
    """Join independent Kaggle output verification with local state decisions."""
    if set(dirs) != set(SLOTS) or set(statuses) != set(SLOTS):
        raise IntegrationError("KAGGLE_AUDIT_SLOT_SET_INCOMPLETE")
    if any(statuses[slug] != "complete" for slug in SLOTS):
        raise IntegrationError("KAGGLE_REMOTE_SESSION_NOT_COMPLETE")

    telegram, control, pin = static_contract(root)

    cpu_a = verify_cpu(dirs["cpu_a"], expected_commit=FROZEN_SHAS["cpu_a"])
    cpu_b = verify_cpu(dirs["cpu_b"], expected_commit=FROZEN_SHAS["cpu_b"])
    if (
        len(cpu_a["completed_task_ids"]) != EXPECTED_TOTAL
        or cpu_a["status"] != "completed"
        or len(cpu_b["completed_task_ids"]) != 80
        or cpu_b["status"] != "in_progress"
    ):
        raise IntegrationError("KAGGLE_CPU_CHECKPOINT_PROGRESS_CONFLICT")
    decision, reason = assess(
        {"a": statuses["cpu_a"], "b": statuses["cpu_b"]},
        {"a": cpu_a, "b": cpu_b},
        quota_signal="unknown",
    )
    if (decision, reason) != ("stop", "ALL_100_TASKS_COMPLETE_NO_RESTART"):
        raise IntegrationError("COMPLETED_PILOT_RESUME_NOT_BLOCKED")
    recovery, _ = reconcile_observed(
        control,
        {"a": statuses["cpu_a"], "b": statuses["cpu_b"]},
        {"a": cpu_a, "b": cpu_b},
    )
    if recovery != "stop_completed_no_duplicate":
        raise IntegrationError("DUPLICATE_RESERVATION_OR_RESUME_GUARD_FAILED")

    # This checks only formatting and exact synthetic progress. Nothing is sent.
    message = build_status(cpu_a)
    if "SYNTHETIC CPU PILOT" not in message or "100/100" not in message:
        raise IntegrationError("SANITIZED_SYNTHETIC_NOTIFICATION_TEMPLATE_INVALID")
    if telegram["last_notified"]["key"] != progress_key(
        len(cpu_a["completed_task_ids"]), cpu_a["status"]
    ):
        raise IntegrationError("TELEGRAM_LEDGER_PROGRESS_NOT_ALIGNED")

    smoke = verify_cuda_smoke(
        dirs["gpu_smoke"], expected_commit=FROZEN_SHAS["gpu_smoke"]
    )
    inference = verify_tiny_model(
        dirs["gpu_inference"], expected_commit=FROZEN_SHAS["gpu_inference"]
    )
    if (
        smoke["device"] != "cuda:0"
        or smoke["gpu_available"] is not True
        or inference["device"] != "cuda:0"
        or inference["model_pretrained"] is not False
        or inference["medical_experiment"] is not False
        or inference["num_samples"] != 4
    ):
        raise IntegrationError("GPU_SYNTHETIC_PROVENANCE_INVALID")

    return {
        "schema_version": 1,
        "status": "historical_synthetic_cloud_outputs_verified",
        "scope": "read_only_historical_synthetic_infrastructure",
        "notebook_statuses": {slug: "complete" for slug in SLOTS},
        "cpu_a": {"completed_tasks": EXPECTED_TOTAL, "source_commit": EXPECTED_COMMIT},
        "cpu_b": {"completed_tasks": 80, "source_commit": EXPECTED_COMMIT},
        "resume_decision": "stop_no_restart",
        "resume_ledger_phase": control["phase"],
        "telegram_last_notified_count": telegram["last_notified"]["count"],
        "telegram_sanitized_message_template_valid": True,
        "gpu_smoke": {
            "cuda_verified_from_saved_checkpoint": True,
            "source_commit": FROZEN_SHAS["gpu_smoke"],
        },
        "gpu_tiny_inference": {
            "untrained_model_checkpoint_verified": True,
            "source_commit": FROZEN_SHAS["gpu_inference"],
            "num_synthetic_sequences": 4,
        },
        "pretrained_minilm": {
            "pinned_model_revision": pin["model_repo_revision_sha"],
            "model_bytes_independently_verified": False,
            "gpu_vram_measured": False,
            "model_execution_authorized": False,
        },
        "new_kaggle_job_submitted": False,
        "gpu_execution_triggered_by_audit": False,
        "telegram_message_sent_by_audit": False,
        "cron_firing_proven_by_audit": False,
        "automated_kaggle_resume_proven_by_audit": False,
        "medical_experiments_executed": False,
    }


def fetch_read_only(work_dir: Path) -> tuple[dict[str, Path], dict[str, str]]:
    """Only Kaggle status and output API calls; every reference is hard-coded."""
    statuses = {slug: check(kernel, "any") for slug, kernel in SLOTS.items()}
    if any(status != "complete" for status in statuses.values()):
        raise IntegrationError("REMOTE_SESSIONS_NOT_ALL_COMPLETE")
    dirs = {}
    for slug, kernel in SLOTS.items():
        target = work_dir / slug
        target.mkdir(parents=True, exist_ok=True)
        proc = subprocess.run(
            ["kaggle", "kernels", "output", kernel, "-p", str(target)],
            capture_output=True, text=True, timeout=180, check=False
        )
        if proc.returncode != 0:
            # Never print remote stderr: it can contain credentials/URLs.
            raise IntegrationError("READ_ONLY_KAGGLE_DOWNLOAD_FAILED")
        dirs[slug] = target
    return dirs, statuses


def make_report(report: dict, target: Path) -> None:
    # No arbitrary remote strings, patient data, token or account identifiers.
    data = dict(report)
    raw = json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")
    data["integrity_sha256"] = hashlib.sha256(raw).hexdigest()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, sort_keys=True, indent=2) + "\n", "utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path, default=Path("build/e2e-readonly"))
    parser.add_argument("--report", type=Path, default=Path("build/e2e-readonly/report.json"))
    args = parser.parse_args()
    try:
        # Check static permission gates before even querying remote Kaggle.
        static_contract(REPO_ROOT)
        paths, statuses = fetch_read_only(args.work_dir)
        report = audit_saved_outputs(REPO_ROOT, paths, statuses)
        make_report(report, args.report)
    except Exception:
        # No traceback or remote response bytes: never reveal secrets.
        print("E2E_READONLY_AUDIT: FAIL_CLOSED_REVIEW_REQUIRED")
        return 1
    print("E2E_READONLY_AUDIT: FOUR_KAGGLE_NOTEBOOKS_COMPLETE_AND_CHECKPOINTS_VERIFIED")
    print("E2E_READONLY_AUDIT: CPU_100_OF_100_STOP_DUPLICATE_RESUME_BLOCKED")
    print("E2E_READONLY_AUDIT: GPU_CUDA_SMOKE_AND_TINY_TRANSFORMER_VERIFIED")
    print("E2E_READONLY_AUDIT: TELEGRAM_LEDGER_AND_SANITIZED_TEMPLATE_CONSISTENT")
    print("E2E_READONLY_AUDIT: PRETRAINED_MODEL_DEPLOYMENT_STILL_BLOCKED")
    print("E2E_READONLY_AUDIT: NO_GPU_NO_SUBMIT_NO_TELEGRAM")
    print("E2E_READONLY_AUDIT: CRON_AND_FULL_AUTOMATIC_RESUME_NOT_PROVEN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
