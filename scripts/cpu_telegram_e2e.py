"""Build and verify a dedicated CPU-only, no-data Kaggle notification smoke job."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

KERNEL = "yeatasimbillah/medical-llm-telegram-cpu-e2e-v1"
PROTOCOL = "telegram-cpu-e2e-v1"
OUTPUT_NAME = "telegram_cpu_e2e_manifest.json"

# Fixed, deterministic, harmless smoke payload. It is NOT a research experiment.
RUNNER = '''import hashlib
import json
from pathlib import Path

def main():
    record = {
        "protocol": "telegram-cpu-e2e-v1",
        "kind": "synthetic_cpu_notification_smoke",
        "tasks": [1, 2, 3, 4, 5],
        "scientific_experiments_executed": False,
        "contains_medical_data": False,
    }
    canonical = json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")
    record["payload_sha256"] = hashlib.sha256(canonical).hexdigest()
    target = Path("/kaggle/working/telegram_cpu_e2e_manifest.json")
    target.write_text(json.dumps(record, sort_keys=True, indent=2) + "\\n", encoding="utf-8")
    print("CPU_TELEGRAM_E2E_SYNTHETIC_OUTPUT_WRITTEN")

if __name__ == "__main__":
    main()
'''
SOURCE_DIGEST = hashlib.sha256(RUNNER.encode("utf-8")).hexdigest()


def build(dest: Path) -> None:
    if dest.exists() and any(dest.iterdir()):
        raise ValueError("STALE_BUILD_DIRECTORY_BLOCKED")
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "runner.py").write_text(RUNNER, encoding="utf-8")
    metadata = {
        "id": KERNEL, "title": "medical-llm-telegram-cpu-e2e-v1",
        "code_file": "runner.py", "language": "python",
        "kernel_type": "script", "is_private": True,
        "enable_gpu": False, "enable_tpu": False,
        "enable_internet": False, "dataset_sources": [],
        "competition_sources": [], "kernel_sources": [], "model_sources": [],
    }
    (dest / "kernel-metadata.json").write_text(
        json.dumps(metadata, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )


def verify(output_dir: Path, expected_digest: str = SOURCE_DIGEST) -> dict:
    if expected_digest != SOURCE_DIGEST:
        raise ValueError("SOURCE_DIGEST_MISMATCH")
    candidates = list(output_dir.rglob(OUTPUT_NAME))
    if len(candidates) != 1 or candidates[0].stat().st_size > 4096:
        raise ValueError("OUTPUT_MISSING_DUPLICATED_OR_TOO_LARGE")
    data = json.loads(candidates[0].read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("OUTPUT_NOT_OBJECT")
    payload = {
        "protocol": PROTOCOL,
        "kind": "synthetic_cpu_notification_smoke",
        "tasks": [1, 2, 3, 4, 5],
        "scientific_experiments_executed": False,
        "contains_medical_data": False,
    }
    expected_hash = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if data != {**payload, "payload_sha256": expected_hash}:
        raise ValueError("OUTPUT_CONTENT_OR_CHECKSUM_MISMATCH")
    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=("build", "verify"))
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    if args.operation == "build":
        build(args.directory)
        print("CPU_TELEGRAM_E2E_PACKAGE_READY_NO_SUBMISSION")
    else:
        verify(args.directory)
        print("CPU_TELEGRAM_E2E_VERIFIED_NO_MEDICAL_DATA")


if __name__ == "__main__":
    main()
