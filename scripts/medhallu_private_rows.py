"""Build a private Kaggle-only MedHallu native-row audit, or verify its aggregate.

No original text or source rows enter a GitHub commit or GitHub Actions job.
A separate, explicitly approved workflow handles notebook submission.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
KERNEL = "yeatasimbillah/medical-llm-medhallu-native-rows-v1"
DATASET = "UTAustin-AIHealth/MedHallu"
DATA_REV = "515060458a945c633debc6fd5baac7764416b724"
REPORT_NAME = "medhallu_native_aggregate.json"
SOURCE_MODULE = ROOT / "src/research_data/medhallu_native_audit.py"
SHA40 = re.compile(r"[0-9a-f]{40}\Z")
SHA64 = re.compile(r"[0-9a-f]{64}\Z")
CONFIG_ROWS = {"pqa_labeled": 1000, "pqa_artificial": 9000}

# The source code above the runner is copied verbatim from the repo.
# Every downloaded dataset byte is kept in /kaggle/temp, which is NOT an
# exported Kaggle working/output artifact. Never save source text to working.
RUNNER_SUFFIX = r'''
import os
import subprocess
import sys
import tempfile
from pathlib import Path
import hashlib
import json

SOURCE_COMMIT = __SOURCE_COMMIT__
DATASET_REVISION = "515060458a945c633debc6fd5baac7764416b724"
EXPECTED_ROWS = {"pqa_labeled": 1000, "pqa_artificial": 9000}
MAX_FILE_BYTES = 40_000_000
MAX_TOTAL_BYTES = 60_000_000
OUTPUT_NAME = "medhallu_native_aggregate.json"
MODULE_SHA256 = "__MODULE_SHA256__"


def _dependencies():
    try:
        from huggingface_hub import snapshot_download
        import pyarrow.parquet as pq
        return snapshot_download, pq
    except ImportError:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-input",
             "--disable-pip-version-check", "--quiet",
             "huggingface_hub==0.36.0", "pyarrow==21.0.0"],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=240,
        )
        from huggingface_hub import snapshot_download
        import pyarrow.parquet as pq
        return snapshot_download, pq


def _run():
    if not Path("/kaggle/temp").is_dir() or not Path("/kaggle/working").is_dir():
        raise NativeAuditError("KAGGLE_PRIVATE_ENVIRONMENT_REQUIRED")
    # Private /kaggle/working must only contain the final sanitized aggregate.
    output = Path("/kaggle/working") / OUTPUT_NAME
    if output.exists() or output.is_symlink():
        raise NativeAuditError("EXISTING_REPORT_BLOCKED")
    with tempfile.TemporaryDirectory(dir="/kaggle/temp", prefix="medhallu-audit-") as tmp:
        base = Path(tmp)
        os.environ["HF_HOME"] = str(base / "hf-home")
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        # The public Hub is the only source. No other dataset or API is accessed.
        snapshot_download, parquet = _dependencies()
        folder = Path(snapshot_download(
            repo_id="UTAustin-AIHealth/MedHallu",
            repo_type="dataset", revision=DATASET_REVISION,
            allow_patterns=["pqa_labeled/*.parquet", "pqa_artificial/*.parquet"],
            local_dir=str(base / "data"), max_workers=2, token=False,
        ))
        approved_files = {}
        rows = {}
        total_bytes = 0
        for config, expected in EXPECTED_ROWS.items():
            files = sorted((folder / config).glob("*.parquet"))
            if len(files) != 1:
                raise NativeAuditError("EXPECTED_ONE_PARQUET_SHARD_PER_CONFIG")
            f = files[0]
            if f.is_symlink() or not f.resolve().is_relative_to(base.resolve()):
                raise NativeAuditError("UNSAFE_DATASET_FILE_PATH")
            size = f.stat().st_size
            total_bytes += size
            if size < 100 or size > MAX_FILE_BYTES or total_bytes > MAX_TOTAL_BYTES:
                raise NativeAuditError("DATASET_BYTES_OUT_OF_POLICY")
            digest = hashlib.sha256()
            with f.open("rb") as stream:
                while chunk := stream.read(1024 * 1024):
                    digest.update(chunk)
            table = parquet.read_table(f)
            if table.num_rows != expected or set(table.column_names) != NATIVE_FIELDS:
                raise NativeAuditError("DATASET_NATIVE_SCHEMA_OR_ROWS_CHANGED")
            rows[config] = table.to_pylist()
            approved_files[config] = {
                "relative_file": f.relative_to(folder).as_posix(),
                "size_bytes": size, "sha256": digest.hexdigest(),
            }
        summary = audit_native_rows(rows)
        verify_report(summary)
        del rows
        envelope = {
            "schema_version": 1,
            "audit_kind": "private_kaggle_medhallu_native_rows",
            "source_git_sha": SOURCE_COMMIT,
            "audit_module_sha256": MODULE_SHA256,
            "dataset_id": "UTAustin-AIHealth/MedHallu",
            "dataset_revision_sha": DATASET_REVISION,
            "source_files": approved_files,
            "summary": summary,
            "dataset_rows_exported": False,
            "scientific_experiments_executed": False,
        }
        raw = json.dumps(envelope, sort_keys=True, separators=(",", ":")).encode("utf-8")
        envelope["integrity_sha256"] = hashlib.sha256(raw).hexdigest()
        if len(json.dumps(envelope).encode("utf-8")) > 8192:
            raise NativeAuditError("REPORT_EXCEEDS_AGGREGATE_ONLY_LIMIT")
        output.write_text(json.dumps(envelope, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        print("MEDHALLU_NATIVE_PRIVATE_AGGREGATE_SAVED")
        print("NO_RAW_ROWS_IN_KAGGLE_WORKING_OR_GITHUB")


if __name__ == "__main__":
    try:
        _run()
    except Exception:
        # Never log dataset text, third-party HTTP responses or private paths.
        print("MEDHALLU_NATIVE_PRIVATE_AUDIT_FAILED_SAFE", file=sys.stderr)
        raise SystemExit(1)
'''


def build(out: Path, source_commit: str) -> dict:
    if not isinstance(source_commit, str) or not SHA40.fullmatch(source_commit):
        raise ValueError("EXACT_SOURCE_GIT_SHA_REQUIRED")
    if out.exists() and (out.is_symlink() or any(out.iterdir())):
        raise ValueError("BUILD_DESTINATION_MUST_BE_EMPTY")
    out.mkdir(parents=True, exist_ok=True)
    source = SOURCE_MODULE.read_text(encoding="utf-8")
    module_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()
    runner = source + "\n" + RUNNER_SUFFIX.replace("__SOURCE_COMMIT__", repr(source_commit)).replace("__MODULE_SHA256__", module_hash)
    (out / "runner.py").write_text(runner, encoding="utf-8")
    metadata = {
        "id": KERNEL, "title": "medical-llm-medhallu-native-rows-v1",
        "code_file": "runner.py", "language": "python", "kernel_type": "script",
        "is_private": True, "enable_gpu": False, "enable_tpu": False,
        "enable_internet": True, "dataset_sources": [],
        "competition_sources": [], "kernel_sources": [], "model_sources": [],
    }
    (out / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    return metadata


def _hash_json(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def verify_aggregate(out: Path, source_commit: str) -> dict:
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from research_data.medhallu_native_audit import verify_report
    if not isinstance(source_commit, str) or not SHA40.fullmatch(source_commit):
        raise ValueError("INVALID_EXPECTED_COMMIT")
    matches = list(out.rglob(REPORT_NAME))
    if len(matches) != 1 or matches[0].is_symlink() or matches[0].stat().st_size > 8192:
        raise ValueError("AGGREGATE_FILE_MISSING_DUPLICATED_OR_OVERSIZED")
    data = json.loads(matches[0].read_text(encoding="utf-8"))
    if not isinstance(data, dict) or set(data) != {
        "schema_version", "audit_kind", "source_git_sha", "audit_module_sha256",
        "dataset_id", "dataset_revision_sha", "source_files", "summary",
        "dataset_rows_exported", "scientific_experiments_executed", "integrity_sha256",
    }:
        raise ValueError("UNEXPECTED_AGGREGATE_SCHEMA")
    if data["schema_version"] != 1 or data["audit_kind"] != "private_kaggle_medhallu_native_rows":
        raise ValueError("UNEXPECTED_REPORT_TYPE")
    if data["source_git_sha"] != source_commit or data["dataset_revision_sha"] != DATA_REV or data["dataset_id"] != DATASET:
        raise ValueError("SOURCE_OR_DATASET_REVISION_MISMATCH")
    actual_module_digest = hashlib.sha256(SOURCE_MODULE.read_bytes()).hexdigest()
    if data["audit_module_sha256"] != actual_module_digest:
        raise ValueError("AUDITOR_SOURCE_DIGEST_MISMATCH")
    if data["dataset_rows_exported"] is not False or data["scientific_experiments_executed"] is not False:
        raise ValueError("UNAUTHORIZED_EXPERIMENT_OR_DATA_EXPORT")
    files = data["source_files"]
    if not isinstance(files, dict) or set(files) != set(CONFIG_ROWS):
        raise ValueError("MISSING_SOURCE_FILE_DIGESTS")
    for config, record in files.items():
        if not isinstance(record, dict) or set(record) != {"relative_file", "size_bytes", "sha256"}:
            raise ValueError("INVALID_FILE_RECORD")
        name = record["relative_file"]
        if not isinstance(name, str) or not re.fullmatch(re.escape(config)+r"/[a-zA-Z0-9_-]+\.parquet", name):
            raise ValueError("UNEXPECTED_FILE_NAME")
        if type(record["size_bytes"]) is not int or not (100 < record["size_bytes"] <= 40_000_000):
            raise ValueError("FILE_SIZE_OUT_OF_POLICY")
        if not isinstance(record["sha256"], str) or not SHA64.fullmatch(record["sha256"]):
            raise ValueError("FILE_DIGEST_INVALID")
    if not isinstance(data.get("integrity_sha256"), str) or not SHA64.fullmatch(data["integrity_sha256"]):
        raise ValueError("REPORT_INTEGRITY_MISSING")
    bare = {k: v for k, v in data.items() if k != "integrity_sha256"}
    if _hash_json(bare) != data["integrity_sha256"]:
        raise ValueError("REPORT_HASH_INCORRECT")
    verify_report(data["summary"])
    if data["summary"]["config_rows"] != CONFIG_ROWS:
        raise ValueError("UNEXPECTED_NATIVE_ROW_COUNTS")
    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=("build", "verify"))
    parser.add_argument("directory", type=Path)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    if args.operation == "build":
        build(args.directory, args.source_commit)
        print("PRIVATE_MEDHALLU_NATIVE_AUDIT_CODE_READY_NO_SUBMISSION")
    else:
        verify_aggregate(args.directory, args.source_commit)
        print("PRIVATE_MEDHALLU_NATIVE_AGGREGATE_VERIFIED_NO_E1_RESULT")


if __name__ == "__main__":
    main()
