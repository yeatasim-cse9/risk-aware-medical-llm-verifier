"""Code-only private Kaggle MedHallu candidate role/split pilot.

Uses SHA256-pinned MedHallu Parquet files from an independently verified prior
private Kaggle audit. NEVER exports original questions, answer texts, IDs,
raw categories, or per-row split assignments. This is NOT E1 training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
DATA_EVIDENCE = ROOT / "configs/medhallu-e1-prior-aggregate-evidence.json"
NATIVE = ROOT / "src/research_data/medhallu_native_audit.py"
CANDIDATE = ROOT / "src/research_data/e1_candidate_splits.py"
KERNEL = "yeatasimbillah/medical-llm-e1-candidate-split-v1"
REPORT = "e1_candidate_split_aggregate.json"
DATASET_REV = "515060458a945c633debc6fd5baac7764416b724"
SEED = 20261125
SHA40 = re.compile(r"[0-9a-f]{40}\Z")
SHA64 = re.compile(r"[0-9a-f]{64}\Z")


def canonical(body: dict) -> bytes:
    return json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def expected_evidence() -> dict:
    p = json.loads(DATA_EVIDENCE.read_text(encoding="utf-8"))
    if (p.get("kind") != "prior_independently_verified_kaggle_aggregate_evidence"
            or p.get("dataset_id") != "UTAustin-AIHealth/MedHallu"
            or p.get("dataset_revision_sha") != DATASET_REV
            or p.get("verified_github_actions_run_id") != 45590480345
            or p.get("artifact_id") != 15010069180
            or p.get("aggregate_report_sha256") != "ac37d9c07b494afd905ea36ecce7dc2eeb7c9e463a7d01871f5e6dbfea053608"):
        raise ValueError("UNVERIFIED_PRIOR_AGGREGATE_REFERENCE")
    files = p.get("source_files")
    if not isinstance(files, dict) or set(files) != {"pqa_labeled", "pqa_artificial"}:
        raise ValueError("EXPECTED_EXACT_TWO_SOURCE_FILES")
    for name, item in files.items():
        if not isinstance(item, dict) or set(item) != {"relative_file", "sha256", "size_bytes", "rows"}:
            raise ValueError("INVALID_SOURCE_FILE_EVIDENCE_SCHEMA")
        if item["relative_file"] != name + "/train-00000-of-00001.parquet":
            raise ValueError("UNEXPECTED_SOURCE_FILE_NAME")
        if not isinstance(item["sha256"], str) or not SHA64.fullmatch(item["sha256"]):
            raise ValueError("INVALID_SOURCE_SHA256")
        if type(item["size_bytes"]) is not int or not 100 < item["size_bytes"] <= 40_000_000:
            raise ValueError("INVALID_SOURCE_BYTE_COUNT")
        if type(item["rows"]) is not int or item["rows"] != (1000 if name == "pqa_labeled" else 9000):
            raise ValueError("INVALID_SOURCE_ROW_COUNT")
    a = p.get("observed_aggregate", {})
    if (a.get("unique_questions_total") != 9999
            or a.get("questions_repeated_within_or_across_configs") != 1
            or a.get("questions_shared_across_configs") != 0
            or a.get("ground_truth_equals_hallucinated_answer_pairs") != 1):
        raise ValueError("PRIOR_NATIVE_AUDIT_COUNTS_CHANGED")
    return p


RUNTIME = r'''
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types

SOURCE_GIT_SHA = __SOURCE_GIT_SHA__
DATASET_REV = "__DATASET_REV__"
FILE_EVIDENCE = __FILE_EVIDENCE__
EXPECTED_ANOMALIES = __EXPECTED_ANOMALIES__
SOURCE_MODULE_SHA256 = __SOURCE_MODULE_SHA256__
OUTPUT_FILE = "e1_candidate_split_aggregate.json"
AUDIT_STAGE = "BOOT"

# Execute the two exact reviewed GitHub modules as code-only Python modules.
# Relative imports remain intact, no dataset material is embedded in them.
pkg = types.ModuleType("research_data")
pkg.__path__ = []
sys.modules["research_data"] = pkg
for name, source in [
    ("medhallu_native_audit", __NATIVE_SOURCE__),
    ("e1_candidate_splits", __CANDIDATE_SOURCE__),
]:
    module = types.ModuleType("research_data." + name)
    module.__package__ = "research_data"
    sys.modules[module.__name__] = module
    exec(compile(source, "<pinned-reviewed-research-module>", "exec"), module.__dict__)

from research_data.medhallu_native_audit import NATIVE_FIELDS, NativeAuditError, audit_native_rows
from research_data.e1_candidate_splits import candidate_plan, verify_summary, SEED


def _stage(name):
    global AUDIT_STAGE
    AUDIT_STAGE = name
    print("E1_CANDIDATE_STAGE=" + name, flush=True)


def _libraries():
    try:
        from huggingface_hub import snapshot_download
        import pyarrow.parquet as parquet
        return snapshot_download, parquet
    except ImportError:
        subprocess.run([
            sys.executable, "-m", "pip", "install", "--no-input",
            "--disable-pip-version-check", "--quiet",
            "huggingface_hub==0.36.0", "pyarrow==21.0.0"
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=240)
        from huggingface_hub import snapshot_download
        import pyarrow.parquet as parquet
        return snapshot_download, parquet


def main():
    _stage("PRIVATE_STORAGE")
    working = Path("/kaggle/working")
    scratch = Path("/kaggle/temp")
    if not working.is_dir() or working.is_symlink() or scratch.is_symlink():
        raise NativeAuditError("KAGGLE_PRIVATE_STORAGE_UNAVAILABLE")
    scratch.mkdir(parents=True, exist_ok=True, mode=0o700)
    if not scratch.is_dir() or not scratch.resolve().is_relative_to(Path("/kaggle").resolve()):
        raise NativeAuditError("KAGGLE_TEMP_NOT_SAFE")
    output = working / OUTPUT_FILE
    if output.exists() or output.is_symlink():
        raise NativeAuditError("OUTPUT_ALREADY_PRESENT_NO_OVERWRITE")
    if SEED != 20261125:
        raise NativeAuditError("CANDIDATE_SEED_CHANGED")
    with tempfile.TemporaryDirectory(dir=str(scratch), prefix="e1-candidate-") as temporary:
        base = Path(temporary)
        os.environ["HF_HOME"] = str(base / "hf-home")
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        _stage("DEPENDENCIES")
        snapshot_download, parquet = _libraries()
        _stage("PINNED_DATASET_DOWNLOAD")
        dataset = Path(snapshot_download(
            repo_id="UTAustin-AIHealth/MedHallu", repo_type="dataset",
            revision=DATASET_REV,
            allow_patterns=["pqa_labeled/*.parquet", "pqa_artificial/*.parquet"],
            local_dir=str(base / "data"), max_workers=2, token=False,
        ))
        rows = {}
        total_bytes = 0
        _stage("FILE_HASH_AND_NATIVE_SCHEMA")
        for config in ("pqa_labeled", "pqa_artificial"):
            evidence = FILE_EVIDENCE[config]
            target = dataset / evidence["relative_file"]
            if (target.is_symlink() or not target.is_file() or
                    not target.resolve().is_relative_to(base.resolve())):
                raise NativeAuditError("EXPECTED_PARQUET_FILE_MISSING_OR_UNSAFE")
            size = target.stat().st_size
            total_bytes += size
            if size != evidence["size_bytes"] or total_bytes > 60_000_000:
                raise NativeAuditError("PINNED_DATASET_BYTE_SIZE_MISMATCH")
            h = hashlib.sha256()
            with target.open("rb") as stream:
                while block := stream.read(1024*1024):
                    h.update(block)
            if h.hexdigest() != evidence["sha256"]:
                raise NativeAuditError("PINNED_DATASET_FILE_SHA256_MISMATCH")
            table = parquet.read_table(target)
            if table.num_rows != evidence["rows"] or set(table.column_names) != NATIVE_FIELDS:
                raise NativeAuditError("PINNED_NATIVE_SCHEMA_MISMATCH")
            rows[config] = table.to_pylist()
            del table
        _stage("CONFIRM_PRIOR_ANOMALIES")
        audit = audit_native_rows(rows)
        if any(audit.get(key) != value for key, value in EXPECTED_ANOMALIES.items()):
            raise NativeAuditError("PRIOR_ANOMALY_COUNT_NOT_REPRODUCIBLE")
        _stage("CANDIDATE_PRIVATE_PLANNING")
        summary, private_assignment = candidate_plan(rows, seed=20261125)
        verify_summary(summary)
        if len(private_assignment) != summary["eligible_parent_rows"]:
            raise NativeAuditError("CANDIDATE_ASSIGNMENT_INCOMPLETE")
        del rows, private_assignment
        _stage("SANITIZED_AGGREGATE_ONLY")
        report = {
            "schema_version": 1,
            "audit_kind": "private_e1_candidate_split_design_only",
            "source_git_sha": SOURCE_GIT_SHA,
            "dataset_id": "UTAustin-AIHealth/MedHallu",
            "dataset_revision_sha": DATASET_REV,
            "prior_verified_aggregate_sha256": "__PRIOR_AGG_SHA256__",
            "module_sha256": SOURCE_MODULE_SHA256,
            "source_files": FILE_EVIDENCE,
            "summary": summary,
            "full_lineage_verified": False,
            "final_splits_approved": False,
            "scientific_experiments_executed": False,
            "raw_source_data_exported": False,
        }
        digest = hashlib.sha256(json.dumps(report, sort_keys=True,
            separators=(",", ":"), ensure_ascii=True).encode("utf-8")).hexdigest()
        report["integrity_sha256"] = digest
        payload = json.dumps(report, sort_keys=True, indent=2) + "\n"
        if len(payload.encode("utf-8")) > 12_000:
            raise NativeAuditError("CANDIDATE_AGGREGATE_TOO_LARGE")
        output.write_text(payload, encoding="utf-8")
        print("E1_CANDIDATE_SAFE_AGGREGATE_SAVED")
        print("E1_CANDIDATE_NOT_A_MODEL_EXPERIMENT")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        # No raw exception, HTTP response, dataset contents or traceback.
        code = str(exc) if isinstance(exc, NativeAuditError) else ""
        if not (code and len(code) <= 64 and code == code.upper()
                and code.replace("_", "").isalnum()):
            code = "SAFE_RUNTIME_FAILURE"
        print("E1_CANDIDATE_FAILED_STAGE=" + AUDIT_STAGE, file=sys.stderr)
        print("E1_CANDIDATE_FAILED_CODE=" + code, file=sys.stderr)
        raise SystemExit(1)
'''


def build(destination: Path, source_git_sha: str) -> dict:
    if not isinstance(source_git_sha, str) or not SHA40.fullmatch(source_git_sha):
        raise ValueError("EXACT_SOURCE_GIT_SHA_REQUIRED")
    if destination.is_symlink() or (destination.exists() and any(destination.iterdir())):
        raise ValueError("BUILD_FOLDER_MUST_BE_EMPTY")
    evidence = expected_evidence()
    native = NATIVE.read_text(encoding="utf-8")
    candidate = CANDIDATE.read_text(encoding="utf-8")
    sha_map = {
        "medhallu_native_audit": hashlib.sha256(native.encode("utf-8")).hexdigest(),
        "e1_candidate_splits": hashlib.sha256(candidate.encode("utf-8")).hexdigest(),
    }
    anomalies = {
        "unique_questions_total": 9999,
        "questions_repeated_within_or_across_configs": 1,
        "questions_shared_across_configs": 0,
        "ground_truth_equals_hallucinated_answer_pairs": 1,
    }
    replacements = {
        "__SOURCE_GIT_SHA__": repr(source_git_sha),
        "__DATASET_REV__": DATASET_REV,
        "__FILE_EVIDENCE__": repr(evidence["source_files"]),
        "__EXPECTED_ANOMALIES__": repr(anomalies),
        "__SOURCE_MODULE_SHA256__": repr(sha_map),
        "__NATIVE_SOURCE__": repr(native),
        "__CANDIDATE_SOURCE__": repr(candidate),
        "__PRIOR_AGG_SHA256__": evidence["aggregate_report_sha256"],
    }
    runner = RUNTIME
    for old, replacement in replacements.items():
        if old not in runner:
            raise ValueError("MISSING_RUNNER_PLACEHOLDER")
        runner = runner.replace(old, replacement)
    compile(runner, "candidate-private-runner.py", "exec")
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "runner.py").write_text(runner, encoding="utf-8")
    metadata = {
        "id": KERNEL, "title": "medical-llm-e1-candidate-split-v1",
        "code_file": "runner.py", "language": "python", "kernel_type": "script",
        "is_private": True, "enable_gpu": False, "enable_tpu": False,
        "enable_internet": True, "dataset_sources": [],
        "competition_sources": [], "kernel_sources": [], "model_sources": [],
    }
    (destination / "kernel-metadata.json").write_text(
        json.dumps(metadata, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return metadata


def verify_aggregate(folder: Path, expected_source_sha: str, *, expected_rows: dict | None = None) -> dict:
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from research_data.e1_candidate_splits import verify_summary
    if not isinstance(expected_source_sha, str) or not SHA40.fullmatch(expected_source_sha):
        raise ValueError("INVALID_EXPECTED_SOURCE_COMMIT")
    files = list(folder.rglob(REPORT))
    if len(files) != 1 or files[0].is_symlink() or files[0].stat().st_size > 12_000:
        raise ValueError("CANDIDATE_REPORT_MISSING_OR_DUPLICATED_OR_OVERSIZED")
    data = json.loads(files[0].read_text(encoding="utf-8"))
    permitted = {
        "schema_version", "audit_kind", "source_git_sha", "dataset_id",
        "dataset_revision_sha", "prior_verified_aggregate_sha256", "module_sha256",
        "source_files", "summary", "full_lineage_verified", "final_splits_approved",
        "scientific_experiments_executed", "raw_source_data_exported",
        "integrity_sha256",
    }
    if not isinstance(data, dict) or set(data) != permitted:
        raise ValueError("CANDIDATE_REPORT_UNEXPECTED_SCHEMA")
    if (type(data["schema_version"]) is not int or data["schema_version"] != 1
            or data["audit_kind"] != "private_e1_candidate_split_design_only"
            or data["source_git_sha"] != expected_source_sha
            or data["dataset_id"] != "UTAustin-AIHealth/MedHallu"
            or data["dataset_revision_sha"] != DATASET_REV):
        raise ValueError("CANDIDATE_PROVENANCE_INVALID")
    evidence = expected_evidence()
    if (data["source_files"] != evidence["source_files"] or
            data["prior_verified_aggregate_sha256"] != evidence["aggregate_report_sha256"]):
        raise ValueError("SOURCE_BYTES_NOT_PINNED_TO_PRIOR_AUDIT")
    hashes = {
        "medhallu_native_audit": hashlib.sha256(NATIVE.read_bytes()).hexdigest(),
        "e1_candidate_splits": hashlib.sha256(CANDIDATE.read_bytes()).hexdigest(),
    }
    if data["module_sha256"] != hashes:
        raise ValueError("AUDITOR_MODULE_HASHES_DIFFER_FROM_REVIEWED_CODE")
    for key in ("full_lineage_verified", "final_splits_approved",
                "scientific_experiments_executed", "raw_source_data_exported"):
        if data[key] is not False:
            raise ValueError("UNSUPPORTED_E1_APPROVAL_OR_DATA_EXPORT")
    integrity = data["integrity_sha256"]
    if not isinstance(integrity, str) or not SHA64.fullmatch(integrity):
        raise ValueError("MISSING_REPORT_DIGEST")
    if hashlib.sha256(canonical({k: v for k, v in data.items() if k != "integrity_sha256"})).hexdigest() != integrity:
        raise ValueError("CANDIDATE_REPORT_DIGEST_MISMATCH")
    summary = data["summary"]
    verify_summary(summary)
    target = {"pqa_artificial": 9000, "pqa_labeled": 1000} if expected_rows is None else expected_rows
    if summary["source_config_rows"] != target:
        raise ValueError("UNEXPECTED_DATASET_CONFIG_ROW_COUNTS")
    return data


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("operation", choices=("build", "verify"))
    ap.add_argument("directory", type=Path)
    ap.add_argument("--source-commit", required=True)
    args = ap.parse_args()
    if args.operation == "build":
        build(args.directory, args.source_commit)
        print("E1_CANDIDATE_PRIVATE_CPU_PACKAGE_READY_NO_KAGGLE_EXECUTION")
    else:
        verify_aggregate(args.directory, args.source_commit)
        print("E1_CANDIDATE_AGGREGATE_INDEPENDENTLY_VERIFIED_NO_SCIENTIFIC_RUN")


if __name__ == "__main__":
    main()
