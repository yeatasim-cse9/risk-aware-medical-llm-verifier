"""Private Kaggle CPU E1 PubMedQA article-linkage and advanced split-leakage audit.

Manual ONLY, ONE new private Kaggle notebook, no model training. GitHub runner
should be fetched at an immutable commit and SHA256-checked by a bootstrap cell.

No raw MedHallu, PubMedQA source, PMID, official test ID or row assignment is
written into /kaggle/working. Raw source data is confined to /kaggle/temp and
cleaned on successful exit or exception. Aggregate counts are provisional.
"""
from __future__ import annotations

from hashlib import sha1, sha256
import importlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import urllib.request

REPO = "yeatasim-cse9/risk-aware-medical-llm-verifier"
MEDHALLU = "UTAustin-AIHealth/MedHallu"
MEDHALLU_REVISION = "515060458a945c633debc6fd5baac7764416b724"
PUBMEDQA = "qiaojin/PubMedQA"
PUBMEDQA_REVISION = "323752ad243ce751ba75c9222d34a3130d080030"
PUBMEDQA_OFFICIAL_GIT_REVISION = "1cbae8e92f72f20c8d3747cbb3bf5bc53554d997"
PUBMEDQA_TEST_GIT_BLOB = "825bf8f262c3cca0e3bd1c6f38eecf5e581b0659"
SOURCE_SHA256 = {
    "pqa_labeled": ("pqa_labeled/train-00000-of-00001.parquet", 1116475,
                    "ac963f9aaeb163f494edd4dbfec04086c3e35c7cc55e3f9c8b1d7b31188fa34c", 1000),
    "pqa_artificial": ("pqa_artificial/train-00000-of-00001.parquet", 10283429,
                       "70de8ce7f89a38314fa231a12e45168f3ee1aa591e8a38f6556b89aa8b1ca513", 9000),
}
PUBMEDQA_SHA256 = {
    "pqa_labeled": ("pqa_labeled/train-00000-of-00001.parquet", 1075513,
                    "3d56bd1abc11884579ecc3aab9dd3cdfce8b7cf54715daeca93b8022b7be231c", 1000),
    "pqa_artificial": ("pqa_artificial/train-00000-of-00001.parquet", 233411194,
                       "cdee2d3c01369f47d2ea7b89cd32ee64b142cc0debcb863bf39b3a81a3cece05", 211269),
}
EXPECTED_MODULE_SHA256 = {
    "medhallu_native_audit": "a23b2b916c8165e4475bebe5c80a8d2aba39eee5494ef0514da234da98b1b30b",
    "e1_candidate_splits": "7f4c8aac27a7b682346cc5c8df92cbceb62cb7821bbda0257931b9fbf545cd4a",
}
CANDIDATE_ASSIGNMENT_SHA256 = "ddd32f57ebc49ec6591cffd2d953a869c30518525bf66d6e2fb931a68bb0945f"
OUTPUT_NAME = "e1_pubmedqa_lineage_advanced_leakage_v1.json"
MAX_OUTPUT_BYTES = 18000
CURRENT_STAGE = "BOOT"


class PrivateLineageError(ValueError):
    """Fixed safe uppercase diagnostics only."""


def stage(step: str) -> None:
    global CURRENT_STAGE
    CURRENT_STAGE = step
    print("E1_LINEAGE_STAGE=" + step, flush=True)


def file_sha256(path: Path) -> str:
    h = sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            h.update(block)
    return h.hexdigest()


def canonical(obj: dict) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def libraries():
    try:
        from huggingface_hub import snapshot_download
        import pyarrow.parquet as pq
        return snapshot_download, pq
    except ImportError:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-input",
             "--disable-pip-version-check", "--quiet",
             "huggingface_hub==0.36.0", "pyarrow==21.0.0"],
            check=True, timeout=300, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        from huggingface_hub import snapshot_download
        import pyarrow.parquet as pq
        return snapshot_download, pq


def checked_modules(source_commit: str, target: Path):
    if re.fullmatch(r"[0-9a-f]{40}", source_commit) is None:
        raise PrivateLineageError("SOURCE_COMMIT_NOT_FROZEN")
    package = target / "research_data"
    package.mkdir(parents=True, exist_ok=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    hashes = {}
    for name in ("medhallu_native_audit", "e1_candidate_splits", "e1_advanced_lineage"):
        url = f"https://raw.githubusercontent.com/{REPO}/{source_commit}/src/research_data/{name}.py"
        request = urllib.request.Request(url, headers={"User-Agent": "e1-lineage-private-research"})
        with urllib.request.urlopen(request, timeout=60) as response:
            body = response.read(100_000)
        if not 1000 < len(body) < 90_000:
            raise PrivateLineageError("RESEARCH_MODULE_NOT_IN_SIZE_POLICY")
        compile(body, name, "exec")
        digest = sha256(body).hexdigest()
        if name in EXPECTED_MODULE_SHA256 and digest != EXPECTED_MODULE_SHA256[name]:
            raise PrivateLineageError("PREVIOUS_REVIEWED_MODULE_SHA256_CHANGED")
        hashes[name] = digest
        (package / f"{name}.py").write_bytes(body)
    sys.path.insert(0, str(target))
    importlib.invalidate_caches()
    native = importlib.import_module("research_data.medhallu_native_audit")
    candidate = importlib.import_module("research_data.e1_candidate_splits")
    advanced = importlib.import_module("research_data.e1_advanced_lineage")
    if candidate.SEED != 20261125 or native.DATASET != MEDHALLU:
        raise PrivateLineageError("CANDIDATE_PROTOCOL_CHANGED")
    return native, candidate, advanced, hashes


def checked_parquet(snapshot_download, pq, dataset_id: str, revision: str,
                    specs: dict, target: Path, *, is_source: bool) -> dict[str, list[dict]]:
    folder = Path(snapshot_download(
        repo_id=dataset_id, repo_type="dataset", revision=revision,
        allow_patterns=[spec[0] for spec in specs.values()],
        local_dir=str(target), max_workers=2, token=False,
    ))
    if folder.resolve() != target.resolve():
        raise PrivateLineageError("UNEXPECTED_DATASET_SNAPSHOT_FOLDER")
    rows_by_config = {}
    total_bytes = 0
    for config in ("pqa_labeled", "pqa_artificial"):
        relative, size, digest, count = specs[config]
        src = folder / relative
        if not src.is_file() or src.is_symlink() or not src.resolve().is_relative_to(target.resolve()):
            raise PrivateLineageError("PINNED_PARQUET_FILE_MISSING")
        total_bytes += src.stat().st_size
        if src.stat().st_size != size or total_bytes > 260_000_000 or file_sha256(src) != digest:
            raise PrivateLineageError("PINNED_PARQUET_SHA256_OR_SIZE_MISMATCH")
        reader = pq.ParquetFile(src)
        if reader.metadata.num_rows != count:
            raise PrivateLineageError("PINNED_PARQUET_ROW_COUNT_CHANGED")
        expected_cols = {"pubid", "question", "context"} if is_source else {
            "Question", "Knowledge", "Ground Truth", "Difficulty Level",
            "Hallucinated Answer", "Category of Hallucination",
        }
        if is_source:
            if not expected_cols <= set(reader.schema_arrow.names):
                raise PrivateLineageError("PUBMEDQA_SOURCE_COLUMNS_UNEXPECTED")
            rows = []
            for batch in reader.iter_batches(batch_size=8192, columns=sorted(expected_cols)):
                rows.extend(batch.to_pylist())
        else:
            if set(reader.schema_arrow.names) != expected_cols:
                raise PrivateLineageError("MEDHALLU_NATIVE_COLUMNS_UNEXPECTED")
            rows = reader.read().to_pylist()
        if len(rows) != count:
            raise PrivateLineageError("PARQUET_DECODED_ROW_COUNT_CHANGED")
        rows_by_config[config] = rows
        del reader
    return rows_by_config


def official_test_ids() -> tuple[set[str], str]:
    url = (
        "https://raw.githubusercontent.com/pubmedqa/pubmedqa/"
        + PUBMEDQA_OFFICIAL_GIT_REVISION + "/data/test_ground_truth.json"
    )
    request = urllib.request.Request(url, headers={"User-Agent": "e1-lineage-private-research"})
    with urllib.request.urlopen(request, timeout=45) as response:
        source = response.read(20_000)
    if not 10_000 <= len(source) <= 15_000:
        raise PrivateLineageError("ORIGINAL_500_TEST_FILE_SIZE_UNEXPECTED")
    git_blob = sha1(b"blob " + str(len(source)).encode() + b"\x00" + source).hexdigest()
    if git_blob != PUBMEDQA_TEST_GIT_BLOB:
        raise PrivateLineageError("ORIGINAL_500_TEST_GIT_BLOB_MISMATCH")
    info = json.loads(source)
    if not isinstance(info, dict) or len(info) != 500:
        raise PrivateLineageError("ORIGINAL_TEST_SET_SCHEMA_CHANGED")
    ids = set(info)
    if len(ids) != 500 or any(not isinstance(k, str) or not k.isdigit() for k in ids):
        raise PrivateLineageError("ORIGINAL_TEST_SET_IDS_INVALID")
    if any(value not in {"yes", "no", "maybe"} for value in info.values()):
        raise PrivateLineageError("ORIGINAL_TEST_SET_LABELS_CHANGED")
    return ids, sha256(source).hexdigest()


def main() -> None:
    git_sha = os.environ.get("E1_LINEAGE_PINNED_GITHUB_COMMIT", "")
    if re.fullmatch(r"[0-9a-f]{40}", git_sha) is None:
        raise PrivateLineageError("REVIEWED_GITHUB_COMMIT_REQUIRED")
    stage("PRIVATE_KAGGLE_CPU_STORAGE")
    scratch = Path("/kaggle/temp")
    working = Path("/kaggle/working")
    if not working.is_dir() or working.is_symlink() or scratch.is_symlink():
        raise PrivateLineageError("PRIVATE_KAGGLE_STORAGE_NOT_AVAILABLE")
    scratch.mkdir(mode=0o700, parents=True, exist_ok=True)
    if not scratch.is_dir() or not scratch.resolve().is_relative_to(Path("/kaggle").resolve()):
        raise PrivateLineageError("KAGGLE_SCRATCH_PATH_NOT_ALLOWED")
    output = working / OUTPUT_NAME
    if output.exists() or output.is_symlink():
        raise PrivateLineageError("EXISTING_REPORT_CANNOT_OVERWRITE")
    with tempfile.TemporaryDirectory(dir=str(scratch), prefix="e1-lineage-") as temp:
        root = Path(temp)
        os.environ["HF_HOME"] = str(root / "hf-cache")
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        stage("CHECKED_GITHUB_MODULES")
        native, candidate, advanced, modules = checked_modules(git_sha, root / "src")
        stage("DEPENDENCIES")
        snapshot_download, pq = libraries()
        stage("PINNED_MEDHALLU")
        med = checked_parquet(snapshot_download, pq, MEDHALLU,
                              MEDHALLU_REVISION, SOURCE_SHA256, root / "medhallu",
                              is_source=False)
        stage("MEDHALLU_CANDIDATE_ASSIGNMENT")
        before = native.audit_native_rows(med)
        if before["config_rows"] != {"pqa_artificial": 9000, "pqa_labeled": 1000}:
            raise PrivateLineageError("MEDHALLU_NATIVE_COUNT_MISMATCH")
        summary, assignments = candidate.candidate_plan(med)
        if (summary["candidate_assignment_sha256"] != CANDIDATE_ASSIGNMENT_SHA256 or
            summary["eligible_parent_rows"] != 9997 or
            summary["candidate_split_parent_rows"] != {"train": 7998, "dev": 1000, "test": 999}):
            raise PrivateLineageError("CANDIDATE_SPLIT_NOT_PREVIOUSLY_VERIFIED")
        stage("PINNED_PUBMEDQA_SOURCE")
        source = checked_parquet(snapshot_download, pq, PUBMEDQA,
                                 PUBMEDQA_REVISION, PUBMEDQA_SHA256, root / "pubmedqa",
                                 is_source=True)
        stage("ORIGINAL_PUBMEDQA_500_TEST_IDS")
        official_ids, official_test_sha256 = official_test_ids()
        stage("SOURCE_PUBID_MATCHING")
        pub_index = advanced.pubmed_index(source)
        findings = advanced.audit_lineage(med, assignments, pub_index, official_ids)
        advanced.validate_report(findings)
        del source, pub_index, official_ids
        stage("ADVANCED_CROSS_SPLIT_NEAR_DUPLICATES")
        near = advanced.advanced_near_screen(med, assignments, candidate_limit=300_000)
        advanced.validate_near_report(near, summary["eligible_parent_rows"])
        if near["candidate_pairs_compared"] > 300_000:
            raise PrivateLineageError("ADVANCED_SCREEN_CANDIDATE_LIMIT_EXCEEDED")
        del med, assignments
        stage("WRITE_SANITIZED_AGGREGATE_ONLY")
        report = {
            "schema_version": 1,
            "audit_kind": "e1_pubmedqa_source_lineage_advanced_leakage_v1",
            "github_source_git_sha": git_sha,
            "medhallu_revision": MEDHALLU_REVISION,
            "pubmedqa_revision": PUBMEDQA_REVISION,
            "pubmedqa_official_test_repo_revision": PUBMEDQA_OFFICIAL_GIT_REVISION,
            "original_pubmedqa_500_test_sha256": official_test_sha256,
            "source_module_sha256": modules,
            "candidate_assignment_sha256": CANDIDATE_ASSIGNMENT_SHA256,
            "medhallu_source_sha256": {k: value[2] for k, value in SOURCE_SHA256.items()},
            "pubmedqa_source_sha256": {k: value[2] for k, value in PUBMEDQA_SHA256.items()},
            "matched_pubmedqa_lineage_screen": findings,
            "advanced_near_duplicate_screen": near,
            "pubmedqa_article_lineage_fully_certified": False,
            "leakage_free_final_splits_approved": False,
            "labels_scientifically_adjudicated": False,
            "e1_scientific_model_experiment_executed": False,
            "source_question_answer_pubids_exported": False,
        }
        report["integrity_sha256"] = sha256(canonical(report)).hexdigest()
        payload = json.dumps(report, sort_keys=True, indent=2) + "\n"
        if len(payload.encode("utf-8")) > MAX_OUTPUT_BYTES:
            raise PrivateLineageError("AGGREGATE_TOO_LARGE")
        output.write_text(payload, encoding="utf-8")
        print("E1_LINEAGE_AGGREGATE_SAVED_PRIVATE_CPU_ONLY")
        print("E1_LINEAGE_REPORT_PATH=/kaggle/working/" + OUTPUT_NAME)
        print("E1_LINEAGE_PUBMEDQA_MATCHED_PARENT_ROWS=" + str(findings["matched_unique_pubid_parent_rows"]))
        print("E1_LINEAGE_OFFICIAL_TEST_TRAIN_OVERLAP=" + str(findings["official_pubmedqa_500_test_overlap_parent_rows_by_candidate_split"]["train"]))
        print("E1_LINEAGE_SOURCE_CROSS_SPLIT_PUBID_CONFLICTS=" + str(findings["source_pubids_spanning_candidate_partitions"]))
        print("E1_LINEAGE_ADVANCED_NEAR_FLAGS=" + str(near["high_overlap_unique_pairs"]))
        print("E1_LINEAGE_FINAL_E1_STATUS=BLOCKED_PENDING_INDEPENDENT_APPROVAL")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        code = str(exc) if isinstance(exc, PrivateLineageError) else ""
        if not (code and 0 < len(code) <= 64 and code == code.upper() and code.replace("_", "").isalnum()):
            code = "SAFE_RUNTIME_FAILURE"
        print("E1_LINEAGE_FAILED_STAGE=" + CURRENT_STAGE, file=sys.stderr)
        print("E1_LINEAGE_FAILED_CODE=" + code, file=sys.stderr)
        raise SystemExit(1)
