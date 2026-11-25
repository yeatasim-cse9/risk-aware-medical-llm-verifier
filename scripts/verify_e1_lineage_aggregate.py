"""Offline fail-closed verifier for aggregate-only E1 PubMedQA lineage reports.

No dataset, Kaggle or network access. This validates a user-supplied *report*
against reviewed local modules and frozen public file hashes. It does NOT
authenticate that an unauthenticated JSON was produced by Kaggle; nor does it
approve E1 scientific labels or final splits.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from research_data.e1_advanced_lineage import validate_report, validate_near_report

from e1_pubmedqa_lineage_kaggle import (
    MEDHALLU_REVISION, PUBMEDQA_REVISION, PUBMEDQA_OFFICIAL_GIT_REVISION,
    SOURCE_SHA256, PUBMEDQA_SHA256, CANDIDATE_ASSIGNMENT_SHA256,
    EXPECTED_MODULE_SHA256, canonical,
)

REPORT_NAME = "e1_pubmedqa_lineage_advanced_leakage_v1.json"
MAX_BYTES = 18000
HASH64 = re.compile(r"[a-f0-9]{64}\Z")
HASH40 = re.compile(r"[a-f0-9]{40}\Z")


def verify(report_file: Path, source_sha: str, *, expected_rows: int = 9997) -> dict:
    if not isinstance(source_sha, str) or not HASH40.fullmatch(source_sha):
        raise ValueError("SOURCE_GIT_COMMIT_MUST_BE_EXACT_40_HEX")
    if report_file.name != REPORT_NAME or report_file.is_symlink() or not report_file.is_file():
        raise ValueError("SAFE_AGGREGATE_FILE_NOT_FOUND")
    if report_file.stat().st_size > MAX_BYTES or report_file.stat().st_size < 500:
        raise ValueError("AGGREGATE_FILE_SIZE_OUT_OF_POLICY")
    body = json.loads(report_file.read_text(encoding="utf-8"))
    allowed = {
        "schema_version", "audit_kind", "github_source_git_sha",
        "medhallu_revision", "pubmedqa_revision",
        "pubmedqa_official_test_repo_revision", "original_pubmedqa_500_test_sha256",
        "source_module_sha256", "candidate_assignment_sha256", "medhallu_source_sha256",
        "pubmedqa_source_sha256", "matched_pubmedqa_lineage_screen",
        "advanced_near_duplicate_screen", "pubmedqa_article_lineage_fully_certified",
        "leakage_free_final_splits_approved", "labels_scientifically_adjudicated",
        "e1_scientific_model_experiment_executed",
        "source_question_answer_pubids_exported", "integrity_sha256",
    }
    if not isinstance(body, dict) or set(body) != allowed:
        raise ValueError("UNEXPECTED_AGGREGATE_CONTENT_FIELDS")
    if (type(body["schema_version"]) is not int or body["schema_version"] != 1
            or body["audit_kind"] != "e1_pubmedqa_source_lineage_advanced_leakage_v1"
            or body["github_source_git_sha"] != source_sha
            or body["medhallu_revision"] != MEDHALLU_REVISION
            or body["pubmedqa_revision"] != PUBMEDQA_REVISION
            or body["pubmedqa_official_test_repo_revision"] != PUBMEDQA_OFFICIAL_GIT_REVISION
            or body["candidate_assignment_sha256"] != CANDIDATE_ASSIGNMENT_SHA256):
        raise ValueError("SOURCE_OR_PROTOCOL_INCONSISTENT")
    observed = body["integrity_sha256"]
    if not isinstance(observed, str) or not HASH64.fullmatch(observed) or (
            sha256(canonical({k: v for k, v in body.items() if k != "integrity_sha256"})).hexdigest() != observed):
        raise ValueError("AGGREGATE_CONTENT_SHA256_MISMATCH")
    original_test_hash = body["original_pubmedqa_500_test_sha256"]
    if not isinstance(original_test_hash, str) or not HASH64.fullmatch(original_test_hash):
        raise ValueError("ORIGINAL_TEST_FILE_DIGEST_INVALID")
    for key in (
        "pubmedqa_article_lineage_fully_certified", "leakage_free_final_splits_approved",
        "labels_scientifically_adjudicated", "e1_scientific_model_experiment_executed",
        "source_question_answer_pubids_exported",
    ):
        if body[key] is not False:
            raise ValueError("UNSUPPORTED_SCIENTIFIC_APPROVAL_OR_SOURCE_EXPORT")
    if body["medhallu_source_sha256"] != {k: value[2] for k, value in SOURCE_SHA256.items()}:
        raise ValueError("MEDHALLU_SOURCE_BYTES_NOT_FROZEN")
    if body["pubmedqa_source_sha256"] != {k: value[2] for k, value in PUBMEDQA_SHA256.items()}:
        raise ValueError("PUBMEDQA_SOURCE_BYTES_NOT_FROZEN")
    expected_modules = {
        "medhallu_native_audit": EXPECTED_MODULE_SHA256["medhallu_native_audit"],
        "e1_candidate_splits": EXPECTED_MODULE_SHA256["e1_candidate_splits"],
        "e1_advanced_lineage": sha256((ROOT/"src/research_data/e1_advanced_lineage.py").read_bytes()).hexdigest(),
    }
    if body["source_module_sha256"] != expected_modules:
        raise ValueError("REVIEWED_SOURCE_MODULE_BYTES_CHANGED")
    lineage = body["matched_pubmedqa_lineage_screen"]
    validate_report(lineage)
    if type(expected_rows) is not int or expected_rows < 1 or lineage["eligible_parent_rows_checked"] != expected_rows:
        raise ValueError("WRONG_ELIGIBLE_ROW_COUNT")
    validate_near_report(body["advanced_near_duplicate_screen"], expected_rows)
    return body


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=Path)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    verify(args.file, args.source_commit)
    print("E1_PUBMEDQA_LINEAGE_AGGREGATE_CONSISTENCY_VERIFIED_NO_SCIENTIFIC_APPROVAL")


if __name__ == "__main__":
    main()
