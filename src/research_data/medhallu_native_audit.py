"""MedHallu native-row scientific gate; aggregate-only and private-environment use.

This module has NO network, file upload, Kaggle submission, dataset download, or
implicit label mapping. It must be supplied already-authorized rows by a separate
reviewed execution adapter. Never print rows, text, labels or identifiers.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import re
import unicodedata

DATASET = "UTAustin-AIHealth/MedHallu"
CONFIGS = ("pqa_artificial", "pqa_labeled")
NATIVE_FIELDS = frozenset({
    "Question", "Knowledge", "Ground Truth", "Difficulty Level",
    "Hallucinated Answer", "Category of Hallucination",
})
MAX_ROWS = 20_000
MAX_CHARS = 50_000
MAX_KNOWLEDGE = 64
MAX_REPORT_BYTES = 8_192


class NativeAuditError(ValueError):
    """Contains error codes, never source text or private identifiers."""


def _norm(text: str) -> str:
    return re.sub(r"[^\w]+", " ", unicodedata.normalize("NFKC", text).casefold()).strip()


def _sha(text: str) -> str:
    return sha256(_norm(text).encode("utf-8")).hexdigest()


def audit_native_rows(rows_by_config: dict[str, list[dict]]) -> dict:
    """Run strictly structural checks and source-question duplication analysis.

    This cannot validate ground-truth medical correctness, dataset rights,
    hallucination category semantics, or train/test split eligibility.
    """
    if not isinstance(rows_by_config, dict) or set(rows_by_config) != set(CONFIGS):
        raise NativeAuditError("CONFIG_SET_INVALID")
    totals = {}
    question_origins: dict[str, set[str]] = defaultdict(set)
    all_questions = Counter()
    conflicting_answers = 0
    all_pair_fingerprints = set()
    identical_answer_pairs = 0
    unique_counts = {}
    for config in CONFIGS:
        rows = rows_by_config[config]
        if not isinstance(rows, list) or not rows or len(rows) > MAX_ROWS:
            raise NativeAuditError("ROW_COUNT_OUT_OF_POLICY")
        q_in_config = Counter()
        counts = Counter()
        for row in rows:
            if not isinstance(row, dict) or set(row) != NATIVE_FIELDS:
                raise NativeAuditError("NATIVE_FIELD_SET_UNEXPECTED")
            for field in ("Question", "Ground Truth", "Hallucinated Answer",
                          "Difficulty Level", "Category of Hallucination"):
                value = row[field]
                if not isinstance(value, str) or not value.strip() or len(value) > MAX_CHARS:
                    raise NativeAuditError("NATIVE_TEXT_FIELD_INVALID")
            knowledge = row["Knowledge"]
            if (not isinstance(knowledge, list) or len(knowledge) > MAX_KNOWLEDGE
                    or any(not isinstance(item, str) or len(item) > MAX_CHARS for item in knowledge)):
                raise NativeAuditError("NATIVE_KNOWLEDGE_FIELD_INVALID")
            qhash = _sha(row["Question"])
            ghash = _sha(row["Ground Truth"])
            hhash = _sha(row["Hallucinated Answer"])
            if not (qhash and ghash and hhash):
                raise NativeAuditError("EMPTY_NORMALIZED_FIELD")
            q_in_config[qhash] += 1
            all_questions[qhash] += 1
            question_origins[qhash].add(config)
            if ghash == hhash:
                conflicting_answers += 1
            pair = (config, qhash, ghash, hhash)
            if pair in all_pair_fingerprints:
                identical_answer_pairs += 1
            all_pair_fingerprints.add(pair)
            counts["rows"] += 1
        totals[config] = counts["rows"]
        unique_counts[config] = len(q_in_config)
    # All report values are counts/statuses. Never emit values of native fields,
    # raw text, question IDs, answer fingerprints, or label names.
    report = {
        "schema_version": 1,
        "dataset_id": DATASET,
        "audit_kind": "native_rows_structural_only",
        "config_rows": totals,
        "unique_questions_by_config": unique_counts,
        "unique_questions_total": len(all_questions),
        "questions_repeated_within_or_across_configs": sum(n > 1 for n in all_questions.values()),
        "questions_shared_across_configs": sum(len(v) > 1 for v in question_origins.values()),
        "ground_truth_equals_hallucinated_answer_pairs": conflicting_answers,
        "identical_record_pairs": identical_answer_pairs,
        "native_fields_count": len(NATIVE_FIELDS),
        "native_label_mapping_approved": False,
        "source_lineage_ids_verified": False,
        "leakage_safe_train_dev_test_created": False,
        "license_and_pubmedqa_rights_verified": False,
        "scientific_experiments_executed": False,
        "raw_rows_or_values_exported": False,
        "status": "REQUIRES_REVIEW_NO_SCIENTIFIC_EXPERIMENT",
    }
    if len(json.dumps(report).encode("utf-8")) > MAX_REPORT_BYTES:
        raise NativeAuditError("AGGREGATE_REPORT_LIMIT_EXCEEDED")
    return report


def verify_report(report: dict) -> None:
    if not isinstance(report, dict) or report.get("audit_kind") != "native_rows_structural_only":
        raise NativeAuditError("INVALID_REPORT_KIND")
    expected = (
        "native_label_mapping_approved", "source_lineage_ids_verified",
        "leakage_safe_train_dev_test_created", "license_and_pubmedqa_rights_verified",
        "scientific_experiments_executed", "raw_rows_or_values_exported",
    )
    if any(report.get(key) is not False for key in expected):
        raise NativeAuditError("UNAUTHORIZED_SCIENTIFIC_OR_LICENSE_CLAIM")
    if report.get("dataset_id") != DATASET or report.get("status") != "REQUIRES_REVIEW_NO_SCIENTIFIC_EXPERIMENT":
        raise NativeAuditError("INVALID_DATASET_OR_STATUS")
    if set(report.get("config_rows", {})) != set(CONFIGS):
        raise NativeAuditError("INVALID_CONFIG_COUNTS")


def write_aggregate(report: dict, destination: Path) -> None:
    verify_report(report)
    if destination.exists():
        raise NativeAuditError("REPORT_OVERWRITE_FORBIDDEN")
    if destination.parent.exists() and destination.parent.is_symlink():
        raise NativeAuditError("SYMLINK_DESTINATION_FORBIDDEN")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
