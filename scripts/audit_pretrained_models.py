"""Validate the Step 10.5C official-model-card shortlist offline.

This gate performs NO downloads, uploads, external requests or GPU jobs.
Licenses are repository/model-card claims, not a complete legal clearance.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
AUDIT_PATH = ROOT / "configs" / "pretrained-model-audit.json"

EXPECTED = {
    "sentence-transformers/all-MiniLM-L6-v2": (
        "retrieval_embedding_baseline", "apache-2.0",
        "sentence_embeddings", None,
    ),
    "cross-encoder/nli-MiniLM2-L6-H768": (
        "three_way_nli_baseline", "apache-2.0",
        "sentence_pair_nli", ["contradiction", "entailment", "neutral"],
    ),
    "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract": (
        "biomedical_domain_encoder_candidate", "mit",
        "masked_language_model", None,
    ),
}
FIELDS = {
    "model_id", "proposed_role", "declared_license", "evidence_url",
    "license_evidence_url", "source_task", "domain", "nli_id2label",
    "limitations", "license_text_inspected", "model_revision_sha",
    "model_file_sha256", "tokenizer_sha256", "package_size_bytes",
    "measured_gpu_peak_memory_bytes", "downstream_data_rights_reviewed",
    "weight_redistribution_reviewed", "execution_approved", "downloads_executed",
}
REQUIRED_FALSE = (
    "downstream_data_rights_reviewed",
    "weight_redistribution_reviewed",
    "execution_approved",
    "downloads_executed",
)
SECURITY = {
    "repo_is_public": True,
    "restricted_medval_allowed_on_public_github": False,
    "restricted_medval_allowed_on_kaggle_without_permission": False,
    "model_weight_upload_allowed_in_this_stage": False,
    "kaggle_gpu_submission_allowed_in_this_stage": False,
    "data_upload_allowed_in_this_stage": False,
    "license_tag_is_full_legal_clearance": False,
    "require_exact_revision_before_next_model_execution": True,
    "require_source_hashes_and_tokenizer_hashes": True,
    "require_dataset_rights_and_weight_distribution_review": True,
    "require_runtime_gpu_peak_memory_measurement": True,
    "require_manual_separate_gpu_approval": True,
}


class AuditError(ValueError):
    pass


def validate(data: object) -> list[dict]:
    if not isinstance(data, dict) or set(data) != {
        "schema_version", "audit_date_utc", "audit_status", "purpose", "models",
        "policy",
    }:
        raise AuditError("AUDIT_SCHEMA_INVALID")
    if type(data["schema_version"]) is not int or data["schema_version"] != 1:
        raise AuditError("AUDIT_VERSION_INVALID")
    if data["audit_date_utc"] != "2026-11-25":
        raise AuditError("AUDIT_REVIEW_DATE_INVALID")
    if data["audit_status"] != "official_model_card_metadata_review_only":
        raise AuditError("AUDIT_MAY_NOT_CLAIM_FULL_LICENSE_CLEARANCE")
    if not isinstance(data["purpose"], str) or len(data["purpose"]) < 30:
        raise AuditError("AUDIT_PURPOSE_MISSING")
    if data["policy"] != SECURITY:
        raise AuditError("AUDIT_SECURITY_POLICY_CHANGED")
    models = data["models"]
    if not isinstance(models, list) or len(models) != len(EXPECTED):
        raise AuditError("AUDIT_CANDIDATE_COUNT_INVALID")
    seen = set()
    for model in models:
        if not isinstance(model, dict) or set(model) != FIELDS:
            raise AuditError("AUDIT_MODEL_SCHEMA_INVALID")
        name = model["model_id"]
        if name not in EXPECTED or name in seen:
            raise AuditError("AUDIT_MODEL_UNLISTED_OR_DUPLICATE")
        seen.add(name)
        role, license_tag, task, labels = EXPECTED[name]
        if (
            model["proposed_role"] != role
            or model["declared_license"] != license_tag
            or model["source_task"] != task
            or model["nli_id2label"] != labels
        ):
            raise AuditError("AUDIT_ROLE_LICENSE_OR_NLI_LABEL_MISMATCH")
        canonical = f"https://huggingface.co/{name}"
        if model["evidence_url"] != canonical:
            raise AuditError("AUDIT_MODEL_SOURCE_URL_INVALID")
        license_url = model["license_evidence_url"]
        if not isinstance(license_url, str) or not license_url.startswith(canonical):
            raise AuditError("AUDIT_LICENSE_EVIDENCE_URL_INVALID")
        if not isinstance(model["domain"], str) or len(model["domain"]) < 10:
            raise AuditError("AUDIT_TRAINING_DOMAIN_MISSING")
        if not isinstance(model["limitations"], str) or len(model["limitations"]) < 50:
            raise AuditError("AUDIT_LIMITATIONS_MISSING")
        if type(model["license_text_inspected"]) is not bool:
            raise AuditError("AUDIT_LICENSE_REVIEW_FLAG_INVALID")
        if name.startswith("microsoft/") and not model["license_text_inspected"]:
            raise AuditError("AUDIT_BIOMED_LICENSE_TEXT_NOT_REVIEWED")
        # This audit does NOT approve an inferred upstream revision or an
        # unexamined model download. A separate stage must freeze both.
        for field in (
            "model_revision_sha", "model_file_sha256", "tokenizer_sha256",
            "package_size_bytes", "measured_gpu_peak_memory_bytes",
        ):
            if model[field] is not None:
                raise AuditError("AUDIT_UNVERIFIED_BINARY_METADATA")
        if any(model[field] is not False for field in REQUIRED_FALSE):
            raise AuditError("AUDIT_GPU_OR_TRANSFER_PREMATURELY_APPROVED")
    return models


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, default=AUDIT_PATH)
    args = parser.parse_args()
    try:
        models = validate(json.loads(args.catalog.read_text(encoding="utf-8")))
    except Exception:
        print("PRETRAINED_MODEL_AUDIT: FAIL SAFE_NO_GPU_OR_MODEL_TRANSFER")
        return 1
    print(f"PRETRAINED_MODEL_AUDIT: MODEL_CARD_METADATA_VERIFIED {len(models)}_CANDIDATES")
    print("PRETRAINED_MODEL_AUDIT: ALL_UPSTREAM_REVISIONS_UNFROZEN")
    print("PRETRAINED_MODEL_AUDIT: LICENSE_AND_DATA_TRANSFER_APPROVAL_PENDING")
    print("PRETRAINED_MODEL_AUDIT: NO_GPU_EXECUTION_NO_MODEL_DOWNLOAD")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
