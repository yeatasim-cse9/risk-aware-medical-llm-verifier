"""Fail-closed, offline E1 readiness gate. Never downloads or processes medical rows."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "configs/experiment-registry.json"
SOURCES = ROOT / "configs/dataset-sources.json"
# This approval artifact is deliberately NOT present until independently reviewed.
APPROVAL = ROOT / "configs/e1-approved-protocol.json"
SHA = re.compile(r"[0-9a-f]{40}\Z")
DIGEST = re.compile(r"[0-9a-f]{64}\Z")


def check(root: Path = ROOT) -> dict:
    """Return an evidence-aware, non-sensitive report; no network, no writes."""
    reasons: list[str] = []
    reg = json.loads((root / "configs/experiment-registry.json").read_text(encoding="utf-8"))
    src = json.loads((root / "configs/dataset-sources.json").read_text(encoding="utf-8"))
    e1 = next((x for x in reg.get("experiments", []) if x.get("id") == "E1"), None)
    med = next((x for x in src.get("datasets", []) if x.get("id") == "medhallu"), None)
    if not isinstance(e1, dict) or e1.get("dataset") != "MedHallu":
        reasons.append("E1_REGISTRY_NOT_VALIDATED")
    if not isinstance(med, dict) or med.get("official_source") != "https://huggingface.co/datasets/UTAustin-AIHealth/MedHallu":
        reasons.append("MEDHALLU_SOURCE_NOT_VALIDATED")
    if isinstance(med, dict):
        if med.get("access_verified") is not True:
            reasons.append("DATASET_ACCESS_NOT_VERIFIED")
        if med.get("downloaded") is not True or not isinstance(med.get("sha256"), str) or not DIGEST.fullmatch(med["sha256"]):
            reasons.append("DATASET_BYTES_NOT_HASH_VERIFIED")
        if not (isinstance(med.get("version"), str) and SHA.fullmatch(med["version"])):
            reasons.append("DATASET_REVISION_NOT_IMMUTABLY_PINNED")
    if not (root / "configs/e1-approved-protocol.json").is_file():
        reasons.append("E1_PROTOCOL_AND_NATIVE_LABEL_MAPPING_NOT_APPROVED")
    else:
        try:
            protocol = json.loads((root / "configs/e1-approved-protocol.json").read_text(encoding="utf-8"))
            required = ("schema_version", "approved", "dataset_revision", "data_sha256",
                        "native_schema_audited", "labels_mapping_approved", "lineage_audited",
                        "leakage_safe_splits_frozen", "baseline_and_evaluator_frozen",
                        "licenses_reviewed", "reproducibility_manifest_reviewed")
            if not isinstance(protocol, dict) or set(protocol) != set(required):
                raise ValueError("invalid approved-protocol schema")
            if type(protocol["schema_version"]) is not int or protocol["schema_version"] != 1 or protocol["approved"] is not True or any(protocol[k] is not True for k in required[4:]):
                raise ValueError("approvals incomplete")
            if not (isinstance(protocol["dataset_revision"], str) and SHA.fullmatch(protocol["dataset_revision"])):
                raise ValueError("bad revision")
            if not (isinstance(protocol["data_sha256"], str) and DIGEST.fullmatch(protocol["data_sha256"])):
                raise ValueError("bad digest")
            if not isinstance(med, dict) or protocol["dataset_revision"] != med.get("version") or protocol["data_sha256"] != med.get("sha256"):
                raise ValueError("approval does not match dataset source")
        except (ValueError, TypeError, KeyError):
            reasons.append("E1_PROTOCOL_APPROVAL_INVALID_OR_MISMATCHED")
    if e1 is not None and e1.get("status") != "ready-for-approved-execution":
        reasons.append("E1_NOT_AUTHORIZED_FOR_EXECUTION")
    # This check is a necessary safety gate, not proof that approvals are genuine.
    return {"experiment": "E1", "ready": not reasons, "status": "READY" if not reasons else "BLOCKED",
            "blocking_reasons": sorted(set(reasons)), "scientific_runs_executed": 0,
            "external_calls": 0, "patient_data_processed": False}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=ROOT)
    args = p.parse_args()
    report = check(args.root)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["ready"]:
        print("E1_PREFLIGHT_READY_REQUIRES_SEPARATE_EXPLICIT_EXECUTION_APPROVAL")
    else:
        print("E1_PREFLIGHT_BLOCKED_NO_SUBMISSION_NO_MEDICAL_DATA")


if __name__ == "__main__":
    main()
