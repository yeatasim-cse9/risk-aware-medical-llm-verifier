"""MC01: fail-closed, metadata-only MedHallu source audit.

Uses only public dataset metadata endpoints. It NEVER requests data rows,
downloads Parquet files, maps clinical labels, or creates test splits.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import urlencode
from urllib.request import Request, urlopen

DATASET_ID = "UTAustin-AIHealth/MedHallu"
CONFIGS = ("pqa_labeled", "pqa_artificial")
HUB = "https://huggingface.co/api/datasets/" + DATASET_ID
VIEWER = "https://datasets-server.huggingface.co"
SHA40 = re.compile(r"[0-9a-f]{40}\Z")
SHA64 = re.compile(r"[0-9a-f]{64}\Z")
MAX_RESPONSE_BYTES = 2_000_000


def canonical(body: dict) -> bytes:
    return json.dumps(body, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode("utf-8")


def fetch_public_metadata(url: str) -> dict:
    """Allowlist endpoints; never fetch /rows, /first-rows, /resolve or data bytes."""
    if not (
        url == HUB
        or url == VIEWER + "/splits?" + urlencode({"dataset": DATASET_ID})
        or any(url == VIEWER + "/info?" + urlencode({"dataset": DATASET_ID, "config": c}) for c in CONFIGS)
    ):
        raise PermissionError("Non-metadata or unapproved network endpoint blocked")
    req = Request(url, headers={"User-Agent": "medverify-metadata-audit/1.0"})
    with urlopen(req, timeout=25) as response:
        content = response.read(MAX_RESPONSE_BYTES + 1)
    if len(content) > MAX_RESPONSE_BYTES:
        raise ValueError("Metadata response unexpectedly large")
    payload = json.loads(content)
    if not isinstance(payload, dict):
        raise ValueError("Expected JSON metadata object")
    return payload


def validate_cloud_dataset(dataset_id: str) -> None:
    """Only MedHallu metadata, never MedVAL or unapproved corpora."""
    if dataset_id != DATASET_ID:
        raise PermissionError("Only approved public MedHallu metadata is allowed in this Kaggle job")


def build_report(source_commit: str, fetch=fetch_public_metadata) -> dict:
    if not SHA40.fullmatch(source_commit):
        raise ValueError("Exact 40-character source Git SHA required")
    validate_cloud_dataset(DATASET_ID)
    hub = fetch(HUB)
    splits = fetch(VIEWER + "/splits?" + urlencode({"dataset": DATASET_ID}))
    if hub.get("id") != DATASET_ID:
        raise ValueError("Hugging Face source identity mismatch")
    revision = hub.get("sha")
    if not isinstance(revision, str) or not SHA40.fullmatch(revision):
        raise ValueError("Remote dataset revision missing or invalid")
    if not isinstance(splits.get("splits"), list):
        raise ValueError("Dataset-viewer split metadata missing")
    listed = splits["splits"]
    configurations = {}
    for config in CONFIGS:
        data = fetch(VIEWER + "/info?" + urlencode({"dataset": DATASET_ID, "config": config}))
        info = data.get("dataset_info")
        if not isinstance(info, dict):
            raise ValueError("Dataset-viewer schema metadata missing for " + config)
        features = info.get("features")
        if not isinstance(features, dict) or not features:
            raise ValueError("Native feature schema unavailable for " + config)
        rows = []
        for entry in listed:
            if not isinstance(entry, dict) or entry.get("config") != config:
                continue
            name, count = entry.get("split"), entry.get("num_rows")
            if not isinstance(name, str) or not name or type(count) is not int or count < 0:
                raise ValueError("Invalid split metadata for " + config)
            rows.append({"name": name, "declared_rows": count})
        if not rows:
            raise ValueError("Configuration missing in published splits: " + config)
        configurations[config] = {
            "native_column_names": sorted(features.keys()),
            "splits": sorted(rows, key=lambda x: x["name"]),
            "native_labels_confirmed": False,
            "grouping_id_confirmed": False,
        }
    card = hub.get("cardData")
    declared_license = card.get("license") if isinstance(card, dict) else None
    body = {
        "schema_version": 1,
        "audit_kind": "public_metadata_only",
        "status": "metadata_observed_schema_and_licence_review_pending",
        "source_code_git_sha": source_commit,
        "dataset_id": DATASET_ID,
        "dataset_revision_sha": revision,
        "declared_license": declared_license if isinstance(declared_license, str) else None,
        "human_license_review": "required",
        "configurations": configurations,
        "dataset_rows_downloaded": False,
        "raw_dataset_files_downloaded": False,
        "data_file_sha256": None,
        "native_label_mapping": None,
        "split_manifest_sha256": None,
        "scientific_experiments_executed": False,
        "limitations": [
            "Hub metadata is not a local data-file integrity proof",
            "Native labels and grouping identifiers require subsequent authorized row audit",
            "Neither original train split is a scientifically frozen evaluation split",
            "No physician-gold or claim-level labels inferred",
        ],
    }
    return {**body, "integrity_sha256": hashlib.sha256(canonical(body)).hexdigest()}


def write_report(report: dict, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "medhallu_metadata_manifest.json"
    if target.exists():
        raise FileExistsError("Refusing to overwrite an existing frozen metadata manifest")
    target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def verify_report(report: dict, source_commit: str) -> None:
    integrity = report.get("integrity_sha256")
    body = {k: v for k, v in report.items() if k != "integrity_sha256"}
    if not isinstance(integrity, str) or not SHA64.fullmatch(integrity):
        raise ValueError("Manifest integrity digest missing")
    if hashlib.sha256(canonical(body)).hexdigest() != integrity:
        raise ValueError("Manifest integrity mismatch")
    if report.get("source_code_git_sha") != source_commit or not SHA40.fullmatch(source_commit):
        raise ValueError("Source provenance mismatch")
    if report.get("dataset_id") != DATASET_ID or not SHA40.fullmatch(str(report.get("dataset_revision_sha"))):
        raise ValueError("Dataset identity/revision mismatch")
    if report.get("audit_kind") != "public_metadata_only":
        raise ValueError("Unexpected audit kind")
    for flag in ("dataset_rows_downloaded", "raw_dataset_files_downloaded", "scientific_experiments_executed"):
        if report.get(flag) is not False:
            raise ValueError("This audit must not consume data rows or run experiments")
    if report.get("human_license_review") != "required" or report.get("native_label_mapping") is not None:
        raise ValueError("Metadata report cannot claim verified licensing/labels")
    configs = report.get("configurations")
    if not isinstance(configs, dict) or set(configs) != set(CONFIGS):
        raise ValueError("Required source configurations missing")
    for cfg in CONFIGS:
        record = configs[cfg]
        if not isinstance(record.get("native_column_names"), list) or not record["native_column_names"]:
            raise ValueError("Feature fields missing")
        if not isinstance(record.get("splits"), list) or not record["splits"]:
            raise ValueError("Split metadata missing")
        if any(type(r.get("declared_rows")) is not int or r["declared_rows"] < 0 for r in record["splits"]):
            raise ValueError("Split counts invalid")


def main() -> None:
    parser = argparse.ArgumentParser(description="MedHallu metadata-only audit (NO dataset download)")
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(args.source_commit)
    verify_report(report, args.source_commit)
    write_report(report, args.output)
    print("MEDHALLU_PUBLIC_METADATA_MANIFEST_SAVED; no dataset rows downloaded")


if __name__ == "__main__":
    main()
