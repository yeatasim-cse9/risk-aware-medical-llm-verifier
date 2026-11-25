"""Verify Kaggle metadata-only manifest without exposing or processing data rows."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from research_data.manager import verify_report

def verify(output_dir: Path, expected_commit: str) -> dict:
    matches = list(output_dir.rglob("medhallu_metadata_manifest.json"))
    if len(matches) != 1:
        raise ValueError("Expected exactly one metadata manifest")
    if matches[0].stat().st_size > 100_000:
        raise ValueError("Metadata-only manifest exceeds size limit")
    data = json.loads(matches[0].read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Unexpected manifest type")
    verify_report(data, expected_commit)
    print("MEDHALLU_KAGGLE_METADATA_INTEGRITY_AND_SOURCE_VERIFIED")
    print("NO_DATASET_ROWS_OR_SCIENTIFIC_EXPERIMENTS_IN_AUDIT")
    return data

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("directory", type=Path)
    p.add_argument("--expected-commit", required=True)
    a = p.parse_args()
    verify(a.directory, a.expected_commit)
