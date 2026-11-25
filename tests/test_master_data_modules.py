"""Synthetic code tests only. These numbers are NOT benchmark results."""
from __future__ import annotations
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from urllib.parse import urlencode

from src.research_data.manager import (
    DATASET_ID, HUB, VIEWER, build_report, validate_cloud_dataset,
    verify_report, write_report, fetch_public_metadata,
)
from src.research_data.audit import audit_rows, audit_jsonl

COMMIT = "b" * 40
REVISION = "a" * 40


def metadata_fixture(url):
    if url == HUB:
        return {"id": DATASET_ID, "sha": REVISION, "cardData": {"license": "mit"}}
    if url == VIEWER + "/splits?" + urlencode({"dataset": DATASET_ID}):
        return {"splits": [
            {"config": "pqa_labeled", "split": "train"},
            {"config": "pqa_artificial", "split": "train"},
        ]}
    if url == VIEWER + "/size?" + urlencode({"dataset": DATASET_ID}):
        return {"size": {"splits": [
            {"config": "pqa_labeled", "split": "train", "num_rows": 1000},
            {"config": "pqa_artificial", "split": "train", "num_rows": 9000},
        ]}, "partial": False}
    if url.startswith(VIEWER + "/info?"):
        return {"dataset_info": {"features": {
            "Question": {"dtype": "string"}, "Hallucinated Answer": {"dtype": "string"},
        }}}
    raise AssertionError("Unexpected network endpoint")


class MasterDataTests(unittest.TestCase):
    def test_metadata_manifest_no_rows(self):
        m = build_report(COMMIT, metadata_fixture)
        verify_report(m, COMMIT)
        self.assertFalse(m["dataset_rows_downloaded"])
        self.assertIsNone(m["native_label_mapping"])
        self.assertEqual(m["configurations"]["pqa_labeled"]["native_column_names"],
                         ["Hallucinated Answer", "Question"])
        self.assertEqual(m["source_code_git_sha"], COMMIT)
        self.assertEqual(m["configurations"]["pqa_artificial"]["splits"][0]["declared_rows"], 9000)

    def test_manifest_tamper_rejected(self):
        m = build_report(COMMIT, metadata_fixture)
        m["dataset_rows_downloaded"] = True
        with self.assertRaises(ValueError):
            verify_report(m, COMMIT)

    def test_cloud_restricted_block(self):
        with self.assertRaises(PermissionError):
            validate_cloud_dataset("medval")
        with self.assertRaises(PermissionError):
            fetch_public_metadata("https://huggingface.co/datasets/UTAustin-AIHealth/MedHallu/resolve/main/data.parquet")

    def test_bad_revision_rejected(self):
        def fake(url):
            result = metadata_fixture(url)
            return {**result, "sha": "main"} if url == HUB else result
        with self.assertRaises(ValueError):
            build_report(COMMIT, fake)

    def test_manifest_overwrite_forbidden(self):
        with tempfile.TemporaryDirectory() as temp:
            manifest = build_report(COMMIT, metadata_fixture)
            write_report(manifest, Path(temp))
            with self.assertRaises(FileExistsError):
                write_report(manifest, Path(temp))

    def test_exact_overlap(self):
        rows = [
            {"id": "a", "split": "train", "text": "Clinical report A."},
            {"id": "b", "split": "test", "text": "clinical report a!"},
        ]
        r = audit_rows(rows)
        self.assertFalse(r["passed"])
        self.assertEqual(r["findings_by_type"]["exact_text"], 1)
        self.assertNotIn("clinical", json.dumps(r).casefold())

    def test_near_overlap(self):
        rows = [
            {"id": "a", "split": "dev", "text": "This medication reduces blood pressure for most adults"},
            {"id": "b", "split": "test", "text": "This medication reduces blood pressure for most adults!"},
        ]
        r = audit_rows(rows)
        self.assertFalse(r["passed"])

    def test_shared_group_without_text_similarity(self):
        rows = [
            {"id": "one", "split": "dev", "text": "Some treatment", "source_id": "same"},
            {"id": "two", "split": "test", "text": "An unrelated measurement", "source_id": "same"},
        ]
        self.assertEqual(audit_rows(rows)["findings_by_type"]["shared_source_group"], 1)

    def test_unrelated_records_pass(self):
        rows = [
            {"id": "one", "split": "dev", "text": "Alpha beta gamma random"},
            {"id": "two", "split": "test", "text": "zebra wild animal travels"},
        ]
        checked = audit_rows(rows)
        self.assertTrue(checked["passed"])
        self.assertFalse(checked["ready_for_frozen_split"])
        self.assertFalse(checked["grouping_gate_passed"])

    def test_fail_closed_candidate_limit(self):
        rows = [
            {"id": "a", "split": "dev", "text": "same words on this page"},
            {"id": "b", "split": "test", "text": "same words on this page"},
            {"id": "c", "split": "test", "text": "same words on this page"},
        ]
        with self.assertRaises(ValueError):
            audit_rows(rows, max_candidate_pairs=1)

    def test_duplicate_ids_and_missing_splits_rejected(self):
        with self.assertRaises(ValueError):
            audit_rows([
                {"id": "a", "split": "dev", "text": "abc"},
                {"id": "a", "split": "dev", "text": "def"},
            ])
        with self.assertRaises(ValueError):
            audit_rows([{"id": "a", "split": "dev", "text": "abc"}])

    def test_medval_cloud_audit_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(PermissionError):
                audit_jsonl(Path(tmp)/"raw.jsonl", Path(tmp)/"audit.json", "medval")

    def test_jsonl_output_aggregate_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = [
                {"id": "s1", "split": "train", "text": "Drug action explained", "source_id": "origin"},
                {"id": "s2", "split": "dev", "text": "Different issue unrelated", "source_id": "other"},
            ]
            (root/"records.jsonl").write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
            audit_jsonl(root/"records.jsonl", root/"audit.json", "medhallu")
            data = (root/"audit.json").read_text(encoding="utf-8")
            self.assertNotIn("Drug action", data)
            self.assertNotIn("origin", data)
            self.assertNotIn("s1", data)

if __name__ == "__main__":
    unittest.main()
