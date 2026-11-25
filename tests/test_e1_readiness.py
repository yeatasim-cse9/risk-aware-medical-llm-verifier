"""Offline E1 gate regression tests: no network and no research execution."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from e1_readiness import check

REV = "a" * 40
HASH = "b" * 64


def fixture(root: Path) -> None:
    (root / "configs").mkdir()
    (root / "configs/experiment-registry.json").write_text(json.dumps({
        "experiments": [{"id": "E1", "dataset": "MedHallu",
                         "status": "ready-for-approved-execution"}]
    }))
    (root / "configs/dataset-sources.json").write_text(json.dumps({
        "datasets": [{"id": "medhallu",
                       "official_source": "https://huggingface.co/datasets/UTAustin-AIHealth/MedHallu",
                       "access_verified": True, "downloaded": True,
                       "version": REV, "sha256": HASH}]
    }))


def approved() -> dict:
    return {
        "schema_version": 1, "approved": True,
        "dataset_revision": REV, "data_sha256": HASH,
        "native_schema_audited": True, "labels_mapping_approved": True,
        "lineage_audited": True, "leakage_safe_splits_frozen": True,
        "baseline_and_evaluator_frozen": True,
        "licenses_reviewed": True, "reproducibility_manifest_reviewed": True,
    }


class ReadinessTests(unittest.TestCase):
    def test_real_repository_remains_blocked(self):
        report = check(ROOT)
        self.assertEqual(report["status"], "BLOCKED")
        self.assertIn("E1_PROTOCOL_AND_NATIVE_LABEL_MAPPING_NOT_APPROVED",
                      report["blocking_reasons"])
        self.assertEqual(report["external_calls"], 0)
        self.assertFalse(report["patient_data_processed"])

    def test_missing_approval_blocks_even_if_other_flags_true(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            fixture(root)
            self.assertFalse(check(root)["ready"])

    def test_complete_structural_approval_is_necessary(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            fixture(root)
            file = root / "configs/e1-approved-protocol.json"
            file.write_text(json.dumps(approved()))
            self.assertTrue(check(root)["ready"])
            for key in ("approved", "native_schema_audited", "labels_mapping_approved",
                        "lineage_audited", "leakage_safe_splits_frozen",
                        "baseline_and_evaluator_frozen", "licenses_reviewed",
                        "reproducibility_manifest_reviewed"):
                bad = approved()
                bad[key] = False
                file.write_text(json.dumps(bad))
                with self.subTest(key=key):
                    self.assertFalse(check(root)["ready"])
            file.write_text(json.dumps(approved()))

    def test_mismatched_pins_and_unknown_fields_block(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            fixture(root)
            file = root / "configs/e1-approved-protocol.json"
            for field, badvalue in (("dataset_revision", "c"*40),
                                    ("data_sha256", "d"*64),
                                    ("schema_version", 2)):
                bad = approved()
                bad[field] = badvalue
                file.write_text(json.dumps(bad))
                with self.subTest(field=field):
                    self.assertFalse(check(root)["ready"])
            bad = approved()
            bad["unexpected"] = True
            file.write_text(json.dumps(bad))
            self.assertFalse(check(root)["ready"])


if __name__ == "__main__":
    unittest.main()
