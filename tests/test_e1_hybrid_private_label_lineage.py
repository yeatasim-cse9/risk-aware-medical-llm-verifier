"""Synthetic-only checks for private manual Kaggle E1 label/lineage screening."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import e1_hybrid_private_label_lineage as hybrid
from research_data.e1_candidate_splits import candidate_plan
from research_data.medhallu_native_audit import _norm


def fake_row(question: str, category: str = "Invented label") -> dict:
    return {
        "Question": question,
        "Knowledge": ["entirely fictional background"],
        "Ground Truth": "fictional answer 1",
        "Hallucinated Answer": "fictional answer 2",
        "Difficulty Level": "invented level",
        "Category of Hallucination": category,
    }


class HybridAuditTests(unittest.TestCase):
    def test_expected_source_commit_and_verified_raw_file_digests(self):
        ev = json.loads((ROOT / "configs/medhallu-e1-prior-aggregate-evidence.json").read_text())
        self.assertEqual(hybrid.REVISION, ev["dataset_revision_sha"])
        self.assertEqual(hybrid.EXPECTED_ASSIGNMENT_SHA,
                         "ddd32f57ebc49ec6591cffd2d953a869c30518525bf66d6e2fb931a68bb0945f")
        for config in hybrid.EXPECTED_FILES:
            name, size, digest, n = hybrid.EXPECTED_FILES[config]
            self.assertEqual({
                "relative_file": name, "size_bytes": size, "sha256": digest, "rows": n
            }, ev["source_files"][config])

    def test_rare_ngram_screen_flags_near_equal_cross_partition(self):
        rows = {
            "pqa_labeled": [
                fake_row("invented notebook topic interesting vivid colorful words"),
                fake_row("unrelated tiny question"),
            ],
            "pqa_artificial": [
                fake_row("invented notebook topic interesting vivid colorful words!"),
            ],
        }
        assignments = {
            ("pqa_labeled", 0): "train",
            ("pqa_labeled", 1): "train",
            ("pqa_artificial", 0): "test",
        }
        x = hybrid.screen_near_duplicate_questions(rows, assignments, _norm)
        self.assertGreaterEqual(x["cross_partition_high_token_trigram_overlap_pairs"], 1)
        self.assertEqual(x["eligible_questions_too_short_for_5_word_screen"], 1)
        self.assertFalse(x["complete_semantic_duplicate_search"])
        self.assertFalse(x["pubmedqa_article_lineage_verified"])
        self.assertNotIn("invented notebook", json.dumps(x))

    def test_rare_ngram_screen_detects_no_unrelated_overlap(self):
        rows = {
            "pqa_labeled": [fake_row("green calm violet sunrise peaceful forest")],
            "pqa_artificial": [fake_row("orange robot castle garden purple city")],
        }
        assignments = {
            ("pqa_labeled", 0): "dev",
            ("pqa_artificial", 0): "test",
        }
        x = hybrid.screen_near_duplicate_questions(rows, assignments, _norm)
        self.assertEqual(x["cross_partition_high_token_trigram_overlap_pairs"], 0)
        self.assertEqual(x["cross_partition_candidate_pairs_compared"], 0)

    def test_cap_marked_incomplete_and_no_patient_rows_exported(self):
        rows = {
            "pqa_labeled": [fake_row(f"common common common common common item {i}") for i in range(20)],
            "pqa_artificial": [fake_row(f"common common common common common sample {i}") for i in range(20)],
        }
        assignments = {("pqa_labeled", i): "train" for i in range(20)}
        assignments.update({("pqa_artificial", i): "test" for i in range(20)})
        limited = hybrid.screen_near_duplicate_questions(rows, assignments, _norm, candidate_limit=1)
        self.assertTrue(limited["candidate_limit_reached"])
        self.assertFalse(limited["complete_semantic_duplicate_search"])
        with self.assertRaises(hybrid.SafeAuditError):
            hybrid.screen_near_duplicate_questions(rows, assignments, _norm, candidate_limit=0)

    def test_file_digest_and_canonical_hashes(self):
        with tempfile.TemporaryDirectory() as d:
            file = Path(d) / "fake-nonmedical"
            file.write_bytes(b"synthetic only")
            self.assertEqual(hybrid.hash_file(file), hashlib.sha256(b"synthetic only").hexdigest())
        one = {"y": 2, "x": 1}
        self.assertEqual(hybrid.canonical(one), b'{"x":1,"y":2}')

    def test_fail_closed_error_constants_and_no_automatic_e1_training(self):
        file = (ROOT / "scripts/e1_hybrid_private_label_lineage.py").read_text()
        compile(file, "hybrid-loader", "exec")
        self.assertIn("E1_HYBRID_FAILED_STAGE=", file)
        self.assertIn("E1_HYBRID_FAILED_CODE=", file)
        self.assertNotIn("traceback.print_exc", file)
        self.assertNotIn("print(exc)", file)
        self.assertIn("E1_HYBRID_LINEAGE_AND_FINAL_E1_STATUS=NOT_APPROVED", file)
        self.assertIn("EXPECTED_ASSIGNMENT_SHA", file)
        self.assertIn('"raw_medical_text_or_source_ids_exported": False', file)
        self.assertIn('"final_train_dev_test_approved": False', file)


if __name__ == "__main__":
    unittest.main()
