"""Synthetic-only regression tests: no MedHallu rows or API calls."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from research_data.e1_candidate_splits import (
    candidate_plan, verify_summary, write_safe_summary, SPLITS,
)
from research_data.medhallu_native_audit import NativeAuditError


def example(q, context="invented passage", truth="reference A", halluc="alternative B"):
    return {
        "Question": q, "Knowledge": [context], "Ground Truth": truth,
        "Hallucinated Answer": halluc, "Difficulty Level": "synthetic",
        "Category of Hallucination": "not a gold label",
    }


def test_rows(count=60):
    return {
        "pqa_labeled": [
            example(f"Dummy question labeled {i}", f"Context labeled {i}")
            for i in range(count // 2)
        ],
        "pqa_artificial": [
            example(f"Dummy question artificial {i}", f"Context artificial {i}")
            for i in range(count // 2)
        ],
    }


class CandidateSplitsTests(unittest.TestCase):
    def test_deterministic_candidate_partition_and_balanced_role_pairs(self):
        records = test_rows(100)
        first, mapping = candidate_plan(records)
        second, same_mapping = candidate_plan(records)
        self.assertEqual(first, second)
        self.assertEqual(mapping, same_mapping)
        self.assertEqual(first["eligible_parent_rows"], 100)
        self.assertEqual(first["candidate_answer_pair_examples"], 200)
        self.assertEqual(first["candidate_role_0_ground_truth_examples"], 100)
        self.assertEqual(first["candidate_role_1_hallucinated_examples"], 100)
        self.assertEqual(set(mapping.values()), set(SPLITS))
        self.assertEqual(sum(first["candidate_split_parent_rows"].values()), 100)
        self.assertEqual(sum(first["candidate_split_answer_examples"].values()), 200)
        self.assertFalse(first["final_train_dev_test_approved"])
        self.assertFalse(first["pubmedqa_lineage_verified"])
        self.assertFalse(first["binary_role_labels_scientifically_adjudicated"])
        serialized = json.dumps(first)
        for forbidden in ("Dummy question", "Context labeled", "reference A", "alternative B"):
            self.assertNotIn(forbidden, serialized)
        verify_summary(first)

    def test_repeat_questions_and_equal_answers_are_quarantined(self):
        rows = test_rows(60)
        rows["pqa_artificial"][0] = example("REPEATED synthetic question", "passage original")
        rows["pqa_artificial"][1] = example("Repeated synthetic QUESTION!", "different passage")
        rows["pqa_labeled"][0] = example("unique collision", "fake source", "same answer", "SAME ANSWER!")
        summary, assignment = candidate_plan(rows)
        self.assertEqual(summary["quarantined_by_reason"], {
            "repeated_question_group": 2, "same_answer_role_collision": 1
        })
        self.assertEqual(summary["quarantined_total_rows"], 3)
        self.assertEqual(summary["eligible_parent_rows"], 57)
        self.assertNotIn(("pqa_artificial", 0), assignment)
        self.assertNotIn(("pqa_artificial", 1), assignment)
        self.assertNotIn(("pqa_labeled", 0), assignment)
        self.assertFalse(summary["final_train_dev_test_approved"])

    def test_identical_knowledge_context_held_in_same_proxy_group(self):
        rows = test_rows(60)
        rows["pqa_labeled"][2]["Knowledge"] = ["Shared invented article passage"]
        rows["pqa_artificial"][4]["Knowledge"] = ["shared invented ARTICLE passage!!!"]
        summary, assignment = candidate_plan(rows)
        self.assertEqual(assignment[("pqa_labeled", 2)], assignment[("pqa_artificial", 4)])
        self.assertTrue(sum(summary["candidate_split_proxy_groups"].values()) < 60)
        self.assertFalse(summary["semantic_near_duplicate_leakage_excluded"])

    def test_seed_changes_reproducible_digest_not_safety_flags(self):
        rows = test_rows(80)
        a, _ = candidate_plan(rows, seed=20261125)
        b, _ = candidate_plan(rows, seed=20261126)
        self.assertNotEqual(a["candidate_assignment_sha256"], b["candidate_assignment_sha256"])
        self.assertFalse(b["final_train_dev_test_approved"])

    def test_invalid_inputs_and_tampered_approval_rejected(self):
        with self.assertRaises(NativeAuditError):
            candidate_plan(test_rows(2))
        with self.assertRaises(NativeAuditError):
            candidate_plan(test_rows(60), seed=True)
        summary, _ = candidate_plan(test_rows(30))
        altered = copy.deepcopy(summary)
        altered["final_train_dev_test_approved"] = True
        with self.assertRaises(NativeAuditError):
            verify_summary(altered)
        altered = copy.deepcopy(summary)
        altered["patient_text"] = "private record"
        with self.assertRaises(NativeAuditError):
            verify_summary(altered)
        altered = copy.deepcopy(summary)
        altered["candidate_split_answer_examples"]["train"] += 1
        with self.assertRaises(NativeAuditError):
            verify_summary(altered)

    def test_only_aggregate_export_and_no_overwrite(self):
        summary, assignment = candidate_plan(test_rows(30))
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "safe_summary.json"
            write_safe_summary(summary, path)
            self.assertEqual(json.loads(path.read_text()), summary)
            self.assertNotIn("pqa_labeled, 1", path.read_text())
            self.assertNotIn("invented", path.read_text())
            self.assertEqual([x.name for x in Path(temp).iterdir()], ["safe_summary.json"])
            with self.assertRaises(NativeAuditError):
                write_safe_summary(summary, path)
        self.assertTrue(assignment)
        self.assertNotIn("assignment", summary)


if __name__ == "__main__":
    unittest.main()
