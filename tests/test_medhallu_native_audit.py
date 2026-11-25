"""All tests use invented, nonmedical text; no network or real dataset rows."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from research_data.medhallu_native_audit import (
    NATIVE_FIELDS, NativeAuditError, audit_native_rows, verify_report, write_aggregate,
)


def sample(question: str, truth: str = "truth alpha", hallucination: str = "false beta") -> dict:
    return {
        "Question": question,
        "Knowledge": ["unrelated invented evidence"],
        "Ground Truth": truth,
        "Difficulty Level": "synthetic_difficulty",
        "Hallucinated Answer": hallucination,
        "Category of Hallucination": "synthetic_category",
    }


class NativeAuditTests(unittest.TestCase):
    def test_duplicate_question_across_configs_is_reported_without_text(self):
        rows = {
            "pqa_labeled": [sample("shared invented question"), sample("independent text")],
            "pqa_artificial": [sample("Shared invented question!"), sample("other invented prompt")],
        }
        report = audit_native_rows(rows)
        verify_report(report)
        self.assertEqual(report["questions_shared_across_configs"], 1)
        self.assertEqual(report["questions_repeated_within_or_across_configs"], 1)
        self.assertEqual(report["unique_questions_total"], 3)
        self.assertEqual(report["config_rows"], {"pqa_artificial": 2, "pqa_labeled": 2})
        self.assertFalse(report["leakage_safe_train_dev_test_created"])
        serialized = json.dumps(report)
        for forbidden in ("shared invented", "truth alpha", "false beta",
                          "synthetic_category", "unrelated invented"):
            self.assertNotIn(forbidden, serialized)

    def test_answer_collision_and_duplicate_record_counts(self):
        row = sample("example dummy", "same text", "SAME text!")
        report = audit_native_rows({
            "pqa_labeled": [row, row], "pqa_artificial": [sample("different question")]
        })
        self.assertEqual(report["ground_truth_equals_hallucinated_answer_pairs"], 2)
        self.assertEqual(report["identical_record_pairs"], 1)

    def test_bad_types_and_native_field_mismatch_fail_closed(self):
        original = sample("invented")
        for key, val in [
            ("Knowledge", "not a list"), ("Knowledge", [42]),
            ("Question", ""), ("Ground Truth", 0),
            ("Category of Hallucination", None), ("Difficulty Level", []),
        ]:
            changed = dict(original)
            changed[key] = val
            with self.subTest(key=key, val=val), self.assertRaises(NativeAuditError):
                audit_native_rows({"pqa_labeled": [changed],
                                   "pqa_artificial": [sample("other")]})
        changed = dict(original)
        changed["unreviewed"] = "potentially sensitive"
        with self.assertRaises(NativeAuditError):
            audit_native_rows({"pqa_labeled": [changed],
                               "pqa_artificial": [sample("other")]})
        self.assertEqual(len(NATIVE_FIELDS), 6)

    def test_no_implicit_label_approval_or_research(self):
        report = audit_native_rows({
            "pqa_labeled": [sample("question a")],
            "pqa_artificial": [sample("question b")],
        })
        for flag in ("native_label_mapping_approved", "source_lineage_ids_verified",
                     "license_and_pubmedqa_rights_verified",
                     "leakage_safe_train_dev_test_created", "scientific_experiments_executed"):
            modified = dict(report)
            modified[flag] = True
            with self.subTest(flag=flag), self.assertRaises(NativeAuditError):
                verify_report(modified)

    def test_only_aggregate_file_written_and_no_overwrite(self):
        report = audit_native_rows({
            "pqa_labeled": [sample("this is synthetic")],
            "pqa_artificial": [sample("and so is this")],
        })
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp) / "aggregate.json"
            write_aggregate(report, p)
            self.assertEqual(json.loads(p.read_text()), report)
            self.assertEqual([x.name for x in Path(temp).iterdir()], ["aggregate.json"])
            with self.assertRaises(NativeAuditError):
                write_aggregate(report, p)

    def test_missing_or_massive_data_rejected(self):
        with self.assertRaises(NativeAuditError):
            audit_native_rows({"pqa_labeled": [], "pqa_artificial": [sample("b")]})
        with self.assertRaises(NativeAuditError):
            audit_native_rows({"pqa_labeled": [sample("a")], "other": [sample("b")]})
        with self.assertRaises(NativeAuditError):
            audit_native_rows({"pqa_labeled": [sample("a")],
                               "pqa_artificial": [sample("x" * 50001)]})


if __name__ == "__main__":
    unittest.main()
