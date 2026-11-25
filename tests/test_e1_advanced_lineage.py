"""Offline synthetic-only source lineage and advanced leakage unit tests."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from research_data.e1_advanced_lineage import (
    pubmed_index, audit_lineage, validate_report, advanced_near_screen,
    LineageAuditError, CONFIGS,
)
import e1_pubmedqa_lineage_kaggle as driver


def row(q, knowledge="Invented study context"):
    return {
        "Question": q, "Knowledge": [knowledge],
        "Ground Truth": "pretend answer A", "Hallucinated Answer": "pretend answer B",
        "Difficulty Level": "fictional", "Category of Hallucination": "synthetic",
    }


def source(pubid, question, knowledge="Invented study context"):
    return {"pubid": pubid, "question": question,
            "context": {"contexts": [knowledge], "labels": [], "meshes": []}}


def official_test():
    return {str(i) for i in range(1, 501)}


class AdvancedLineageTests(unittest.TestCase):
    def test_pubid_link_and_heldout_overlap_produce_only_counts(self):
        med = {
            "pqa_labeled": [
                row("invented red blue green title", "fictional context one"),
                row("invented green yellow title", "fictional context two"),
                row("an unrelated title never sourced", "fictional other"),
            ],
            "pqa_artificial": [
                row("second invented title", "fictional context one"),
                row("artificial other title", "context changed deliberately"),
            ],
        }
        pqa = {
            "pqa_labeled": [
                source(5, "Invented red blue green title!", "fictional context one"),
                source(700, "Invented green yellow title", "fictional context two"),
            ],
            "pqa_artificial": [
                source(5, "second invented title", "fictional context one"),
                source(555, "artificial other title", "different reference context"),
            ],
        }
        assignment = {
            ("pqa_labeled", 0): "train",
            ("pqa_labeled", 1): "dev",
            ("pqa_labeled", 2): "train",
            ("pqa_artificial", 0): "test",
            ("pqa_artificial", 1): "test",
        }
        idx = pubmed_index(pqa)
        report = audit_lineage(med, assignment, idx, official_test())
        validate_report(report)
        self.assertEqual(report["eligible_parent_rows_checked"], 5)
        self.assertEqual(report["matched_unique_pubid_parent_rows"], 4)
        self.assertEqual(report["matched_same_configuration"], 4)
        self.assertEqual(report["unmatched_parent_rows"], 1)
        self.assertEqual(report["source_pubids_spanning_candidate_partitions"], 1)
        self.assertEqual(report["parent_rows_affected_by_cross_split_source_pubid"], 2)
        self.assertEqual(report["official_pubmedqa_500_test_overlap_parent_rows_by_candidate_split"],
                         {"train": 1, "dev": 0, "test": 1})
        self.assertEqual(report["pubmedqa_context_exact_agreement_rows"], 3)
        self.assertEqual(report["pubmedqa_context_not_exact_rows"], 1)
        self.assertFalse(report["complete_article_provenance_certified"])
        self.assertNotIn("invented red", json.dumps(report))
        self.assertNotIn("fictional context one", json.dumps(report))
        self.assertNotIn('"pubid"', json.dumps(report))

    def test_ambiguous_pubid_and_cross_configuration_fallback(self):
        med = {
            "pqa_labeled": [row("cross config source question"), row("ambiguous question")],
            "pqa_artificial": [row("source unrelated artificial question")],
        }
        pqa = {
            "pqa_labeled": [source(801, "ambiguous question"),
                            source(802, "Ambiguous question!")],
            "pqa_artificial": [source(901, "cross config source question")],
        }
        assignment = {
            ("pqa_labeled", 0): "train",
            ("pqa_labeled", 1): "dev",
            ("pqa_artificial", 0): "test",
        }
        d = audit_lineage(med, assignment, pubmed_index(pqa), official_test())
        validate_report(d)
        self.assertEqual(d["matched_cross_configuration_fallback"], 1)
        self.assertEqual(d["ambiguous_multiple_pubids_parent_rows"], 1)
        self.assertEqual(d["unmatched_parent_rows"], 1)

    def test_exact_text_hash_across_partitions_detected(self):
        med = {
            "pqa_labeled": [row("identical fake question", "same evidence text")],
            "pqa_artificial": [row("IDENTICAL fake QUESTION!", "same evidence text")],
        }
        pqa = {
            "pqa_labeled": [source(802, "identical fake question")],
            "pqa_artificial": [source(802, "identical fake question")],
        }
        assignments = {("pqa_labeled", 0): "train", ("pqa_artificial", 0): "test"}
        report = audit_lineage(med, assignments, pubmed_index(pqa), official_test())
        self.assertEqual(report["exact_question_fingerprint_groups_spanning_splits"], 1)
        self.assertEqual(report["exact_knowledge_fingerprint_groups_spanning_splits"], 1)
        self.assertEqual(report["source_pubids_spanning_candidate_partitions"], 1)

    def test_near_screen_catches_question_paraphrase_and_short(self):
        med = {
            "pqa_labeled": [
                row("a fictional orange tree grows near a blue river"),
                row("tiny"),
            ],
            "pqa_artificial": [
                row("A fictional orange tree grows near a blue river!"),
                row("tiny?"),
            ],
        }
        assignment = {
            ("pqa_labeled", 0): "train",
            ("pqa_labeled", 1): "dev",
            ("pqa_artificial", 0): "test",
            ("pqa_artificial", 1): "test",
        }
        report = advanced_near_screen(med, assignment)
        self.assertGreaterEqual(report["candidate_pairs_compared"], 2)
        self.assertGreaterEqual(report["high_overlap_unique_pairs"], 2)
        self.assertEqual(report["short_questions_in_scope"], 2)
        self.assertFalse(report["complete_pairwise_search"])
        self.assertFalse(report["semantic_paraphrase_search_performed"])
        self.assertNotIn("fictional orange", json.dumps(report))

    def test_invalid_source_and_fabricated_approval_rejected(self):
        with self.assertRaises(LineageAuditError):
            pubmed_index({"pqa_labeled": [source(True, "invalid")],
                          "pqa_artificial": [source(12, "fine")]})
        with self.assertRaises(LineageAuditError):
            pubmed_index({"pqa_labeled": [], "pqa_artificial": []})
        med = {"pqa_labeled": [row("fake title")],
               "pqa_artificial": [row("separate title")]}
        pqa = {"pqa_labeled": [source(900, "fake title")],
               "pqa_artificial": [source(800, "separate title")]}
        with self.assertRaises(LineageAuditError):
            audit_lineage(med, {("pqa_labeled", 0): "train"}, pubmed_index(pqa), {"1"})
        report = audit_lineage(med, {("pqa_labeled", 0): "train",
                                     ("pqa_artificial", 0): "test"},
                               pubmed_index(pqa), official_test())
        altered = copy.deepcopy(report)
        altered["final_leakage_safe_splits_approved"] = True
        with self.assertRaises(LineageAuditError):
            validate_report(altered)
        altered = copy.deepcopy(report)
        altered["source_pubmedqa_pubids_exported"] = True
        with self.assertRaises(LineageAuditError):
            validate_report(altered)

    def test_official_source_provenance_and_no_model_training(self):
        self.assertEqual(driver.PUBMEDQA_REVISION, "323752ad243ce751ba75c9222d34a3130d080030")
        self.assertEqual(driver.PUBMEDQA_TEST_GIT_BLOB,
                         "825bf8f262c3cca0e3bd1c6f38eecf5e581b0659")
        self.assertEqual(driver.PUBMEDQA_SHA256["pqa_artificial"][3], 211269)
        self.assertEqual(driver.PUBMEDQA_SHA256["pqa_labeled"][3], 1000)
        self.assertEqual(driver.SOURCE_SHA256["pqa_labeled"][3], 1000)
        self.assertEqual(driver.SOURCE_SHA256["pqa_artificial"][3], 9000)
        self.assertFalse(any(k in driver.OUTPUT_NAME for k in ("pmid", "raw_question")))
        body = (ROOT / "scripts/e1_pubmedqa_lineage_kaggle.py").read_text()
        self.assertIn("E1_LINEAGE_FAILED_STAGE=", body)
        self.assertIn("E1_LINEAGE_FAILED_CODE=", body)
        self.assertIn("E1_LINEAGE_FINAL_E1_STATUS=BLOCKED_PENDING_INDEPENDENT_APPROVAL", body)
        self.assertIn('"source_question_answer_pubids_exported": False', body)
        self.assertIn('"leakage_free_final_splits_approved": False', body)
        self.assertNotIn("traceback.print_exc", body)

if __name__ == "__main__":
    unittest.main()
