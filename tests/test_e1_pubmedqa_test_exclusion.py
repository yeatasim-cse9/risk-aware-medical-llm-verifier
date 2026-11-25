"""Only synthetic nonmedical data. Never access real PubMedQA/MedHallu."""
from __future__ import annotations

import copy
from hashlib import sha256
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
sys.path.insert(0, str(ROOT/"scripts"))

from research_data.e1_advanced_lineage import pubmed_index
from research_data.e1_pubmedqa_test_exclusion import (
    candidate_remediation, verify_summary, RemediationError
)
import e1_pubmedqa_test_exclusion_kaggle as runner


def fake_question(config, i):
    return f"Invented colored nonmedical story about {config} number {i}"


def fake_row(config, i):
    return {
        "Question": fake_question(config, i),
        "Knowledge": [f"Fictional context {config} {i}"],
        "Ground Truth": "made-up correct answer",
        "Hallucinated Answer": "made-up incorrect answer",
        "Difficulty Level": "invented easy",
        "Category of Hallucination": "pretend category A" if i % 3 else "pretend category B",
    }


def fixture():
    med = {
        c: [fake_row(c, i) for i in range(20)]
        for c in ("pqa_labeled", "pqa_artificial")
    }
    # One repeated exact context from a distinct synthetic source, to
    # ensure grouping respects exact Knowledge too, not just PMID.
    med["pqa_labeled"][3]["Knowledge"] = ["Shared fictional document"]
    med["pqa_artificial"][3]["Knowledge"] = ["shared FICTIONAL document!!!"]
    pqa = {"pqa_labeled": [], "pqa_artificial": []}
    for c in pqa:
        for i in range(20):
            pubid = 1000 + i + (0 if c == "pqa_labeled" else 100)
            if c == "pqa_labeled" and i == 0:
                pubid = 1  # official holdout
            if c == "pqa_artificial" and i == 0:
                pubid = 2  # official holdout
            ctx = med[c][i]["Knowledge"][0]
            if c == "pqa_artificial" and i == 2:
                ctx = "a source document not matching MedHallu context"
            pqa[c].append({
                "pubid": pubid, "question": med[c][i]["Question"],
                "context": {"contexts": [ctx]},
            })
    pqa["pqa_artificial"].append({
        "pubid": 9000, "question": med["pqa_artificial"][1]["Question"],
        "context": {"contexts": [med["pqa_artificial"][1]["Knowledge"][0]]},
    }) # ambiguous
    old = {}
    for c in med:
        for i in range(20):
            old[(c, i)] = ("train", "dev", "test")[i % 3]
    return med, old, pubmed_index(pqa), {str(i) for i in range(1, 501)}


class RemediationTests(unittest.TestCase):
    def test_official_500_and_unresolved_provenance_quarantined(self):
        med, old, index, ids = fixture()
        summary, assignment = candidate_remediation(med, old, index, ids)
        self.assertEqual(summary["previous_candidate_parent_records"], 40)
        self.assertEqual(summary["excluded_parent_records_by_reason"], {
            "original_official_500_test_heldout": 2,
            "multiple_pubmedqa_pubids": 1,
            "no_pubmedqa_pubid": 0,
            "source_context_not_exact": 1,
        })
        self.assertEqual(summary["source_reviewed_candidate_parent_records"], 36)
        self.assertEqual(len(assignment), 36)
        for key in (("pqa_labeled", 0), ("pqa_artificial", 0),
                    ("pqa_artificial", 1), ("pqa_artificial", 2)):
            self.assertNotIn(key, assignment)
        self.assertEqual(assignment[("pqa_labeled", 3)], assignment[("pqa_artificial", 3)])
        self.assertEqual(summary["exact_pubid_groups_spanning_revised_splits"], 0)
        self.assertEqual(summary["exact_knowledge_groups_spanning_revised_splits"], 0)
        self.assertFalse(summary["original_official_500_test_removed_from_all_new_candidate_splits"])
        self.assertFalse(summary["final_train_dev_test_approved"])
        self.assertFalse(summary["scientific_e1_experiment_executed"])
        self.assertEqual(sum(summary["provisional_parent_split_counts"].values()), 36)
        self.assertEqual(sum(summary["provisional_answer_example_split_counts"].values()), 72)
        verify_summary(summary)
        exported = json.dumps(summary)
        for forbidden in ("Invented colored", "Shared fictional document",
                          "made-up correct", "pretend category"):
            self.assertNotIn(forbidden, exported)
        self.assertNotIn('"pubid"', exported)

    def test_deterministic_and_seed_specific_not_fake_ready(self):
        med, old, index, ids = fixture()
        a, aa = candidate_remediation(med, old, index, ids)
        b, bb = candidate_remediation(med, old, index, ids)
        self.assertEqual((a, aa), (b, bb))
        c, cc = candidate_remediation(med, old, index, ids, seed=20261126)
        self.assertNotEqual(a["private_mapping_sha256"], c["private_mapping_sha256"])
        self.assertFalse(c["complete_semantic_leakage_excluded"])

    def test_report_rejects_malicious_extra_raw_fields_or_approval(self):
        med, old, index, ids = fixture()
        good, _ = candidate_remediation(med, old, index, ids)
        for name, value in (
            ("patient_text", "sensitive"),
            ("final_train_dev_test_approved", True),
            ("scientific_e1_experiment_executed", True),
            ("source_reviewed_candidate_parent_records", 99),
            ("exact_pubid_groups_spanning_revised_splits", 1),
        ):
            bad = copy.deepcopy(good)
            bad[name] = value
            with self.subTest(name=name), self.assertRaises(RemediationError):
                verify_summary(bad)

    def test_no_auto_use_nonverified_original_test_set(self):
        med, old, index, ids = fixture()
        with self.assertRaises(RemediationError):
            candidate_remediation(med, old, index, set())
        with self.assertRaises(RemediationError):
            candidate_remediation(med, old, index, {str(i) for i in range(1,500)})
        with self.assertRaises(RemediationError):
            candidate_remediation(med, old, index, ids, seed=True)

    def test_manual_kaggle_runner_has_hard_blocker_and_no_training(self):
        body = (ROOT/"scripts/e1_pubmedqa_test_exclusion_kaggle.py").read_text()
        compile(body, "e1-remediation-runner", "exec")
        self.assertIn('["ambiguous_multiple_pubids_parent_rows"] != 23', body)
        self.assertIn('["original_official_500_test_heldout"] != 500', body)
        self.assertIn('["source_reviewed_candidate_parent_records"] != 9474', body)
        self.assertIn("E1_REMEDIATION_FINAL_E1_STATUS=BLOCKED_PENDING_SCIENTIFIC_APPROVAL", body)
        self.assertIn('"medical_text_pubids_private_assignments_exported": False', body)
        self.assertNotIn("print(exc)", body)
        self.assertNotIn("traceback.print_exc", body)


if __name__ == "__main__":
    unittest.main()
