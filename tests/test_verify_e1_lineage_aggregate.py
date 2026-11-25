"""Fabricated nonmedical aggregate verification tests, no PubMedQA download."""
from __future__ import annotations

import copy
from hashlib import sha256
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import e1_pubmedqa_lineage_kaggle as driver
from verify_e1_lineage_aggregate import REPORT_NAME, verify
from research_data.e1_advanced_lineage import (
    pubmed_index, audit_lineage, advanced_near_screen,
)

COMMIT = "a" * 40


def synthetic_row(question):
    return {
        "Question": question, "Knowledge": ["completely fictional context"],
        "Ground Truth": "fictional correct", "Hallucinated Answer": "fictional false",
        "Difficulty Level": "fake", "Category of Hallucination": "fake",
    }


def new_synthetic_report():
    med = {
        "pqa_labeled": [synthetic_row("invented green forest station")],
        "pqa_artificial": [synthetic_row("invented purple cloud station")],
    }
    index = pubmed_index({
        "pqa_labeled": [{"pubid": 501, "question": "invented green forest station",
                         "context": {"contexts": ["completely fictional context"]}}],
        "pqa_artificial": [{"pubid": 502, "question": "invented purple cloud station",
                            "context": {"contexts": ["completely fictional context"]}}],
    })
    assignment = {("pqa_labeled", 0): "train", ("pqa_artificial", 0): "test"}
    lineage = audit_lineage(med, assignment, index, {str(i) for i in range(1,501)})
    near = advanced_near_screen(med, assignment)
    mods = {
        **driver.EXPECTED_MODULE_SHA256,
        "e1_advanced_lineage": sha256(
            (ROOT/"src/research_data/e1_advanced_lineage.py").read_bytes()
        ).hexdigest(),
    }
    body = {
        "schema_version": 1, "audit_kind": "e1_pubmedqa_source_lineage_advanced_leakage_v1",
        "github_source_git_sha": COMMIT,
        "medhallu_revision": driver.MEDHALLU_REVISION,
        "pubmedqa_revision": driver.PUBMEDQA_REVISION,
        "pubmedqa_official_test_repo_revision": driver.PUBMEDQA_OFFICIAL_GIT_REVISION,
        "original_pubmedqa_500_test_sha256": "b"*64,
        "source_module_sha256": mods,
        "candidate_assignment_sha256": driver.CANDIDATE_ASSIGNMENT_SHA256,
        "medhallu_source_sha256": {k: x[2] for k,x in driver.SOURCE_SHA256.items()},
        "pubmedqa_source_sha256": {k: x[2] for k,x in driver.PUBMEDQA_SHA256.items()},
        "matched_pubmedqa_lineage_screen": lineage,
        "advanced_near_duplicate_screen": near,
        "pubmedqa_article_lineage_fully_certified": False,
        "leakage_free_final_splits_approved": False,
        "labels_scientifically_adjudicated": False,
        "e1_scientific_model_experiment_executed": False,
        "source_question_answer_pubids_exported": False,
    }
    body["integrity_sha256"] = sha256(driver.canonical(body)).hexdigest()
    return body


class VerifiedAggregateTests(unittest.TestCase):
    def test_synthetic_user_supplied_aggregate_consistency_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / REPORT_NAME
            report = new_synthetic_report()
            file.write_text(json.dumps(report), encoding="utf-8")
            out = verify(file, COMMIT, expected_rows=2)
            self.assertEqual(out["matched_pubmedqa_lineage_screen"]["eligible_parent_rows_checked"], 2)
            self.assertFalse(out["leakage_free_final_splits_approved"])
            self.assertNotIn("invented green forest", file.read_text())
            with self.assertRaises(ValueError):
                verify(file, COMMIT)

    def test_changed_or_injected_report_denied_even_if_digest_updated(self):
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / REPORT_NAME
            for case in ("private_id", "false_ready", "source_tamper", "code_tamper",
                         "near_false_semantic", "wrong_counts", "hash_corrupt"):
                report = new_synthetic_report()
                if case == "private_id":
                    report["raw_question_text"] = "private"
                elif case == "false_ready":
                    report["leakage_free_final_splits_approved"] = True
                elif case == "source_tamper":
                    report["pubmedqa_source_sha256"]["pqa_labeled"] = "0"*64
                elif case == "code_tamper":
                    report["source_module_sha256"]["e1_advanced_lineage"] = "0"*64
                elif case == "near_false_semantic":
                    report["advanced_near_duplicate_screen"]["semantic_paraphrase_search_performed"] = True
                elif case == "wrong_counts":
                    report["matched_pubmedqa_lineage_screen"]["matched_unique_pubid_parent_rows"] += 1
                elif case == "hash_corrupt":
                    report["integrity_sha256"] = "0"*64
                if case != "hash_corrupt":
                    report["integrity_sha256"] = sha256(driver.canonical({
                        k:v for k,v in report.items() if k != "integrity_sha256"
                    })).hexdigest()
                file.write_text(json.dumps(report), encoding="utf-8")
                with self.subTest(case=case), self.assertRaises((ValueError, Exception)):
                    verify(file, COMMIT, expected_rows=2)

    def test_expected_source_sha_is_mandatory(self):
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / REPORT_NAME
            file.write_text(json.dumps(new_synthetic_report()), encoding="utf-8")
            with self.assertRaises(ValueError):
                verify(file, "main", expected_rows=2)
            with self.assertRaises(ValueError):
                verify(file, "c"*40, expected_rows=2)

if __name__ == "__main__":
    unittest.main()
