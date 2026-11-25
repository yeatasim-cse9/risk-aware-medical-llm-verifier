"""Synthetic-safe manifest contract tests. Never load private MedHallu rows."""
from __future__ import annotations

import json
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "configs/e1-hybrid-audit-evidence-2026-11-25.json"


class HybridEvidenceManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = json.loads(PATH.read_text(encoding="utf-8"))

    def test_evidence_kind_and_checksum_are_only_user_supplied_provenance(self):
        d = self.data
        self.assertEqual(d["schema_version"], 1)
        self.assertEqual(
            d["report_integrity_sha256"],
            "b060a7a3f7bc8c449f36961bbdee10b12545feaf9f502b89edf0dc49369b7c3e",
        )
        self.assertIn("independent Kaggle output retrieval NOT performed",
                      d["verification_scope"])
        self.assertEqual(d["source_git_sha"],
                         "5e5b08d8f3818d4557051db9f5badadebd69ea78")
        self.assertEqual(d["pinned_dataset_revision_sha"],
                         "515060458a945c633debc6fd5baac7764416b724")
        for digest in (d["report_integrity_sha256"],
                       d["candidate_assignment_sha256"],
                       *d["pinned_source_parquet_sha256"].values(),
                       *d["matching_github_module_sha256"].values()):
            self.assertRegex(digest, r"\A[0-9a-f]{64}\Z")

    def test_provenance_source_digests_preserved(self):
        d = self.data
        p = json.loads((ROOT / "configs/medhallu-e1-prior-aggregate-evidence.json").read_text())
        self.assertEqual(d["pinned_dataset_revision_sha"], p["dataset_revision_sha"])
        self.assertEqual(d["pinned_dataset_id"], p["dataset_id"])
        for config, digest in d["pinned_source_parquet_sha256"].items():
            self.assertEqual(digest, p["source_files"][config]["sha256"])

    def test_counts_reconcile_and_label_frequencies_are_only_counts(self):
        x = self.data["counts"]
        self.assertEqual(sum(x["source_configs"].values()), x["original_parent_records"])
        self.assertEqual(x["original_parent_records"],
                         x["eligible_parent_records"] + x["quarantined_parent_records"])
        self.assertEqual(x["quarantined_parent_records"],
                         x["quarantined_repeated_question_rows"] +
                         x["quarantined_same_answer_rows"])
        self.assertEqual(sum(x["parent_splits"].values()), x["eligible_parent_records"])
        self.assertEqual(sum(x["answer_examples_splits"].values()), x["answer_pair_examples"])
        for split in ("train", "dev", "test"):
            self.assertEqual(x["answer_examples_splits"][split], 2*x["parent_splits"][split])
        self.assertEqual(x["answer_pair_examples"], 2*x["eligible_parent_records"])
        for config in x["source_configs"]:
            self.assertEqual(sum(x["category_frequencies_without_names"][config]),
                             x["source_configs"][config])
            self.assertEqual(sum(x["difficulty_frequencies_without_names"][config]),
                             x["source_configs"][config])
            self.assertEqual(len(x["category_frequencies_without_names"][config]),
                             x["category_unique_counts"][config])
            self.assertEqual(len(x["difficulty_frequencies_without_names"][config]),
                             x["difficulty_unique_counts"][config])
        self.assertEqual(x["cross_split_near_duplicate_candidate_pairs_compared"], 1223)
        self.assertEqual(x["cross_split_high_trigram_overlap_flags"], 0)
        self.assertFalse(x["candidate_cap_reached"])

    def test_all_scientific_approval_and_data_export_flags_remain_false(self):
        flags = self.data["scientific_claims"]
        self.assertEqual(set(flags), {
            "pubmedqa_article_lineage_verified",
            "full_semantic_near_duplicate_search_completed",
            "dataset_roles_scientifically_adjudicated",
            "final_splits_approved",
            "scientific_model_experiment_executed",
            "medical_text_or_source_ids_exported",
        })
        self.assertTrue(all(value is False for value in flags.values()))
        self.assertGreaterEqual(len(self.data["next_required_gates"]), 4)


if __name__ == "__main__":
    unittest.main()
