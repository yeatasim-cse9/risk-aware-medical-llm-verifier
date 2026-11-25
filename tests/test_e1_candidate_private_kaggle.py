"""All fixtures are invented. No dataset rows, Kaggle jobs, or network calls."""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
from e1_candidate_private_kaggle import (
    DATASET_REV, KERNEL, REPORT, RUNTIME, NATIVE, CANDIDATE,
    canonical, expected_evidence, build, verify_aggregate,
)
from research_data.e1_candidate_splits import candidate_plan

SOURCE_COMMIT = "b" * 40


def fake_record(i: int, config: str) -> dict:
    return {
        "Question": f"FAKE NONMEDICAL question {config} {i}",
        "Knowledge": [f"FAKE artificial evidence {config} {i}"],
        "Ground Truth": "pretend true",
        "Hallucinated Answer": "pretend false",
        "Difficulty Level": "fake",
        "Category of Hallucination": "fake only",
    }


def synthetic_envelope():
    rows = {
        "pqa_labeled": [fake_record(i, "labeled") for i in range(30)],
        "pqa_artificial": [fake_record(i, "artificial") for i in range(30)],
    }
    summary, assignment = candidate_plan(rows)
    assert len(assignment) == 60
    evidence = expected_evidence()
    result = {
        "schema_version": 1,
        "audit_kind": "private_e1_candidate_split_design_only",
        "source_git_sha": SOURCE_COMMIT,
        "dataset_id": "UTAustin-AIHealth/MedHallu",
        "dataset_revision_sha": DATASET_REV,
        "prior_verified_aggregate_sha256": evidence["aggregate_report_sha256"],
        "module_sha256": {
            "medhallu_native_audit": hashlib.sha256(NATIVE.read_bytes()).hexdigest(),
            "e1_candidate_splits": hashlib.sha256(CANDIDATE.read_bytes()).hexdigest(),
        },
        "source_files": evidence["source_files"],
        "summary": summary,
        "full_lineage_verified": False,
        "final_splits_approved": False,
        "scientific_experiments_executed": False,
        "raw_source_data_exported": False,
    }
    result["integrity_sha256"] = hashlib.sha256(canonical(result)).hexdigest()
    return result


class PrivateCandidatePilotTests(unittest.TestCase):
    def test_evidence_matches_original_verified_aggregate_metadata(self):
        ev = expected_evidence()
        self.assertEqual(ev["dataset_revision_sha"], DATASET_REV)
        self.assertEqual(ev["artifact_id"], 15010069180)
        self.assertEqual(ev["source_files"]["pqa_labeled"]["rows"], 1000)
        self.assertEqual(ev["source_files"]["pqa_artificial"]["rows"], 9000)
        self.assertEqual(ev["observed_aggregate"]["unique_questions_total"], 9999)
        self.assertFalse(ev["scientific_experiments_executed"])

    def test_code_only_private_cpu_package_compiles_and_is_pinned(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "package"
            meta = build(target, SOURCE_COMMIT)
            self.assertEqual(meta["id"], KERNEL)
            self.assertIs(meta["is_private"], True)
            self.assertIs(meta["enable_gpu"], False)
            self.assertIs(meta["enable_tpu"], False)
            self.assertIs(meta["enable_internet"], True)
            for field in ("dataset_sources", "competition_sources", "kernel_sources", "model_sources"):
                self.assertEqual(meta[field], [])
            self.assertEqual({p.name for p in target.iterdir()}, {"kernel-metadata.json", "runner.py"})
            runner = (target / "runner.py").read_text()
            compile(runner, "pinned-candidate-split-runner", "exec")
            self.assertIn(DATASET_REV, runner)
            self.assertIn(SOURCE_COMMIT, runner)
            for value in expected_evidence()["source_files"].values():
                self.assertIn(value["sha256"], runner)
            self.assertIn("PINNED_DATASET_FILE_SHA256_MISMATCH", runner)
            self.assertIn("E1_CANDIDATE_FAILED_STAGE", runner)
            with self.assertRaises(ValueError):
                build(target, SOURCE_COMMIT)
            with self.assertRaises(ValueError):
                build(Path(directory) / "invalid", "main")

    def test_aggregate_report_verifies_without_raw_records(self):
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / REPORT
            body = synthetic_envelope()
            file.write_text(json.dumps(body))
            result = verify_aggregate(
                Path(directory), SOURCE_COMMIT,
                expected_rows={"pqa_labeled": 30, "pqa_artificial": 30},
            )
            self.assertEqual(result["summary"]["candidate_answer_pair_examples"], 120)
            self.assertFalse(result["summary"]["final_train_dev_test_approved"])
            self.assertNotIn("FAKE NONMEDICAL question", file.read_text())
            self.assertNotIn("pretend true", file.read_text())

    def test_malicious_fields_and_false_approval_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / REPORT
            for mutation in ("raw_question", "final_approved", "wrong_raw_sha", "bad_code",
                             "bad_integrity", "scientific_run", "unapproved_count"):
                body = synthetic_envelope()
                if mutation == "raw_question":
                    body["text"] = "raw private data blocked"
                elif mutation == "final_approved":
                    body["summary"]["final_train_dev_test_approved"] = True
                elif mutation == "wrong_raw_sha":
                    body["source_files"]["pqa_labeled"]["sha256"] = "0" * 64
                elif mutation == "bad_code":
                    body["module_sha256"]["e1_candidate_splits"] = "0" * 64
                elif mutation == "bad_integrity":
                    body["integrity_sha256"] = "0" * 64
                elif mutation == "scientific_run":
                    body["scientific_experiments_executed"] = True
                elif mutation == "unapproved_count":
                    body["summary"]["eligible_parent_rows"] += 1
                if mutation != "bad_integrity":
                    body["integrity_sha256"] = hashlib.sha256(canonical({
                        k: v for k, v in body.items() if k != "integrity_sha256"
                    })).hexdigest()
                file.write_text(json.dumps(body))
                with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                    verify_aggregate(Path(directory), SOURCE_COMMIT,
                                     expected_rows={"pqa_labeled": 30, "pqa_artificial": 30})
            file.write_text(json.dumps(synthetic_envelope()))
            with self.assertRaises(ValueError):
                verify_aggregate(Path(directory), "c"*40)

    def test_production_rejects_non_10k_simulated_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / REPORT).write_text(json.dumps(synthetic_envelope()))
            with self.assertRaises(ValueError):
                verify_aggregate(Path(directory), SOURCE_COMMIT)

    def test_only_safe_stage_codes_exported_and_no_automatic_approval(self):
        self.assertIn("E1_CANDIDATE_FAILED_CODE=", RUNTIME)
        self.assertIn("E1_CANDIDATE_FAILED_STAGE=", RUNTIME)
        self.assertIn('raise NativeAuditError("PINNED_DATASET_FILE_SHA256_MISMATCH")', RUNTIME)
        self.assertNotIn("print(exc)", RUNTIME)
        self.assertNotIn("traceback.print_exc", RUNTIME)
        self.assertIn("del rows, private_assignment", RUNTIME)
        self.assertIn('"final_splits_approved": False', RUNTIME)


if __name__ == "__main__":
    unittest.main()
