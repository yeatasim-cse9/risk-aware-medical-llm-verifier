"""All inputs are invented; no real MedHallu rows, no Kaggle or network."""
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
from medhallu_private_rows import (
    KERNEL, DATA_REV, SOURCE_MODULE, REPORT_NAME,
    RUNNER_SUFFIX, _hash_json, build, verify_aggregate,
)
from research_data.medhallu_native_audit import audit_native_rows

COMMIT = "a" * 40


def record(question):
    return {
        "Question": question, "Knowledge": ["made-up nonmedical source"],
        "Ground Truth": "safe example A",
        "Difficulty Level": "dummy", "Hallucinated Answer": "safe example B",
        "Category of Hallucination": "dummy_category",
    }


def aggregate_fixture():
    summary = audit_native_rows({
        "pqa_labeled": [record("made-up question one")],
        "pqa_artificial": [record("made-up question two")],
    })
    # Simulator only: the real Kaggle verifier independently requires 1000+9000
    # and this test creates count-only summary without biomedical samples.
    summary["config_rows"] = {"pqa_labeled": 1000, "pqa_artificial": 9000}
    module_sha = hashlib.sha256(SOURCE_MODULE.read_bytes()).hexdigest()
    envelope = {
        "schema_version": 1, "audit_kind": "private_kaggle_medhallu_native_rows",
        "source_git_sha": COMMIT, "audit_module_sha256": module_sha,
        "dataset_id": "UTAustin-AIHealth/MedHallu",
        "dataset_revision_sha": DATA_REV,
        "source_files": {
            "pqa_artificial": {
                "relative_file": "pqa_artificial/train-00000-of-00001.parquet",
                "size_bytes": 12999, "sha256": "b"*64,
            },
            "pqa_labeled": {
                "relative_file": "pqa_labeled/train-00000-of-00001.parquet",
                "size_bytes": 2000, "sha256": "c"*64,
            },
        },
        "summary": summary,
        "dataset_rows_exported": False,
        "scientific_experiments_executed": False,
    }
    envelope["integrity_sha256"] = _hash_json(envelope)
    return envelope


class PrivateMedhalluNativeAuditTests(unittest.TestCase):
    def test_private_cpu_package_has_pinned_public_source_and_no_embedded_rows(self):
        with tempfile.TemporaryDirectory() as t:
            dest = Path(t) / "new"
            m = build(dest, COMMIT)
            self.assertEqual(m["id"], KERNEL)
            self.assertIs(m["is_private"], True)
            self.assertIs(m["enable_gpu"], False)
            self.assertIs(m["enable_tpu"], False)
            self.assertIs(m["enable_internet"], True)
            for k in ("dataset_sources", "model_sources", "kernel_sources", "competition_sources"):
                self.assertEqual(m[k], [])
            runner = (dest / "runner.py").read_text()
            compile(runner, "kaggle-runner", "exec")
            self.assertIn(DATA_REV, runner)
            self.assertIn(COMMIT, runner)
            self.assertIn("TemporaryDirectory(dir=\"/kaggle/temp\"", runner)
            self.assertIn("dataset_rows_exported", runner)
            self.assertNotIn("MedVAL-Bench", runner)
            with self.assertRaises(ValueError):
                build(dest, COMMIT)
            with self.assertRaises(ValueError):
                build(Path(t) / "other", "main")

    def test_verify_aggregate_accepts_known_good_synthetic_envelope(self):
        with tempfile.TemporaryDirectory() as t:
            f = Path(t) / REPORT_NAME
            f.write_text(json.dumps(aggregate_fixture()))
            data = verify_aggregate(Path(t), COMMIT)
            self.assertFalse(data["dataset_rows_exported"])
            self.assertFalse(data["scientific_experiments_executed"])

    def test_tampering_and_unexpected_source_file_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            f = Path(t) / REPORT_NAME
            for mutation in ("wrong_sha", "wrong_commit", "added_text",
                             "wrong_dataset_sha", "untrusted_file", "made_up_approval"):
                data = aggregate_fixture()
                if mutation == "wrong_sha":
                    data["integrity_sha256"] = "0"*64
                elif mutation == "wrong_commit":
                    data["source_git_sha"] = "0"*40
                elif mutation == "added_text":
                    data["source_raw_question"] = "private text should never pass"
                elif mutation == "wrong_dataset_sha":
                    data["dataset_revision_sha"] = "0"*40
                elif mutation == "untrusted_file":
                    data["source_files"]["pqa_labeled"]["relative_file"] = "../private.parquet"
                else:
                    data["summary"]["native_label_mapping_approved"] = True
                data["integrity_sha256"] = _hash_json(
                    {k: v for k, v in data.items() if k != "integrity_sha256"}
                ) if mutation != "wrong_sha" else "0"*64
                f.write_text(json.dumps(data))
                with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                    verify_aggregate(Path(t), COMMIT)

    def test_source_code_only_and_safe_dependency_constraints(self):
        self.assertIn('allow_patterns=["pqa_labeled/*.parquet", "pqa_artificial/*.parquet"]', RUNNER_SUFFIX)
        self.assertIn('"huggingface_hub==0.36.0"', RUNNER_SUFFIX)
        self.assertIn('"pyarrow==21.0.0"', RUNNER_SUFFIX)
        self.assertIn('raise NativeAuditError("KAGGLE_PRIVATE_ENVIRONMENT_REQUIRED")', RUNNER_SUFFIX)
        self.assertIn('stderr=subprocess.DEVNULL', RUNNER_SUFFIX)
        self.assertNotIn("kaggle kernels push", RUNNER_SUFFIX)


if __name__ == "__main__":
    unittest.main()
