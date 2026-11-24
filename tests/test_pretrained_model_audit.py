"""Pure, no-network model-card audit and safety-gate tests."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from audit_pretrained_models import validate, AuditError, AUDIT_PATH, main


class ModelAuditTests(unittest.TestCase):
    def setUp(self):
        self.original = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))

    def test_curated_models_and_no_execution(self):
        models = validate(self.original)
        self.assertEqual(len(models), 3)
        self.assertTrue(all(m["model_revision_sha"] is None for m in models))
        self.assertTrue(all(m["downloads_executed"] is False for m in models))
        self.assertTrue(all(m["execution_approved"] is False for m in models))

    def test_nli_label_order_locked(self):
        c = copy.deepcopy(self.original)
        nli = next(m for m in c["models"] if m["source_task"] == "sentence_pair_nli")
        self.assertEqual(nli["nli_id2label"], [
            "contradiction", "entailment", "neutral"
        ])
        nli["nli_id2label"] = ["entailment", "neutral", "contradiction"]
        with self.assertRaises(AuditError):
            validate(c)

    def test_model_licenses_cannot_be_silently_changed(self):
        c = copy.deepcopy(self.original)
        c["models"][2]["declared_license"] = "apache-2.0"
        with self.assertRaises(AuditError):
            validate(c)

    def test_reject_unreviewed_file_revision_or_fake_gpu_measurements(self):
        for name, value in (
            ("model_revision_sha", "a"*40),
            ("model_file_sha256", "b"*64),
            ("tokenizer_sha256", "c"*64),
            ("package_size_bytes", 123),
            ("measured_gpu_peak_memory_bytes", 800000),
        ):
            with self.subTest(name=name):
                c=copy.deepcopy(self.original)
                c["models"][0][name]=value
                with self.assertRaises(AuditError):
                    validate(c)

    def test_reject_unapproved_model_or_unsafe_license_link(self):
        c=copy.deepcopy(self.original)
        c["models"][0]["model_id"]="random-account/medical-model"
        with self.assertRaises(AuditError):
            validate(c)
        c=copy.deepcopy(self.original)
        c["models"][0]["license_evidence_url"]="http://bad.example/license"
        with self.assertRaises(AuditError):
            validate(c)

    def test_reject_unverified_approval_or_data_transfer(self):
        for field in (
            "execution_approved", "downloads_executed",
            "weight_redistribution_reviewed", "downstream_data_rights_reviewed",
        ):
            with self.subTest(field=field):
                c=copy.deepcopy(self.original)
                c["models"][0][field]=True
                with self.assertRaises(AuditError):
                    validate(c)

    def test_restricted_medval_and_kaggle_privacy_gates(self):
        for field in (
            "restricted_medval_allowed_on_kaggle_without_permission",
            "restricted_medval_allowed_on_public_github",
            "model_weight_upload_allowed_in_this_stage",
            "kaggle_gpu_submission_allowed_in_this_stage",
        ):
            c=copy.deepcopy(self.original)
            c["policy"][field]=True
            with self.subTest(field=field), self.assertRaises(AuditError):
                validate(c)

    def test_script_succeeds_without_network_or_credentials(self):
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/"audit.json"
            source.write_text(json.dumps(self.original),encoding="utf-8")
            original_argv=sys.argv
            try:
                sys.argv=["audit_pretrained_models.py","--catalog",str(source)]
                self.assertEqual(main(),0)
            finally:
                sys.argv=original_argv


if __name__ == "__main__":
    unittest.main()
