"""Safety assertions for immutable first-model provenance snapshot. No network."""
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
PIN = ROOT / "configs" / "model-pins" / "all-minilm-l6-v2.json"


class PinnedMiniLmTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = json.loads(PIN.read_text(encoding="utf-8"))

    def test_exact_immutable_hub_commit_and_weights_metadata(self):
        d = self.data
        self.assertEqual(d["model_id"], "sentence-transformers/all-MiniLM-L6-v2")
        self.assertEqual(d["model_repo_revision_sha"], "1110a243fdf4706b3f48f1d95db1a4f5529b4d41")
        self.assertEqual(d["hub_model_safetensors_lfs_sha256_metadata"], "53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db")
        self.assertEqual(d["hub_model_safetensors_file_size_bytes"], 90868376)
        self.assertEqual(d["github_verification_run"], 45487590017)
        self.assertTrue(d["pinned_tree_url"].endswith(d["model_repo_revision_sha"]))

    def test_parameter_storage_math_is_not_a_gpu_vram_measurement(self):
        d = self.data
        self.assertEqual(d["fp32_theoretical_parameter_bytes"], d["hub_safetensors_parameter_count"] * 4)
        self.assertEqual(d["fp16_theoretical_parameter_bytes"], d["hub_safetensors_parameter_count"] * 2)
        self.assertIsNone(d["max_cuda_memory_allocated_bytes_measured"])
        self.assertFalse(d["kaggle_accelerator_fit_proven"])

    def test_no_unreviewed_model_or_restricted_data_transfer_approval(self):
        d = self.data
        for field in (
            "model_weights_downloaded",
            "license_notice_and_training_data_rights_fully_reviewed",
            "model_execution_authorized",
            "patient_or_medval_data_transfer_authorized",
        ):
            self.assertIs(d[field], False, field)
        self.assertIsNone(d["tokenizer_sha256_independently_checked"])
        self.assertIsNone(d["weight_sha256_independently_checked"])


if __name__ == "__main__":
    unittest.main()
