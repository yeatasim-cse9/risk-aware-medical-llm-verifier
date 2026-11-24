"""No-network, deterministic HF metadata validation tests."""
import copy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from pin_minilm_provenance import (
    ROOT, MODEL, FILES, ProvenanceError, inspect, validate_metadata, request_json,
)

SHA = "a"*40
FILE_BYTES = {
    "model.safetensors": 90_000_000,
    "config.json": 640,
    "tokenizer.json": 466_000,
    "tokenizer_config.json": 390,
    "vocab.txt": 231_000,
    "modules.json": 470,
    "1_Pooling/config.json": 190,
}


def fixture():
    info = {
        "id": MODEL, "sha": SHA, "private": False, "gated": False,
        "pipeline_tag": "sentence-similarity",
        "cardData": {"license": "apache-2.0"},
        "safetensors": {"total": 22_713_728},
    }
    tree = [
        {
            "type": "file",
            "path": name,
            "size": value,
            "lfs": {"oid": "f"*64} if name == "model.safetensors" else None,
        }
        for name, value in FILE_BYTES.items()
    ]
    return info, tree


class MiniLmProvenanceTests(unittest.TestCase):
    def test_valid_revision_and_conservative_storage_estimate(self):
        info, tree = fixture()
        data = validate_metadata(info, tree)
        self.assertEqual(data["revision_sha"], SHA)
        self.assertEqual(data["model_parameter_count_from_hub"], 22_713_728)
        self.assertEqual(data["theoretical_parameter_storage_bytes_fp32"], 90_854_912)
        self.assertEqual(data["theoretical_parameter_storage_bytes_fp16"], 45_427_456)
        self.assertEqual(data["required_files"]["model.safetensors"]["lfs_sha256_metadata"], "f"*64)
        self.assertIsNone(data["gpu_peak_vram_bytes_measured"])
        self.assertFalse(data["kaggle_gpu_fit_verified"])
        self.assertFalse(data["model_loading_approved"])

    def test_never_claim_license_fully_cleared_or_model_downloaded(self):
        data = validate_metadata(*fixture())
        self.assertFalse(data["license_full_text_audit_complete"])
        self.assertFalse(data["model_or_tokenizer_content_downloaded"])
        self.assertFalse(data["medical_evaluation_performed"])
        self.assertFalse(data["datasets_processed"])

    def test_changed_identity_license_and_gated_model_rejected(self):
        for field, val in (
            ("id", "somebody/other-model"), ("private", True),
            ("gated", True), ("pipeline_tag", "text-generation"),
            ("sha", "main"), ("sha", "x"*40),
        ):
            info, tree = fixture()
            info[field] = val
            with self.subTest(field=field, val=val), self.assertRaises(ProvenanceError):
                validate_metadata(info, tree)
        info, tree = fixture()
        info["cardData"]["license"] = "unknown"
        with self.assertRaises(ProvenanceError):
            validate_metadata(info, tree)

    def test_missing_required_files_and_malicious_file_size(self):
        info, tree = fixture()
        with self.assertRaises(ProvenanceError):
            validate_metadata(info, tree[:-1])
        for invalid in (None, 0, -1, True, 10**12):
            with self.subTest(value=invalid):
                info, tree = fixture()
                tree[0]["size"] = invalid
                with self.assertRaises(ProvenanceError):
                    validate_metadata(info, tree)

    def test_duplicate_files_and_corrupt_lfs(self):
        info, tree = fixture()
        with self.assertRaises(ProvenanceError):
            validate_metadata(info, tree + [copy.deepcopy(tree[0])])
        info, tree = fixture()
        tree[0]["lfs"] = {"oid": ["not-a-string"]}
        # Non-hex non-string LFS field cannot be represented as a checked digest.
        data=validate_metadata(info, tree)
        self.assertIsNone(data["required_files"]["model.safetensors"]["lfs_sha256_metadata"])
        self.assertFalse(data["model_loading_approved"])

    def test_query_uses_immutable_revision_in_file_tree(self):
        info, tree = fixture()
        seen = []
        def fake_get(url):
            seen.append(url)
            return info if len(seen)==1 else tree
        with patch("pin_minilm_provenance.request_json", side_effect=fake_get):
            self.assertEqual(inspect()["revision_sha"], SHA)
        self.assertEqual(seen, [
            ROOT + "/revision/main",
            ROOT + "/tree/" + SHA + "?recursive=true",
        ])

    def test_no_unapproved_urls(self):
        for bad in (
            "https://somewhere.example/model",
            ROOT + "/revision/other",
            ROOT + "/tree/../../secrets?recursive=true",
        ):
            with self.subTest(bad=bad), self.assertRaises(ProvenanceError):
                request_json(bad)

    def test_missing_safetensors_parameter_count_is_failure(self):
        info, tree = fixture()
        info["safetensors"] = {}
        with self.assertRaises(ProvenanceError):
            validate_metadata(info, tree)


if __name__ == "__main__":
    unittest.main()
