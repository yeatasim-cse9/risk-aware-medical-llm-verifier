"""Offline tests: no GPU, no Kaggle submission, no model downloads."""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from build_tiny_gpu_inference import build, KERNEL
from tiny_gpu_inference_runner import (
    PILOT_ID, canonical, seal, input_tokens, LABELS, SEED, WEIGHTS_FILE,
    MANIFEST_FILE,
)
from verify_tiny_gpu_inference import verify


SHA = "a"*40
WEIGHTS_FIXTURE = bytes(range(256))


def simulated_manifest(weights=WEIGHTS_FIXTURE):
    logits = [
        [0.1, 1.0, -0.4],
        [1.5, 0.0, 0.1],
        [-0.3, -0.8, 2.0],
        [0.0, 0.2, -0.1],
    ]
    probs = []
    for row in logits:
        shifted = [math.exp(v-max(row)) for v in row]
        probs.append([round(v/sum(shifted), 8) for v in shifted])
    return seal({
        "schema_version": 1,
        "pilot_id": PILOT_ID,
        "source_commit": SHA,
        "status": "completed",
        "model": "untrained_tiny_transformer_one_encoder_layer",
        "model_pretrained": False,
        "medical_experiment": False,
        "device": "cuda:0",
        "gpu_count": 1,
        "gpu_name": "SYNTHETIC_GPU_FIXTURE_NOT_HARDWARE_PROOF",
        "torch_version": "test-only",
        "torch_cuda_version": "test-only",
        "parameter_count": 12000,
        "seed": SEED,
        "num_samples": 4,
        "sequence_length": 8,
        "labels": list(LABELS),
        "input_sha256": hashlib.sha256(canonical(input_tokens())).hexdigest(),
        "logits": logits,
        "logits_sha256": hashlib.sha256(canonical(logits)).hexdigest(),
        "probabilities": probs,
        "predictions": [1, 0, 2, 1],
        "weights_file": WEIGHTS_FILE,
        "weights_size_bytes": len(weights),
        "weights_sha256": hashlib.sha256(weights).hexdigest(),
        "restore_max_abs_error": 0.0,
        "completed_at_utc": datetime(2026, 11, 25, tzinfo=timezone.utc).isoformat(),
    })


def write_fixture(directory, manifest=None, weights=WEIGHTS_FIXTURE):
    target = Path(directory)/"checkpoints"
    target.mkdir(parents=True, exist_ok=True)
    (target/WEIGHTS_FILE).write_bytes(weights)
    (target/MANIFEST_FILE).write_text(
        json.dumps(manifest or simulated_manifest(weights)), encoding="utf-8"
    )


class TinyGpuInferenceTests(unittest.TestCase):
    def test_private_gpu_build_offline_and_free_of_data_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            info = build(SHA, p)
            self.assertEqual(info["id"], KERNEL)
            self.assertEqual(info["id"], "yeatasimbillah/medical-llm-gpu-inference-pilot")
            self.assertIs(info["is_private"], True)
            self.assertIs(info["enable_gpu"], True)
            self.assertIs(info["enable_tpu"], False)
            self.assertIs(info["enable_internet"], False)
            for key in ("dataset_sources", "model_sources", "kernel_sources", "competition_sources"):
                self.assertEqual(info[key], [])
            code = (p/"runner.py").read_text(encoding="utf-8")
            self.assertIn(SHA, code)
            self.assertNotIn("__SOURCE_COMMIT__", code)
            subprocess.run(
                [sys.executable, "-m", "py_compile", str(p/"runner.py")],
                check=True, capture_output=True
            )
            self.assertEqual(json.loads((p/"kernel-metadata.json").read_text()), info)

    def test_unpinned_version_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            for bad in ("main", "A"*40, "a"*39, "z"*40, ""):
                with self.subTest(bad=bad):
                    with self.assertRaises(ValueError):
                        build(bad, Path(tmp))

    def test_valid_synthetic_fixture(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_fixture(tmp)
            state = verify(Path(tmp), SHA)
            self.assertEqual(state["predictions"], [1,0,2,1])
            self.assertFalse(state["model_pretrained"])

    def test_wrong_git_commit_or_missing_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                verify(Path(tmp), SHA)
            write_fixture(tmp)
            with self.assertRaises(ValueError):
                verify(Path(tmp), "b"*40)
            (Path(tmp)/"checkpoints"/WEIGHTS_FILE).unlink()
            with self.assertRaises(ValueError):
                verify(Path(tmp), SHA)

    def test_corrupted_weights_or_manifest_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_fixture(tmp)
            (Path(tmp)/"checkpoints"/WEIGHTS_FILE).write_bytes(b"x"*256)
            with self.assertRaisesRegex(ValueError, "SHA256"):
                verify(Path(tmp), SHA)
            (Path(tmp)/"checkpoints"/WEIGHTS_FILE).write_bytes(WEIGHTS_FIXTURE)
            p=Path(tmp)/"checkpoints"/MANIFEST_FILE
            v=json.loads(p.read_text())
            v["status"]="failed"
            p.write_text(json.dumps(v),encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "checksum"):
                verify(Path(tmp), SHA)

    def test_resealed_cpu_or_pretrained_claim_rejected(self):
        for name,value in (("device","cpu"),("model_pretrained",True),
                           ("medical_experiment",True),("gpu_count",0),
                           ("restore_max_abs_error",1.0)):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                payload=simulated_manifest()
                body={k:v for k,v in payload.items() if k!="integrity_sha256"}
                body[name]=value
                write_fixture(tmp, seal(body))
                with self.assertRaises(ValueError):
                    verify(Path(tmp), SHA)

    def test_invalid_softmax_or_predictions_rejected_even_with_new_checksum(self):
        for name,new in (("probabilities", [[0.0,1.0,0.0]]*4),
                         ("predictions", [0,0,0,0]),
                         ("logits_sha256","f"*64)):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                payload=simulated_manifest()
                body={k:v for k,v in payload.items() if k!="integrity_sha256"}
                body[name]=new
                write_fixture(tmp, seal(body))
                with self.assertRaises(ValueError):
                    verify(Path(tmp), SHA)

    def test_no_duplicate_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_fixture(tmp)
            second=Path(tmp)/"other"
            second.mkdir()
            (second/MANIFEST_FILE).write_text("{}",encoding="utf-8")
            with self.assertRaises(ValueError):
                verify(Path(tmp),SHA)


if __name__ == "__main__":
    unittest.main()
