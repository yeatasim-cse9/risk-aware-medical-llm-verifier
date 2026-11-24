"""GPU smoke tests use synthetic fixtures and never allocate a GPU."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from build_gpu_smoke import build
from verify_gpu_smoke import verify
from gpu_smoke_runner import seal

SHA = "a" * 40


def fake_checkpoint():
    return seal({
        "schema_version": 1,
        "test_id": "medical-llm-private-gpu-smoke-v1",
        "source_commit": SHA,
        "status": "completed",
        "device": "cuda:0",
        "gpu_available": True,
        "gpu_count": 1,
        "gpu_name": "SYNTHETIC_TEST_GPU",
        "torch_version": "2.test",
        "torch_cuda_version": "12.test",
        "matrix_size": 256,
        "expected_sum": 256**3,
        "observed_sum": 256**3,
        "max_abs_error": 0.0,
        "completed_at_utc": "2026-11-25T00:00:00+00:00",
    })


class GpuSmokeTests(unittest.TestCase):
    def test_builder_private_gpu_offline_no_data_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            dest = Path(directory)
            data = build(SHA, dest)
            self.assertEqual(data["id"], "yeatasimbillah/medical-llm-gpu-smoke")
            self.assertIs(data["is_private"], True)
            self.assertIs(data["enable_gpu"], True)
            self.assertIs(data["enable_internet"], False)
            self.assertIs(data["enable_tpu"], False)
            for key in ("kernel_sources", "dataset_sources", "model_sources", "competition_sources"):
                self.assertEqual(data[key], [])
            runner = dest / "runner.py"
            self.assertIn(SHA, runner.read_text(encoding="utf-8"))
            self.assertNotIn("__SOURCE_COMMIT__", runner.read_text(encoding="utf-8"))
            subprocess.run(
                [sys.executable, "-m", "py_compile", str(runner)],
                check=True, capture_output=True
            )

    def test_build_rejects_unpinned_versions(self):
        with tempfile.TemporaryDirectory() as directory:
            for source in ("main", "a"*39, "A"*40, "z"*40):
                with self.subTest(source=source), self.assertRaises(ValueError):
                    build(source, Path(directory))

    def test_verified_gpu_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "gpu-smoke-result.json"
            p.write_text(json.dumps(fake_checkpoint()), encoding="utf-8")
            self.assertEqual(verify(Path(directory), SHA)["gpu_available"], True)

    def test_cpu_fallback_fails_even_with_recomputed_sha(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "gpu-smoke-result.json"
            fake = fake_checkpoint()
            data = {k:v for k,v in fake.items() if k != "integrity_sha256"}
            data["gpu_available"] = False
            p.write_text(json.dumps(seal(data)), encoding="utf-8")
            with self.assertRaises(ValueError):
                verify(Path(directory), SHA)

    def test_checksum_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "gpu-smoke-result.json"
            p.write_text(json.dumps({**fake_checkpoint(), "observed_sum": 0}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "integrity"):
                verify(Path(directory), SHA)

    def test_wrong_source_and_missing_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                verify(Path(directory), SHA)
            p = Path(directory) / "gpu-smoke-result.json"
            p.write_text(json.dumps(fake_checkpoint()), encoding="utf-8")
            with self.assertRaises(ValueError):
                verify(Path(directory), "b"*40)


if __name__ == "__main__":
    unittest.main()
