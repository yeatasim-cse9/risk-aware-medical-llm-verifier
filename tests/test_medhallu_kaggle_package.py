"""Offline safety/package tests with synthetic source SHA only."""
from __future__ import annotations
import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("build_medhallu_audit", ROOT / "scripts" / "build_medhallu_audit.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)

class KaggleMetadataPackageTests(unittest.TestCase):
    def test_private_cpu_metadata_only(self):
        with tempfile.TemporaryDirectory() as temp:
            d = Path(temp) / "kernel"
            record = builder.build("f" * 40, d)
            self.assertEqual({p.name for p in d.iterdir()},
                             {"runner.py", "kernel-metadata.json"})
            self.assertTrue(record["is_private"])
            self.assertFalse(record["enable_gpu"])
            self.assertTrue(record["enable_internet"])
            self.assertEqual(record["dataset_sources"], [])
            self.assertIn("f" * 40, (d / "runner.py").read_text(encoding="utf-8"))
            self.assertNotIn("load_dataset", (d / "runner.py").read_text(encoding="utf-8"))
            runner = (d / "runner.py").read_text(encoding="utf-8")
            self.assertNotIn("from manager", runner)
            self.assertIn("def fetch_public_metadata", runner)
            standalone_spec = importlib.util.spec_from_file_location("standalone_metadata_runner", d / "runner.py")
            standalone = importlib.util.module_from_spec(standalone_spec)
            standalone_spec.loader.exec_module(standalone)
            self.assertEqual(standalone.SOURCE_COMMIT, "f" * 40)
            self.assertTrue(callable(standalone.build_report))

    def test_stale_directory_and_bad_sha_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            d = Path(temp) / "kernel"
            with self.assertRaises(ValueError):
                builder.build("main", d)
            builder.build("f" * 40, d)
            with self.assertRaises(FileExistsError):
                builder.build("f" * 40, d)

if __name__ == "__main__":
    unittest.main()
