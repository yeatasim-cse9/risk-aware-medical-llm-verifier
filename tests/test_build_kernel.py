import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from build_kernel import build


class KernelBuildTests(unittest.TestCase):
    def test_build_cpu_private_and_pinned_commit(self):
        with tempfile.TemporaryDirectory() as tmp:
            commit = "a" * 40
            path = Path(tmp)
            metadata = build(commit, "b", True, path)
            self.assertFalse(metadata["enable_gpu"])
            self.assertFalse(metadata["enable_internet"])
            self.assertTrue(metadata["is_private"])
            self.assertEqual(metadata["kernel_sources"], ["yeatasimbillah/medical-llm-cpu-auto-a"])
            self.assertIn(commit, (path / "runner.py").read_text())
            json.loads((path / "kernel-metadata.json").read_text())
            subprocess.run([sys.executable, "-m", "py_compile", str(path / "runner.py")], check=True)

    def test_reject_unpinned_version(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                build("main", "a", False, Path(tmp))


if __name__ == "__main__":
    unittest.main()
