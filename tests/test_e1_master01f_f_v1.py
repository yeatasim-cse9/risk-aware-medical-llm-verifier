"""Synthetic-only hardening regression; actual D subprocess negative probe."""
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import e1_master01f_f_preflight_v1 as f


class Hardening(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.scripts = self.root / "scripts"
        self.scripts.mkdir()
        self.module = self.scripts / "a.py"
        self.module.write_text("pass")
        self.pins = {"scripts/a.py": "a" * 64}

    def test_reject_cache_extension_and_extra_file(self):
        f.strict_inventory(self.root, self.pins)
        for name in ("__pycache__", "a.so", "a.pyc"):
            p = self.scripts / name
            p.mkdir() if name == "__pycache__" else p.write_bytes(b"x")
            with self.assertRaisesRegex(ValueError, "IMPORT_INVENTORY"):
                f.strict_inventory(self.root, self.pins)
            p.rmdir() if p.is_dir() else p.unlink()

    def test_reject_symlink_module(self):
        self.module.unlink()
        self.module.symlink_to(self.root / "missing")
        with self.assertRaisesRegex(ValueError, "IMPORT_FILE_POLICY"):
            f.strict_inventory(self.root, self.pins)

    def test_external_manifest_bytes_required(self):
        raw = json.dumps(self.pins).encode()
        with patch.object(f, "CODE_MANIFEST_PIN", hashlib.sha256(raw).hexdigest()), patch.object(f, "MODULES", 1):
            self.assertEqual(f.code_pins(raw), self.pins)
            with self.assertRaisesRegex(ValueError, "CODE_MANIFEST_PIN"):
                f.code_pins(raw + b" ")

    def test_duplicate_manifest_key(self):
        with self.assertRaisesRegex(ValueError, "DUPLICATE_CODE_PIN"):
            f.unique([("a", 1), ("a", 2)])

    def test_child_environment_and_exact_dispatch(self):
        with patch.dict(os.environ, {"MASTER01FD_AUTHORIZED_ONE_GPU_PILOT": "YES",
                                     "PYTHONPATH": "/untrusted", "HF_TOKEN": "invented"}):
            with patch.object(f.subprocess, "run") as run:
                f.child(self.root, self.root, self.root, "a" * 64)
        args, kw = run.call_args
        self.assertIn("preflight", args[0])
        self.assertNotIn("run", args[0])
        self.assertEqual(kw["env"]["CUDA_VISIBLE_DEVICES"], "")
        for key in ("MASTER01FD_AUTHORIZED_ONE_GPU_PILOT", "PYTHONPATH", "HF_TOKEN"):
            self.assertNotIn(key, kw["env"])

    def test_real_immutable_child_rejects_missing_manifest(self):
        path = os.environ.get("MASTER01FF_SYNTHETIC_D_CHECKOUT")
        if not path:
            self.skipTest("Dedicated CI supplies immutable D worktree")
        root = Path(path)
        pins = f.old.hashes(root, f.D_COMMIT)
        self.assertEqual(len(pins), f.MODULES)
        raw = (json.dumps(pins, sort_keys=True, indent=2) + "\n").encode()
        self.assertEqual(f.code_pins(raw), pins)
        f.strict_inventory(root, pins)
        bundle = self.root / "empty_bundle"
        assets = self.root / "empty_assets"
        bundle.mkdir()
        assets.mkdir()
        result = f.child(root, bundle, assets, "a" * 64)
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stderr, b"")
        self.assertEqual(result.stdout.splitlines(), [
            b"MASTER01FD_STAGE=PREFLIGHT", b"MASTER01FD_FAILED_STAGE=PREFLIGHT"])
        with self.assertRaisesRegex(ValueError, "PRIVATE_PREFLIGHT_FAILED"):
            f.preflight(root, raw, bundle, assets, "a" * 64)


if __name__ == "__main__":
    unittest.main()
