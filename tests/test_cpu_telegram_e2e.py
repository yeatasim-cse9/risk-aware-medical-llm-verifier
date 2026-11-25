"""Offline safety tests; never contacts Kaggle, Telegram or GitHub APIs."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from cpu_telegram_e2e import build, verify, RUNNER, SOURCE_DIGEST, OUTPUT_NAME, KERNEL
from universal_telegram_alerts import WATCHED, check_checkpoint, validate_ledger, poll_kaggle


class CpuTelegramE2ETests(unittest.TestCase):
    def test_private_cpu_no_network_no_datasets(self):
        with tempfile.TemporaryDirectory() as t:
            dest = Path(t) / "package"
            build(dest)
            meta = json.loads((dest / "kernel-metadata.json").read_text())
            self.assertEqual(meta["id"], KERNEL)
            self.assertTrue(meta["is_private"])
            for field in ("enable_gpu", "enable_tpu", "enable_internet"):
                self.assertIs(meta[field], False)
            for field in ("dataset_sources", "competition_sources", "kernel_sources", "model_sources"):
                self.assertEqual(meta[field], [])
            self.assertEqual((dest / "runner.py").read_text(), RUNNER)
            self.assertEqual(hashlib.sha256(RUNNER.encode()).hexdigest(), SOURCE_DIGEST)
            with self.assertRaises(ValueError):
                build(dest)

    def test_deterministic_runner_and_independent_verifier(self):
        with tempfile.TemporaryDirectory() as t:
            output = Path(t) / "out"
            output.mkdir()
            # Runner itself writes to Kaggle working; adapt only the fixed output path
            # in memory for this no-network offline verification.
            runner = RUNNER.replace("/kaggle/working/telegram_cpu_e2e_manifest.json",
                                    str(output / OUTPUT_NAME))
            subprocess.run([sys.executable, "-c", runner], check=True, capture_output=True)
            data = verify(output)
            self.assertEqual(data["tasks"], [1, 2, 3, 4, 5])
            self.assertIs(data["scientific_experiments_executed"], False)
            self.assertIs(data["contains_medical_data"], False)
            self.assertIn("5/5", check_checkpoint("cpu_telegram_e2e", output, "fixed-protocol-v1"))
            data["tasks"] = [1, 2, 3]
            (output / OUTPUT_NAME).write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                verify(output)

    def test_missing_duplicate_and_wrong_source_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            out = Path(t)
            with self.assertRaises(ValueError):
                verify(out)
            (out / OUTPUT_NAME).write_text("{}")
            with self.assertRaises(ValueError):
                verify(out, expected_digest="incorrect")
            with self.assertRaises(ValueError):
                verify(out)
            other = out / "other"
            other.mkdir()
            (other / OUTPUT_NAME).write_text("{}")
            with self.assertRaises(ValueError):
                verify(out)

    def test_ledger_registered_and_monitor_is_read_only(self):
        data = json.loads((ROOT / "monitoring/universal-telegram-alert-ledger.json").read_text())
        validate_ledger(data)
        key = "cpu-telegram-e2e-v1"
        self.assertEqual(WATCHED[key]["kernel"], KERNEL)
        self.assertEqual(data["kaggle_events"][key], {"last_event": None, "delivery": "none"})
        with patch("universal_telegram_alerts.read_ledger", return_value=(data, "a"*40)), \
             patch("universal_telegram_alerts.check", return_value="missing") as check, \
             patch("universal_telegram_alerts.send_message") as send, \
             patch("universal_telegram_alerts.subprocess.run") as execute:
            self.assertEqual(poll_kaggle("GH", "BOT", "123456", Path("unused")),
                             "kaggle_new_alerts_0")
            check.assert_called_once_with(KERNEL, "any")
            send.assert_not_called()
            execute.assert_not_called()


if __name__ == "__main__":
    unittest.main()
