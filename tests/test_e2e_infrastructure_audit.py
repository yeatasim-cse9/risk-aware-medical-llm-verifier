"""Offline integration fixtures; strict separation from actual Kaggle evidence."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from e2e_infrastructure_audit import (
    SLOTS, FROZEN_SHAS, REPO_ROOT, IntegrationError,
    audit_saved_outputs, fetch_read_only, static_contract, make_report,
)


def cpu_state(n):
    return {
        "experiment_id": "cpu-pilot-automation-001",
        "git_commit": FROZEN_SHAS["cpu_a"],
        "total_tasks": 100,
        "completed_task_ids": list(range(n)),
        "status": "completed" if n == 100 else "in_progress",
    }


def gpu_smoke():
    return {
        "device": "cuda:0", "gpu_available": True,
        "source_commit": FROZEN_SHAS["gpu_smoke"]
    }


def tiny_model():
    return {
        "device": "cuda:0", "model_pretrained": False,
        "medical_experiment": False, "num_samples": 4,
        "source_commit": FROZEN_SHAS["gpu_inference"],
    }


def dirs():
    return {key: Path("/tmp/" + key) for key in SLOTS}


def statuses():
    return {key: "complete" for key in SLOTS}


class ReadOnlyE2ETests(unittest.TestCase):
    def test_static_completion_ledgers_and_model_gates(self):
        telegram, control, pin = static_contract(REPO_ROOT)
        self.assertEqual(telegram["last_notified"]["count"], 100)
        self.assertEqual(control["phase"], "completed")
        self.assertFalse(pin["model_execution_authorized"])
        self.assertIsNone(pin["weight_sha256_independently_checked"])

    @patch("e2e_infrastructure_audit.verify_tiny_model")
    @patch("e2e_infrastructure_audit.verify_cuda_smoke")
    @patch("e2e_infrastructure_audit.verify_cpu")
    def test_e2e_happy_path_read_only_report(self, verify_cpu, verify_smoke, verify_tiny):
        verify_cpu.side_effect = [cpu_state(100), cpu_state(80)]
        verify_smoke.return_value = gpu_smoke()
        verify_tiny.return_value = tiny_model()
        report = audit_saved_outputs(REPO_ROOT, dirs(), statuses())
        self.assertEqual(report["status"], "historical_synthetic_cloud_outputs_verified")
        self.assertEqual(report["cpu_a"]["completed_tasks"], 100)
        self.assertEqual(report["cpu_b"]["completed_tasks"], 80)
        self.assertFalse(report["new_kaggle_job_submitted"])
        self.assertFalse(report["gpu_execution_triggered_by_audit"])
        self.assertFalse(report["telegram_message_sent_by_audit"])
        self.assertFalse(report["cron_firing_proven_by_audit"])
        self.assertFalse(report["medical_experiments_executed"])
        verify_cpu.assert_any_call(dirs()["cpu_a"], expected_commit=FROZEN_SHAS["cpu_a"])
        verify_cpu.assert_any_call(dirs()["cpu_b"], expected_commit=FROZEN_SHAS["cpu_b"])

    @patch("e2e_infrastructure_audit.verify_cpu")
    def test_missing_or_running_remote_refuses_outputs(self, cpu):
        for bad in ("running", "queued", "error", "missing"):
            with self.subTest(status=bad):
                s = statuses()
                s["gpu_inference"] = bad
                with self.assertRaises(IntegrationError):
                    audit_saved_outputs(REPO_ROOT, dirs(), s)
        cpu.assert_not_called()

    @patch("e2e_infrastructure_audit.verify_cpu")
    def test_cpu_wrong_checkpoint_progress_fails_closed(self, verify_cpu):
        verify_cpu.side_effect = [cpu_state(80), cpu_state(80)]
        with self.assertRaises(IntegrationError):
            audit_saved_outputs(REPO_ROOT, dirs(), statuses())

    @patch("e2e_infrastructure_audit.verify_tiny_model")
    @patch("e2e_infrastructure_audit.verify_cuda_smoke")
    @patch("e2e_infrastructure_audit.verify_cpu")
    def test_never_allow_forged_gpu_fallback(self, cpu, smoke, tiny):
        cpu.side_effect = [cpu_state(100), cpu_state(80)]
        smoke.return_value = {**gpu_smoke(), "device": "cpu"}
        tiny.return_value = tiny_model()
        with self.assertRaises(IntegrationError):
            audit_saved_outputs(REPO_ROOT, dirs(), statuses())

    @patch("e2e_infrastructure_audit.verify_tiny_model")
    @patch("e2e_infrastructure_audit.verify_cuda_smoke")
    @patch("e2e_infrastructure_audit.verify_cpu")
    def test_never_claim_pretrained_model_as_synthetic_pilot(self, cpu, smoke, tiny):
        cpu.side_effect = [cpu_state(100), cpu_state(80)]
        smoke.return_value = gpu_smoke()
        tiny.return_value = {**tiny_model(), "model_pretrained": True}
        with self.assertRaises(IntegrationError):
            audit_saved_outputs(REPO_ROOT, dirs(), statuses())

    @patch("e2e_infrastructure_audit.subprocess.run")
    @patch("e2e_infrastructure_audit.check")
    def test_four_remote_status_checks_and_only_readonly_output_calls(self, check, run):
        check.return_value = "complete"
        run.return_value = MagicMock(returncode=0)
        with tempfile.TemporaryDirectory() as d:
            paths, remote = fetch_read_only(Path(d))
        self.assertEqual(len(paths), 4)
        self.assertEqual(remote, statuses())
        self.assertEqual(check.call_count, 4)
        self.assertEqual(run.call_count, 4)
        for call in run.call_args_list:
            argv = call.args[0]
            self.assertEqual(argv[:3], ["kaggle", "kernels", "output"])
            self.assertTrue(call.kwargs["capture_output"])
            self.assertNotIn("push", argv)

    @patch("e2e_infrastructure_audit.subprocess.run")
    @patch("e2e_infrastructure_audit.check")
    def test_unknown_kaggle_status_skips_all_downloads(self, check, run):
        check.side_effect = ["complete", "complete", "queued", "complete"]
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(IntegrationError):
                fetch_read_only(Path(d))
        run.assert_not_called()

    @patch("e2e_infrastructure_audit.subprocess.run")
    @patch("e2e_infrastructure_audit.check")
    def test_download_failure_no_raw_response_leak(self, check, run):
        check.return_value = "complete"
        run.return_value = MagicMock(returncode=1, stderr="SUPER_SECRET_TOKEN")
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(IntegrationError, "^READ_ONLY_KAGGLE_DOWNLOAD_FAILED$") as e:
                fetch_read_only(Path(d))
        self.assertNotIn("SUPER_SECRET_TOKEN", str(e.exception))

    def test_report_integrity_and_no_sensitive_content(self):
        with tempfile.TemporaryDirectory() as d:
            report_path = Path(d)/"report.json"
            data = {
                "schema_version": 1, "status": "historical_synthetic_cloud_outputs_verified",
                "new_kaggle_job_submitted": False,
                "telegram_message_sent_by_audit": False,
            }
            make_report(data, report_path)
            saved = json.loads(report_path.read_text("utf-8"))
            digest = saved.pop("integrity_sha256")
            self.assertEqual(digest, hashlib.sha256(
                json.dumps(saved, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest())
            self.assertNotIn("TELEGRAM_BOT_TOKEN", report_path.read_text("utf-8"))


if __name__ == "__main__":
    unittest.main()
