"""No-network unit tests for scheduled synthetic Telegram monitoring."""
import base64
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from periodic_pilot_monitor import (
    LEDGER_PATH, MonitorError, ledger_with_progress, validate_ledger,
    progress_key, process_verified_checkpoint, schedule_guard,
    read_ledger, write_ledger,
)
from notify_synthetic_pilot_status import SOURCE_COMMIT


def state(count=100):
    return {
        "experiment_id": "cpu-pilot-automation-001",
        "git_commit": SOURCE_COMMIT,
        "total_tasks": 100,
        "completed_task_ids": list(range(count)),
        "status": "completed" if count == 100 else "in_progress",
    }


class PeriodicMonitorTests(unittest.TestCase):
    def test_ledger_validation(self):
        valid = ledger_with_progress(100, "completed")
        self.assertEqual(validate_ledger(valid), valid)
        for bad in (
            {},
            {**valid, "scope": "clinical_dataset"},
            {**valid, "last_notified": {**valid["last_notified"], "key": "bad"}},
        ):
            with self.subTest(bad=bad):
                with self.assertRaises(MonitorError):
                    validate_ledger(bad)

    def test_fingerprint_stable(self):
        self.assertEqual(
            progress_key(100, "completed"),
            f"{SOURCE_COMMIT}:a:100:completed",
        )
        with self.assertRaises(MonitorError):
            progress_key(101, "completed")

    @patch("periodic_pilot_monitor.read_ledger")
    def test_scheduled_terminal_skips_kaggle(self, read):
        read.return_value = (ledger_with_progress(100, "completed"), "a" * 40)
        self.assertFalse(schedule_guard("TOKEN", "schedule"))
        self.assertTrue(schedule_guard("TOKEN", "workflow_dispatch"))
        self.assertTrue(schedule_guard("TOKEN", "push"))

    @patch("periodic_pilot_monitor.read_ledger")
    def test_scheduled_partial_runs(self, read):
        read.return_value = (ledger_with_progress(40, "in_progress"), "a" * 40)
        self.assertTrue(schedule_guard("TOKEN", "schedule"))

    @patch("periodic_pilot_monitor.send_message")
    @patch("periodic_pilot_monitor.write_ledger")
    @patch("periodic_pilot_monitor.read_ledger")
    @patch("periodic_pilot_monitor.verify")
    def test_duplicate_is_suppressed(self, verify, read, write, send):
        verify.return_value = state(100)
        read.return_value = (ledger_with_progress(100, "completed"), "a" * 40)
        result = process_verified_checkpoint(Path("fake"), "GH_TOKEN", "", "")
        self.assertEqual(result, "duplicate")
        send.assert_not_called()
        write.assert_not_called()

    @patch("periodic_pilot_monitor.send_message")
    @patch("periodic_pilot_monitor.write_ledger")
    @patch("periodic_pilot_monitor.read_ledger")
    @patch("periodic_pilot_monitor.verify")
    def test_new_progress_sends_once_then_persists(self, verify, read, write, send):
        verify.return_value = state(80)
        read.return_value = (ledger_with_progress(40, "in_progress"), "a" * 40)
        events = []
        send.side_effect = lambda *a: events.append("telegram")
        write.side_effect = lambda *a: events.append("ledger")
        result = process_verified_checkpoint(Path("fake"), "GH_TOKEN", "BOT", "123456")
        self.assertEqual(result, "notified")
        self.assertEqual(events, ["telegram", "ledger"])
        sent = send.call_args.args[2]
        self.assertIn("80/100", sent)
        new_ledger = write.call_args.args[2]
        self.assertEqual(new_ledger["last_notified"]["count"], 80)

    @patch("periodic_pilot_monitor.send_message")
    @patch("periodic_pilot_monitor.write_ledger")
    @patch("periodic_pilot_monitor.read_ledger")
    @patch("periodic_pilot_monitor.verify")
    def test_failed_delivery_does_not_persist(self, verify, read, write, send):
        verify.return_value = state(100)
        read.return_value = (ledger_with_progress(80, "in_progress"), "a" * 40)
        send.side_effect = ValueError("failed telegram")
        with self.assertRaises(ValueError):
            process_verified_checkpoint(Path("fake"), "GH_TOKEN", "BOT", "123456")
        write.assert_not_called()

    @patch("periodic_pilot_monitor.send_message")
    @patch("periodic_pilot_monitor.write_ledger")
    @patch("periodic_pilot_monitor.read_ledger")
    @patch("periodic_pilot_monitor.verify")
    def test_regression_blocks_all_delivery(self, verify, read, write, send):
        verify.return_value = state(40)
        read.return_value = (ledger_with_progress(100, "completed"), "a" * 40)
        with self.assertRaisesRegex(MonitorError, "PROGRESS_REGRESSION"):
            process_verified_checkpoint(Path("fake"), "GH_TOKEN", "BOT", "123456")
        write.assert_not_called()
        send.assert_not_called()

    @patch("periodic_pilot_monitor.request_github")
    def test_ledger_decode_and_ref(self, request):
        data = ledger_with_progress(100, "completed")
        encoded = base64.b64encode(json.dumps(data).encode()).decode()
        request.return_value = {"sha": "a"*40, "encoding": "base64", "content": encoded}
        found, sha = read_ledger("TOKEN")
        self.assertEqual(found, data)
        self.assertEqual(sha, "a"*40)
        self.assertEqual(request.call_args.args[1], "GET")

    @patch("periodic_pilot_monitor.request_github")
    def test_write_ledger_uses_sha_and_fixed_branch(self, request):
        request.return_value = {"commit": {"sha": "f" * 40}}
        write_ledger("TOKEN", "a"*40, ledger_with_progress(100, "completed"))
        args = request.call_args.args
        self.assertEqual(args[1], "PUT")
        self.assertEqual(args[2]["sha"], "a"*40)
        self.assertEqual(args[2]["branch"], "main")
        self.assertIn("synthetic Telegram progress", args[2]["message"])

    @patch("periodic_pilot_monitor.request_github")
    def test_corrupt_ledger_never_falls_back(self, request):
        request.side_effect = MonitorError("GITHUB_LEDGER_MISSING")
        with self.assertRaises(MonitorError):
            read_ledger("TOKEN")


if __name__ == "__main__":
    unittest.main()
