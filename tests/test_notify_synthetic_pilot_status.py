"""Offline status notification safety tests; no Kaggle or Telegram network."""
import unittest
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from notify_synthetic_pilot_status import (
    build_status, notify_verified_checkpoint, SOURCE_COMMIT, TOTAL
)


def good_state():
    return {
        "experiment_id": "cpu-pilot-automation-001",
        "git_commit": SOURCE_COMMIT,
        "total_tasks": TOTAL,
        "completed_task_ids": list(range(TOTAL)),
        "status": "completed",
    }


class StatusTests(unittest.TestCase):
    def test_final_progress(self):
        text = build_status(good_state())
        self.assertIn("100/100 (100%)", text)
        self.assertIn("synthetic test data", text.lower())
        self.assertNotIn("patient", text.lower())

    def test_partial_progress(self):
        s = good_state()
        s["completed_task_ids"] = list(range(80))
        s["status"] = "in_progress"
        self.assertIn("80/100 (80%)", build_status(s))

    def test_invalid_experiment_rejected(self):
        s = good_state()
        s["experiment_id"] = "restricted-medical-experiment"
        with self.assertRaises(ValueError):
            build_status(s)

    def test_wrong_commit_rejected(self):
        s = good_state()
        s["git_commit"] = "0" * 40
        with self.assertRaises(ValueError):
            build_status(s)

    def test_wrong_ids_or_status_rejected(self):
        s = good_state()
        s["completed_task_ids"] = list(range(99)) + [99, 99]
        with self.assertRaises(ValueError):
            build_status(s)
        s = good_state()
        s["status"] = "in_progress"
        with self.assertRaises(ValueError):
            build_status(s)

    @patch("notify_synthetic_pilot_status.send_message")
    @patch("notify_synthetic_pilot_status.verify")
    @patch.dict("notify_synthetic_pilot_status.os.environ", {
        "TELEGRAM_BOT_TOKEN": "DUMMY:SECRET",
        "TELEGRAM_CHAT_ID": "123456",
    })
    def test_verified_checkpoint_only_and_recipient(self, verify, send):
        verify.return_value = good_state()
        notify_verified_checkpoint(Path("build/verified"))
        verify.assert_called_once_with(Path("build/verified"), expected_commit=SOURCE_COMMIT)
        self.assertEqual(send.call_args.args[1], "123456")
        self.assertIn("100/100", send.call_args.args[2])

    @patch("notify_synthetic_pilot_status.send_message")
    @patch("notify_synthetic_pilot_status.verify")
    @patch.dict("notify_synthetic_pilot_status.os.environ", {
        "TELEGRAM_BOT_TOKEN": "DUMMY:SECRET",
        "TELEGRAM_CHAT_ID": "123456",
    })
    def test_never_send_failed_validation(self, verify, send):
        verify.side_effect = ValueError("Invalid checksum")
        with self.assertRaises(ValueError):
            notify_verified_checkpoint(Path("build/bad"))
        send.assert_not_called()


if __name__ == "__main__":
    unittest.main()
