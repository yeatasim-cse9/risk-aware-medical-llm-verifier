import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from safe_monitor_decision import decide, decide_from_checkpoint, EXPECTED_COMMIT


def state(count=100, status=None):
    return {
        "experiment_id": "cpu-pilot-automation-001",
        "git_commit": EXPECTED_COMMIT,
        "total_tasks": 100,
        "completed_task_ids": list(range(count)),
        "status": status or ("completed" if count == 100 else "in_progress"),
    }


class SafeMonitorTests(unittest.TestCase):
    def test_finished_experiment_never_restarts(self):
        self.assertEqual(
            decide("complete", state(100), "a"),
            ("stop", "ALL_TASKS_COMPLETE_NO_RESTART"),
        )

    def test_partial_checkpoint_only_ready_for_review(self):
        self.assertEqual(
            decide("complete", state(40), "a"),
            ("ready_for_review", "CHECKPOINT_VALID_NEXT_SLOT_B"),
        )
        self.assertEqual(
            decide("complete", state(80), "b"),
            ("ready_for_review", "CHECKPOINT_VALID_NEXT_SLOT_A"),
        )

    def test_running_waits_and_failed_pauses(self):
        self.assertEqual(decide("running", None, "a")[0], "wait")
        self.assertEqual(decide("queued", None, "b")[0], "wait")
        self.assertEqual(decide("failed", state(40), "a")[0], "pause")
        self.assertEqual(decide("missing", None, "a")[0], "pause")
        self.assertEqual(decide("mystery", None, "a")[0], "pause")

    def test_missing_or_corrupted_metadata_blocked(self):
        self.assertEqual(decide("complete", None, "a")[0], "pause")
        invalid = state(40)
        invalid["git_commit"] = "0" * 40
        self.assertEqual(decide("complete", invalid, "a")[0], "pause")
        invalid = state(40)
        invalid["completed_task_ids"] = [0, 1, 1]
        self.assertEqual(decide("complete", invalid, "a")[0], "pause")
        invalid = state(100, "in_progress")
        self.assertEqual(decide("complete", invalid, "a")[0], "pause")
        self.assertEqual(decide("complete", state(100), "unknown")[0], "pause")

    @patch("safe_monitor_decision.verify")
    def test_verify_checkpoint_before_decision(self, verify):
        verify.return_value = state(100)
        decision = decide_from_checkpoint(Path("/tmp/pilot"), "complete", "a")
        self.assertEqual(decision[0], "stop")
        verify.assert_called_once_with(Path("/tmp/pilot"), expected_commit=EXPECTED_COMMIT)

    @patch("safe_monitor_decision.verify")
    def test_invalid_checkpoint_never_accepted(self, verify):
        verify.side_effect = ValueError("Checksum mismatch")
        with self.assertRaises(ValueError):
            decide_from_checkpoint(Path("/tmp/pilot"), "complete", "a")


if __name__ == "__main__":
    unittest.main()
