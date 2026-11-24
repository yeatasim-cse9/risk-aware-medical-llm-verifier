"""Unit tests for read-only two-slot resume and quota-safety checks."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from auto_resume_preflight import assess, status_and_checkpoints
from safe_monitor_decision import EXPECTED_COMMIT


def checkpoint(count, status=None):
    return {
        "experiment_id": "cpu-pilot-automation-001",
        "git_commit": EXPECTED_COMMIT,
        "total_tasks": 100,
        "completed_task_ids": list(range(count)),
        "status": status or ("completed" if count == 100 else "in_progress"),
    }


class AutoResumePreflightTests(unittest.TestCase):
    def test_actual_finished_pilot_must_stop_regardless_of_quota(self):
        remote = {"a": "complete", "b": "complete"}
        states = {"a": checkpoint(100), "b": checkpoint(80)}
        for signal in ("unknown", "insufficient", "operator_attested"):
            self.assertEqual(
                assess(remote, states, signal),
                ("stop", "ALL_100_TASKS_COMPLETE_NO_RESTART"),
            )

    def test_first_session_with_unknown_quota_pauses(self):
        remote = {"a": "complete", "b": "missing"}
        states = {"a": checkpoint(40)}
        self.assertEqual(
            assess(remote, states),
            ("pause", "QUOTA_NOT_INDEPENDENTLY_VERIFIED"),
        )
        self.assertEqual(
            assess(remote, states, "insufficient"),
            ("pause", "QUOTA_INSUFFICIENT"),
        )
        self.assertEqual(
            assess(remote, states, "operator_attested"),
            ("ready_for_manual_review", "LATEST_A_40_NEXT_B"),
        )

    def test_second_session_a_b_resume_candidate_review_only(self):
        remote = {"a": "complete", "b": "complete"}
        self.assertEqual(
            assess(remote, {"a": checkpoint(40), "b": checkpoint(80)}, "operator_attested"),
            ("ready_for_manual_review", "LATEST_B_80_NEXT_A"),
        )

    def test_active_and_failed_never_launch(self):
        for status in ("running", "queued", "pending", "starting", "initializing"):
            self.assertEqual(
                assess({"a": "complete", "b": status}, {"a": checkpoint(40)})[0],
                "wait",
            )
        for status in ("failed", "error", "cancelled", "cancelacknowledged"):
            self.assertEqual(
                assess({"a": "complete", "b": status}, {"a": checkpoint(40)})[0],
                "pause",
            )

    def test_missing_ambiguous_invalid_and_regression_cases(self):
        cases = [
            ({"a": "missing", "b": "missing"}, {}),
            ({"a": "complete", "b": "complete"}, {"a": checkpoint(40)}),
            ({"a": "complete", "b": "complete"}, {"a": checkpoint(80), "b": checkpoint(80)}),
            ({"a": "complete", "b": "missing"}, {"a": checkpoint(40), "b": checkpoint(80)}),
            ({"a": "complete", "b": "missing"}, {"a": checkpoint(40, "completed")}),
            ({"a": "complete", "b": "missing"}, {"a": checkpoint(100, "in_progress")}),
            ({"a": "complete", "b": "unrecognized"}, {"a": checkpoint(40)}),
        ]
        for remote, states in cases:
            with self.subTest(remote=remote, states=states):
                self.assertEqual(assess(remote, states, "operator_attested")[0], "pause")

    def test_invalid_quota_signal_fails(self):
        self.assertEqual(
            assess({"a": "complete", "b": "missing"}, {"a": checkpoint(40)}, "fake_available")[0],
            "pause",
        )

    @patch("auto_resume_preflight.verify")
    @patch("auto_resume_preflight.subprocess.run")
    @patch("auto_resume_preflight.check")
    def test_checks_both_slots_and_verifies_outputs(self, check, run, verify):
        check.side_effect = ["complete", "complete"]
        run.return_value = MagicMock(returncode=0)
        verify.side_effect = [checkpoint(100), checkpoint(80)]
        remote, states = status_and_checkpoints(Path("build/test-resume"))
        self.assertEqual(remote, {"a": "complete", "b": "complete"})
        self.assertEqual(states["a"]["completed_task_ids"], list(range(100)))
        self.assertEqual(verify.call_count, 2)
        for call in verify.call_args_list:
            self.assertEqual(call.kwargs["expected_commit"], EXPECTED_COMMIT)
        self.assertEqual(run.call_count, 2)
        for call in run.call_args_list:
            self.assertEqual(call.args[0][:3], ["kaggle", "kernels", "output"])
            self.assertTrue(call.kwargs["capture_output"])

    @patch("auto_resume_preflight.subprocess.run")
    @patch("auto_resume_preflight.check")
    def test_active_never_downloads_output(self, check, run):
        check.side_effect = ["complete", "running"]
        remote, states = status_and_checkpoints(Path("build/test-resume"))
        self.assertEqual(remote, {"a": "complete", "b": "running"})
        self.assertEqual(states, {})
        run.assert_not_called()

    @patch("auto_resume_preflight.subprocess.run")
    @patch("auto_resume_preflight.check")
    def test_failed_download_refuses_to_continue(self, check, run):
        check.side_effect = ["complete", "missing"]
        run.return_value = MagicMock(returncode=1, stdout="SENSITIVE", stderr="SENSITIVE")
        with self.assertRaisesRegex(RuntimeError, "KAGGLE_CHECKPOINT_DOWNLOAD_FAILED"):
            status_and_checkpoints(Path("build/test-resume"))


if __name__ == "__main__":
    unittest.main()
