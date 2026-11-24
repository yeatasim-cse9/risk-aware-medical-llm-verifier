"""Offline tests for quota, cooldown, retry-backoff and safe-stop policy."""
import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from quota_cooldown_guard import (
    PILOT_SCOPE, evaluate, check_history, GuardError, COOLDOWN_HOURS,
    MAX_ATTEMPTS_24H, MAX_CONSECUTIVE_FAILURES, MAX_ATTESTATION_AGE_MINUTES,
)
from safe_monitor_decision import EXPECTED_COMMIT

NOW = datetime(2026, 11, 25, 12, 0, 0, tzinfo=timezone.utc)


def iso(dt):
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def checkpoint(count):
    return {
        "experiment_id": "cpu-pilot-automation-001",
        "git_commit": EXPECTED_COMMIT,
        "total_tasks": 100,
        "completed_task_ids": list(range(count)),
        "status": "completed" if count == 100 else "in_progress",
    }


def history(*items):
    return {
        "schema_version": 1,
        "scope": PILOT_SCOPE,
        "source": "operator_record",
        "complete": True,
        "attempts": [
            {"id": f"ghrun-{i+1}", "submitted_at_utc": iso(t), "outcome": status}
            for i, (t, status) in enumerate(items)
        ],
    }


def manual(remote=None, states=None, **kwargs):
    if remote is None:
        remote = {"a": "complete", "b": "missing"}
    if states is None:
        states = {"a": checkpoint(40)}
    return evaluate(
        remote, states, NOW, quota_signal="operator_attested",
        quota_attested_at_utc=iso(NOW - timedelta(minutes=2)), **kwargs,
    )


class QuotaCooldownTests(unittest.TestCase):
    def test_completed_session_stops_even_without_quota_or_history(self):
        self.assertEqual(
            evaluate(
                {"a": "complete", "b": "complete"},
                {"a": checkpoint(100), "b": checkpoint(80)}, NOW,
            ),
            ("stop", "ALL_100_TASKS_COMPLETE_NO_RESTART"),
        )

    def test_unknown_quota_does_not_authorize_partial_resume(self):
        self.assertEqual(
            evaluate({"a": "complete", "b": "missing"}, {"a": checkpoint(40)}, NOW),
            ("pause", "QUOTA_NOT_INDEPENDENTLY_VERIFIED"),
        )
        self.assertEqual(
            evaluate({"a": "complete", "b": "missing"}, {"a": checkpoint(40)}, NOW,
                     quota_signal="insufficient"),
            ("pause", "QUOTA_INSUFFICIENT"),
        )

    def test_active_waits_and_failed_pauses_even_with_attestation(self):
        for state in ("running", "queued", "pending", "starting"):
            with self.subTest(state=state):
                self.assertEqual(
                    manual({"a": "complete", "b": state}, {"a": checkpoint(40)},
                           history=history())[0],
                    "wait",
                )
        self.assertEqual(
            manual({"a": "complete", "b": "failed"}, {"a": checkpoint(40)},
                   history=history())[0],
            "pause",
        )

    def test_missing_or_invalid_quota_attestation(self):
        self.assertEqual(
            evaluate(
                {"a": "complete", "b": "missing"}, {"a": checkpoint(40)}, NOW,
                quota_signal="operator_attested", history=history(),
            ),
            ("pause", "QUOTA_ATTESTATION_REQUIRED"),
        )
        for stamp in (
            iso(NOW - timedelta(minutes=MAX_ATTESTATION_AGE_MINUTES + 1)),
            iso(NOW + timedelta(seconds=1)), "bad",
        ):
            with self.subTest(stamp=stamp):
                result = evaluate(
                    {"a": "complete", "b": "missing"}, {"a": checkpoint(40)}, NOW,
                    quota_signal="operator_attested",
                    quota_attested_at_utc=stamp, history=history(),
                )
                self.assertEqual(result[0], "pause")

    def test_history_missing_untrusted_and_malformed_are_blocked(self):
        self.assertEqual(manual()[0], "pause")
        for bad in (
            {}, {"schema_version": 1},
            {**history(), "complete": False},
            {**history(), "scope": "clinical"},
            {**history(), "source": "unknown"},
        ):
            with self.subTest(bad=bad):
                self.assertEqual(manual(history=bad)[0], "pause")

    def test_six_hour_cooldown_and_equality_boundary(self):
        last = NOW - timedelta(hours=COOLDOWN_HOURS-1)
        self.assertEqual(
            manual(history=history((last, "completed"))),
            ("pause", "COOLDOWN_OR_FAILURE_BACKOFF_ACTIVE"),
        )
        self.assertEqual(
            manual(history=history((NOW - timedelta(hours=COOLDOWN_HOURS),
                                    "completed")))[0],
            "ready_for_manual_review",
        )

    def test_failure_backoff_is_exponential_and_bounded(self):
        # One failed run yields 12h backoff.
        last = NOW - timedelta(hours=8)
        self.assertEqual(
            manual(history=history((last, "failed"))),
            ("pause", "COOLDOWN_OR_FAILURE_BACKOFF_ACTIVE"),
        )
        self.assertEqual(
            manual(history=history((NOW - timedelta(hours=12), "failed")))[0],
            "ready_for_manual_review",
        )
        # Two consecutive failures require review regardless of age.
        self.assertEqual(
            manual(history=history(
                (NOW - timedelta(hours=48), "failed"),
                (NOW - timedelta(hours=36), "failed"),
            )),
            ("pause", "REPEATED_FAILURES_REQUIRE_MANUAL_DIAGNOSIS"),
        )

    def test_max_two_attempts_rolling_24h(self):
        records = history(
            (NOW - timedelta(hours=22), "completed"),
            (NOW - timedelta(hours=8), "completed"),
        )
        self.assertEqual(
            manual(history=records),
            ("pause", "ROLLING_24H_ATTEMPT_LIMIT_REACHED"),
        )
        self.assertEqual(MAX_ATTEMPTS_24H, 2)
        self.assertEqual(MAX_CONSECUTIVE_FAILURES, 2)

    def test_complete_history_and_timestamp_integrity(self):
        good = history(
            (NOW - timedelta(hours=48), "completed"),
            (NOW - timedelta(hours=7), "completed"),
        )
        self.assertEqual(len(check_history(good, NOW)), 2)
        tests = [
            history((NOW + timedelta(minutes=1), "completed")),
            history((NOW - timedelta(hours=2), "completed"),
                    (NOW - timedelta(hours=3), "completed")),
            history((NOW - timedelta(hours=2), "unknown")),
        ]
        duplicate = history((NOW - timedelta(hours=4), "completed"),
                            (NOW - timedelta(hours=2), "failed"))
        duplicate["attempts"][1]["id"] = duplicate["attempts"][0]["id"]
        tests.append(duplicate)
        for bad in tests:
            with self.subTest(bad=bad):
                with self.assertRaises(GuardError):
                    check_history(bad, NOW)

    def test_passes_only_for_manual_review_not_auto_submit(self):
        self.assertEqual(
            manual(history=history((NOW-timedelta(hours=30), "completed"))),
            ("ready_for_manual_review", "COOLDOWN_AND_LOCAL_POLICY_CHECKS_PASSED"),
        )

    def test_invalid_clock_fails(self):
        self.assertEqual(
            evaluate(
                {"a": "complete", "b": "missing"},
                {"a": checkpoint(40)}, NOW.replace(tzinfo=None),
            ), ("pause", "INVALID_CURRENT_UTC_TIME"),
        )

    @patch("quota_cooldown_guard.status_and_checkpoints")
    def test_live_runner_only_reads_status_and_checkpoint(self, read):
        read.return_value = (
            {"a": "complete", "b": "complete"},
            {"a": checkpoint(100), "b": checkpoint(80)},
        )
        from quota_cooldown_guard import main
        with patch("sys.argv", ["quota_cooldown_guard.py", "--work-dir", "build/one"]):
            self.assertEqual(main(), 0)
        read.assert_called_once_with(Path("build/one"))


if __name__ == "__main__":
    unittest.main()
