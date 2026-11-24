"""Offline simulations of duplicate-proof intent and recovery state machine."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from idempotent_resume_controller import (
    make_state, validate_state, reserve_once, mark_before_external_submission,
    require_review, reconcile_observed, proposed_compare_and_swap, live_inspect,
    ResumeSafetyError,
)
from safe_monitor_decision import EXPECTED_COMMIT


def checkpoint(count: int):
    return {
        "experiment_id": "cpu-pilot-automation-001",
        "git_commit": EXPECTED_COMMIT,
        "total_tasks": 100,
        "completed_task_ids": list(range(count)),
        "status": "completed" if count == 100 else "in_progress",
    }


def reserved_a_to_b():
    decision, reserved = reserve_once(make_state(40, "a"), "a", 40, "b", "12345")
    assert decision == "reservation_proposed_not_dispatched"
    return reserved


class IdempotentResumeTests(unittest.TestCase):
    def test_final_100_checkpoint_never_submits(self):
        finished = make_state(100, "a", "completed")
        self.assertEqual(reserve_once(finished, "b", 80, "a", "55555")[0],
                         "completed_no_submit")
        out, same = reconcile_observed(
            finished, {"a": "complete", "b": "complete"},
            {"a": checkpoint(100), "b": checkpoint(80)}
        )
        self.assertEqual(out, "stop_completed_no_duplicate")
        self.assertEqual(same, finished)

    def test_duplicate_reservation_blocked_even_new_run(self):
        reserved = reserved_a_to_b()
        decision, current = reserve_once(reserved, "a", 40, "b", "22222")
        self.assertEqual(decision, "duplicate_reservation_blocked")
        self.assertEqual(current, reserved)

    def test_different_or_stale_reservation_blocked(self):
        reserved = reserved_a_to_b()
        self.assertEqual(reserve_once(reserved, "b", 80, "a", "22222")[0],
                         "different_reservation_blocked")
        self.assertEqual(reserve_once(make_state(40, "a"), "b", 80, "a", "22222")[0],
                         "stale_or_regressed_checkpoint_blocked")

    def test_submitted_marker_before_external_call(self):
        reserved = reserved_a_to_b()
        with self.assertRaisesRegex(ResumeSafetyError, "CLAIM_NOT_OWNED"):
            mark_before_external_submission(reserved, "99999")
        submitted = mark_before_external_submission(reserved, "12345")
        self.assertEqual(submitted["phase"], "submitted")
        self.assertTrue(submitted["intent"]["submission_attempted"])
        self.assertEqual(reserve_once(submitted, "a", 40, "b", "55555")[0],
                         "duplicate_reservation_blocked")
        with self.assertRaisesRegex(ResumeSafetyError, "CLAIM_NOT_OWNED"):
            mark_before_external_submission(submitted, "12345")

    def test_ambiguous_submission_never_retried(self):
        submitted = mark_before_external_submission(reserved_a_to_b(), "12345")
        flagged = require_review(submitted, "AMBIGUOUS_KAGGLE_SUBMISSION")
        self.assertEqual(flagged["phase"], "needs_review")
        self.assertEqual(reserve_once(flagged, "a", 40, "b", "22222")[0],
                         "duplicate_reservation_blocked")
        outcome, _ = reconcile_observed(
            flagged, {"a": "complete", "b": "missing"}, {"a": checkpoint(40)}
        )
        self.assertEqual(outcome, "manual_review_required")

    def test_active_kaggle_blocks_new_submission(self):
        outcome, _ = reconcile_observed(
            reserved_a_to_b(), {"a": "complete", "b": "running"},
            {"a": checkpoint(40)}
        )
        self.assertEqual(outcome, "wait_active_kaggle_job")

    def test_successful_resume_reconciles_40_to_80(self):
        submitted = mark_before_external_submission(reserved_a_to_b(), "12345")
        outcome, new = reconcile_observed(
            submitted, {"a": "complete", "b": "complete"},
            {"a": checkpoint(40), "b": checkpoint(80)}
        )
        self.assertEqual(outcome, "reconciled_advanced")
        self.assertEqual(new, make_state(80, "b", "idle"))

    def test_successful_resume_reconciles_80_to_100(self):
        _, reserved = reserve_once(make_state(80, "b"), "b", 80, "a", "12345")
        submitted = mark_before_external_submission(reserved, "12345")
        outcome, new = reconcile_observed(
            submitted, {"a": "complete", "b": "complete"},
            {"a": checkpoint(100), "b": checkpoint(80)}
        )
        self.assertEqual(outcome, "reconciled_completed")
        self.assertEqual(new, make_state(100, "a", "completed"))

    def test_failure_becomes_manual_review_without_retry(self):
        submitted = mark_before_external_submission(reserved_a_to_b(), "12345")
        out, flagged = reconcile_observed(
            submitted, {"a": "complete", "b": "failed"}, {"a": checkpoint(40)}
        )
        self.assertEqual(out, "needs_manual_failure_review")
        self.assertEqual(flagged["phase"], "needs_review")
        self.assertEqual(flagged["review_reason"], "REMOTE_SESSION_FAILED")

    def test_target_missing_or_unadvanced_requires_review(self):
        submitted = mark_before_external_submission(reserved_a_to_b(), "12345")
        outcome, flagged = reconcile_observed(
            submitted, {"a": "complete", "b": "missing"}, {"a": checkpoint(40)}
        )
        self.assertEqual(outcome, "submission_unknown_manual_reconciliation")
        self.assertEqual(flagged["phase"], "needs_review")
        outcome, flagged = reconcile_observed(
            submitted, {"a": "complete", "b": "complete"},
            {"a": checkpoint(40), "b": checkpoint(40)}
        )
        self.assertEqual(outcome, "checkpoint_invalid_manual_review")
        self.assertEqual(flagged["phase"], "needs_review")

    def test_corrupt_ledger_rejected(self):
        good = make_state(100, "a", "completed")
        for bad in (
            {**good, "scope": "restricted-medical-data"},
            {**good, "phase": "idle"},
            {**good, "intent": {"anything": "value"}},
            {**good, "confirmed_completed_tasks": 101},
        ):
            with self.subTest(bad=bad):
                with self.assertRaises(ResumeSafetyError):
                    validate_state(bad)

    def test_cas_persistence_conflict_never_submits(self):
        saved = []
        def writer(sha, record):
            saved.append((sha, record))
            return False
        outcome, _ = proposed_compare_and_swap(
            lambda: (make_state(40, "a"), "a" * 40),
            writer, "a", 40, "b", "12345"
        )
        self.assertEqual(outcome, "concurrent_claim_conflict_no_submit")
        self.assertEqual(len(saved), 1)

    def test_cas_reservation_is_not_dispatch_permission(self):
        actual = []
        outcome, proposal = proposed_compare_and_swap(
            lambda: (make_state(40, "a"), "a" * 40),
            lambda sha, data: actual.append(data) or True,
            "a", 40, "b", "12345"
        )
        self.assertEqual(outcome, "durably_reserved_no_submit")
        self.assertEqual(actual, [proposal])
        self.assertEqual(proposal["phase"], "reserved")

    @patch("idempotent_resume_controller.status_and_checkpoints")
    def test_live_guard_rejects_corrupt_remote(self, getter):
        getter.return_value = ({"a": "complete", "b": "failed"}, {})
        import tempfile
        import json
        with tempfile.TemporaryDirectory() as tmp:
            control = Path(tmp) / "control.json"
            control.write_text(json.dumps(make_state(100, "a", "completed")))
            self.assertEqual(
                live_inspect(control, Path(tmp) / "output")[0], "PAUSE"
            )

    @patch("idempotent_resume_controller.status_and_checkpoints")
    def test_live_guard_checks_real_final_checkpoint(self, getter):
        getter.return_value = (
            {"a": "complete", "b": "complete"},
            {"a": checkpoint(100), "b": checkpoint(80)}
        )
        import tempfile
        import json
        with tempfile.TemporaryDirectory() as tmp:
            control = Path(tmp) / "control.json"
            control.write_text(json.dumps(make_state(100, "a", "completed")))
            self.assertEqual(
                live_inspect(control, Path(tmp) / "output"),
                ("STOP", "ALREADY_COMPLETE_DUPLICATE_SUBMISSION_BLOCKED")
            )


if __name__ == "__main__":
    unittest.main()
