"""Step 10.5F tests — no Telegram sends or live Kaggle calls."""
from __future__ import annotations

import base64
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from telegram_notification_test import NotificationError
from universal_telegram_alerts import (
    REPO, LEDGER_PATH, WATCHED, AlertError, validate_ledger,
    bootstrap, github_alert_payload, handle_github_event, poll_kaggle,
    read_ledger, write_ledger, check_checkpoint,
)

BASELINE = ROOT / LEDGER_PATH


def ledger():
    # Independent deterministic fixture, even after the live bootstrap has
    # legitimately changed the persistent file to bootstrap_sent=true.
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    baseline["bootstrap_sent"] = False
    baseline["github_events"] = []
    return baseline


def run_event(name="Kaggle GPU Tiny Transformer Inference Controller",
              conclusion="success", event="workflow_dispatch", branch="main",
              repo=REPO, attempt=1, number=1234567):
    return {
        "action": "completed",
        "workflow_run": {
            "name": name, "conclusion": conclusion, "event": event,
            "head_branch": branch,
            "head_repository": {"full_name": repo},
            "head_commit": {"message": "feat: test safe change"},
            "id": number, "run_attempt": attempt,
        },
    }


class UniversalAlertsTests(unittest.TestCase):
    def test_seed_historical_completed_notebooks_suppresses_backfill(self):
        data = validate_ledger(ledger())
        self.assertFalse(data["bootstrap_sent"])
        self.assertEqual(set(data["kaggle_events"]), set(WATCHED))
        self.assertTrue(all(x["delivery"] == "historical_suppressed"
                            for x in data["kaggle_events"].values()))

    @patch("universal_telegram_alerts.read_ledger")
    @patch("universal_telegram_alerts.send_message")
    @patch("universal_telegram_alerts.write_ledger")
    def test_bootstrap_one_message_then_persist(self, write, send, read):
        data=ledger()
        read.return_value = (data, "a"*40)
        self.assertEqual(bootstrap("GH", "BOT", "123456"), "bootstrap_sent")
        self.assertEqual(send.call_count, 1)
        self.assertIn("research alerts enabled", send.call_args.args[2])
        self.assertNotIn("MedVAL", send.call_args.args[2])
        saved = write.call_args.args[2]
        self.assertTrue(saved["bootstrap_sent"])
        self.assertFalse(data["bootstrap_sent"])
        read.return_value = (saved, "b"*40)
        self.assertEqual(bootstrap("GH", "BOT", "123456"), "bootstrap_already_recorded")
        self.assertEqual(send.call_count, 1)

    def test_allowed_success_is_github_only_not_gpu_success_claim(self):
        key, msg = github_alert_payload(run_event())
        self.assertEqual(key, "github:1234567:1:success")
        self.assertIn("GitHub workflow finished", msg)
        self.assertIn("not Kaggle runtime completion", msg)

    def test_failure_and_cancel_send_bounded_safe_strings(self):
        for outcome in ("failure", "timed_out", "cancelled"):
            with self.subTest(outcome=outcome):
                key, msg = github_alert_payload(run_event(conclusion=outcome))
                self.assertTrue(key.endswith(":" + outcome))
                self.assertIn(outcome, msg)
                self.assertLessEqual(len(msg), 500)
                self.assertNotIn("BOT", msg)

    def test_untrusted_or_nonactionable_github_events_silent(self):
        events=[
            run_event(name="Unregistered workflow"),
            run_event(branch="some-feature"),
            run_event(repo="attacker/untrusted"),
            run_event(event="pull_request"),
            run_event(conclusion="skipped"),
            run_event(name="Research Repository CI", event="push"),
            run_event(),
            {"action": "requested", "workflow_run": run_event()["workflow_run"]},
        ]
        events[6]["workflow_run"]["head_commit"]["message"] = (
            "chore(alerts): remember sanitized synthetic status delivery"
        )
        for event in events:
            with self.subTest(event=event):
                self.assertIsNone(github_alert_payload(event))

    @patch("universal_telegram_alerts.read_ledger")
    @patch("universal_telegram_alerts.send_message")
    @patch("universal_telegram_alerts.write_ledger")
    def test_github_run_is_sent_once_and_deduplicated(self, write, send, read):
        original=ledger()
        read.return_value=(original, "a"*40)
        event=run_event(conclusion="failure")
        self.assertEqual(handle_github_event("GH","BOT","123456",event), "github_event_sent")
        self.assertEqual(send.call_count, 1)
        saved=write.call_args.args[2]
        self.assertEqual(saved["github_events"], ["github:1234567:1:failure"])
        read.return_value=(saved,"b"*40)
        self.assertEqual(handle_github_event("GH","BOT","123456",event),
                         "github_run_duplicate_skipped")
        send.assert_called_once()
        self.assertEqual(original["github_events"], [])

    def test_rerun_attempt_changes_dedup_key(self):
        key1, _ = github_alert_payload(run_event(conclusion="failure", attempt=1))
        key2, _ = github_alert_payload(run_event(conclusion="success", attempt=2))
        self.assertNotEqual(key1, key2)

    @patch("universal_telegram_alerts.read_ledger")
    @patch("universal_telegram_alerts.check")
    @patch("universal_telegram_alerts.send_message")
    def test_all_old_kaggle_notebooks_skipped_without_api(self, send, check, read):
        read.return_value=(ledger(),"a"*40)
        result=poll_kaggle("GH","BOT","123456",Path("build/unused"))
        self.assertEqual(result,"all_registered_historical_jobs_already_complete")
        check.assert_not_called()
        send.assert_not_called()

    @patch("universal_telegram_alerts.read_ledger")
    @patch("universal_telegram_alerts.check")
    @patch("universal_telegram_alerts.send_message")
    def test_new_running_job_does_not_claim_success(self, send, check, read):
        data=ledger()
        data["kaggle_events"]["gpu-tiny"]={"last_event":None,"delivery":"none"}
        read.return_value=(data,"a"*40)
        check.return_value="running"
        self.assertEqual(poll_kaggle("GH","BOT","123456",Path("build/unused")),
                         "kaggle_new_alerts_0")
        send.assert_not_called()

    @patch("universal_telegram_alerts.read_ledger")
    @patch("universal_telegram_alerts.check")
    @patch("universal_telegram_alerts.send_message")
    @patch("universal_telegram_alerts.write_ledger")
    def test_new_kaggle_failed_job_sends_warning_once(self, write, send, check, read):
        data=ledger()
        data["kaggle_events"]["gpu-tiny"]={"last_event":None,"delivery":"none"}
        read.return_value=(data,"a"*40)
        check.return_value="failed"
        self.assertEqual(poll_kaggle("GH","BOT","123456",Path("build/unused")),
                         "kaggle_new_alerts_1")
        self.assertIn("requires attention",send.call_args.args[2])
        self.assertEqual(write.call_args.args[2]["kaggle_events"]["gpu-tiny"],
                         {"last_event":"gpu-tiny:failed","delivery":"sent"})

    @patch("universal_telegram_alerts.read_ledger")
    @patch("universal_telegram_alerts.check")
    @patch("universal_telegram_alerts.subprocess.run")
    @patch("universal_telegram_alerts.check_checkpoint")
    @patch("universal_telegram_alerts.send_message")
    @patch("universal_telegram_alerts.write_ledger")
    def test_completion_requires_saved_verified_checkpoint(
        self, write, send, verify, run, check, read,
    ):
        data=ledger()
        data["kaggle_events"]["gpu-tiny"]={"last_event":None,"delivery":"none"}
        read.return_value=(data,"a"*40)
        check.return_value="complete"
        run.return_value=MagicMock(returncode=0)
        verify.return_value="Untrained Transformer CUDA checkpoint verified."
        with tempfile.TemporaryDirectory() as tmp:
            result=poll_kaggle("GH","BOT","123456",Path(tmp))
        self.assertEqual(result,"kaggle_new_alerts_1")
        self.assertIn("Kaggle job completed",send.call_args.args[2])
        verify.assert_called_once()
        self.assertEqual(run.call_args.args[0][:3],["kaggle","kernels","output"])
        self.assertNotIn("push",run.call_args.args[0])
        self.assertEqual(write.call_args.args[2]["kaggle_events"]["gpu-tiny"],
                         {"last_event":"gpu-tiny:complete","delivery":"sent"})

    @patch("universal_telegram_alerts.read_ledger")
    @patch("universal_telegram_alerts.check")
    @patch("universal_telegram_alerts.subprocess.run")
    @patch("universal_telegram_alerts.check_checkpoint")
    @patch("universal_telegram_alerts.send_message")
    @patch("universal_telegram_alerts.write_ledger")
    def test_corrupt_checkpoint_never_issues_success_notification(
        self, write, send, verify, run, check, read,
    ):
        data=ledger()
        data["kaggle_events"]["gpu-smoke"]={"last_event":None,"delivery":"none"}
        read.return_value=(data,"a"*40)
        check.return_value="complete"
        run.return_value=MagicMock(returncode=0)
        verify.side_effect=ValueError("sensitive remote error never printed")
        with tempfile.TemporaryDirectory() as tmp:
            poll_kaggle("GH","BOT","123456",Path(tmp))
        self.assertNotIn("Kaggle job completed",send.call_args.args[2])
        self.assertIn("requires attention",send.call_args.args[2])
        self.assertEqual(write.call_args.args[2]["kaggle_events"]["gpu-smoke"]["last_event"],
                         "gpu-smoke:verification_failed")

    @patch("universal_telegram_alerts.read_ledger")
    @patch("universal_telegram_alerts.send_message")
    @patch("universal_telegram_alerts.write_ledger")
    def test_failed_delivery_will_not_claim_ledger_success(self, write, send, read):
        read.return_value=(ledger(),"a"*40)
        send.side_effect=NotificationError("BOT_BLOCKED_OR_NOT_STARTED")
        with self.assertRaises(NotificationError):
            bootstrap("GH","BOT","123456")
        write.assert_not_called()

    @patch("universal_telegram_alerts.github_content_api")
    def test_github_ledger_persist_uses_original_sha_and_main_branch(self, api):
        original=ledger()
        api.side_effect=[
            {"sha":"a"*40,"encoding":"base64",
             "content":base64.b64encode(json.dumps(original).encode()).decode()},
            {"commit":{"sha":"b"*40}},
        ]
        stored, sha=read_ledger("GH")
        self.assertEqual(sha,"a"*40)
        write_ledger("GH",sha,stored)
        self.assertEqual(api.call_args.args[1],"PUT")
        body=api.call_args.args[2]
        self.assertEqual(body["sha"],"a"*40)
        self.assertEqual(body["branch"],"main")
        self.assertTrue(body["message"].startswith("chore(alerts):"))

    def test_fail_closed_corrupt_kaggle_ledger_history(self):
        for field, value in (
            ("scope","medical_records"),
            ("kaggle_events", {}),
            ("github_events",["github:1234567:1:success"]*2),
        ):
            with self.subTest(field=field):
                data=ledger()
                data[field]=value
                with self.assertRaises(AlertError):
                    validate_ledger(data)

    @patch("universal_telegram_alerts.verify_cpu")
    def test_cpu_b_distinguishes_session_completion_from_100_tasks(self, verify):
        verify.return_value={"completed_task_ids":list(range(80)),"status":"in_progress"}
        result=check_checkpoint("cpu_b",Path("fake"),"a"*40)
        self.assertIn("80/100",result)
        self.assertIn("session",result)
        verify.return_value={"completed_task_ids":list(range(100)),"status":"completed"}
        with self.assertRaises(AlertError):
            check_checkpoint("cpu_b",Path("fake"),"a"*40)


if __name__=="__main__":
    unittest.main()
