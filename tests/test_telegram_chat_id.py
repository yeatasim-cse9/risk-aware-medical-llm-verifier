"""Offline tests for Telegram private-delivery guardrails; no real Telegram API."""
import io
import sys
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from telegram_chat_id import (
    DeliveryError, deliver_private_id, main, private_start_chat_id,
    telegram_request, unique_private_start_chat_id,
)


def start(chat_id=12345):
    return {
        "update_id": 42,
        "message": {
            "text": "/start",
            "chat": {"type": "private", "id": chat_id},
            "from": {"id": chat_id, "is_bot": False},
        },
    }


class PrivateSelectionTests(unittest.TestCase):
    def test_valid_private_start(self):
        self.assertEqual(private_start_chat_id([start()]), 12345)
        self.assertEqual(unique_private_start_chat_id([start()]), 12345)

    def test_duplicates_one_sender(self):
        self.assertEqual(private_start_chat_id([start(), start()]), 12345)

    def test_empty_queue_reports_specific_code(self):
        with self.assertRaisesRegex(DeliveryError, "NO_PENDING_UPDATES"):
            private_start_chat_id([])

    def test_multiple_senders_fail_closed(self):
        with self.assertRaisesRegex(DeliveryError, "AMBIGUOUS_RECIPIENT"):
            private_start_chat_id([start(12345), start(99999)])

    def test_no_valid_private_start(self):
        group = start()
        group["message"]["chat"]["type"] = "group"
        for value in ([group], [{"message": {"text": "hello"}}]):
            with self.subTest(value=value):
                with self.assertRaisesRegex(DeliveryError, "NO_PRIVATE_START"):
                    private_start_chat_id(value)

    def test_spoofed_sender_and_bot_rejected(self):
        spoofed = start()
        spoofed["message"]["from"]["id"] = 3333
        bot = start()
        bot["message"]["from"]["is_bot"] = True
        for value in (spoofed, bot):
            with self.assertRaisesRegex(DeliveryError, "NO_PRIVATE_START"):
                private_start_chat_id([value])

    def test_invalid_format(self):
        with self.assertRaisesRegex(DeliveryError, "INVALID_UPDATE_FORMAT"):
            private_start_chat_id({})
        with self.assertRaisesRegex(DeliveryError, "INVALID_UPDATE_FORMAT"):
            private_start_chat_id([None])

    @patch("telegram_chat_id.telegram_request")
    def test_success_sends_id_only_to_selected_private_chat(self, req):
        req.side_effect = [
            {"is_bot": True}, {"url": ""}, [start()],
            {"chat": {"id": 12345}},
        ]
        deliver_private_id("SECRET_NOT_FOR_LOGS")
        self.assertEqual(req.call_count, 4)
        self.assertEqual(req.call_args.args[1], "sendMessage")
        self.assertEqual(req.call_args.args[2]["chat_id"], "12345")

    @patch("telegram_chat_id.telegram_request")
    def test_webhook_prevents_update_poll(self, req):
        req.side_effect = [{"is_bot": True}, {"url": "https://example.org/hook"}]
        with self.assertRaisesRegex(DeliveryError, "WEBHOOK_ACTIVE"):
            deliver_private_id("dummy")
        self.assertEqual(req.call_count, 2)

    @patch("telegram_chat_id.telegram_request")
    def test_no_updates_never_send(self, req):
        req.side_effect = [{"is_bot": True}, {"url": ""}, []]
        with self.assertRaisesRegex(DeliveryError, "NO_PENDING_UPDATES"):
            deliver_private_id("dummy")
        self.assertEqual(req.call_count, 3)

    @patch("telegram_chat_id.telegram_request")
    def test_failed_delivery_detected(self, req):
        req.side_effect = [
            {"is_bot": True}, {"url": ""}, [start()],
            {"chat": {"id": 11111}},
        ]
        with self.assertRaisesRegex(DeliveryError, "MESSAGE_DELIVERY_NOT_CONFIRMED"):
            deliver_private_id("dummy")


class SafeErrorHandlingTests(unittest.TestCase):
    @patch("telegram_chat_id.urllib.request.urlopen")
    def test_token_never_leaks_from_http_errors(self, openurl):
        openurl.side_effect = urllib.error.HTTPError(
            "https://api.telegram.org/botVERY_SECRET_TOKEN/getUpdates",
            401, "Unauthorized", hdrs=None, fp=None
        )
        with self.assertRaisesRegex(DeliveryError, "INVALID_OR_REVOKED_BOT_TOKEN") as got:
            telegram_request("VERY_SECRET_TOKEN", "getUpdates", {})
        self.assertNotIn("VERY_SECRET_TOKEN", str(got.exception))

    @patch("telegram_chat_id.deliver_private_id")
    @patch.dict("telegram_chat_id.os.environ", {"TELEGRAM_BOT_TOKEN": "TEST_TOKEN"})
    def test_main_prints_safe_code_only(self, delivery):
        delivery.side_effect = DeliveryError("NO_PENDING_UPDATES")
        err = io.StringIO()
        with patch("sys.stderr", err):
            self.assertEqual(main(), 1)
        self.assertIn("NO_PENDING_UPDATES", err.getvalue())
        self.assertNotIn("TEST_TOKEN", err.getvalue())


if __name__ == "__main__":
    unittest.main()
