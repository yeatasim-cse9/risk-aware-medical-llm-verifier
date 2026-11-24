"""Tests for the private Telegram notification workflow (no network)."""
import io
import json
import sys
import unittest
import urllib.error
import urllib.parse
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from telegram_notification_test import NotificationError, main, send_test


class FakeResponse:
    status = 200

    def __init__(self, data):
        self.data = data

    def read(self, *args):
        return json.dumps(self.data).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class NotificationTests(unittest.TestCase):
    @patch("telegram_notification_test.urllib.request.urlopen")
    def test_private_delivery_confirms_correct_recipient(self, urlopen):
        urlopen.return_value = FakeResponse({
            "ok": True,
            "result": {"chat": {"id": 123456, "type": "private"}, "message_id": 90},
        })
        send_test("DUMMY:SECRET", "123456")
        request = urlopen.call_args.args[0]
        self.assertEqual(request.get_method(), "POST")
        params = urllib.parse.parse_qs(request.data.decode())
        self.assertEqual(params["chat_id"], ["123456"])
        self.assertIn("Test notification", params["text"][0])

    @patch("telegram_notification_test.urllib.request.urlopen")
    def test_rejects_different_recipient(self, urlopen):
        urlopen.return_value = FakeResponse({
            "ok": True,
            "result": {"chat": {"id": 999, "type": "private"}, "message_id": 10},
        })
        with self.assertRaisesRegex(NotificationError, "PRIVATE_RECIPIENT_CONFIRMATION"):
            send_test("DUMMY:SECRET", "123456")

    @patch("telegram_notification_test.urllib.request.urlopen")
    def test_rejects_group_chat(self, urlopen):
        urlopen.return_value = FakeResponse({
            "ok": True,
            "result": {"chat": {"id": 123456, "type": "group"}, "message_id": 10},
        })
        with self.assertRaises(NotificationError):
            send_test("DUMMY:SECRET", "123456")

    @patch("telegram_notification_test.urllib.request.urlopen")
    def test_http_error_does_not_expose_token(self, urlopen):
        urlopen.side_effect = urllib.error.HTTPError(
            "https://api.telegram.org/botSUPER_SECRET_TOKEN/sendMessage",
            401, "Unauthorized", None, None,
        )
        with self.assertRaises(NotificationError) as context:
            send_test("SUPER_SECRET_TOKEN", "123456")
        self.assertNotIn("SUPER_SECRET_TOKEN", str(context.exception))
        self.assertEqual(str(context.exception), "INVALID_OR_REVOKED_BOT_TOKEN")

    def test_missing_secrets_and_non_private_id(self):
        with self.assertRaisesRegex(NotificationError, "MISSING_BOT_TOKEN"):
            send_test("", "123456")
        for chat in ("-100123", "chat_name", "", " 123456", "123456\n"):
            with self.subTest(chat=chat):
                with self.assertRaises(NotificationError):
                    send_test("TOKEN", chat)

    @patch("telegram_notification_test.send_test")
    @patch.dict("telegram_notification_test.os.environ", {
        "TELEGRAM_BOT_TOKEN": "TEST_SECRET",
        "TELEGRAM_CHAT_ID": "123456",
    })
    def test_main_safe_failure_message(self, send):
        send.side_effect = NotificationError("BAD_REQUEST_CHECK_CHAT_ID")
        err = io.StringIO()
        with patch("sys.stderr", err):
            self.assertEqual(main(), 1)
        self.assertNotIn("TEST_SECRET", err.getvalue())
        self.assertNotIn("123456", err.getvalue())


if __name__ == "__main__":
    unittest.main()
