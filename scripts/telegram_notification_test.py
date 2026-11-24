"""Private, one-time Telegram delivery test.

This script does not log bot tokens, chat IDs, Telegram response bodies or
HTTP exceptions (exception URLs can contain the bot token).
"""
from __future__ import annotations

import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request


class NotificationError(Exception):
    """Safe diagnostic message with no private values."""


def send_test(token: str, chat_id: str) -> None:
    if not token:
        raise NotificationError("MISSING_BOT_TOKEN")
    if not re.fullmatch(r"[1-9][0-9]{0,19}", chat_id):
        raise NotificationError("INVALID_PRIVATE_CHAT_ID")
    message = (
        "✅ Medical LLM Research Automation\n"
        "Test notification delivered from GitHub Actions.\n"
        "No medical data was sent."
    )
    body = urllib.parse.urlencode({
        "chat_id": chat_id,
        "text": message,
        "disable_notification": "false",
    }).encode("utf-8")
    request = urllib.request.Request(
        f"https://api.telegram.org/bot{token}/sendMessage",
        data=body,
        method="POST",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            if response.status != 200:
                raise NotificationError("TELEGRAM_NON_200_RESPONSE")
            answer = json.load(response)
    except urllib.error.HTTPError as exc:
        error_code = {
            400: "BAD_REQUEST_CHECK_CHAT_ID",
            401: "INVALID_OR_REVOKED_BOT_TOKEN",
            403: "BOT_BLOCKED_OR_NOT_STARTED",
            429: "TELEGRAM_RATE_LIMIT",
        }.get(exc.code, "TELEGRAM_HTTP_FAILURE")
        raise NotificationError(error_code) from None
    except (urllib.error.URLError, OSError, TimeoutError):
        raise NotificationError("TELEGRAM_NETWORK_OR_TIMEOUT_FAILURE") from None
    except (json.JSONDecodeError, ValueError):
        raise NotificationError("INVALID_TELEGRAM_RESPONSE") from None

    if not isinstance(answer, dict) or answer.get("ok") is not True:
        raise NotificationError("TELEGRAM_REJECTED_MESSAGE")
    result = answer.get("result")
    if not isinstance(result, dict):
        raise NotificationError("MESSAGE_CONFIRMATION_MISSING")
    chat = result.get("chat")
    if (
        not isinstance(chat, dict)
        or chat.get("type") != "private"
        or type(chat.get("id")) is not int
        or str(chat["id"]) != chat_id
        or type(result.get("message_id")) is not int
    ):
        raise NotificationError("PRIVATE_RECIPIENT_CONFIRMATION_FAILED")


def main() -> int:
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    try:
        send_test(token, chat_id)
    except NotificationError as exc:
        print(f"TELEGRAM_TEST_FAILED: {exc}", file=sys.stderr)
        return 1
    except Exception:
        # Unknown exceptions can contain bot tokens in request URLs.
        print("TELEGRAM_TEST_FAILED: UNEXPECTED_ERROR", file=sys.stderr)
        return 1
    print("TELEGRAM TEST NOTIFICATION DELIVERED SUCCESSFULLY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
