"""Read-only Telegram bot diagnostics without printing identifying information.

No update offsets are advanced and no messages are sent.
Never print the bot token, bot username, chat IDs or Telegram response bodies.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request


def request(token: str, method: str):
    req = urllib.request.Request(
        f"https://api.telegram.org/bot{token}/{method}",
        data=urllib.parse.urlencode({}).encode("utf-8"),
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=12) as response:
            data = json.load(response)
    except urllib.error.HTTPError as exc:
        # Do not print the HTTPError: it contains a URL with the secret token.
        raise RuntimeError(f"HTTP_STATUS_{exc.code}") from None
    except (urllib.error.URLError, TimeoutError):
        raise RuntimeError("NETWORK_OR_TIMEOUT_ERROR") from None
    except (json.JSONDecodeError, ValueError):
        raise RuntimeError("UNEXPECTED_API_JSON") from None
    if not isinstance(data, dict) or data.get("ok") is not True:
        raise RuntimeError("UNEXPECTED_API_RESULT")
    return data.get("result")


def inspect(token: str) -> int:
    try:
        user = request(token, "getMe")
        if not isinstance(user, dict) or user.get("is_bot") is not True:
            raise RuntimeError("NOT_A_BOT_ACCOUNT")
        print("BOT_AUTHENTICATION: OK")
    except RuntimeError as exc:
        print(f"BOT_AUTHENTICATION: FAILED ({exc})")
        return 1

    try:
        webhook = request(token, "getWebhookInfo")
        if not isinstance(webhook, dict):
            raise RuntimeError("INVALID_WEBHOOK_STATUS")
        active = bool(webhook.get("url"))
        print("WEBHOOK_CONFIGURED:", "YES" if active else "NO")
        if active:
            print("DIAGNOSIS: Existing webhook prevents getUpdates polling. No settings changed.")
            return 0
    except RuntimeError as exc:
        print(f"WEBHOOK_STATUS: FAILED ({exc})")
        return 1

    try:
        updates = request(token, "getUpdates")
        if not isinstance(updates, list):
            raise RuntimeError("INVALID_UPDATE_LIST")
        unique = set()
        messages = 0
        starts = 0
        for entry in updates:
            if not isinstance(entry, dict):
                continue
            msg = entry.get("message")
            if not isinstance(msg, dict):
                continue
            messages += 1
            chat = msg.get("chat", {})
            sender = msg.get("from", {})
            if (
                isinstance(chat, dict)
                and isinstance(sender, dict)
                and chat.get("type") == "private"
                and type(chat.get("id")) is int
                and type(sender.get("id")) is int
                and chat.get("id") == sender.get("id")
                and sender.get("is_bot") is False
                and isinstance(msg.get("text"), str)
                and msg.get("text").strip().split(" ", 1)[0] in ("/start",)
            ):
                unique.add(chat["id"])
                starts += 1
        print("PENDING_UPDATES_COUNT:", len(updates))
        print("MESSAGE_UPDATES_COUNT:", messages)
        print("PRIVATE_START_MESSAGES_COUNT:", starts)
        print("DISTINCT_PRIVATE_START_SENDERS:", len(unique))
        if not updates:
            print("DIAGNOSIS: No pending updates; send a fresh /start in bot private chat.")
        elif not unique:
            print("DIAGNOSIS: Updates exist but no valid private /start; send a fresh /start.")
        elif len(unique) > 1:
            print("DIAGNOSIS: Multiple /start senders; ambiguous. No Chat ID delivery.")
        else:
            print("DIAGNOSIS: Exactly one private /start candidate; safe delivery is possible.")
        return 0
    except RuntimeError as exc:
        print(f"GET_UPDATES: FAILED ({exc})")
        return 1


if __name__ == "__main__":
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    if not token:
        raise SystemExit("BOT_AUTHENTICATION: Missing repository secret")
    raise SystemExit(inspect(token))
