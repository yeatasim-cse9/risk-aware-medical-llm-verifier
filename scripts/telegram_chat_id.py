"""Safely deliver a newly created Telegram bot's private chat ID to that chat.

Does not log token, chat ID, Telegram response content, or request errors.
Only an exact /start from one unique private human chat is accepted.
"""
from __future__ import annotations

import json
import os
import sys
import urllib.parse
import urllib.request


def unique_private_start_chat_id(updates: object) -> int:
    """Require one distinct private human /start sender in bot updates."""
    if not isinstance(updates, list):
        raise ValueError("Unexpected Telegram updates structure")
    ids: set[int] = set()
    for update in updates:
        if not isinstance(update, dict):
            raise ValueError("Unexpected update object")
        message = update.get("message")
        if not isinstance(message, dict) or message.get("text") != "/start":
            continue
        chat = message.get("chat")
        sender = message.get("from")
        if not isinstance(chat, dict) or not isinstance(sender, dict):
            continue
        chat_id = chat.get("id")
        sender_id = sender.get("id")
        if (
            chat.get("type") == "private"
            and type(chat_id) is int
            and chat_id > 0
            and type(sender_id) is int
            and sender_id == chat_id
            and sender.get("is_bot") is False
        ):
            ids.add(chat_id)
    if len(ids) != 1:
        raise ValueError("Expected exactly one private /start sender")
    return next(iter(ids))


def call_telegram(token: str, method: str, data: dict[str, str]) -> object:
    """Use POST requests; never propagate HTTP exceptions with token URLs."""
    payload = urllib.parse.urlencode(data).encode("utf-8")
    request = urllib.request.Request(
        f"https://api.telegram.org/bot{token}/{method}",
        data=payload,
        method="POST",
    )
    # Caller catches exceptions without printing URL, token, or response body.
    with urllib.request.urlopen(request, timeout=15) as response:
        answer = json.load(response)
    if not isinstance(answer, dict) or answer.get("ok") is not True:
        raise ValueError("Unexpected Telegram API response")
    return answer.get("result")


def main() -> int:
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    if not token:
        print("ERROR: TELEGRAM_BOT_TOKEN secret is missing.", file=sys.stderr)
        return 1

    try:
        updates = call_telegram(
            token, "getUpdates",
            {"limit": "100", "timeout": "0", "allowed_updates": '["message"]'},
        )
        chat_id = unique_private_start_chat_id(updates)
        sent = call_telegram(
            token, "sendMessage",
            {
                "chat_id": str(chat_id),
                "text": (
                    "Medical LLM Research Monitor: Your private Telegram Chat ID is "
                    f"{chat_id}.\n\n"
                    "Store this number as the GitHub Actions repository secret "
                    "TELEGRAM_CHAT_ID. Do not paste it into a public GitHub file."
                ),
            },
        )
        if (
            not isinstance(sent, dict)
            or not isinstance(sent.get("chat"), dict)
            or sent["chat"].get("id") != chat_id
        ):
            raise ValueError("Telegram delivery was not confirmed")
    except Exception:
        # urllib exceptions can contain bot token in their request URL.
        # Intentionally hide all exception text and remote response bodies.
        print(
            "ERROR: Chat ID delivery could not be verified. Ensure the bot has "
            "received exactly one private /start and has no active webhook. "
            "Do not share any credentials or identifiers in logs.",
            file=sys.stderr,
        )
        return 1

    print("TELEGRAM PRIVATE CHAT ID DELIVERED SUCCESSFULLY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
