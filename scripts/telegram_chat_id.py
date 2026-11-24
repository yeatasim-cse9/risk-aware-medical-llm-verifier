"""Privately deliver a Telegram chat ID without leaking bot tokens or chat IDs.

Does not expose HTTP URLs, raw Telegram responses, sender identities or tokens
in GitHub Actions logs. Read-only discovery; only send to a unique private
human sender who issued /start. Does not delete/modify webhooks or offsets.
"""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request


class DeliveryError(Exception):
    """A controlled public error code; never contains secret values."""


def private_start_chat_id(updates: object) -> int:
    if not isinstance(updates, list):
        raise DeliveryError("INVALID_UPDATE_FORMAT")
    if not updates:
        raise DeliveryError("NO_PENDING_UPDATES: Send a new /start to the intended bot.")
    ids: set[int] = set()
    for update in updates:
        if not isinstance(update, dict):
            raise DeliveryError("INVALID_UPDATE_FORMAT")
        message = update.get("message")
        if not isinstance(message, dict):
            continue
        command = message.get("text")
        if not isinstance(command, str) or command.strip() != "/start":
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
    if not ids:
        raise DeliveryError(
            "NO_PRIVATE_START: Send a new /start in the intended bot's private chat."
        )
    if len(ids) != 1:
        raise DeliveryError(
            "AMBIGUOUS_RECIPIENT: More than one private /start sender; "
            "refusing to choose a recipient."
        )
    return next(iter(ids))


def unique_private_start_chat_id(updates: object) -> int:
    """Backwards-compatible selection helper for tests and callers."""
    return private_start_chat_id(updates)


def telegram_request(token: str, method: str, payload: dict[str, str]):
    req = urllib.request.Request(
        f"https://api.telegram.org/bot{token}/{method}",
        data=urllib.parse.urlencode(payload).encode("utf-8"),
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as response:
            data = json.load(response)
    except urllib.error.HTTPError as exc:
        category = {
            400: "BAD_REQUEST",
            401: "INVALID_OR_REVOKED_BOT_TOKEN",
            403: "BOT_BLOCKED_OR_NOT_ALLOWED",
            409: "WEBHOOK_OR_POLLING_CONFLICT",
            429: "TELEGRAM_RATE_LIMIT",
        }.get(exc.code, "REMOTE_HTTP_FAILURE")
        # An HTTPError can contain a URL with the token. NEVER print it.
        raise DeliveryError(category) from None
    except (urllib.error.URLError, TimeoutError, OSError):
        raise DeliveryError("NETWORK_OR_TIMEOUT_FAILURE") from None
    except (ValueError, TypeError):
        raise DeliveryError("INVALID_TELEGRAM_JSON") from None
    if not isinstance(data, dict) or data.get("ok") is not True:
        raise DeliveryError("UNEXPECTED_TELEGRAM_API_RESULT")
    return data.get("result")


def deliver_private_id(token: str) -> None:
    me = telegram_request(token, "getMe", {})
    if not isinstance(me, dict) or me.get("is_bot") is not True:
        raise DeliveryError("TOKEN_NOT_A_BOT_ACCOUNT")
    webhook = telegram_request(token, "getWebhookInfo", {})
    if not isinstance(webhook, dict):
        raise DeliveryError("INVALID_WEBHOOK_INFO")
    if webhook.get("url"):
        raise DeliveryError(
            "WEBHOOK_ACTIVE: Existing webhook must be reviewed before getUpdates."
        )
    # No offset is supplied: this does not acknowledge/consume existing updates.
    # Do not filter allowed_updates here, since prior bot configuration might
    # differ. Fetch most recent 100 only.
    updates = telegram_request(
        token, "getUpdates", {"limit": "100", "timeout": "0"}
    )
    chat_id = private_start_chat_id(updates)
    result = telegram_request(
        token,
        "sendMessage",
        {
            "chat_id": str(chat_id),
            "text": (
                "Medical LLM Research Monitor\n"
                f"Your Telegram Chat ID: {chat_id}\n\n"
                "Save this number privately as GitHub Actions repository "
                "secret TELEGRAM_CHAT_ID. Do not post it in a public repo "
                "or this chat."
            ),
        },
    )
    if (
        not isinstance(result, dict)
        or not isinstance(result.get("chat"), dict)
        or result["chat"].get("id") != chat_id
    ):
        raise DeliveryError("MESSAGE_DELIVERY_NOT_CONFIRMED")


def main() -> int:
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    if not token:
        print("TELEGRAM_ERROR: MISSING_BOT_TOKEN_SECRET", file=sys.stderr)
        return 1
    try:
        deliver_private_id(token)
    except DeliveryError as exc:
        print(f"TELEGRAM_ERROR: {exc}", file=sys.stderr)
        return 1
    except Exception:
        # Never print unknown exception text; it might contain a token URL.
        print("TELEGRAM_ERROR: UNEXPECTED_PRIVATE_DELIVERY_FAILURE", file=sys.stderr)
        return 1
    print("TELEGRAM PRIVATE CHAT ID DELIVERED SUCCESSFULLY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
