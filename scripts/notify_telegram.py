"""Send optional status-only Telegram notification, never patient data."""
import os
import sys
import urllib.parse
import urllib.request


def notify(status: str) -> None:
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat = os.environ.get("TELEGRAM_CHAT_ID", "")
    if not token or not chat:
        print("Telegram notification skipped (secrets not configured)")
        return
    if status not in {"prepared", "submitted", "checked", "collected", "failed"}:
        raise ValueError("Unexpected notification status")
    msg = f"Medical LLM CPU PILOT: {status}. No medical data involved."
    encoded = urllib.parse.urlencode({"chat_id": chat, "text": msg}).encode()
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    request = urllib.request.Request(url, data=encoded, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            if response.status != 200:
                raise RuntimeError("Unexpected Telegram HTTP status")
        print("Telegram status notification sent")
    except Exception:
        # The bot token appears in the HTTP URL; never print exception text.
        print("Telegram status notification failed (details hidden)", file=sys.stderr)


if __name__ == "__main__":
    notify(sys.argv[1])
