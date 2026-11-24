"""No network calls; tests private Telegram Chat ID extraction."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from telegram_chat_id import unique_private_start_chat_id


def private_start(chat_id=12345):
    return {
        "update_id": 42,
        "message": {
            "text": "/start",
            "chat": {"type": "private", "id": chat_id},
            "from": {"id": chat_id, "is_bot": False},
        },
    }


class TelegramChatIdTests(unittest.TestCase):
    def test_one_private_start(self):
        self.assertEqual(unique_private_start_chat_id([private_start()]), 12345)

    def test_duplicate_updates_from_same_sender(self):
        self.assertEqual(
            unique_private_start_chat_id([private_start(), private_start()]), 12345
        )

    def test_multiple_private_chats_refused(self):
        with self.assertRaises(ValueError):
            unique_private_start_chat_id([private_start(12345), private_start(99999)])

    def test_group_chat_and_other_text_are_ignored(self):
        group = private_start()
        group["message"]["chat"]["type"] = "group"
        with self.assertRaises(ValueError):
            unique_private_start_chat_id([group])
        with self.assertRaises(ValueError):
            unique_private_start_chat_id([{"message": {"text": "hello"}}])

    def test_spoofed_sender_and_bot_blocked(self):
        spoofed = private_start()
        spoofed["message"]["from"]["id"] = 22222
        with self.assertRaises(ValueError):
            unique_private_start_chat_id([spoofed])
        bot = private_start()
        bot["message"]["from"]["is_bot"] = True
        with self.assertRaises(ValueError):
            unique_private_start_chat_id([bot])

    def test_malformed_updates_refused(self):
        with self.assertRaises(ValueError):
            unique_private_start_chat_id({"result": []})


if __name__ == "__main__":
    unittest.main()
