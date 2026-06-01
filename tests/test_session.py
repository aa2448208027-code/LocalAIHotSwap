from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from hotmodel.session import SessionStore


class SessionStoreTests(unittest.TestCase):
    def test_max_session_messages_keeps_recent_history(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            store = SessionStore(Path(raw) / "state.json", "preset", max_session_messages=3)

            store.append_messages(
                "s",
                [
                    {"role": "user", "content": "one"},
                    {"role": "assistant", "content": "two"},
                    {"role": "user", "content": "three"},
                    {"role": "assistant", "content": "four"},
                ],
            )

            session = store.get_or_create("s")
            self.assertEqual(
                session.messages,
                [
                    {"role": "assistant", "content": "two"},
                    {"role": "user", "content": "three"},
                    {"role": "assistant", "content": "four"},
                ],
            )

    def test_prompt_budget_keeps_system_and_incoming(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            store = SessionStore(Path(raw) / "state.json", "system", max_prompt_chars=45)
            store.append_messages(
                "s",
                [
                    {"role": "user", "content": "old message that should drop"},
                    {"role": "assistant", "content": "recent"},
                ],
            )

            messages = store.build_prompt_messages("s", [{"role": "user", "content": "new"}])

            self.assertEqual(
                messages,
                [
                    {"role": "system", "content": "system"},
                    {"role": "assistant", "content": "recent"},
                    {"role": "user", "content": "new"},
                ],
            )


if __name__ == "__main__":
    unittest.main()
