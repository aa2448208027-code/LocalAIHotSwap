from __future__ import annotations

from typing import Any
import unittest
from unittest import mock

from hotmodel.backend import LlamaHttpBackend


class LlamaHttpBackendTests(unittest.TestCase):
    def test_count_chat_tokens_uses_template_then_tokenize(self) -> None:
        calls: list[tuple[str, dict[str, Any]]] = []

        def fake_post_json(base_url: str, path: str, payload: dict[str, Any], timeout_seconds: float) -> dict[str, Any]:
            calls.append((path, payload))
            if path.startswith("/apply-template"):
                return {"prompt": "rendered prompt"}
            if path.startswith("/tokenize"):
                return {"tokens": [1, 2, 3]}
            raise AssertionError(path)

        backend = LlamaHttpBackend("http://127.0.0.1:28000")
        with mock.patch("hotmodel.backend.post_json", fake_post_json):
            count = backend.count_chat_tokens("qwen3-small", [{"role": "user", "content": "hello"}])

        self.assertEqual(count, 3)
        self.assertEqual(calls[0][0], "/apply-template?model=qwen3-small")
        self.assertEqual(calls[0][1]["model"], "qwen3-small")
        self.assertEqual(calls[1][0], "/tokenize?model=qwen3-small")
        self.assertEqual(calls[1][1]["content"], "rendered prompt")


if __name__ == "__main__":
    unittest.main()
