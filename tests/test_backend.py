from __future__ import annotations

from typing import Any
import unittest
from unittest import mock

from hotmodel.backend import LlamaHttpBackend, post_json_stream


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

    def test_post_json_stream_yields_raw_sse_lines(self) -> None:
        class FakeResponse:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def __iter__(self):
                yield b"data: one\n\n"
                yield b"data: [DONE]\n\n"

        with mock.patch("urllib.request.urlopen", return_value=FakeResponse()):
            lines = list(post_json_stream("http://127.0.0.1:28000", "/v1/chat/completions", {"stream": True}))

        self.assertEqual(lines, [b"data: one\n\n", b"data: [DONE]\n\n"])


if __name__ == "__main__":
    unittest.main()
