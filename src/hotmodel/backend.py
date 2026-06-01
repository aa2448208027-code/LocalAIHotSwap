from __future__ import annotations

from typing import Any, Iterator
import json
import urllib.parse
import urllib.error
import urllib.request


def post_json(base_url: str, path: str, payload: dict[str, Any], timeout_seconds: float = 600) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}{path}",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"llama-server returned HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"llama-server request failed: {exc}") from exc


def post_json_stream(
    base_url: str,
    path: str,
    payload: dict[str, Any],
    timeout_seconds: float = 600,
) -> Iterator[bytes]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}{path}",
        data=body,
        headers={
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            for line in response:
                yield line
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"llama-server returned HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"llama-server request failed: {exc}") from exc


def get_json(base_url: str, path: str, timeout_seconds: float = 30) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(f"{base_url.rstrip('/')}{path}", timeout=timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"llama-server returned HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"llama-server request failed: {exc}") from exc


class LlamaHttpBackend:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    def chat_completions(self, payload: dict[str, Any], timeout_seconds: float = 600) -> dict[str, Any]:
        return post_json(self.base_url, "/v1/chat/completions", payload, timeout_seconds)

    def chat_completions_stream(self, payload: dict[str, Any], timeout_seconds: float = 600) -> Iterator[bytes]:
        return post_json_stream(self.base_url, "/v1/chat/completions", payload, timeout_seconds)

    def count_chat_tokens(
        self,
        model: str,
        messages: list[dict[str, Any]],
        timeout_seconds: float = 60,
    ) -> int:
        prompt = self.apply_template(model, messages, timeout_seconds=timeout_seconds)
        tokens = self.tokenize(model, prompt, timeout_seconds=timeout_seconds)
        return len(tokens)

    def apply_template(
        self,
        model: str,
        messages: list[dict[str, Any]],
        timeout_seconds: float = 60,
    ) -> str:
        payload = {"model": model, "messages": messages}
        encoded_model = urllib.parse.quote(model, safe="")
        for path in (f"/apply-template?model={encoded_model}", "/apply-template"):
            try:
                data = post_json(self.base_url, path, payload, timeout_seconds)
                prompt = data.get("prompt")
                if isinstance(prompt, str):
                    return prompt
            except RuntimeError:
                continue
        raise RuntimeError("llama-server apply-template request failed")

    def tokenize(self, model: str, content: str, timeout_seconds: float = 60) -> list[Any]:
        payload = {
            "model": model,
            "content": content,
            "add_special": False,
            "parse_special": True,
            "with_pieces": False,
        }
        encoded_model = urllib.parse.quote(model, safe="")
        for path in (f"/tokenize?model={encoded_model}", "/tokenize"):
            try:
                data = post_json(self.base_url, path, payload, timeout_seconds)
                tokens = data.get("tokens")
                if isinstance(tokens, list):
                    return tokens
            except RuntimeError:
                continue
        raise RuntimeError("llama-server tokenize request failed")
