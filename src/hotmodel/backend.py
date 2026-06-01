from __future__ import annotations

from typing import Any
import json
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
