from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

try:
    from fastapi.testclient import TestClient
except RuntimeError:
    TestClient = None
except ImportError:
    TestClient = None

from hotmodel.api import create_app
from hotmodel.config import ModelSpec, RuntimeConfig


@unittest.skipIf(TestClient is None, "FastAPI TestClient dependencies are not installed")
class ApiIntegrationTests(unittest.TestCase):
    def test_http_routes_parse_json_bodies(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            config = _config(Path(raw))

            with TestClient(create_app(config)) as client:
                health = client.get("/health")
                models = client.get("/v1/models")
                unknown_switch = client.post("/admin/switch", json={"model": "missing"})
                chat = client.post(
                    "/v1/chat/completions",
                    json={"messages": [{"role": "user", "content": "hi"}]},
                )
                stream = client.post(
                    "/v1/chat/completions",
                    json={"stream": True, "messages": [{"role": "user", "content": "hi"}]},
                )

            self.assertEqual(health.status_code, 200)
            self.assertEqual(models.status_code, 200)
            self.assertEqual(models.json()["data"][0]["id"], "small")
            self.assertEqual(unknown_switch.status_code, 404)
            self.assertEqual(chat.status_code, 503)
            self.assertEqual(stream.status_code, 503)


def _config(root: Path) -> RuntimeConfig:
    return RuntimeConfig(
        host="127.0.0.1",
        port=18080,
        state_path=root / "state.json",
        active_model=None,
        switch_policy="zero_overlap",
        backend_mode="process",
        switch_drain_timeout_seconds=1,
        gpu_settle_timeout_seconds=0,
        gpu_settle_memory_mb=None,
        system_prompt="preset",
        max_session_messages=None,
        max_prompt_chars=None,
        max_prompt_tokens=None,
        token_budget_mode="auto",
        token_budget_chars_per_token=4.0,
        router=None,
        models={"small": ModelSpec(name="small", path=root / "small.gguf", port=28080)},
    )


if __name__ == "__main__":
    unittest.main()
