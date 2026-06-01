from __future__ import annotations

from pathlib import Path
from typing import Any
import tempfile
import threading
import time
import unittest

from hotmodel.config import ModelSpec, RouterSpec, RuntimeConfig
from hotmodel.orchestrator import Orchestrator
from hotmodel.session import SessionStore


class FakeProcess:
    events: list[str] = []

    def __init__(self, model: ModelSpec) -> None:
        self.model = model
        self.running = False

    def start(self) -> None:
        FakeProcess.events.append(f"start:{self.model.name}")
        self.running = True

    def stop(self) -> None:
        FakeProcess.events.append(f"stop:{self.model.name}")
        self.running = False

    def is_running(self) -> bool:
        return self.running

    def wait_ready(self, timeout_seconds: float) -> bool:
        return self.running


class FakeChatBackend:
    calls: list[dict[str, Any]] = []

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url

    def chat_completions(self, payload: dict[str, Any], timeout_seconds: float = 600) -> dict[str, Any]:
        FakeChatBackend.calls.append({"base_url": self.base_url, "payload": payload})
        return {
            "id": "fake",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": f"from {payload['model']}"},
                    "finish_reason": "stop",
                }
            ],
        }


class FakeRouter:
    def __init__(self, release_load: threading.Event | None = None, entered_load: threading.Event | None = None) -> None:
        self.events: list[str] = []
        self.running = False
        self.base_url = "http://127.0.0.1:28000"
        self.release_load = release_load
        self.entered_load = entered_load

    def start(self) -> None:
        if not self.running:
            self.events.append("router:start")
            self.running = True

    def stop(self) -> None:
        self.events.append("router:stop")
        self.running = False

    def is_running(self) -> bool:
        return self.running

    def load_model(self, model: ModelSpec) -> None:
        self.events.append(f"load:{model.llama_model_id}")
        if self.entered_load is not None:
            self.entered_load.set()
        if self.release_load is not None:
            self.release_load.wait(timeout=5)

    def unload_model(self, model: ModelSpec) -> None:
        self.events.append(f"unload:{model.llama_model_id}")


class OrchestratorTests(unittest.TestCase):
    def test_switch_stops_previous_before_starting_next(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            tmp_path = Path(raw)
            FakeProcess.events = []
            orchestrator = Orchestrator(
                _config(tmp_path),
                sessions=SessionStore(tmp_path / "state.json", "preset"),
                process_factory=lambda model: FakeProcess(model),
                chat_backend_factory=lambda base_url: FakeChatBackend(base_url),
            )

            orchestrator.switch_model("small")
            orchestrator.switch_model("tiny")

            self.assertEqual(FakeProcess.events, ["start:small", "stop:small", "start:tiny"])
            self.assertEqual(orchestrator.state()["active_model"], "tiny")
            self.assertIs(orchestrator.state()["process_running"], True)

    def test_context_and_preset_survive_model_switch(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            tmp_path = Path(raw)
            FakeChatBackend.calls = []
            orchestrator = Orchestrator(
                _config(tmp_path),
                sessions=SessionStore(tmp_path / "state.json", "fixed system prompt"),
                process_factory=lambda model: FakeProcess(model),
                chat_backend_factory=lambda base_url: FakeChatBackend(base_url),
            )

            orchestrator.switch_model("small")
            first = orchestrator.chat(
                "session-a",
                [{"role": "user", "content": "remember alpha"}],
                {"temperature": 0.1},
            )
            orchestrator.switch_model("tiny")
            second = orchestrator.chat(
                first["hotmodel"]["session_id"],
                [{"role": "user", "content": "what did I ask?"}],
                {"temperature": 0.1},
            )

            self.assertEqual(second["hotmodel"]["active_model"], "tiny")
            messages = FakeChatBackend.calls[-1]["payload"]["messages"]
            self.assertEqual(
                messages,
                [
                    {"role": "system", "content": "fixed system prompt"},
                    {"role": "user", "content": "remember alpha"},
                    {"role": "assistant", "content": "from small"},
                    {"role": "user", "content": "what did I ask?"},
                ],
            )

    def test_chat_is_rejected_without_active_model(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            tmp_path = Path(raw)
            orchestrator = Orchestrator(
                _config(tmp_path),
                sessions=SessionStore(tmp_path / "state.json", "preset"),
                process_factory=lambda model: FakeProcess(model),
                chat_backend_factory=lambda base_url: FakeChatBackend(base_url),
            )

            with self.assertRaisesRegex(RuntimeError, "no active model"):
                orchestrator.chat("s", [{"role": "user", "content": "hi"}], {})

    def test_router_mode_uses_unload_load_and_router_model_id(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            tmp_path = Path(raw)
            FakeChatBackend.calls = []
            router = FakeRouter()
            orchestrator = Orchestrator(
                _router_config(tmp_path),
                sessions=SessionStore(tmp_path / "state.json", "preset"),
                chat_backend_factory=lambda base_url: FakeChatBackend(base_url),
            )
            orchestrator._router = router

            orchestrator.switch_model("small")
            orchestrator.switch_model("tiny")
            orchestrator.chat("s", [{"role": "user", "content": "hi"}], {})

            self.assertEqual(
                router.events,
                ["router:start", "load:router-small", "unload:router-small", "load:router-tiny"],
            )
            self.assertEqual(FakeChatBackend.calls[-1]["base_url"], router.base_url)
            self.assertEqual(FakeChatBackend.calls[-1]["payload"]["model"], "router-tiny")

    def test_chat_rejects_immediately_while_switch_loads(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            tmp_path = Path(raw)
            release_load = threading.Event()
            entered_load = threading.Event()
            router = FakeRouter(release_load=release_load, entered_load=entered_load)
            orchestrator = Orchestrator(
                _router_config(tmp_path),
                sessions=SessionStore(tmp_path / "state.json", "preset"),
                chat_backend_factory=lambda base_url: FakeChatBackend(base_url),
            )
            orchestrator._router = router

            switch_thread = threading.Thread(target=lambda: orchestrator.switch_model("small"))
            switch_thread.start()
            self.assertTrue(entered_load.wait(timeout=2))
            started = time.monotonic()
            with self.assertRaisesRegex(RuntimeError, "model switch in progress"):
                orchestrator.chat("s", [{"role": "user", "content": "hi"}], {})
            self.assertLess(time.monotonic() - started, 0.5)
            release_load.set()
            switch_thread.join(timeout=2)

    def test_repeated_router_switch_is_noop(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            tmp_path = Path(raw)
            router = FakeRouter()
            orchestrator = Orchestrator(
                _router_config(tmp_path),
                sessions=SessionStore(tmp_path / "state.json", "preset"),
                chat_backend_factory=lambda base_url: FakeChatBackend(base_url),
            )
            orchestrator._router = router

            orchestrator.switch_model("small")
            orchestrator.switch_model("small")

            self.assertEqual(router.events, ["router:start", "load:router-small"])

    def test_switch_waits_for_inflight_chat_before_unload(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            tmp_path = Path(raw)
            router = FakeRouter()
            release = threading.Event()
            entered = threading.Event()
            events: list[str] = []

            class SlowBackend:
                def __init__(self, base_url: str) -> None:
                    self.base_url = base_url

                def chat_completions(self, payload: dict[str, Any], timeout_seconds: float = 600) -> dict[str, Any]:
                    events.append("chat:start")
                    entered.set()
                    release.wait(timeout=5)
                    events.append("chat:end")
                    return {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}

            orchestrator = Orchestrator(
                _router_config(tmp_path),
                sessions=SessionStore(tmp_path / "state.json", "preset"),
                chat_backend_factory=lambda base_url: SlowBackend(base_url),
            )
            orchestrator._router = router
            original_unload = router.unload_model

            def recording_unload(model: ModelSpec) -> None:
                events.append(f"unload:{model.llama_model_id}")
                original_unload(model)

            router.unload_model = recording_unload
            orchestrator.switch_model("small")

            thread = threading.Thread(target=lambda: orchestrator.chat("s", [{"role": "user", "content": "hi"}], {}))
            thread.start()
            self.assertTrue(entered.wait(timeout=2))
            switch_thread = threading.Thread(target=lambda: orchestrator.switch_model("tiny"))
            switch_thread.start()
            time.sleep(0.05)
            self.assertNotIn("unload:router-small", events)
            release.set()
            thread.join(timeout=2)
            switch_thread.join(timeout=2)

            self.assertEqual(events, ["chat:start", "chat:end", "unload:router-small"])


def _config(tmp_path: Path) -> RuntimeConfig:
    return RuntimeConfig(
        host="127.0.0.1",
        port=18080,
        state_path=tmp_path / "state.json",
        active_model=None,
        switch_policy="zero_overlap",
        backend_mode="process",
        switch_drain_timeout_seconds=300,
        gpu_settle_timeout_seconds=0,
        gpu_settle_memory_mb=None,
        system_prompt="preset",
        max_session_messages=None,
        max_prompt_chars=None,
        router=None,
        models={
            "small": ModelSpec(name="small", path=tmp_path / "small.gguf", port=28080),
            "tiny": ModelSpec(name="tiny", path=tmp_path / "tiny.gguf", port=28081),
        },
    )


def _router_config(tmp_path: Path) -> RuntimeConfig:
    return RuntimeConfig(
        host="127.0.0.1",
        port=18080,
        state_path=tmp_path / "state.json",
        active_model=None,
        switch_policy="zero_overlap",
        backend_mode="router",
        switch_drain_timeout_seconds=300,
        gpu_settle_timeout_seconds=0,
        gpu_settle_memory_mb=None,
        system_prompt="preset",
        max_session_messages=None,
        max_prompt_chars=None,
        router=RouterSpec(port=28000, models_max=1, models_autoload=False),
        models={
            "small": ModelSpec(
                name="small",
                path=tmp_path / "small.gguf",
                port=28080,
                router_id="router-small",
            ),
            "tiny": ModelSpec(
                name="tiny",
                path=tmp_path / "tiny.gguf",
                port=28081,
                router_id="router-tiny",
            ),
        },
    )


if __name__ == "__main__":
    unittest.main()
