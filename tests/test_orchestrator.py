from __future__ import annotations

from pathlib import Path
from typing import Any
import tempfile
import threading
import time
import unittest

from hotmodel.config import ModelSpec, RouterSpec, RuntimeConfig
from hotmodel.gpu import GpuMemorySnapshot
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
    token_counts: list[dict[str, Any]] = []

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

    def chat_completions_stream(self, payload: dict[str, Any], timeout_seconds: float = 600):
        FakeChatBackend.calls.append({"base_url": self.base_url, "payload": payload})
        yield b'data: {"choices":[{"delta":{"content":"hel"}}]}\n\n'
        yield b'data: {"choices":[{"delta":{"content":"lo"}}]}\n\n'
        yield b"data: [DONE]\n\n"

    def count_chat_tokens(self, model: str, messages: list[dict[str, Any]]) -> int:
        FakeChatBackend.token_counts.append({"model": model, "messages": messages})
        return sum(len(str(message.get("content", ""))) for message in messages)


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


class FakeGpuProbe:
    def __init__(self, values: list[int | None]) -> None:
        self.values = values
        self.index = 0

    def snapshot(self) -> GpuMemorySnapshot:
        value = self.values[min(self.index, len(self.values) - 1)]
        self.index += 1
        per_gpu = [value] if value is not None else None
        return GpuMemorySnapshot(total_mb=value, per_gpu_mb=per_gpu, captured_at=float(self.index))

    def wait_until_below(self, threshold_mb: int, timeout_seconds: float) -> bool:
        return True


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

    def test_switch_report_includes_gpu_memory_snapshots(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            tmp_path = Path(raw)
            orchestrator = Orchestrator(
                _config(tmp_path),
                sessions=SessionStore(tmp_path / "state.json", "preset"),
                process_factory=lambda model: FakeProcess(model),
                chat_backend_factory=lambda base_url: FakeChatBackend(base_url),
                gpu_probe=FakeGpuProbe([900, 100, 80, 700]),
            )

            report = orchestrator.switch_model("small")

            self.assertEqual(report.gpu_memory["before_unload"]["total_mb"], 900)
            self.assertEqual(report.gpu_memory["after_unload"]["total_mb"], 100)
            self.assertEqual(report.gpu_memory["after_settle"]["total_mb"], 80)
            self.assertEqual(report.gpu_memory["after_load"]["total_mb"], 700)

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

    def test_token_budget_uses_backend_token_counter(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            tmp_path = Path(raw)
            FakeChatBackend.calls = []
            FakeChatBackend.token_counts = []
            config = _config(tmp_path, max_prompt_tokens=30, token_budget_mode="llama")
            orchestrator = Orchestrator(
                config,
                sessions=SessionStore(tmp_path / "state.json", "preset"),
                process_factory=lambda model: FakeProcess(model),
                chat_backend_factory=lambda base_url: FakeChatBackend(base_url),
            )

            orchestrator.switch_model("small")
            orchestrator.chat("s", [{"role": "user", "content": "old message that should be dropped"}], {})
            response = orchestrator.chat("s", [{"role": "user", "content": "new"}], {})

            self.assertGreater(len(FakeChatBackend.token_counts), 0)
            messages = FakeChatBackend.calls[-1]["payload"]["messages"]
            self.assertEqual(
                messages,
                [
                    {"role": "system", "content": "preset"},
                    {"role": "assistant", "content": "from small"},
                    {"role": "user", "content": "new"},
                ],
            )
            self.assertEqual(response["hotmodel"]["prompt_budget"]["unit"], "llama_tokens")
            self.assertEqual(response["hotmodel"]["prompt_budget"]["dropped_messages"], 1)

    def test_auto_token_budget_skips_backend_counter_when_estimate_fits(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            tmp_path = Path(raw)
            FakeChatBackend.calls = []
            FakeChatBackend.token_counts = []
            config = _config(tmp_path, max_prompt_tokens=4096, token_budget_mode="auto")
            orchestrator = Orchestrator(
                config,
                sessions=SessionStore(tmp_path / "state.json", "preset"),
                process_factory=lambda model: FakeProcess(model),
                chat_backend_factory=lambda base_url: FakeChatBackend(base_url),
            )

            orchestrator.switch_model("small")
            response = orchestrator.chat("s", [{"role": "user", "content": "short"}], {})

            self.assertEqual(FakeChatBackend.token_counts, [])
            self.assertEqual(response["hotmodel"]["prompt_budget"]["unit"], "estimated_tokens")

    def test_streaming_chat_tracks_inflight_and_persists_completed_response(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            tmp_path = Path(raw)
            FakeChatBackend.calls = []
            orchestrator = Orchestrator(
                _config(tmp_path),
                sessions=SessionStore(tmp_path / "state.json", "preset"),
                process_factory=lambda model: FakeProcess(model),
                chat_backend_factory=lambda base_url: FakeChatBackend(base_url),
            )
            orchestrator.switch_model("small")

            stream = orchestrator.chat_stream("s", [{"role": "user", "content": "hi"}], {"stream": True})
            self.assertEqual(orchestrator.state()["inflight_chats"], 1)
            lines = list(stream)

            self.assertEqual(lines[-1], b"data: [DONE]\n\n")
            self.assertEqual(orchestrator.state()["inflight_chats"], 0)
            session = orchestrator.sessions.get_or_create("s")
            self.assertEqual(
                session.messages,
                [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "hello"},
                ],
            )
            self.assertIs(FakeChatBackend.calls[-1]["payload"]["stream"], True)

    def test_cancelled_stream_releases_inflight_without_partial_history(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            tmp_path = Path(raw)
            orchestrator = Orchestrator(
                _config(tmp_path),
                sessions=SessionStore(tmp_path / "state.json", "preset"),
                process_factory=lambda model: FakeProcess(model),
                chat_backend_factory=lambda base_url: FakeChatBackend(base_url),
            )
            orchestrator.switch_model("small")

            stream = orchestrator.chat_stream("s", [{"role": "user", "content": "hi"}], {"stream": True})
            next(stream)
            stream.close()

            self.assertEqual(orchestrator.state()["inflight_chats"], 0)
            self.assertEqual(orchestrator.sessions.get_or_create("s").messages, [])

    def test_unconsumed_stream_close_releases_inflight(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            tmp_path = Path(raw)
            orchestrator = Orchestrator(
                _config(tmp_path),
                sessions=SessionStore(tmp_path / "state.json", "preset"),
                process_factory=lambda model: FakeProcess(model),
                chat_backend_factory=lambda base_url: FakeChatBackend(base_url),
            )
            orchestrator.switch_model("small")

            stream = orchestrator.chat_stream("s", [{"role": "user", "content": "hi"}], {"stream": True})
            self.assertEqual(orchestrator.state()["inflight_chats"], 1)
            stream.close()

            self.assertEqual(orchestrator.state()["inflight_chats"], 0)
            self.assertEqual(orchestrator.sessions.get_or_create("s").messages, [])

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


def _config(
    tmp_path: Path,
    max_prompt_tokens: int | None = None,
    token_budget_mode: str = "auto",
) -> RuntimeConfig:
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
        max_prompt_tokens=max_prompt_tokens,
        token_budget_mode=token_budget_mode,
        token_budget_chars_per_token=4.0,
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
        max_prompt_tokens=None,
        token_budget_mode="auto",
        token_budget_chars_per_token=4.0,
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
