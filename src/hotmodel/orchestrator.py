from __future__ import annotations

from dataclasses import dataclass
from threading import Condition, RLock
from typing import Any, Callable
import time

from .backend import LlamaHttpBackend
from .config import ModelSpec, RuntimeConfig
from .gpu import GpuMemoryProbe
from .llama_process import LlamaRouterProcess, LlamaServerProcess, ManagedBackend
from .session import Message, SessionStore


BackendFactory = Callable[[ModelSpec], ManagedBackend]
ChatBackendFactory = Callable[[str], LlamaHttpBackend]


@dataclass(frozen=True)
class SwitchReport:
    previous_model: str | None
    active_model: str
    elapsed_seconds: float
    gpu_settled: bool | None


class Orchestrator:
    def __init__(
        self,
        config: RuntimeConfig,
        sessions: SessionStore | None = None,
        process_factory: BackendFactory | None = None,
        chat_backend_factory: ChatBackendFactory | None = None,
        gpu_probe: GpuMemoryProbe | None = None,
    ) -> None:
        self.config = config
        self.sessions = sessions or SessionStore(
            config.state_path,
            config.system_prompt,
            max_session_messages=config.max_session_messages,
            max_prompt_chars=config.max_prompt_chars,
        )
        self._process_factory = process_factory or (lambda model: LlamaServerProcess(model))
        self._chat_backend_factory = chat_backend_factory or (lambda base_url: LlamaHttpBackend(base_url))
        self._gpu_probe = gpu_probe or GpuMemoryProbe()
        self._lock = RLock()
        self._condition = Condition(self._lock)
        self._process: ManagedBackend | None = None
        self._router: LlamaRouterProcess | None = (
            LlamaRouterProcess(config.router) if config.backend_mode == "router" and config.router else None
        )
        self._active_model: str | None = None
        self._switching = False
        self._inflight_chats = 0

    @property
    def active_model(self) -> str | None:
        with self._lock:
            return self._active_model

    @property
    def switching(self) -> bool:
        with self._lock:
            return self._switching

    def start_default(self) -> SwitchReport | None:
        if self.config.active_model is None:
            return None
        return self.switch_model(self.config.active_model)

    def switch_model(self, target_model: str) -> SwitchReport:
        if target_model not in self.config.models:
            raise KeyError(f"unknown model '{target_model}'")

        started_at = time.monotonic()
        with self._condition:
            if self._is_target_already_active(target_model):
                return SwitchReport(
                    previous_model=target_model,
                    active_model=target_model,
                    elapsed_seconds=0,
                    gpu_settled=None,
                )

            previous_name = self._active_model
            previous_model = self.config.models.get(previous_name) if previous_name else None
            self._switching = True

            if not self._wait_for_inflight_chats_locked(self.config.switch_drain_timeout_seconds):
                self._switching = False
                self._condition.notify_all()
                raise RuntimeError(
                    f"timed out waiting for {self._inflight_chats} in-flight chat request(s) before switching"
                )

            self._active_model = None

            if self._router is not None:
                self._router.start()
                if previous_model is not None:
                    self._router.unload_model(previous_model)
            else:
                previous_process = self._process
                self._process = None
                if previous_process is not None:
                    previous_process.stop()

            gpu_settled = self._wait_for_gpu_settle()
            target = self.config.models[target_model]
            try:
                if self._router is not None:
                    self._router.load_model(target)
                else:
                    next_process = self._process_factory(target)
                    next_process.start()
            except Exception:
                if previous_model is not None:
                    if self._router is not None:
                        self._router.load_model(previous_model)
                    else:
                        rollback = self._process_factory(previous_model)
                        rollback.start()
                    self._process = rollback
                    self._active_model = previous_model.name
                self._switching = False
                self._condition.notify_all()
                raise

            if self._router is not None:
                self._process = None
            else:
                self._process = next_process
            self._active_model = target_model
            self._switching = False
            self._condition.notify_all()

        return SwitchReport(
            previous_model=previous_name,
            active_model=target_model,
            elapsed_seconds=time.monotonic() - started_at,
            gpu_settled=gpu_settled,
        )

    def stop(self) -> None:
        with self._lock:
            if self._process is not None:
                self._process.stop()
            if self._router is not None:
                self._router.stop()
            self._process = None
            self._active_model = None
            self._switching = False

    def chat(self, session_id: str | None, incoming: list[Message], params: dict[str, Any]) -> dict[str, Any]:
        with self._condition:
            if self._switching:
                raise RuntimeError("model switch in progress")
            if self._active_model is None:
                raise RuntimeError("no active model")
            active = self.config.models[self._active_model]
            resolved_session = self.sessions.get_or_create(session_id)
            messages = self.sessions.build_prompt_messages(resolved_session.session_id, incoming)
            payload = dict(params)
            payload["model"] = active.llama_model_id
            payload["messages"] = messages
            backend_base_url = self._backend_base_url(active)
            self._inflight_chats += 1

        try:
            backend = self._chat_backend_factory(backend_base_url)
            response = backend.chat_completions(payload)

            assistant_message = _extract_assistant_message(response)
            stored_messages = [message for message in incoming if message.get("role") != "system"]
            if assistant_message is not None:
                stored_messages.append(assistant_message)
            if stored_messages:
                self.sessions.append_messages(resolved_session.session_id, stored_messages)
            response.setdefault("hotmodel", {})
            response["hotmodel"]["session_id"] = resolved_session.session_id
            response["hotmodel"]["active_model"] = active.name
            response["hotmodel"]["prompt_messages"] = len(messages)
            return response
        finally:
            with self._condition:
                self._inflight_chats -= 1
                self._condition.notify_all()

    def state(self) -> dict[str, Any]:
        with self._lock:
            return {
                "active_model": self._active_model,
                "switching": self._switching,
                "models": sorted(self.config.models),
                "backend_mode": self.config.backend_mode,
                "process_running": self._process.is_running() if self._process else False,
                "router_running": self._router.is_running() if self._router else False,
                "inflight_chats": self._inflight_chats,
            }

    def _wait_for_gpu_settle(self) -> bool | None:
        threshold = self.config.gpu_settle_memory_mb
        timeout = self.config.gpu_settle_timeout_seconds
        if threshold is None or timeout <= 0:
            return None
        return self._gpu_probe.wait_until_below(threshold, timeout)

    def _backend_base_url(self, active: ModelSpec) -> str:
        if self._router is not None:
            return self._router.base_url
        return active.base_url

    def _is_target_already_active(self, target_model: str) -> bool:
        if self._active_model != target_model:
            return False
        if self._router is not None:
            return self._router.is_running()
        return bool(self._process and self._process.is_running())

    def _wait_for_inflight_chats_locked(self, timeout_seconds: float) -> bool:
        if self._inflight_chats == 0:
            return True
        if timeout_seconds <= 0:
            return False
        deadline = time.monotonic() + timeout_seconds
        while self._inflight_chats > 0:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            self._condition.wait(timeout=remaining)
        return True


def _extract_assistant_message(response: dict[str, Any]) -> Message | None:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    first = choices[0]
    if not isinstance(first, dict):
        return None
    message = first.get("message")
    if not isinstance(message, dict):
        return None
    role = message.get("role")
    content = message.get("content")
    if role != "assistant" or content is None:
        return None
    return {"role": "assistant", "content": content}
