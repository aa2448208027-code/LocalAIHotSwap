from __future__ import annotations

from dataclasses import dataclass
from threading import Condition, RLock
from typing import Any, Callable, Iterator
import json
import time

from .backend import LlamaHttpBackend
from .budget import BudgetResult, EstimatedTokenMeasurer, LlamaTokenMeasurer, fit_messages_to_budget
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
    gpu_memory: dict[str, dict[str, object]]


@dataclass(frozen=True)
class PreparedChat:
    active: ModelSpec
    session_id: str
    incoming: list[Message]
    payload: dict[str, Any]
    backend_base_url: str
    budget: BudgetResult


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
                    gpu_memory={},
                )
            if self._switching:
                raise RuntimeError("model switch in progress")

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
            previous_process = self._process
            if self._router is None:
                self._process = None
            self._condition.notify_all()

        gpu_settled: bool | None = None
        gpu_memory: dict[str, dict[str, object]] = {}
        next_process: ManagedBackend | None = None
        try:
            gpu_memory["before_unload"] = self._gpu_probe.snapshot().as_dict()
            if self._router is not None:
                self._router.start()
                if previous_model is not None:
                    self._router.unload_model(previous_model)
            else:
                if previous_process is not None:
                    previous_process.stop()

            gpu_memory["after_unload"] = self._gpu_probe.snapshot().as_dict()
            gpu_settled = self._wait_for_gpu_settle()
            gpu_memory["after_settle"] = self._gpu_probe.snapshot().as_dict()
            target = self.config.models[target_model]
            if self._router is not None:
                self._router.load_model(target)
            else:
                next_process = self._process_factory(target)
                next_process.start()
            gpu_memory["after_load"] = self._gpu_probe.snapshot().as_dict()
        except Exception:
            self._rollback_switch(previous_model)
            raise

        with self._condition:
            if self._router is None:
                self._process = next_process
            self._active_model = target_model
            self._switching = False
            self._condition.notify_all()

        return SwitchReport(
            previous_model=previous_name,
            active_model=target_model,
            elapsed_seconds=time.monotonic() - started_at,
            gpu_settled=gpu_settled,
            gpu_memory=gpu_memory,
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
        prepared = self._prepare_chat(session_id, incoming, params)
        try:
            backend = self._chat_backend_factory(prepared.backend_base_url)
            response = backend.chat_completions(prepared.payload)

            assistant_message = _extract_assistant_message(response)
            stored_messages = [message for message in prepared.incoming if message.get("role") != "system"]
            if assistant_message is not None:
                stored_messages.append(assistant_message)
            if stored_messages:
                self.sessions.append_messages(prepared.session_id, stored_messages)
            response.setdefault("hotmodel", {})
            response["hotmodel"]["session_id"] = prepared.session_id
            response["hotmodel"]["active_model"] = prepared.active.name
            response["hotmodel"]["prompt_messages"] = len(prepared.budget.messages)
            response["hotmodel"]["prompt_budget"] = {
                "cost": prepared.budget.cost,
                "unit": prepared.budget.unit,
                "dropped_messages": prepared.budget.dropped_messages,
            }
            return response
        finally:
            self._finish_chat()

    def chat_stream(self, session_id: str | None, incoming: list[Message], params: dict[str, Any]) -> Iterator[bytes]:
        payload = dict(params)
        payload["stream"] = True
        prepared = self._prepare_chat(session_id, incoming, payload)

        def _stream() -> Iterator[bytes]:
            content_parts: list[str] = []
            completed = False
            try:
                backend = self._chat_backend_factory(prepared.backend_base_url)
                for line in backend.chat_completions_stream(prepared.payload):
                    content = _extract_stream_content(line)
                    if content:
                        content_parts.append(content)
                    if _is_done_stream_line(line):
                        completed = True
                    yield line
                completed = True
            finally:
                if completed:
                    stored_messages = [message for message in prepared.incoming if message.get("role") != "system"]
                    if content_parts:
                        stored_messages.append({"role": "assistant", "content": "".join(content_parts)})
                    if stored_messages:
                        self.sessions.append_messages(prepared.session_id, stored_messages)
                self._finish_chat()

        return _stream()

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

    def _prepare_chat(
        self,
        session_id: str | None,
        incoming: list[Message],
        params: dict[str, Any],
    ) -> PreparedChat:
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
            backend_base_url = self._backend_base_url(active)
            self._inflight_chats += 1

        try:
            backend = self._chat_backend_factory(backend_base_url)
            budget = self._apply_token_budget(backend, active, messages, incoming_count=len(_without_system(incoming)))
            payload["messages"] = budget.messages
            return PreparedChat(
                active=active,
                session_id=resolved_session.session_id,
                incoming=[dict(message) for message in incoming],
                payload=payload,
                backend_base_url=backend_base_url,
                budget=budget,
            )
        except Exception:
            self._finish_chat()
            raise

    def _finish_chat(self) -> None:
        with self._condition:
            self._inflight_chats -= 1
            self._condition.notify_all()

    def _apply_token_budget(
        self,
        backend: LlamaHttpBackend,
        active: ModelSpec,
        messages: list[Message],
        incoming_count: int,
    ) -> BudgetResult:
        if self.config.max_prompt_tokens is None:
            return BudgetResult(messages=messages, cost=len(messages), dropped_messages=0, unit="messages")

        estimator = EstimatedTokenMeasurer(self.config.token_budget_chars_per_token)
        estimated = fit_messages_to_budget(
            messages,
            incoming_count=incoming_count,
            budget=self.config.max_prompt_tokens,
            measurer=estimator,
            unit=estimator.unit,
        )
        if self.config.token_budget_mode == "estimate":
            return estimated
        if self.config.token_budget_mode == "auto" and estimated.dropped_messages == 0:
            return estimated

        if self.config.token_budget_mode in {"auto", "llama"}:
            measurer = LlamaTokenMeasurer(
                active.llama_model_id,
                lambda model, items: backend.count_chat_tokens(model, items),
            )
            try:
                return fit_messages_to_budget(
                    messages,
                    incoming_count=incoming_count,
                    budget=self.config.max_prompt_tokens,
                    measurer=measurer,
                    unit=measurer.unit,
                )
            except RuntimeError:
                if self.config.token_budget_mode == "llama":
                    raise
        return estimated

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

    def _rollback_switch(self, previous_model: ModelSpec | None) -> None:
        rollback_process: ManagedBackend | None = None
        rollback_model_name: str | None = None
        if previous_model is not None:
            try:
                if self._router is not None:
                    self._router.load_model(previous_model)
                else:
                    rollback_process = self._process_factory(previous_model)
                    rollback_process.start()
                rollback_model_name = previous_model.name
            except Exception:
                rollback_process = None
                rollback_model_name = None

        with self._condition:
            self._process = rollback_process if self._router is None else None
            self._active_model = rollback_model_name
            self._switching = False
            self._condition.notify_all()


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


def _without_system(messages: list[Message]) -> list[Message]:
    return [message for message in messages if message.get("role") != "system"]


def _extract_stream_content(line: bytes) -> str | None:
    data = _stream_data(line)
    if data is None or data == "[DONE]":
        return None
    try:
        item = json.loads(data)
    except json.JSONDecodeError:
        return None
    choices = item.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    first = choices[0]
    if not isinstance(first, dict):
        return None
    delta = first.get("delta")
    if isinstance(delta, dict) and isinstance(delta.get("content"), str):
        return delta["content"]
    message = first.get("message")
    if isinstance(message, dict) and isinstance(message.get("content"), str):
        return message["content"]
    return None


def _is_done_stream_line(line: bytes) -> bool:
    return _stream_data(line) == "[DONE]"


def _stream_data(line: bytes) -> str | None:
    text = line.decode("utf-8", errors="replace").strip()
    if not text.startswith("data:"):
        return None
    return text.removeprefix("data:").strip()
