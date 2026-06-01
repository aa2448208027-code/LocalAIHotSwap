from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Any
import json
import time
import uuid

from .budget import CharMeasurer, fit_messages_to_budget


Message = dict[str, Any]


@dataclass
class Session:
    session_id: str
    messages: list[Message] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


class SessionStore:
    def __init__(
        self,
        path: Path,
        system_prompt: str,
        max_session_messages: int | None = None,
        max_prompt_chars: int | None = None,
    ) -> None:
        self.path = path
        self.system_prompt = system_prompt
        self.max_session_messages = max_session_messages
        self.max_prompt_chars = max_prompt_chars
        self._lock = RLock()
        self._sessions: dict[str, Session] = {}
        self._load()

    def get_or_create(self, session_id: str | None) -> Session:
        with self._lock:
            sid = session_id or str(uuid.uuid4())
            session = self._sessions.get(sid)
            if session is None:
                session = Session(session_id=sid)
                self._sessions[sid] = session
                self._save()
            return session

    def append_messages(self, session_id: str, messages: list[Message]) -> None:
        with self._lock:
            session = self.get_or_create(session_id)
            session.messages.extend(_copy_messages(messages))
            session.messages = self._trim_session_messages(session.messages)
            session.updated_at = time.time()
            self._save()

    def build_prompt_messages(self, session_id: str, incoming: list[Message]) -> list[Message]:
        with self._lock:
            session = self.get_or_create(session_id)
            messages: list[Message] = []
            if self.system_prompt:
                messages.append({"role": "system", "content": self.system_prompt})
            messages.extend(_without_system(session.messages))
            messages.extend(_without_system(incoming))
            return self._fit_prompt_messages(messages, len(_without_system(incoming)))

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "sessions": {
                    sid: {
                        "session_id": session.session_id,
                        "messages": _copy_messages(session.messages),
                        "created_at": session.created_at,
                        "updated_at": session.updated_at,
                    }
                    for sid, session in self._sessions.items()
                }
            }

    def _load(self) -> None:
        if not self.path.exists():
            return
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        sessions = raw.get("sessions", {})
        for sid, item in sessions.items():
            self._sessions[sid] = Session(
                session_id=str(item.get("session_id", sid)),
                messages=_copy_messages(item.get("messages", [])),
                created_at=float(item.get("created_at", time.time())),
                updated_at=float(item.get("updated_at", time.time())),
            )

    def _trim_session_messages(self, messages: list[Message]) -> list[Message]:
        if self.max_session_messages is None:
            return _copy_messages(messages)
        return _copy_messages(messages[-self.max_session_messages :])

    def _fit_prompt_messages(self, messages: list[Message], incoming_count: int) -> list[Message]:
        result = fit_messages_to_budget(
            messages,
            incoming_count=incoming_count,
            budget=self.max_prompt_chars,
            measurer=CharMeasurer(),
            unit="chars",
        )
        return result.messages

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(self.snapshot(), indent=2), encoding="utf-8")
        tmp_path.replace(self.path)


def _without_system(messages: list[Message]) -> list[Message]:
    return [message for message in messages if message.get("role") != "system"]


def _copy_messages(messages: list[Message]) -> list[Message]:
    return [dict(message) for message in messages]
