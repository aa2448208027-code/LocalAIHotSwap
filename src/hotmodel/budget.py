from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol
import json
import math

Message = dict[str, Any]


class MessageMeasurer(Protocol):
    def cost(self, messages: list[Message]) -> int:
        ...


@dataclass(frozen=True)
class BudgetResult:
    messages: list[Message]
    cost: int
    dropped_messages: int
    unit: str


@dataclass(frozen=True)
class CharMeasurer:
    unit: str = "chars"

    def cost(self, messages: list[Message]) -> int:
        return sum(_message_chars(message) for message in messages)


@dataclass(frozen=True)
class EstimatedTokenMeasurer:
    chars_per_token: float = 4.0
    unit: str = "estimated_tokens"

    def cost(self, messages: list[Message]) -> int:
        chars = sum(_message_chars(message) for message in messages)
        return max(1, math.ceil(chars / self.chars_per_token)) if chars else 0


@dataclass(frozen=True)
class LlamaTokenMeasurer:
    model: str
    count_chat_tokens: Callable[[str, list[Message]], int]
    unit: str = "llama_tokens"

    def cost(self, messages: list[Message]) -> int:
        return self.count_chat_tokens(self.model, messages)


def fit_messages_to_budget(
    messages: list[Message],
    incoming_count: int,
    budget: int | None,
    measurer: MessageMeasurer,
    unit: str,
) -> BudgetResult:
    copied = _copy_messages(messages)
    if budget is None:
        return BudgetResult(messages=copied, cost=measurer.cost(copied), dropped_messages=0, unit=unit)

    incoming_count = max(0, incoming_count)
    prefix_count = 1 if copied and copied[0].get("role") == "system" else 0
    prefix = copied[:prefix_count]
    tail = copied[len(copied) - incoming_count :] if incoming_count else []
    history = copied[prefix_count : len(copied) - incoming_count if incoming_count else len(copied)]

    required = prefix + tail
    required_cost = measurer.cost(required)
    if required_cost >= budget or not history:
        return BudgetResult(
            messages=required,
            cost=required_cost,
            dropped_messages=len(history),
            unit=unit,
        )

    low = 0
    high = len(history)
    best_count = 0
    best_cost = required_cost
    while low <= high:
        mid = (low + high) // 2
        candidate = prefix + history[len(history) - mid :] + tail
        candidate_cost = measurer.cost(candidate)
        if candidate_cost <= budget:
            best_count = mid
            best_cost = candidate_cost
            low = mid + 1
        else:
            high = mid - 1

    selected = history[len(history) - best_count :] if best_count else []
    return BudgetResult(
        messages=_copy_messages(prefix + selected + tail),
        cost=best_cost,
        dropped_messages=len(history) - best_count,
        unit=unit,
    )


def _copy_messages(messages: list[Message]) -> list[Message]:
    return [dict(message) for message in messages]


def _message_chars(message: Message) -> int:
    role = str(message.get("role", ""))
    content = message.get("content", "")
    if isinstance(content, str):
        content_text = content
    else:
        content_text = json.dumps(content, ensure_ascii=False, sort_keys=True)
    return len(role) + len(content_text)
