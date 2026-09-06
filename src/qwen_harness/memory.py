"""Bounded, in-process working memory for typed model-message turns."""

from copy import deepcopy
from dataclasses import dataclass
from threading import RLock
from typing import Sequence

from pydantic_ai.messages import (
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    ToolCallPart,
    ToolReturnPart,
)


@dataclass(frozen=True)
class MemoryUpdate:
    """Sanitized outcome of adding one complete turn."""

    evicted_turns: int = 0
    dropped_turns: int = 0
    turn_count: int = 0
    message_count: int = 0
    serialized_bytes: int = 0


@dataclass(frozen=True)
class MemorySnapshot:
    """One atomic, immutable view of memory and its revision."""

    messages: tuple[ModelMessage, ...]
    revision: int
    turn_count: int
    serialized_bytes: int

    @property
    def message_count(self) -> int:
        return len(self.messages)


class WorkingMemory:
    """Retain newest whole turns under turn-count and serialized-byte bounds."""

    def __init__(self, *, max_turns: int, max_serialized_bytes: int) -> None:
        if max_turns <= 0:
            raise ValueError("max_turns must be greater than zero")
        if max_serialized_bytes <= 0:
            raise ValueError("max_serialized_bytes must be greater than zero")
        self.max_turns = max_turns
        self.max_serialized_bytes = max_serialized_bytes
        self._turns: list[tuple[ModelMessage, ...]] = []
        self._serialized_bytes = 0
        self._revision = 0
        self._lock = RLock()

    def remember(
        self,
        turn: Sequence[ModelMessage],
        *,
        expected_revision: int | None = None,
    ) -> MemoryUpdate:
        """Add one complete turn, evicting only whole older turns when needed."""
        retained_turn = tuple(deepcopy(tuple(turn)))
        if not _is_complete_turn(retained_turn):
            return self._update(dropped_turns=1)
        new_size = _serialized_size(retained_turn)
        if new_size > self.max_serialized_bytes:
            return self._update(dropped_turns=1)

        with self._lock:
            if (
                expected_revision is not None
                and expected_revision != self._revision
            ):
                return self._update_locked(dropped_turns=1)
            candidate = [*self._turns, retained_turn]
            evicted = 0
            while len(candidate) > self.max_turns or (
                _serialized_size(_flatten(candidate)) > self.max_serialized_bytes
            ):
                candidate.pop(0)
                evicted += 1
            self._turns = candidate
            self._serialized_bytes = _serialized_size(_flatten(candidate))
            self._revision += 1
            return self._update_locked(evicted_turns=evicted)

    def inspect(self) -> MemorySnapshot:
        """Return messages and sanitized counts from one memory revision."""
        with self._lock:
            return MemorySnapshot(
                messages=tuple(deepcopy(_flatten(self._turns))),
                revision=self._revision,
                turn_count=len(self._turns),
                serialized_bytes=self._serialized_bytes,
            )

    def snapshot(self) -> tuple[ModelMessage, ...]:
        """Return an immutable flattened snapshot suitable for message_history."""
        return self.inspect().messages

    @property
    def turn_count(self) -> int:
        return self.inspect().turn_count

    @property
    def message_count(self) -> int:
        return self.inspect().message_count

    @property
    def serialized_bytes(self) -> int:
        return self.inspect().serialized_bytes

    def clear(self) -> None:
        """Forget all working state without touching persistent storage."""
        with self._lock:
            if self._turns:
                self._revision += 1
            self._turns.clear()
            self._serialized_bytes = 0

    def _update(self, *, dropped_turns: int = 0) -> MemoryUpdate:
        with self._lock:
            return self._update_locked(dropped_turns=dropped_turns)

    def _update_locked(
        self,
        *,
        evicted_turns: int = 0,
        dropped_turns: int = 0,
    ) -> MemoryUpdate:
        return MemoryUpdate(
            evicted_turns=evicted_turns,
            dropped_turns=dropped_turns,
            turn_count=len(self._turns),
            message_count=sum(len(turn) for turn in self._turns),
            serialized_bytes=self._serialized_bytes,
        )


def _flatten(
    turns: Sequence[Sequence[ModelMessage]],
) -> tuple[ModelMessage, ...]:
    return tuple(message for turn in turns for message in turn)


def _serialized_size(messages: Sequence[ModelMessage]) -> int:
    return len(ModelMessagesTypeAdapter.dump_json(list(messages)))


def _is_complete_turn(messages: Sequence[ModelMessage]) -> bool:
    """Return whether a full turn has correctly paired tool calls and results."""
    if (
        not messages
        or not isinstance(messages[0], ModelRequest)
        or not isinstance(messages[-1], ModelResponse)
    ):
        return False

    pending: dict[str, str] = {}
    seen_call_ids: set[str] = set()
    for message in messages:
        if isinstance(message, ModelResponse):
            for part in message.parts:
                if isinstance(part, ToolCallPart):
                    call_id = part.tool_call_id
                    if not call_id or call_id in seen_call_ids:
                        return False
                    seen_call_ids.add(call_id)
                    pending[call_id] = part.tool_name
        elif isinstance(message, ModelRequest):
            for part in message.parts:
                if isinstance(part, ToolReturnPart) or (
                    isinstance(part, RetryPromptPart)
                    and part.tool_call_id is not None
                ):
                    call_id = part.tool_call_id
                    if not call_id or call_id not in pending:
                        return False
                    if part.tool_name != pending[call_id]:
                        return False
                    del pending[call_id]
    return not pending
