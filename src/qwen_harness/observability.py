"""Run-scoped event propagation shared by the harness and tool boundary."""

from contextvars import ContextVar, Token
from typing import Any, Callable, Mapping


RunEventEmitter = Callable[[str, Mapping[str, Any] | None], None]

_current_emitter: ContextVar[RunEventEmitter | None] = ContextVar(
    "qwen_harness_run_event_emitter", default=None
)


def bind_run_event_emitter(emitter: RunEventEmitter) -> Token[RunEventEmitter | None]:
    """Bind an emitter to the current async/thread context."""
    return _current_emitter.set(emitter)


def reset_run_event_emitter(token: Token[RunEventEmitter | None]) -> None:
    """Restore the previous run emitter."""
    _current_emitter.reset(token)


def emit_current_run_event(
    name: str, data: Mapping[str, Any] | None = None
) -> bool:
    """Emit through the active run, returning whether a run was bound."""
    emitter = _current_emitter.get()
    if emitter is None:
        return False
    emitter(name, data)
    return True
