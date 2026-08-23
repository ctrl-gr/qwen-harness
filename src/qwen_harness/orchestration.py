"""Typed decisions and Python-owned run lifecycle primitives."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field


class RunState(str, Enum):
    """Observable states in a single harness run."""

    CREATED = "created"
    BUILDING_CONTEXT = "building_context"
    CALLING_MODEL = "calling_model"
    VERIFYING = "verifying"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class InvalidStateTransition(RuntimeError):
    """Raised when the orchestrator attempts an illegal state transition."""


@dataclass(frozen=True)
class TransitionEvent:
    """One accepted transition in the execution trace."""

    from_state: RunState
    to_state: RunState


@dataclass(frozen=True)
class RunEvent:
    """One sanitized, externally observable boundary in a harness run."""

    run_id: str
    timestamp: datetime
    name: str
    data: Mapping[str, Any]


_ALLOWED_TRANSITIONS: dict[RunState, frozenset[RunState]] = {
    RunState.CREATED: frozenset({RunState.BUILDING_CONTEXT, RunState.FAILED}),
    RunState.BUILDING_CONTEXT: frozenset({RunState.CALLING_MODEL, RunState.FAILED}),
    RunState.CALLING_MODEL: frozenset({RunState.VERIFYING, RunState.FAILED}),
    RunState.VERIFYING: frozenset({RunState.SUCCEEDED, RunState.FAILED}),
    RunState.SUCCEEDED: frozenset(),
    RunState.FAILED: frozenset(),
}


@dataclass
class RunLifecycle:
    """Own and validate state changes; the model never changes state directly."""

    state: RunState = RunState.CREATED
    _events: list[TransitionEvent] = field(default_factory=list, repr=False)

    @property
    def trace(self) -> tuple[TransitionEvent, ...]:
        """Return an immutable snapshot of accepted transitions."""
        return tuple(self._events)

    def transition(self, destination: RunState) -> None:
        """Move to a legal destination and append it to the trace."""
        if destination not in _ALLOWED_TRANSITIONS[self.state]:
            raise InvalidStateTransition(
                f"illegal run-state transition: {self.state.name} -> {destination.name}"
            )
        event = TransitionEvent(from_state=self.state, to_state=destination)
        self.state = destination
        self._events.append(event)


class FinishDecision(BaseModel):
    """Return the final user-facing content for a successful Phase 1 run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    action: Literal["finish"]
    content: str = Field(min_length=1)


ModelDecision = FinishDecision
