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
    INCOMPLETE = "incomplete"
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
    RunState.VERIFYING: frozenset(
        {RunState.INCOMPLETE, RunState.SUCCEEDED, RunState.FAILED}
    ),
    RunState.INCOMPLETE: frozenset(),
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


class TaskContract(BaseModel):
    """Python-owned requirements that must be evidenced before success."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    required_tools: frozenset[str] = Field(default_factory=frozenset)


class ToolCallStatus(str, Enum):
    """Terminal outcome recorded for one model-initiated tool call."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass(frozen=True)
class ToolCallEvidence:
    """Sanitized evidence for one tool call; arguments and output are excluded."""

    tool_call_id: str
    tool_name: str
    status: ToolCallStatus
    risk: str | None = None
    elapsed_seconds: float | None = None
    error_category: str | None = None


@dataclass
class ToolEvidenceLedger:
    """Collect sanitized terminal tool outcomes from observable run events."""

    _calls: dict[str, ToolCallEvidence] = field(default_factory=dict)
    _order: list[str] = field(default_factory=list)
    _attempted: set[str] = field(default_factory=set)

    def observe(self, name: str, data: Mapping[str, Any]) -> None:
        tool_name = data.get("tool_name")
        if name == "tool.call.started" and isinstance(tool_name, str):
            self._attempted.add(tool_name)
            return
        if name not in {"tool.call.completed", "tool.call.failed"}:
            return
        call_id = data.get("tool_call_id")
        if not isinstance(call_id, str) or not isinstance(tool_name, str):
            return
        self._attempted.add(tool_name)
        if call_id not in self._calls:
            self._order.append(call_id)
        self._calls[call_id] = ToolCallEvidence(
            tool_call_id=call_id,
            tool_name=tool_name,
            status=(
                ToolCallStatus.SUCCEEDED
                if name == "tool.call.completed"
                else ToolCallStatus.FAILED
            ),
            risk=data.get("risk") if isinstance(data.get("risk"), str) else None,
            elapsed_seconds=(
                float(data["elapsed_seconds"])
                if isinstance(data.get("elapsed_seconds"), (int, float))
                else None
            ),
            error_category=(
                data.get("error_category")
                if isinstance(data.get("error_category"), str)
                else None
            ),
        )

    @property
    def evidence(self) -> tuple[ToolCallEvidence, ...]:
        return tuple(self._calls[call_id] for call_id in self._order)

    @property
    def attempted_tools(self) -> frozenset[str]:
        return frozenset(self._attempted)

    @property
    def successful_tools(self) -> frozenset[str]:
        return frozenset(
            item.tool_name
            for item in self.evidence
            if item.status is ToolCallStatus.SUCCEEDED
        )
