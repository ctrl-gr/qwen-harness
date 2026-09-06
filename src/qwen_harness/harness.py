"""Application boundary around the agent framework."""

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from types import MappingProxyType
from typing import Any, Callable, Mapping, Protocol
from uuid import uuid4

from pydantic import TypeAdapter

from qwen_harness.context import ContextBuilder
from qwen_harness.metrics import (
    RunMetrics,
    TokenUsage,
    collect_token_usage,
)
from qwen_harness.memory import WorkingMemory
from qwen_harness.orchestration import (
    ModelDecision,
    FinishDecision,
    RunEvent,
    RunLifecycle,
    RunState,
    TaskContract,
    ToolCallEvidence,
    ToolEvidenceLedger,
    TransitionEvent,
)
from qwen_harness.observability import (
    bind_run_event_emitter,
    reset_run_event_emitter,
)


EventSink = Callable[[RunEvent], None]
MetricsSink = Callable[[RunMetrics], None]


class HarnessError(Exception):
    """Base class for expected harness failures."""


class HarnessConfigurationError(HarnessError):
    """Raised when runtime configuration cannot build a harness."""


class InvalidPromptError(HarnessError):
    """Raised when a prompt is empty."""


class InvalidTaskContractError(HarnessError, ValueError):
    """Raised before model execution when a task contract is unsupported."""


class HarnessRunError(HarnessError):
    """A failed run with its final state and immutable execution trace."""

    def __init__(self, message: str, lifecycle: RunLifecycle | None = None) -> None:
        super().__init__(message)
        self.final_state = lifecycle.state if lifecycle else None
        self.trace = lifecycle.trace if lifecycle else ()


class HarnessTimeoutError(HarnessRunError):
    """Raised when the complete agent run exceeds its deadline."""


class HarnessExecutionError(HarnessRunError):
    """Raised when the model boundary fails during a run."""


class InvalidModelDecisionError(HarnessRunError):
    """Raised when model output does not satisfy the decision protocol."""


class IncompleteTaskError(HarnessRunError):
    """Raised when the answer lacks the tool evidence required for success."""

    def __init__(
        self,
        message: str,
        lifecycle: RunLifecycle | None = None,
        *,
        evidence: tuple[ToolCallEvidence, ...] = (),
        unmet_tools: frozenset[str] = frozenset(),
    ) -> None:
        super().__init__(message, lifecycle)
        self.tool_evidence = evidence
        self.evidence = evidence
        self.unmet_tools = unmet_tools


class RunResult(Protocol):
    """The small part of a PydanticAI run result consumed by this application."""

    output: Any


class AgentBoundary(Protocol):
    """An injectable boundary that keeps tests independent from a live model."""

    async def run(self, prompt: str, **kwargs: Any) -> RunResult: ...


@dataclass(frozen=True)
class HarnessRun:
    """The user-visible result and immutable execution trace of one run."""

    output: str
    decision: ModelDecision
    final_state: RunState
    trace: tuple[TransitionEvent, ...]
    tool_evidence: tuple[ToolCallEvidence, ...]
    token_usage: TokenUsage | None = None

    @property
    def evidence(self) -> tuple[ToolCallEvidence, ...]:
        """Backward-compatible short name for the tool evidence snapshot."""
        return self.tool_evidence


@dataclass(frozen=True)
class Harness:
    """Run bounded, synchronous chat turns against an injected agent."""

    agent: AgentBoundary
    max_steps: int
    timeout_seconds: float
    max_output_tokens: int
    max_tool_calls: int = 4
    event_sink: EventSink | None = None
    workspace_root: Path | None = None
    available_tools: frozenset[str] = frozenset()
    context_capacity: int = 2_048
    model_name: str = "unknown"
    metrics_sink: MetricsSink | None = None

    def __post_init__(self) -> None:
        if self.max_steps <= 0:
            raise ValueError("max_steps must be greater than zero")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero")
        if self.max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be greater than zero")
        if self.max_tool_calls <= 0:
            raise ValueError("max_tool_calls must be greater than zero")
        if self.context_capacity <= 0:
            raise ValueError("context_capacity must be greater than zero")

    def chat(
        self,
        prompt: str,
        *,
        contract: TaskContract | None = None,
        memory: WorkingMemory | None = None,
    ) -> str:
        """Return one assistant response for a non-empty user prompt."""
        return self.run(prompt, contract=contract, memory=memory).output

    def run(
        self,
        prompt: str,
        *,
        contract: TaskContract | None = None,
        memory: WorkingMemory | None = None,
    ) -> HarnessRun:
        """Execute one prompt and return its structured result and trace."""
        return asyncio.run(self._run(prompt, contract=contract, memory=memory))

    async def _run(
        self,
        prompt: str,
        *,
        contract: TaskContract | None = None,
        memory: WorkingMemory | None = None,
    ) -> HarnessRun:
        """Run one turn under a Python-owned lifecycle and deadline."""
        from pydantic_ai.exceptions import UnexpectedModelBehavior
        from pydantic_ai.settings import ModelSettings
        from pydantic_ai.usage import UsageLimits

        if not prompt.strip():
            raise InvalidPromptError("prompt must not be empty")
        contract = contract or TaskContract()
        unknown_tools = contract.required_tools - self.available_tools
        if unknown_tools:
            names = ", ".join(sorted(unknown_tools))
            raise InvalidTaskContractError(f"unknown required tool(s): {names}")

        run_id = uuid4().hex
        run_timestamp = datetime.now(timezone.utc)
        run_started = perf_counter()
        model_started = run_started
        verification_started = run_started
        evidence_ledger = ToolEvidenceLedger()
        token_usage: TokenUsage | None = None

        def emit(name: str, data: Mapping[str, Any] | None = None) -> None:
            evidence_ledger.observe(name, data or {})
            if self.event_sink is None:
                return
            try:
                # Sinks are an optional, non-blocking observability boundary.
                # Their failures must never change the result of a harness run.
                self.event_sink(
                    RunEvent(
                        run_id=run_id,
                        timestamp=datetime.now(timezone.utc),
                        name=name,
                        data=MappingProxyType(dict(data or {})),
                    )
                )
            except Exception:
                pass

        lifecycle = RunLifecycle()

        def transition(destination: RunState) -> None:
            previous = lifecycle.state
            lifecycle.transition(destination)
            emit(
                "state.transition",
                {"from_state": previous.value, "to_state": destination.value},
            )

        def emit_failure(category: str, message: str) -> None:
            emit(
                "run.failed",
                {
                    "error_category": category,
                    "message": message,
                    "elapsed_seconds": perf_counter() - run_started,
                },
            )

        def persist_metrics(final_state: RunState) -> None:
            if self.metrics_sink is None or token_usage is None:
                return
            try:
                self.metrics_sink(
                    RunMetrics(
                        run_id=run_id,
                        timestamp=run_timestamp,
                        model=self.model_name,
                        final_state=final_state.value,
                        elapsed_seconds=perf_counter() - run_started,
                        token_usage=token_usage,
                    )
                )
            except Exception:
                # Local telemetry is best-effort and cannot alter task outcome.
                pass

        emit(
            "run.started",
            {
                "prompt_characters": len(prompt),
                "max_model_calls": self.max_steps,
                "timeout_seconds": self.timeout_seconds,
                "max_output_tokens": self.max_output_tokens,
                "max_tool_calls": self.max_tool_calls,
            },
        )
        transition(RunState.BUILDING_CONTEXT)
        task_context = ContextBuilder().build(
            contract=contract,
            available_tools=self.available_tools,
        )
        emit(
            "context.built",
            {"context_version": task_context.version},
        )
        memory_history: tuple[Any, ...] = ()
        memory_revision: int | None = None
        if memory is not None:
            memory_snapshot = memory.inspect()
            memory_history = memory_snapshot.messages
            memory_revision = memory_snapshot.revision
            emit(
                "memory.loaded",
                {
                    "turn_count": memory_snapshot.turn_count,
                    "message_count": memory_snapshot.message_count,
                    "serialized_bytes": memory_snapshot.serialized_bytes,
                },
            )
        transition(RunState.CALLING_MODEL)
        model_started = perf_counter()
        emit("model.call.started")
        run_arguments: dict[str, Any] = {
            "output_type": str,
            "instructions": task_context.instructions,
            "usage_limits": UsageLimits(
                request_limit=self.max_steps,
                tool_calls_limit=self.max_tool_calls,
            ),
            "model_settings": ModelSettings(
                timeout=self.timeout_seconds,
                max_tokens=self.max_output_tokens,
                thinking=False,
                extra_body={"reasoning_effort": "none"},
            ),
        }
        if memory_history:
            run_arguments["message_history"] = memory_history
        run = self.agent.run(
            prompt,
            # Phase 1 has only one legal action. Let the small model generate
            # plain content and let Python construct the terminal decision.
            **run_arguments,
        )
        emitter_token = bind_run_event_emitter(emit)
        try:
            result = await asyncio.wait_for(run, timeout=self.timeout_seconds)
        except asyncio.TimeoutError as exc:
            transition(RunState.FAILED)
            emit_failure("timeout", "The model call timed out.")
            raise HarnessTimeoutError(
                f"agent run exceeded {self.timeout_seconds:g} seconds",
                lifecycle,
            ) from exc
        except UnexpectedModelBehavior as exc:
            transition(RunState.FAILED)
            emit_failure(
                "invalid_model_output",
                "The model returned an invalid response.",
            )
            raise InvalidModelDecisionError(str(exc), lifecycle) from exc
        except Exception as exc:
            transition(RunState.FAILED)
            if isinstance(exc, (ConnectionError, OSError)):
                emit_failure(
                    "connection_error",
                    "The model service is unavailable.",
                )
            else:
                emit_failure("model_error", "The model call failed.")
            raise HarnessExecutionError(str(exc), lifecycle) from exc
        finally:
            reset_run_event_emitter(emitter_token)

        emit(
            "model.call.completed",
            {
                "elapsed_seconds": perf_counter() - model_started,
                "output_characters": len(str(result.output)),
            },
        )
        token_usage = collect_token_usage(
            result,
            context_capacity=self.context_capacity,
        )
        if token_usage is not None:
            emit(
                "model.usage",
                {
                    "input_tokens": token_usage.input_tokens,
                    "output_tokens": token_usage.output_tokens,
                    "total_tokens": token_usage.total_tokens,
                    "requests": token_usage.requests,
                    "tool_calls": token_usage.tool_calls,
                    "peak_context_tokens": token_usage.peak_context_tokens,
                    "context_capacity": token_usage.context_capacity,
                    "context_utilization": token_usage.context_utilization,
                },
            )
        transition(RunState.VERIFYING)
        verification_started = perf_counter()
        emit("verification.started")
        try:
            decision = _coerce_decision(result.output)
            _verify_decision(decision)
        except Exception as exc:
            transition(RunState.FAILED)
            persist_metrics(RunState.FAILED)
            emit_failure(
                "invalid_model_output",
                "The model returned an invalid response.",
            )
            raise InvalidModelDecisionError(
                f"model returned an invalid decision: {exc}", lifecycle
            ) from exc
        required_or_attempted = (
            contract.required_tools | evidence_ledger.attempted_tools
        )
        unmet_tools = required_or_attempted - evidence_ledger.successful_tools
        if unmet_tools:
            emit(
                "verification.failed",
                {
                    "reason": "required_tool_not_completed",
                    "unmet_tools": tuple(sorted(unmet_tools)),
                },
            )
            transition(RunState.INCOMPLETE)
            persist_metrics(RunState.INCOMPLETE)
            emit(
                "run.incomplete",
                {
                    "reason": "required_tool_not_completed",
                    "unmet_tools": tuple(sorted(unmet_tools)),
                    "elapsed_seconds": perf_counter() - run_started,
                },
            )
            raise IncompleteTaskError(
                "Task requirements were not completed.",
                lifecycle,
                evidence=evidence_ledger.evidence,
                unmet_tools=unmet_tools,
            )
        emit(
            "verification.completed",
            {"elapsed_seconds": perf_counter() - verification_started},
        )
        transition(RunState.SUCCEEDED)
        if memory is not None:
            new_messages = getattr(result, "new_messages", None)
            if callable(new_messages):
                try:
                    update = memory.remember(
                        new_messages(),
                        expected_revision=memory_revision,
                    )
                except Exception:
                    emit(
                        "memory.update.failed",
                        {"error_category": "memory_serialization_error"},
                    )
                else:
                    emit(
                        "memory.updated",
                        {
                            "turn_count": update.turn_count,
                            "message_count": update.message_count,
                            "serialized_bytes": update.serialized_bytes,
                            "evicted_turn_count": update.evicted_turns,
                            "dropped_turn_count": update.dropped_turns,
                        },
                    )
        persist_metrics(RunState.SUCCEEDED)
        emit(
            "run.completed",
            {"elapsed_seconds": perf_counter() - run_started},
        )
        return HarnessRun(
            output=decision.content,
            decision=decision,
            final_state=lifecycle.state,
            trace=lifecycle.trace,
            tool_evidence=evidence_ledger.evidence,
            token_usage=token_usage,
        )


def _coerce_decision(output: Any) -> ModelDecision:
    """Validate model output while retaining compatibility with string fakes."""
    if isinstance(output, str):
        return FinishDecision(action="finish", content=output)
    return TypeAdapter(ModelDecision).validate_python(output)


def _verify_decision(decision: ModelDecision) -> None:
    """Reject user-visible content that contains leaked model reasoning."""
    normalized = decision.content.lstrip().casefold()
    if not normalized:
        raise ValueError("model response is empty")

    has_leaked_prefix = normalized.startswith("/think") and (
        len(normalized) == len("/think") or normalized[len("/think")].isspace()
    )
    if has_leaked_prefix:
        raise ValueError("model response contains leaked or incomplete reasoning")
