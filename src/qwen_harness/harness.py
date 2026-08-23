"""Application boundary around the agent framework."""

import asyncio
from dataclasses import dataclass
from typing import Any, Protocol

from pydantic import TypeAdapter

from qwen_harness.orchestration import (
    ModelDecision,
    FinishDecision,
    RunLifecycle,
    RunState,
    TransitionEvent,
)


class HarnessError(Exception):
    """Base class for expected harness failures."""


class HarnessConfigurationError(HarnessError):
    """Raised when runtime configuration cannot build a harness."""


class InvalidPromptError(HarnessError):
    """Raised when a prompt is empty."""


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


@dataclass(frozen=True)
class Harness:
    """Run bounded, synchronous chat turns against an injected agent."""

    agent: AgentBoundary
    max_steps: int
    timeout_seconds: float
    max_output_tokens: int

    def __post_init__(self) -> None:
        if self.max_steps <= 0:
            raise ValueError("max_steps must be greater than zero")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero")
        if self.max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be greater than zero")

    def chat(self, prompt: str) -> str:
        """Return one assistant response for a non-empty user prompt."""
        return self.run(prompt).output

    def run(self, prompt: str) -> HarnessRun:
        """Execute one prompt and return its structured result and trace."""
        return asyncio.run(self._run(prompt))

    async def _run(self, prompt: str) -> HarnessRun:
        """Run one turn under a Python-owned lifecycle and deadline."""
        from pydantic_ai.exceptions import UnexpectedModelBehavior
        from pydantic_ai.settings import ModelSettings
        from pydantic_ai.usage import UsageLimits

        if not prompt.strip():
            raise InvalidPromptError("prompt must not be empty")

        lifecycle = RunLifecycle()
        lifecycle.transition(RunState.BUILDING_CONTEXT)
        lifecycle.transition(RunState.CALLING_MODEL)
        run = self.agent.run(
            prompt,
            # Phase 1 has only one legal action. Let the small model generate
            # plain content and let Python construct the terminal decision.
            output_type=str,
            usage_limits=UsageLimits(request_limit=self.max_steps),
            model_settings=ModelSettings(
                timeout=self.timeout_seconds,
                max_tokens=self.max_output_tokens,
                thinking=False,
                extra_body={"reasoning_effort": "none"},
            ),
        )
        try:
            result = await asyncio.wait_for(run, timeout=self.timeout_seconds)
        except asyncio.TimeoutError as exc:
            lifecycle.transition(RunState.FAILED)
            raise HarnessTimeoutError(
                f"agent run exceeded {self.timeout_seconds:g} seconds",
                lifecycle,
            ) from exc
        except UnexpectedModelBehavior as exc:
            lifecycle.transition(RunState.FAILED)
            raise InvalidModelDecisionError(str(exc), lifecycle) from exc
        except Exception as exc:
            lifecycle.transition(RunState.FAILED)
            raise HarnessExecutionError(str(exc), lifecycle) from exc

        lifecycle.transition(RunState.VERIFYING)
        try:
            decision = _coerce_decision(result.output)
            _verify_decision(decision)
        except Exception as exc:
            lifecycle.transition(RunState.FAILED)
            raise InvalidModelDecisionError(
                f"model returned an invalid decision: {exc}", lifecycle
            ) from exc
        lifecycle.transition(RunState.SUCCEEDED)
        return HarnessRun(
            output=decision.content,
            decision=decision,
            final_state=lifecycle.state,
            trace=lifecycle.trace,
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
