"""Application boundary around the agent framework."""

import asyncio
from dataclasses import dataclass
from typing import Any, Protocol


class HarnessError(Exception):
    """Base class for expected harness failures."""


class HarnessConfigurationError(HarnessError):
    """Raised when runtime configuration cannot build a harness."""


class InvalidPromptError(HarnessError):
    """Raised when a prompt is empty."""


class HarnessTimeoutError(HarnessError):
    """Raised when the complete agent run exceeds its deadline."""


class RunResult(Protocol):
    """The small part of a PydanticAI run result consumed by this application."""

    output: Any


class AgentBoundary(Protocol):
    """An injectable boundary that keeps tests independent from a live model."""

    async def run(self, prompt: str, **kwargs: Any) -> RunResult: ...


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
        return asyncio.run(self._chat(prompt))

    async def _chat(self, prompt: str) -> str:
        """Run one turn under a cancellable wall-clock deadline."""
        from pydantic_ai.settings import ModelSettings
        from pydantic_ai.usage import UsageLimits

        if not prompt.strip():
            raise InvalidPromptError("prompt must not be empty")

        run = self.agent.run(
            prompt,
            usage_limits=UsageLimits(request_limit=self.max_steps),
            model_settings=ModelSettings(
                timeout=self.timeout_seconds,
                max_tokens=self.max_output_tokens,
                thinking=False,
            ),
        )
        try:
            result = await asyncio.wait_for(run, timeout=self.timeout_seconds)
        except asyncio.TimeoutError as exc:
            raise HarnessTimeoutError(
                f"agent run exceeded {self.timeout_seconds:g} seconds"
            ) from exc
        return str(result.output)
