import asyncio
from dataclasses import dataclass
from typing import Any

import pytest

from qwen_harness.harness import Harness
from qwen_harness.harness import HarnessTimeoutError


@dataclass
class FakeRunResult:
    output: str


class RecordingAgent:
    def __init__(self, output: str) -> None:
        self.output = output
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def run(self, prompt: str, **kwargs: Any) -> FakeRunResult:
        self.calls.append((prompt, kwargs))
        return FakeRunResult(output=self.output)


def test_harness_passes_prompt_to_injected_agent_and_returns_output() -> None:
    agent = RecordingAgent(output="local response")
    harness = Harness(
        agent=agent,
        max_steps=4,
        timeout_seconds=20,
        max_output_tokens=128,
    )

    output = harness.chat("Hello, Qwen")

    assert output == "local response"
    assert agent.calls[0][0] == "Hello, Qwen"
    model_settings = agent.calls[0][1]["model_settings"]
    assert model_settings["thinking"] is False
    assert model_settings["max_tokens"] == 128
    assert model_settings["timeout"] == 20
    assert agent.calls[0][1]["usage_limits"].request_limit == 4


def test_harness_exposes_its_execution_bounds() -> None:
    harness = Harness(
        agent=RecordingAgent(output="unused"),
        max_steps=2,
        timeout_seconds=7.5,
        max_output_tokens=64,
    )

    assert harness.max_steps == 2
    assert harness.timeout_seconds == 7.5
    assert harness.max_output_tokens == 64


class SlowAgent:
    async def run(self, prompt: str, **kwargs: Any) -> FakeRunResult:
        await asyncio.sleep(0.1)
        return FakeRunResult(output="too late")


def test_harness_enforces_a_whole_run_deadline() -> None:
    harness = Harness(
        agent=SlowAgent(),
        max_steps=2,
        timeout_seconds=0.01,
        max_output_tokens=64,
    )

    with pytest.raises(HarnessTimeoutError, match="0.01"):
        harness.chat("Hello")
