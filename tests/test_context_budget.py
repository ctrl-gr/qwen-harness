from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Sequence

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

from qwen_harness.config import Settings
import qwen_harness.harness as harness_module
from qwen_harness.harness import Harness, HarnessExecutionError
from qwen_harness.memory import WorkingMemory


NOW = datetime(2026, 9, 6, tzinfo=timezone.utc)


def _text_turn(prompt: str, answer: str) -> tuple[ModelMessage, ...]:
    return (
        ModelRequest(parts=[UserPromptPart(prompt, timestamp=NOW)]),
        ModelResponse(parts=[TextPart(answer)], timestamp=NOW),
    )


def _tool_turn(marker: str = "newest") -> tuple[ModelMessage, ...]:
    call_id = f"call-{marker}"
    return (
        ModelRequest(parts=[UserPromptPart(marker, timestamp=NOW)]),
        ModelResponse(
            parts=[ToolCallPart("list_files", {"path": "."}, call_id)],
            timestamp=NOW,
        ),
        ModelRequest(
            parts=[
                ToolReturnPart(
                    "list_files",
                    [f"{marker}.txt"],
                    call_id,
                    timestamp=NOW,
                )
            ]
        ),
        ModelResponse(parts=[TextPart(f"found {marker}")], timestamp=NOW),
    )


def _serialized_size(messages: Sequence[ModelMessage]) -> int:
    return len(ModelMessagesTypeAdapter.dump_json(list(messages)))


@dataclass
class FakeRunResult:
    output: str


class RecordingAgent:
    def __init__(self, output: str = "answer") -> None:
        self.output = output
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def run(self, prompt: str, **kwargs: Any) -> FakeRunResult:
        self.calls.append((prompt, kwargs))
        return FakeRunResult(self.output)


def _harness(
    agent: RecordingAgent,
    *,
    context_capacity: int,
    context_overhead_tokens: int = 16,
    events: list[Any] | None = None,
) -> Harness:
    return Harness(
        agent=agent,
        max_steps=3,
        max_tool_calls=3,
        timeout_seconds=20,
        max_output_tokens=32,
        context_capacity=context_capacity,
        context_overhead_tokens=context_overhead_tokens,
        available_tools=frozenset({"list_files"}),
        event_sink=events.append if events is not None else None,
    )


def test_prompt_and_instructions_that_cannot_fit_are_rejected_before_model_call(
) -> None:
    agent = RecordingAgent()
    harness = _harness(agent, context_capacity=80)

    with pytest.raises(
        harness_module.ContextBudgetError,
        match="context|budget|fit",
    ):
        harness.chat("essential prompt")

    assert agent.calls == []


def test_only_newest_contiguous_whole_memory_turns_are_sent_when_budget_is_tight(
) -> None:
    oldest = _text_turn("oldest-" + "x" * 5_000, "old answer")
    middle = _text_turn("middle-" + "y" * 5_000, "middle answer")
    newest = _tool_turn()
    memory = WorkingMemory(max_turns=3, max_serialized_bytes=50_000)
    memory.remember(oldest)
    memory.remember(middle)
    memory.remember(newest)
    before = memory.inspect()
    agent = RecordingAgent()

    # The essential context plus this allowance can hold the compact newest
    # turn, but not either deliberately large older turn.
    harness = _harness(
        agent,
        context_capacity=1_200 + _serialized_size(newest),
    )

    assert harness.chat("continue", memory=memory) == "answer"

    assert tuple(agent.calls[0][1]["message_history"]) == newest
    assert isinstance(agent.calls[0][1]["message_history"][1].parts[0], ToolCallPart)
    assert isinstance(agent.calls[0][1]["message_history"][2].parts[0], ToolReturnPart)
    after = memory.inspect()
    assert after.messages == before.messages
    assert after.revision == before.revision
    assert after.turn_count == before.turn_count
    assert after.serialized_bytes == before.serialized_bytes


def test_context_budget_event_reports_sanitized_estimates_and_trim_counts() -> None:
    secret = "private-budget-marker-9274"
    old = _text_turn(secret + "z" * 1_500, "old")
    newest = _text_turn("new", "recent")
    memory = WorkingMemory(max_turns=2, max_serialized_bytes=50_000)
    memory.remember(old)
    memory.remember(newest)
    events: list[Any] = []
    agent = RecordingAgent()

    _harness(
        agent,
        context_capacity=1_000 + _serialized_size(newest),
        events=events,
    ).chat("continue", memory=memory)

    budget_events = [event for event in events if event.name == "context.budgeted"]
    assert len(budget_events) == 1
    data = budget_events[0].data
    assert {
        "context_capacity",
        "reserved_output_tokens",
        "reserved_overhead_tokens",
        "available_input_tokens",
        "estimated_input_tokens",
        "selected_memory_turn_count",
        "selected_memory_message_count",
        "trimmed_memory_turn_count",
        "trimmed_memory_message_count",
    } <= set(data)
    assert data["reserved_output_tokens"] == 32
    assert data["reserved_overhead_tokens"] == 16
    assert data["selected_memory_turn_count"] == 1
    assert data["selected_memory_message_count"] == len(newest)
    assert data["trimmed_memory_turn_count"] == 1
    assert data["trimmed_memory_message_count"] == len(old)
    assert data["estimated_input_tokens"] <= data["available_input_tokens"]
    assert all(isinstance(value, int) for value in data.values())
    assert secret not in repr(dict(data))
    names = [event.name for event in events]
    assert names.index("context.budgeted") < names.index("model.call.started")


def test_sufficient_no_memory_call_preserves_prompt_output_and_omits_history() -> None:
    prompt = "Explain agent harnesses"
    agent = RecordingAgent(output="plain answer")

    output = _harness(agent, context_capacity=2_048).chat(prompt)

    assert output == "plain answer"
    assert agent.calls[0][0] == prompt
    assert "message_history" not in agent.calls[0][1]


def _context_budget_cause(error: BaseException) -> BaseException:
    if isinstance(error, harness_module.ContextBudgetError):
        return error
    assert isinstance(error, HarnessExecutionError)
    assert error.__cause__ is not None
    return error.__cause__


def test_large_tool_result_is_rejected_before_a_second_provider_request() -> None:
    provider_requests: list[list[ModelMessage]] = []

    def model_function(
        messages: list[ModelMessage], agent_info: AgentInfo
    ) -> ModelResponse:
        provider_requests.append(messages)
        if len(provider_requests) == 1:
            return ModelResponse(
                parts=[ToolCallPart("large_result", {}, "large-result-call")],
                timestamp=NOW,
            )
        return ModelResponse(parts=[TextPart("should not be reached")], timestamp=NOW)

    agent = Agent(FunctionModel(model_function))

    @agent.tool_plain
    def large_result() -> str:
        """Return deterministic content that cannot fit the second request."""
        return "tool-result-" + "x" * 8_000

    harness = Harness(
        agent=agent,
        max_steps=3,
        max_tool_calls=2,
        timeout_seconds=20,
        max_output_tokens=32,
        context_capacity=1_600,
        context_overhead_tokens=16,
    )

    with pytest.raises(
        (harness_module.ContextBudgetError, HarnessExecutionError)
    ) as caught:
        harness.chat("Use the tool")

    assert isinstance(
        _context_budget_cause(caught.value),
        harness_module.ContextBudgetError,
    )
    assert len(provider_requests) == 1


def test_actual_function_tool_schema_is_included_in_each_request_budget() -> None:
    small_requests: list[list[ModelMessage]] = []
    large_requests: list[list[ModelMessage]] = []

    def small_model(
        messages: list[ModelMessage], agent_info: AgentInfo
    ) -> ModelResponse:
        small_requests.append(messages)
        return ModelResponse(parts=[TextPart("small schema fits")], timestamp=NOW)

    def large_model(
        messages: list[ModelMessage], agent_info: AgentInfo
    ) -> ModelResponse:
        large_requests.append(messages)
        return ModelResponse(parts=[TextPart("must not be called")], timestamp=NOW)

    small_agent = Agent(FunctionModel(small_model))
    large_agent = Agent(FunctionModel(large_model))

    @small_agent.tool_plain
    def inspect_value(value: str) -> str:
        """Inspect one value."""
        return value

    large_description = "Detailed schema constraint. " + "s" * 4_000

    @large_agent.tool_plain(description=large_description)
    def inspect_value(value: str) -> str:
        return value

    def build(agent: Any) -> Harness:
        return Harness(
            agent=agent,
            max_steps=2,
            max_tool_calls=1,
            timeout_seconds=20,
            max_output_tokens=32,
            context_capacity=1_400,
            context_overhead_tokens=16,
        )

    assert build(small_agent).chat("Answer directly") == "small schema fits"
    with pytest.raises(
        (harness_module.ContextBudgetError, HarnessExecutionError)
    ) as caught:
        build(large_agent).chat("Answer directly")

    assert isinstance(
        _context_budget_cause(caught.value),
        harness_module.ContextBudgetError,
    )
    assert len(small_requests) == 1
    assert large_requests == []


def test_cpu_model_settings_reject_context_that_diverges_from_modelfile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OLLAMA_MODEL", "qwen3.5:0.8b-cpu")
    monkeypatch.setenv("HARNESS_CONTEXT_WINDOW_TOKENS", "4096")

    with pytest.raises(ValueError, match="context|2048|Modelfile"):
        Settings()


@pytest.mark.parametrize(
    "arguments",
    [
        {"context_capacity": 100, "max_output_tokens": 100},
        {
            "context_capacity": 120,
            "max_output_tokens": 100,
            "context_overhead_tokens": 20,
        },
        {"context_capacity": 2_048, "context_overhead_tokens": 0},
    ],
)
def test_harness_rejects_impossible_context_budget_configuration(
    arguments: dict[str, int],
) -> None:
    defaults = {
        "agent": RecordingAgent(),
        "max_steps": 3,
        "timeout_seconds": 20,
        "max_output_tokens": 32,
        "context_capacity": 2_048,
        "context_overhead_tokens": 16,
    }
    defaults.update(arguments)

    with pytest.raises(ValueError, match="context|output|overhead"):
        Harness(**defaults)


@pytest.mark.parametrize(
    "arguments",
    [
        {"context_window_tokens": 100, "max_output_tokens": 100},
        {
            "context_window_tokens": 120,
            "max_output_tokens": 100,
            "context_overhead_tokens": 20,
        },
        {"context_overhead_tokens": 0},
    ],
)
def test_settings_reject_impossible_context_budget_configuration(
    arguments: dict[str, int],
) -> None:
    with pytest.raises(ValueError):
        Settings(**arguments)
