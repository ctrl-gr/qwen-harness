from dataclasses import dataclass
from datetime import timezone
from pathlib import Path
from typing import Any

import pytest
from pydantic_ai import ModelRetry

from qwen_harness.harness import Harness, HarnessExecutionError
from qwen_harness.tools import WorkspaceToolRegistry


@dataclass
class TextRunResult:
    output: str


class TextAgent:
    def __init__(self, output: str = "A concise answer") -> None:
        self.output = output

    async def run(self, prompt: str, **kwargs: Any) -> TextRunResult:
        return TextRunResult(output=self.output)


SENSITIVE_ERROR = (
    "Ollama connection failed\n"
    "Authorization: Bearer super-secret-token\x1b[31m"
)


class FailingAgent:
    async def run(self, prompt: str, **kwargs: Any) -> TextRunResult:
        raise ConnectionError(SENSITIVE_ERROR)


def _build_harness(agent: Any, events: list[Any]) -> Harness:
    return Harness(
        agent=agent,
        max_steps=2,
        timeout_seconds=20,
        max_output_tokens=64,
        event_sink=events.append,
    )


def test_successful_run_emits_structured_events_for_each_observable_boundary() -> None:
    events: list[Any] = []
    harness = _build_harness(TextAgent(), events)

    run = harness.run("Explain an agent harness")

    assert run.output == "A concise answer"
    assert [event.name for event in events] == [
        "run.started",
        "state.transition",
        "context.built",
        "state.transition",
        "model.call.started",
        "model.call.completed",
        "state.transition",
        "verification.started",
        "verification.completed",
        "state.transition",
        "run.completed",
    ]
    assert [
        (event.data["from_state"], event.data["to_state"])
        for event in events
        if event.name == "state.transition"
    ] == [
        ("created", "building_context"),
        ("building_context", "calling_model"),
        ("calling_model", "verifying"),
        ("verifying", "succeeded"),
    ]

    run_ids = {event.run_id for event in events}
    assert len(run_ids) == 1
    assert next(iter(run_ids))
    assert all(event.timestamp.tzinfo is timezone.utc for event in events)
    assert events[-1].data["elapsed_seconds"] >= 0


def test_failed_model_call_emits_failed_transition_and_sanitized_error() -> None:
    events: list[Any] = []
    harness = _build_harness(FailingAgent(), events)

    with pytest.raises(HarnessExecutionError):
        harness.run("Hello")

    assert [event.name for event in events][-2:] == [
        "state.transition",
        "run.failed",
    ]
    assert events[-2].data == {
        "from_state": "calling_model",
        "to_state": "failed",
    }
    assert events[-1].data["error_category"] == "connection_error"
    assert events[-1].data["message"] == "The model service is unavailable."
    assert events[-1].data["elapsed_seconds"] >= 0
    serialized_failure = repr(events[-1])
    assert "super-secret-token" not in serialized_failure
    assert "Authorization" not in serialized_failure
    assert "\n" not in serialized_failure
    assert "\x1b" not in serialized_failure
    assert "error" not in events[-1].data
    assert "error_type" not in events[-1].data
    for value in events[-1].data.values():
        if isinstance(value, str):
            assert "\n" not in value
            assert "\r" not in value
            assert "\x1b" not in value


def test_observability_does_not_publish_hidden_reasoning_content() -> None:
    events: list[Any] = []
    leaked_reasoning = "/think private chain of thought"
    harness = _build_harness(TextAgent(leaked_reasoning), events)

    with pytest.raises(Exception, match="reasoning|invalid decision"):
        harness.run("Hello")

    serialized_events = repr(events)
    assert leaked_reasoning not in serialized_events
    assert "private chain of thought" not in serialized_events


@pytest.mark.parametrize(
    "throw_on_event",
    ["run.started", "state.transition", "run.completed"],
)
def test_event_sink_failures_do_not_change_a_successful_run(
    throw_on_event: str,
) -> None:
    received_names: list[str] = []

    def broken_sink(event: Any) -> None:
        received_names.append(event.name)
        if event.name == throw_on_event:
            raise RuntimeError("logging backend failed")

    harness = Harness(
        agent=TextAgent("The actual answer"),
        max_steps=2,
        timeout_seconds=20,
        max_output_tokens=64,
        event_sink=broken_sink,
    )

    run = harness.run("Hello")

    assert run.output == "The actual answer"
    assert run.final_state.value == "succeeded"
    assert throw_on_event in received_names


def test_event_sink_failure_does_not_replace_the_original_run_failure() -> None:
    def broken_sink(event: Any) -> None:
        if event.name == "run.failed":
            raise RuntimeError("logging backend failed")

    harness = Harness(
        agent=FailingAgent(),
        max_steps=2,
        timeout_seconds=20,
        max_output_tokens=64,
        event_sink=broken_sink,
    )

    with pytest.raises(HarnessExecutionError) as captured:
        harness.run("Hello")

    assert isinstance(captured.value.__cause__, ConnectionError)
    assert SENSITIVE_ERROR in str(captured.value)
    assert "logging backend failed" not in str(captured.value)


def test_successful_tool_call_emits_sanitized_started_and_completed_events(
    tmp_path: Path,
) -> None:
    sensitive_content = "TOP-SECRET file content"
    sensitive_path = "TOP-SECRET-name.txt"
    (tmp_path / sensitive_path).write_text(sensitive_content, encoding="utf-8")
    events: list[Any] = []
    registry = WorkspaceToolRegistry(tmp_path, event_sink=events.append)

    result = registry.read_file(sensitive_path)

    assert result.content == sensitive_content
    assert [event.name for event in events] == [
        "tool.call.started",
        "tool.call.completed",
    ]
    assert events[0].data["tool_name"] == "read_file"
    assert events[0].data["risk"] == "read"
    tool_call_id = events[0].data["tool_call_id"]
    assert tool_call_id
    assert events[1].data["tool_call_id"] == tool_call_id
    assert events[0].run_id == events[1].run_id
    assert events[1].data["tool_name"] == "read_file"
    assert events[1].data["risk"] == "read"
    assert events[1].data["elapsed_seconds"] >= 0
    serialized = repr(events)
    assert sensitive_path not in serialized
    assert sensitive_content not in serialized
    assert "arguments" not in serialized


def test_failed_tool_call_emits_a_sanitized_failure_event(
    tmp_path: Path,
) -> None:
    events: list[Any] = []
    registry = WorkspaceToolRegistry(tmp_path, event_sink=events.append)

    with pytest.raises(ModelRetry):
        registry.read_file("../TOP-SECRET-outside.txt")

    assert [event.name for event in events] == [
        "tool.call.started",
        "tool.call.failed",
    ]
    failure = events[-1]
    assert failure.data["tool_name"] == "read_file"
    assert failure.data["risk"] == "read"
    assert failure.data["tool_call_id"] == events[0].data["tool_call_id"]
    assert failure.data["tool_call_id"]
    assert failure.run_id == events[0].run_id
    assert failure.data["elapsed_seconds"] >= 0
    assert failure.data["error_category"] == "workspace_tool_error"
    serialized = repr(failure)
    assert "TOP-SECRET" not in serialized
    assert "outside.txt" not in serialized
    assert "arguments" not in serialized


def test_separate_tool_invocations_have_unique_call_ids(tmp_path: Path) -> None:
    (tmp_path / "one.txt").write_bytes(b"one")
    (tmp_path / "two.txt").write_bytes(b"two")
    events: list[Any] = []
    registry = WorkspaceToolRegistry(tmp_path, event_sink=events.append)

    registry.read_file("one.txt")
    registry.read_file("two.txt")

    started = [event for event in events if event.name == "tool.call.started"]
    assert len(started) == 2
    assert started[0].data["tool_call_id"] != started[1].data["tool_call_id"]
