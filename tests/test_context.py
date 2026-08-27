from dataclasses import dataclass
from importlib import import_module
import re
from typing import Any

from qwen_harness.harness import Harness
from qwen_harness.observability import emit_current_run_event
from qwen_harness.orchestration import TaskContract


@dataclass
class TextResult:
    output: str


class RecordingAgent:
    def __init__(self, output: str = "ordinary answer") -> None:
        self.output = output
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def run(self, prompt: str, **kwargs: Any) -> TextResult:
        self.calls.append((prompt, kwargs))
        return TextResult(self.output)


class SuccessfulListAgent(RecordingAgent):
    async def run(self, prompt: str, **kwargs: Any) -> TextResult:
        self.calls.append((prompt, kwargs))
        event_data = {
            "tool_name": "list_files",
            "tool_call_id": "context-test-call",
            "risk": "read",
        }
        emit_current_run_event("tool.call.started", event_data)
        emit_current_run_event(
            "tool.call.completed",
            {**event_data, "elapsed_seconds": 0.01},
        )
        return TextResult(self.output)


def _harness(agent: RecordingAgent, *, event_sink: Any = None) -> Harness:
    return Harness(
        agent=agent,
        max_steps=3,
        max_tool_calls=3,
        timeout_seconds=20,
        max_output_tokens=128,
        event_sink=event_sink,
        available_tools=frozenset({"list_files", "read_file", "search_text"}),
    )


def test_context_builder_creates_concise_versioned_invariant_context() -> None:
    context_module = import_module("qwen_harness.context")
    builder = context_module.ContextBuilder()

    context = builder.build(
        contract=TaskContract(),
        available_tools=frozenset({"list_files", "read_file", "search_text"}),
    )

    normalized = context.instructions.casefold()
    assert "workspace" in normalized
    assert "read-only" in normalized
    assert "modified" in normalized
    assert len(context.instructions) <= 500
    assert re.fullmatch(r"[a-z0-9]+(?:[._-][a-z0-9]+)*", context.version)


def test_plain_chat_receives_context_without_changing_prompt_or_output() -> None:
    secret_prompt = "Explain harnesses; private-marker-7391"
    agent = RecordingAgent(output="same user-visible answer")

    output = _harness(agent).chat(secret_prompt)

    assert output == "same user-visible answer"
    assert agent.calls[0][0] == secret_prompt
    instructions = agent.calls[0][1]["instructions"]
    normalized = instructions.casefold()
    assert "workspace" in normalized
    assert "read-only" in normalized
    assert secret_prompt not in instructions


def test_required_tool_context_names_only_explicitly_required_tools_once() -> None:
    agent = SuccessfulListAgent()
    secret_prompt = "List files for private-project-8824"

    _harness(agent).chat(
        secret_prompt,
        contract=TaskContract(required_tools={"list_files"}),
    )

    instructions = agent.calls[0][1]["instructions"]
    assert instructions.count("list_files") == 1
    assert "read_file" not in instructions
    assert "search_text" not in instructions
    assert "delete_file" not in instructions
    assert secret_prompt not in instructions


def test_irrelevant_available_tools_are_not_enumerated_for_plain_chat() -> None:
    agent = RecordingAgent()

    _harness(agent).chat("What is dependency injection?")

    instructions = agent.calls[0][1]["instructions"]
    assert "list_files" not in instructions
    assert "read_file" not in instructions
    assert "search_text" not in instructions
    assert "delete_file" not in instructions


def test_context_version_is_logged_without_prompt_or_instructions() -> None:
    events: list[Any] = []
    secret_prompt = "Explain Python private-marker-1543"

    _harness(RecordingAgent(), event_sink=events.append).chat(secret_prompt)

    context_events = [event for event in events if event.name == "context.built"]
    assert len(context_events) == 1
    event = context_events[0]
    assert re.fullmatch(
        r"[a-z0-9]+(?:[._-][a-z0-9]+)*",
        event.data["context_version"],
    )
    assert "prompt" not in event.data
    assert "instructions" not in event.data
    assert secret_prompt not in repr(event)

    names = [item.name for item in events]
    assert names.index("context.built") < names.index("model.call.started")
