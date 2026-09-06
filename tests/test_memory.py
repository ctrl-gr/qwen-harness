from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from threading import Barrier, Lock
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
from pydantic_ai.models.test import TestModel

from qwen_harness.harness import Harness, IncompleteTaskError, InvalidModelDecisionError
from qwen_harness.memory import WorkingMemory
from qwen_harness.observability import emit_current_run_event


NOW = datetime(2026, 8, 31, tzinfo=timezone.utc)


def _text_turn(prompt: str, answer: str) -> tuple[ModelMessage, ...]:
    return (
        ModelRequest(parts=[UserPromptPart(prompt, timestamp=NOW)]),
        ModelResponse(parts=[TextPart(answer)], timestamp=NOW),
    )


def _tool_turn() -> tuple[ModelMessage, ...]:
    return (
        ModelRequest(parts=[UserPromptPart("List files", timestamp=NOW)]),
        ModelResponse(
            parts=[ToolCallPart("list_files", {"path": "."}, "call-1")],
            timestamp=NOW,
        ),
        ModelRequest(
            parts=[
                ToolReturnPart(
                    "list_files",
                    ["README.md"],
                    "call-1",
                    timestamp=NOW,
                )
            ]
        ),
        ModelResponse(parts=[TextPart("README.md")], timestamp=NOW),
    )


def _serialized_size(messages: Sequence[ModelMessage]) -> int:
    return len(ModelMessagesTypeAdapter.dump_json(list(messages)))


@dataclass
class FakeRunResult:
    output: str
    messages: tuple[ModelMessage, ...]

    def new_messages(self) -> list[ModelMessage]:
        return list(self.messages)


class SequencedAgent:
    def __init__(self, results: Sequence[FakeRunResult]) -> None:
        self.results = list(results)
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def run(self, prompt: str, **kwargs: Any) -> FakeRunResult:
        self.calls.append((prompt, kwargs))
        return self.results.pop(0)


def _harness(agent: Any, *, events: list[Any] | None = None) -> Harness:
    return Harness(
        agent=agent,
        max_steps=4,
        max_tool_calls=4,
        timeout_seconds=20,
        max_output_tokens=128,
        available_tools=frozenset({"list_files"}),
        event_sink=events.append if events is not None else None,
    )


def test_successful_turn_is_loaded_into_the_next_model_call_as_typed_history() -> None:
    first_turn = _tool_turn()
    second_turn = _text_turn("Thanks", "You are welcome")
    agent = SequencedAgent(
        [
            FakeRunResult("README.md", first_turn),
            FakeRunResult("You are welcome", second_turn),
        ]
    )
    memory = WorkingMemory(max_turns=3, max_serialized_bytes=20_000)
    harness = _harness(agent)

    assert harness.chat("List files", memory=memory) == "README.md"
    assert harness.chat("Thanks", memory=memory) == "You are welcome"

    assert "message_history" not in agent.calls[0][1]
    assert agent.calls[1][1]["message_history"] == first_turn
    assert all(
        isinstance(message, (ModelRequest, ModelResponse))
        for message in agent.calls[1][1]["message_history"]
    )


def test_call_without_memory_omits_message_history_and_remains_compatible() -> None:
    agent = SequencedAgent(
        [FakeRunResult("ordinary answer", _text_turn("question", "ordinary answer"))]
    )

    output = _harness(agent).chat("question")

    assert output == "ordinary answer"
    assert "message_history" not in agent.calls[0][1]


def test_memory_retains_a_complete_tool_turn_including_call_and_result_pair() -> None:
    turn = _tool_turn()
    memory = WorkingMemory(max_turns=2, max_serialized_bytes=20_000)

    memory.remember(turn)

    snapshot = memory.snapshot()
    assert snapshot == turn
    assert isinstance(snapshot, tuple)
    assert isinstance(snapshot[1].parts[0], ToolCallPart)
    assert isinstance(snapshot[2].parts[0], ToolReturnPart)


@pytest.mark.parametrize(
    "malformed_turn",
    [
        (
            ModelRequest(parts=[UserPromptPart("List files", timestamp=NOW)]),
            ModelResponse(
                parts=[ToolCallPart("list_files", {"path": "."}, "unmatched")],
                timestamp=NOW,
            ),
        ),
        (
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        "list_files",
                        ["README.md"],
                        "unmatched",
                        timestamp=NOW,
                    )
                ]
            ),
            ModelResponse(parts=[TextPart("README.md")], timestamp=NOW),
        ),
    ],
    ids=["unmatched-tool-call", "unmatched-tool-return"],
)
def test_memory_drops_turns_with_unmatched_tool_messages(
    malformed_turn: tuple[ModelMessage, ...],
) -> None:
    memory = WorkingMemory(max_turns=2, max_serialized_bytes=20_000)

    update = memory.remember(malformed_turn)

    assert update.dropped_turns == 1
    assert update.evicted_turns == 0
    assert memory.snapshot() == ()


def test_valid_paired_tool_turn_is_accepted_by_structural_validation() -> None:
    memory = WorkingMemory(max_turns=2, max_serialized_bytes=20_000)
    valid_turn = _tool_turn()

    update = memory.remember(valid_turn)

    assert update.dropped_turns == 0
    assert memory.snapshot() == valid_turn


@pytest.mark.parametrize(
    "malformed_turn",
    [
        (
            ModelRequest(parts=[UserPromptPart("List files", timestamp=NOW)]),
            ModelResponse(
                parts=[ToolCallPart("list_files", {"path": "."}, "call-1")],
                timestamp=NOW,
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        "read_file",
                        "wrong tool for this call ID",
                        "call-1",
                        timestamp=NOW,
                    )
                ]
            ),
            ModelResponse(parts=[TextPart("done")], timestamp=NOW),
        ),
        (ModelRequest(parts=[UserPromptPart("request only", timestamp=NOW)]),),
        (ModelResponse(parts=[TextPart("response only")], timestamp=NOW),),
    ],
    ids=[
        "tool-name-mismatch-for-call-id",
        "request-only-sequence",
        "response-only-sequence",
    ],
)
def test_memory_drops_incomplete_or_mismatched_turn_sequences(
    malformed_turn: tuple[ModelMessage, ...],
) -> None:
    memory = WorkingMemory(max_turns=2, max_serialized_bytes=20_000)

    update = memory.remember(malformed_turn)

    assert update.dropped_turns == 1
    assert update.evicted_turns == 0
    assert memory.snapshot() == ()


def test_valid_text_request_response_turn_passes_structural_validation() -> None:
    memory = WorkingMemory(max_turns=2, max_serialized_bytes=20_000)
    valid_turn = _text_turn("question", "answer")

    update = memory.remember(valid_turn)

    assert update.dropped_turns == 0
    assert memory.snapshot() == valid_turn


def test_turn_limit_evicts_the_oldest_whole_turn() -> None:
    oldest = _text_turn("one", "first")
    newest = _text_turn("two", "second")
    memory = WorkingMemory(max_turns=1, max_serialized_bytes=20_000)

    memory.remember(oldest)
    update = memory.remember(newest)

    assert memory.snapshot() == newest
    assert update.evicted_turns == 1


def test_exact_byte_limit_is_inclusive_and_one_byte_less_evicts_whole_turn() -> None:
    oldest = _text_turn("one", "first")
    newest = _text_turn("two", "second")
    combined = oldest + newest
    exact_limit = _serialized_size(combined)

    exact = WorkingMemory(max_turns=2, max_serialized_bytes=exact_limit)
    exact.remember(oldest)
    exact.remember(newest)
    assert exact.snapshot() == combined
    assert _serialized_size(exact.snapshot()) == exact_limit

    constrained = WorkingMemory(max_turns=2, max_serialized_bytes=exact_limit - 1)
    constrained.remember(oldest)
    update = constrained.remember(newest)
    assert constrained.snapshot() == newest
    assert update.evicted_turns == 1
    assert _serialized_size(constrained.snapshot()) <= exact_limit - 1


def test_single_oversized_new_turn_is_dropped_without_losing_existing_memory() -> None:
    retained = _text_turn("short", "kept")
    oversized = _text_turn("large", "x" * 2_000)
    limit = _serialized_size(retained)
    memory = WorkingMemory(max_turns=2, max_serialized_bytes=limit)
    memory.remember(retained)

    update = memory.remember(oversized)

    assert memory.snapshot() == retained
    assert update.dropped_turns == 1
    assert _serialized_size(memory.snapshot()) <= limit


def test_only_successfully_verified_runs_are_remembered() -> None:
    invalid_turn = _text_turn("bad", "/think unfinished")
    incomplete_turn = _tool_turn()
    memory = WorkingMemory(max_turns=3, max_serialized_bytes=20_000)

    with pytest.raises(InvalidModelDecisionError):
        _harness(SequencedAgent([FakeRunResult("/think unfinished", invalid_turn)])).run(
            "bad", memory=memory
        )

    class FailedToolAgent(SequencedAgent):
        async def run(self, prompt: str, **kwargs: Any) -> FakeRunResult:
            emit_current_run_event(
                "tool.call.failed",
                {
                    "tool_name": "list_files",
                    "tool_call_id": "call-1",
                    "risk": "read",
                },
            )
            return await super().run(prompt, **kwargs)

    with pytest.raises(IncompleteTaskError):
        _harness(
            FailedToolAgent([FakeRunResult("Could not list files", incomplete_turn)])
        ).run("list", memory=memory)

    assert memory.snapshot() == ()


def test_memory_events_are_sanitized_and_contain_only_counts() -> None:
    secret = "do-not-log-this-memory-content"
    events: list[Any] = []
    memory = WorkingMemory(max_turns=2, max_serialized_bytes=20_000)
    agent = SequencedAgent([FakeRunResult("answer", _text_turn(secret, "answer"))])

    _harness(agent, events=events).run(secret, memory=memory)

    memory_events = [event for event in events if event.name.startswith("memory.")]
    assert [event.name for event in memory_events] == [
        "memory.loaded",
        "memory.updated",
    ]
    assert set(memory_events[0].data) == {
        "turn_count",
        "message_count",
        "serialized_bytes",
    }
    assert set(memory_events[1].data) == {
        "turn_count",
        "message_count",
        "serialized_bytes",
        "evicted_turn_count",
        "dropped_turn_count",
    }
    assert all(
        isinstance(value, int)
        for event in memory_events
        for value in event.data.values()
    )
    assert secret not in repr([dict(event.data) for event in memory_events])


def test_unserializable_new_messages_do_not_change_a_successful_run_or_memory() -> None:
    unserializable = (
        ModelResponse(
            parts=[TextPart("answer")],
            timestamp=NOW,
            metadata={"unsupported": object()},
        ),
    )
    memory = WorkingMemory(max_turns=2, max_serialized_bytes=20_000)

    run = _harness(
        SequencedAgent([FakeRunResult("answer", unserializable)])
    ).run("question", memory=memory)

    assert run.output == "answer"
    assert run.final_state.value == "succeeded"
    assert memory.snapshot() == ()


def test_clear_removes_all_history_and_memory_is_not_persistent() -> None:
    memory = WorkingMemory(max_turns=2, max_serialized_bytes=20_000)
    memory.remember(_text_turn("one", "first"))

    memory.clear()

    assert memory.snapshot() == ()
    assert WorkingMemory(max_turns=2, max_serialized_bytes=20_000).snapshot() == ()


def test_snapshot_cannot_mutate_the_messages_retained_by_memory() -> None:
    memory = WorkingMemory(max_turns=2, max_serialized_bytes=20_000)
    memory.remember(_text_turn("original", "answer"))

    snapshot = memory.snapshot()
    snapshot[0].parts[0].content = "mutated outside memory"

    assert memory.snapshot()[0].parts[0].content == "original"


def test_caller_mutation_during_serialization_cannot_change_retained_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    caller_turn = list(_text_turn("original", "answer"))
    original_size = _serialized_size(caller_turn)
    original_dump_json = ModelMessagesTypeAdapter.dump_json
    mutated = False

    def dump_json_then_mutate(messages: Any, *args: Any, **kwargs: Any) -> bytes:
        nonlocal mutated
        payload = original_dump_json(messages, *args, **kwargs)
        if not mutated:
            mutated = True
            caller_turn[0].parts[0].content = "x" * 2_000
        return payload

    monkeypatch.setattr(ModelMessagesTypeAdapter, "dump_json", dump_json_then_mutate)
    memory = WorkingMemory(max_turns=1, max_serialized_bytes=original_size)

    update = memory.remember(caller_turn)

    assert update.dropped_turns == 0
    assert memory.snapshot()[0].parts[0].content == "original"
    assert memory.serialized_bytes <= original_size


class BarrierAgent:
    def __init__(self, barrier: Barrier) -> None:
        self.barrier = barrier
        self.histories: list[tuple[ModelMessage, ...]] = []
        self._lock = Lock()

    async def run(self, prompt: str, **kwargs: Any) -> FakeRunResult:
        history = tuple(kwargs.get("message_history", ()))
        with self._lock:
            self.histories.append(history)
        await asyncio.to_thread(self.barrier.wait)
        return FakeRunResult(f"answer-{prompt}", _text_turn(prompt, f"answer-{prompt}"))


def test_concurrent_runs_from_same_memory_revision_commit_only_one_turn() -> None:
    initial_turn = _text_turn("initial", "context")
    memory = WorkingMemory(max_turns=4, max_serialized_bytes=40_000)
    memory.remember(initial_turn)
    events: list[Any] = []
    agent = BarrierAgent(Barrier(2))
    harness = _harness(agent, events=events)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(harness.run, prompt, memory=memory)
            for prompt in ("alpha", "beta")
        ]
        runs = [future.result(timeout=5) for future in futures]

    assert {run.output for run in runs} == {"answer-alpha", "answer-beta"}
    assert agent.histories == [initial_turn, initial_turn]
    retained = memory.snapshot()
    assert retained[: len(initial_turn)] == initial_turn
    retained_text = repr(retained[len(initial_turn) :])
    assert ("answer-alpha" in retained_text) ^ ("answer-beta" in retained_text)
    assert memory.turn_count == 2

    loaded_events = [event for event in events if event.name == "memory.loaded"]
    assert len(loaded_events) == 2
    assert all(
        dict(event.data)
        == {
            "turn_count": 1,
            "message_count": len(initial_turn),
            "serialized_bytes": _serialized_size(initial_turn),
        }
        for event in loaded_events
    )
    updated_events = [event for event in events if event.name == "memory.updated"]
    assert len(updated_events) == 2
    assert sorted(event.data["dropped_turn_count"] for event in updated_events) == [
        0,
        1,
    ]
    assert all(event.data["turn_count"] == 2 for event in updated_events)
    assert all(event.data["message_count"] == len(retained) for event in updated_events)
    assert all(
        event.data["serialized_bytes"] == _serialized_size(retained)
        for event in updated_events
    )


def test_real_pydantic_agent_result_round_trips_through_working_memory() -> None:
    memory = WorkingMemory(max_turns=2, max_serialized_bytes=20_000)
    harness = _harness(Agent(TestModel(custom_output_text="answer")))

    assert harness.chat("first", memory=memory) == "answer"
    assert harness.chat("second", memory=memory) == "answer"

    assert memory.turn_count == 2
    assert memory.message_count == 4


@pytest.mark.parametrize(
    ("max_turns", "max_serialized_bytes", "field_name"),
    [
        (0, 100, "max_turns"),
        (-1, 100, "max_turns"),
        (1, 0, "max_serialized_bytes"),
        (1, -1, "max_serialized_bytes"),
    ],
)
def test_constructor_rejects_non_positive_limits(
    max_turns: int, max_serialized_bytes: int, field_name: str
) -> None:
    with pytest.raises(ValueError, match=field_name):
        WorkingMemory(
            max_turns=max_turns,
            max_serialized_bytes=max_serialized_bytes,
        )
