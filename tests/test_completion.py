from dataclasses import dataclass
from typing import Any

import pytest

from qwen_harness.harness import Harness, IncompleteTaskError
from qwen_harness.observability import emit_current_run_event
from qwen_harness.orchestration import RunState, TaskContract, ToolCallStatus


@dataclass
class TextResult:
    output: str


class EvidenceAgent:
    def __init__(self, outcomes: list[tuple[str, str]], output: str = "done") -> None:
        self.outcomes = outcomes
        self.output = output
        self.calls: list[dict[str, Any]] = []

    async def run(self, prompt: str, **kwargs: Any) -> TextResult:
        self.calls.append(kwargs)
        for index, (tool_name, outcome) in enumerate(self.outcomes):
            call_id = f"call-{index}"
            common = {
                "tool_name": tool_name,
                "tool_call_id": call_id,
                "risk": "read",
            }
            emit_current_run_event("tool.call.started", common)
            emit_current_run_event(
                f"tool.call.{outcome}",
                {
                    **common,
                    "elapsed_seconds": 0.01,
                    **(
                        {"error_category": "workspace_tool_error"}
                        if outcome == "failed"
                        else {}
                    ),
                },
            )
        return TextResult(self.output)


def _harness(agent: EvidenceAgent) -> Harness:
    return Harness(
        agent=agent,
        max_steps=4,
        max_tool_calls=4,
        timeout_seconds=20,
        max_output_tokens=128,
        available_tools=frozenset({"list_files", "read_file", "search_text"}),
    )


def test_required_tool_success_produces_a_successful_run_with_evidence() -> None:
    agent = EvidenceAgent([("list_files", "completed")])
    contract = TaskContract(required_tools={"list_files"})

    run = _harness(agent).run("List files", contract=contract)

    assert run.final_state is RunState.SUCCEEDED
    assert len(run.tool_evidence) == 1
    assert run.tool_evidence[0].tool_name == "list_files"
    assert run.tool_evidence[0].status is ToolCallStatus.SUCCEEDED
    assert "list_files" in agent.calls[0]["instructions"]


def test_failed_then_successful_retry_for_same_tool_satisfies_contract() -> None:
    run = _harness(
        EvidenceAgent(
            [("list_files", "failed"), ("list_files", "completed")]
        )
    ).run("List files", contract=TaskContract(required_tools={"list_files"}))

    assert run.final_state is RunState.SUCCEEDED
    assert [item.status for item in run.tool_evidence] == [
        ToolCallStatus.FAILED,
        ToolCallStatus.SUCCEEDED,
    ]


def test_required_tool_without_success_ends_incomplete_and_hides_model_excuse() -> None:
    events: list[Any] = []
    harness = _harness(EvidenceAgent([], output="I could not do it, but all is fine"))
    object.__setattr__(harness, "event_sink", events.append)

    with pytest.raises(IncompleteTaskError) as captured:
        harness.run(
            "List files",
            contract=TaskContract(required_tools={"list_files"}),
        )

    assert captured.value.final_state is RunState.INCOMPLETE
    assert "I could not do it" not in str(captured.value)
    assert captured.value.unmet_tools == frozenset({"list_files"})
    assert [event.to_state for event in captured.value.trace][-2:] == [
        RunState.VERIFYING,
        RunState.INCOMPLETE,
    ]
    assert [event.name for event in events][-3:] == [
        "verification.failed",
        "state.transition",
        "run.incomplete",
    ]


def test_attempted_tool_that_never_succeeds_is_automatically_required() -> None:
    harness = _harness(EvidenceAgent([("read_file", "failed")]))

    with pytest.raises(IncompleteTaskError) as captured:
        harness.run("Inspect the file")

    assert captured.value.unmet_tools == frozenset({"read_file"})
    assert captured.value.tool_evidence[0].status is ToolCallStatus.FAILED


def test_started_tool_without_terminal_event_cannot_be_reported_as_success() -> None:
    class StartedOnlyAgent(EvidenceAgent):
        async def run(self, prompt: str, **kwargs: Any) -> TextResult:
            emit_current_run_event(
                "tool.call.started",
                {
                    "tool_name": "read_file",
                    "tool_call_id": "interrupted-call",
                    "risk": "read",
                },
            )
            return TextResult("done")

    with pytest.raises(IncompleteTaskError) as captured:
        _harness(StartedOnlyAgent([])).run("Inspect a file")

    assert captured.value.unmet_tools == frozenset({"read_file"})
    assert captured.value.tool_evidence == ()


def test_plain_chat_without_tool_activity_remains_compatible() -> None:
    harness = _harness(EvidenceAgent([], output="ordinary answer"))

    assert harness.chat("What is Python?") == "ordinary answer"


def test_unknown_required_tool_is_rejected_before_calling_model() -> None:
    agent = EvidenceAgent([])

    with pytest.raises(ValueError, match="delete_file"):
        _harness(agent).run(
            "Delete it",
            contract=TaskContract(required_tools={"delete_file"}),
        )

    assert agent.calls == []
