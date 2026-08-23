from dataclasses import dataclass
from typing import Any

import pytest
from pydantic import TypeAdapter, ValidationError
from pydantic_ai.exceptions import UnexpectedModelBehavior

from qwen_harness.harness import (
    Harness,
    HarnessError,
    HarnessTimeoutError,
    InvalidModelDecisionError,
)
from qwen_harness.orchestration import (
    FinishDecision,
    InvalidStateTransition,
    ModelDecision,
    RunLifecycle,
    RunState,
)


def test_model_decision_accepts_the_terminal_finish_action() -> None:
    payload = {"action": "finish", "content": "The task is complete."}

    decision = TypeAdapter(ModelDecision).validate_python(payload)

    assert isinstance(decision, FinishDecision)
    assert decision.content == payload["content"]


@pytest.mark.parametrize("action", ["respond", "delete_everything"])
def test_model_decision_rejects_a_non_terminal_action(action: str) -> None:
    with pytest.raises(ValidationError):
        TypeAdapter(ModelDecision).validate_python(
            {"action": action, "content": "not a terminal decision"}
        )


def test_model_decision_schema_is_one_strict_terminal_object() -> None:
    schema = TypeAdapter(ModelDecision).json_schema()

    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {"action", "content"}
    assert schema["properties"]["action"]["const"] == "finish"
    assert "oneOf" not in schema


def test_model_decision_rejects_a_missing_action() -> None:
    with pytest.raises(ValidationError):
        TypeAdapter(ModelDecision).validate_python({"content": "ambiguous"})


def test_model_decision_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        TypeAdapter(ModelDecision).validate_python(
            {"action": "finish", "content": "valid", "unexpected": "not allowed"}
        )


def test_run_lifecycle_records_each_legal_transition() -> None:
    lifecycle = RunLifecycle()

    assert lifecycle.state is RunState.CREATED

    lifecycle.transition(RunState.BUILDING_CONTEXT)
    lifecycle.transition(RunState.CALLING_MODEL)
    lifecycle.transition(RunState.VERIFYING)
    lifecycle.transition(RunState.SUCCEEDED)

    assert lifecycle.state is RunState.SUCCEEDED
    assert [
        (event.from_state, event.to_state) for event in lifecycle.trace
    ] == [
        (RunState.CREATED, RunState.BUILDING_CONTEXT),
        (RunState.BUILDING_CONTEXT, RunState.CALLING_MODEL),
        (RunState.CALLING_MODEL, RunState.VERIFYING),
        (RunState.VERIFYING, RunState.SUCCEEDED),
    ]


def test_run_lifecycle_rejects_illegal_transition_without_mutating_trace() -> None:
    lifecycle = RunLifecycle()

    with pytest.raises(
        InvalidStateTransition,
        match=r"CREATED.*VERIFYING",
    ):
        lifecycle.transition(RunState.VERIFYING)

    assert lifecycle.state is RunState.CREATED
    assert lifecycle.trace == ()


@dataclass
class TextRunResult:
    output: str


class TextAgent:
    def __init__(self, output: str) -> None:
        self.output = output
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def run(self, prompt: str, **kwargs: Any) -> TextRunResult:
        self.calls.append((prompt, kwargs))
        return TextRunResult(output=self.output)


def test_harness_wraps_plain_model_text_in_a_strict_terminal_decision() -> None:
    content = "A verified final response"
    agent = TextAgent(content)
    harness = Harness(
        agent=agent,
        max_steps=4,
        timeout_seconds=20,
        max_output_tokens=128,
    )

    run = harness.run("Hello, Qwen")

    assert run.output == content
    assert run.decision == FinishDecision(action="finish", content=content)
    assert run.final_state is RunState.SUCCEEDED
    assert [event.to_state for event in run.trace] == [
        RunState.BUILDING_CONTEXT,
        RunState.CALLING_MODEL,
        RunState.VERIFYING,
        RunState.SUCCEEDED,
    ]
    assert agent.calls[0][1]["output_type"] is str
    assert harness.chat("Hello again") == content


def test_harness_rejects_the_observed_leading_think_envelope() -> None:
    agent = TextAgent("/think I need...")
    harness = Harness(
        agent=agent,
        max_steps=2,
        timeout_seconds=20,
        max_output_tokens=64,
    )

    with pytest.raises(
        InvalidModelDecisionError,
        match="reasoning|incomplete",
    ) as captured:
        harness.run("Explain an agent harness")

    _assert_failed_run_error(
        captured.value,
        [
            RunState.BUILDING_CONTEXT,
            RunState.CALLING_MODEL,
            RunState.VERIFYING,
            RunState.FAILED,
        ],
    )


@pytest.mark.parametrize(
    "content",
    [
        "The tags `<think>...</think>` can delimit reasoning in some model outputs.",
        (
            "For example, a raw model response might contain:\n"
            "```text\n<think>I should inspect the request.</think>\n```"
        ),
    ],
)
def test_harness_allows_legitimate_explanations_of_think_tags(
    content: str,
) -> None:
    harness = Harness(
        agent=TextAgent(content),
        max_steps=2,
        timeout_seconds=20,
        max_output_tokens=128,
    )

    run = harness.run("Explain think tags")

    assert run.output == content
    assert run.decision == FinishDecision(action="finish", content=content)
    assert run.final_state is RunState.SUCCEEDED


def test_harness_accepts_a_normal_answer_containing_the_word_think() -> None:
    content = (
        "I think an agent harness is the software that controls a model, "
        "its tools, and its execution limits."
    )
    harness = Harness(
        agent=TextAgent(content),
        max_steps=2,
        timeout_seconds=20,
        max_output_tokens=64,
    )

    run = harness.run("Explain an agent harness")

    assert run.output == content
    assert run.final_state is RunState.SUCCEEDED


def test_harness_rejects_whitespace_only_model_output_during_verification() -> None:
    harness = Harness(
        agent=TextAgent("  \n\t"),
        max_steps=2,
        timeout_seconds=20,
        max_output_tokens=64,
    )

    with pytest.raises(
        InvalidModelDecisionError,
        match="empty|whitespace|invalid decision",
    ) as captured:
        harness.run("Explain an agent harness")

    _assert_failed_run_error(
        captured.value,
        [
            RunState.BUILDING_CONTEXT,
            RunState.CALLING_MODEL,
            RunState.VERIFYING,
            RunState.FAILED,
        ],
    )


class SlowTextAgent:
    async def run(self, prompt: str, **kwargs: Any) -> TextRunResult:
        import asyncio

        await asyncio.sleep(0.1)
        return TextRunResult(output="too late")


class RaisingTextAgent:
    async def run(self, prompt: str, **kwargs: Any) -> TextRunResult:
        raise RuntimeError("model boundary failed")


class InvalidModelOutputAgent:
    async def run(self, prompt: str, **kwargs: Any) -> TextRunResult:
        raise UnexpectedModelBehavior(
            "model output did not match the decision schema",
            body='{"content":"missing action"}',
        )


@dataclass
class UnvalidatedRunResult:
    output: Any


class MalformedDecisionAgent:
    async def run(self, prompt: str, **kwargs: Any) -> UnvalidatedRunResult:
        return UnvalidatedRunResult(
            output={"action": "unknown", "content": "cannot dispatch"}
        )


def _assert_failed_run_error(
    error: HarnessError,
    expected_states: list[RunState],
) -> None:
    assert error.final_state is RunState.FAILED
    assert isinstance(error.trace, tuple)
    assert [event.to_state for event in error.trace] == expected_states


def test_timeout_exposes_failed_state_and_immutable_trace() -> None:
    harness = Harness(
        agent=SlowTextAgent(),
        max_steps=2,
        timeout_seconds=0.01,
        max_output_tokens=64,
    )

    with pytest.raises(HarnessTimeoutError, match="0.01") as captured:
        harness.run("Hello")

    _assert_failed_run_error(
        captured.value,
        [
            RunState.BUILDING_CONTEXT,
            RunState.CALLING_MODEL,
            RunState.FAILED,
        ],
    )


def test_model_exception_exposes_failed_state_and_immutable_trace() -> None:
    harness = Harness(
        agent=RaisingTextAgent(),
        max_steps=2,
        timeout_seconds=20,
        max_output_tokens=64,
    )

    with pytest.raises(HarnessError, match="model boundary failed") as captured:
        harness.run("Hello")

    _assert_failed_run_error(
        captured.value,
        [
            RunState.BUILDING_CONTEXT,
            RunState.CALLING_MODEL,
            RunState.FAILED,
        ],
    )


def test_agent_model_format_failure_is_an_invalid_decision_with_failed_trace() -> None:
    harness = Harness(
        agent=InvalidModelOutputAgent(),
        max_steps=2,
        timeout_seconds=20,
        max_output_tokens=64,
    )

    with pytest.raises(
        InvalidModelDecisionError,
        match="did not match the decision schema",
    ) as captured:
        harness.run("Hello")

    assert isinstance(captured.value.__cause__, UnexpectedModelBehavior)
    _assert_failed_run_error(
        captured.value,
        [
            RunState.BUILDING_CONTEXT,
            RunState.CALLING_MODEL,
            RunState.FAILED,
        ],
    )


def test_malformed_decision_exposes_verification_then_failed_trace() -> None:
    harness = Harness(
        agent=MalformedDecisionAgent(),
        max_steps=2,
        timeout_seconds=20,
        max_output_tokens=64,
    )

    with pytest.raises(HarnessError) as captured:
        harness.run("Hello")

    _assert_failed_run_error(
        captured.value,
        [
            RunState.BUILDING_CONTEXT,
            RunState.CALLING_MODEL,
            RunState.VERIFYING,
            RunState.FAILED,
        ],
    )
