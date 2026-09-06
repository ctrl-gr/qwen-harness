"""Minimal, deterministic per-run context construction."""

from dataclasses import dataclass
from math import ceil
from typing import Sequence

from pydantic import TypeAdapter
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter
from pydantic_ai.models import ModelRequestContext, ModelRequestParameters
from pydantic_ai.tools import RunContext

from qwen_harness.observability import emit_current_run_event
from qwen_harness.orchestration import TaskContract


CONTEXT_VERSION = "context.v1"
DEFAULT_CONTEXT_OVERHEAD_TOKENS = 256


class ContextBudgetError(ValueError):
    """Raised when mandatory context cannot fit before the model call."""


@dataclass(frozen=True)
class TaskContext:
    """Instructions and identity for one model-call context."""

    version: str
    instructions: str


@dataclass(frozen=True)
class ContextBudgetPlan:
    """Sanitized preflight plan for one bounded model context."""

    selected_messages: tuple[ModelMessage, ...]
    context_capacity: int
    reserved_output_tokens: int
    reserved_overhead_tokens: int
    available_input_tokens: int
    estimated_input_tokens: int
    selected_memory_turn_count: int
    selected_memory_message_count: int
    trimmed_memory_turn_count: int
    trimmed_memory_message_count: int


class ContextBudgetPlanner:
    """Fit the newest contiguous whole memory turns around essential context."""

    def plan(
        self,
        *,
        instructions: str,
        prompt: str,
        memory_turns: Sequence[Sequence[ModelMessage]],
        context_capacity: int,
        reserved_output_tokens: int,
        reserved_overhead_tokens: int,
    ) -> ContextBudgetPlan:
        available_input_tokens = (
            context_capacity
            - reserved_output_tokens
            - reserved_overhead_tokens
        )
        essential_tokens = _conservative_text_tokens(instructions) + (
            _conservative_text_tokens(prompt)
        )
        if essential_tokens > available_input_tokens:
            raise ContextBudgetError(
                "essential prompt and instructions do not fit the context budget"
            )

        selected_reversed: list[tuple[ModelMessage, ...]] = []
        estimated_input_tokens = essential_tokens
        for turn in reversed(memory_turns):
            complete_turn = tuple(turn)
            turn_tokens = _conservative_message_tokens(complete_turn)
            if estimated_input_tokens + turn_tokens > available_input_tokens:
                break
            selected_reversed.append(complete_turn)
            estimated_input_tokens += turn_tokens

        selected_turns = tuple(reversed(selected_reversed))
        selected_messages = tuple(
            message for turn in selected_turns for message in turn
        )
        trimmed_turns = len(memory_turns) - len(selected_turns)
        trimmed_messages = sum(
            len(turn) for turn in memory_turns[:trimmed_turns]
        )
        return ContextBudgetPlan(
            selected_messages=selected_messages,
            context_capacity=context_capacity,
            reserved_output_tokens=reserved_output_tokens,
            reserved_overhead_tokens=reserved_overhead_tokens,
            available_input_tokens=available_input_tokens,
            estimated_input_tokens=estimated_input_tokens,
            selected_memory_turn_count=len(selected_turns),
            selected_memory_message_count=len(selected_messages),
            trimmed_memory_turn_count=trimmed_turns,
            trimmed_memory_message_count=trimmed_messages,
        )


class ContextBudgetCapability(AbstractCapability[object]):
    """Reject every prepared provider request that exceeds the local budget."""

    def __init__(
        self,
        *,
        context_capacity: int,
        reserved_output_tokens: int,
        reserved_overhead_tokens: int,
    ) -> None:
        self.context_capacity = context_capacity
        self.reserved_output_tokens = reserved_output_tokens
        self.reserved_overhead_tokens = reserved_overhead_tokens

    async def before_model_request(
        self,
        ctx: RunContext[object],
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        available = (
            self.context_capacity
            - self.reserved_output_tokens
            - self.reserved_overhead_tokens
        )
        estimated = _prepared_request_tokens(
            request_context.messages,
            request_context.model_request_parameters,
        )
        emit_current_run_event(
            "model.context.checked",
            {
                "request_number": int(ctx.run_step),
                "context_capacity": self.context_capacity,
                "reserved_output_tokens": self.reserved_output_tokens,
                "reserved_overhead_tokens": self.reserved_overhead_tokens,
                "available_input_tokens": available,
                "estimated_input_tokens": estimated,
                "function_tool_count": len(
                    request_context.model_request_parameters.function_tools
                ),
                "output_tool_count": len(
                    request_context.model_request_parameters.output_tools
                ),
            },
        )
        if estimated > available:
            raise ContextBudgetError(
                "prepared model request does not fit the context budget"
            )
        return request_context


class ContextBuilder:
    """Build concise instructions without copying user or tool-result content."""

    def build(
        self,
        *,
        contract: TaskContract,
        available_tools: frozenset[str],
    ) -> TaskContext:
        unknown_tools = contract.required_tools - available_tools
        if unknown_tools:
            names = ", ".join(sorted(unknown_tools))
            raise ValueError(f"unknown required tool(s): {names}")

        instructions = (
            "You are a careful local software assistant. Answer directly when "
            "workspace inspection is unnecessary. Workspace tools are read-only "
            "and restricted to the configured workspace. Never claim files were "
            "modified. State uncertainty plainly."
        )
        if contract.required_tools:
            required_names = ", ".join(sorted(contract.required_tools))
            instructions += (
                " Completion contract: successfully call these tools before "
                f"answering: {required_names}. Retry correctable tool errors. "
                "Do not claim completion without successful tool evidence."
            )

        return TaskContext(
            version=CONTEXT_VERSION,
            instructions=instructions,
        )


def _conservative_text_tokens(text: str) -> int:
    """Upper-bound Qwen text tokens using UTF-8 bytes without a model tokenizer."""
    return len(text.encode("utf-8"))


def _conservative_message_tokens(messages: Sequence[ModelMessage]) -> int:
    """Upper-bound message payload tokens using stable public serialization."""
    structural_reserve = 16 * len(messages)
    return structural_reserve + len(_serialized_message_payload(messages))


def _prepared_request_tokens(
    messages: Sequence[ModelMessage],
    parameters: ModelRequestParameters,
) -> int:
    """Estimate a rendered request, including instructions and actual tool schemas."""
    message_bytes = len(_serialized_message_payload(messages))
    parameter_bytes = len(
        TypeAdapter(ModelRequestParameters).dump_json(parameters)
    )
    return ceil((message_bytes + parameter_bytes) / 3)


def _serialized_message_payload(messages: Sequence[ModelMessage]) -> bytes:
    """Serialize model-facing payload while excluding storage-only metadata."""
    return ModelMessagesTypeAdapter.dump_json(
        list(messages),
        exclude={
            "__all__": {
                "timestamp": True,
                "instructions": True,
                "kind": True,
                "run_id": True,
                "conversation_id": True,
                "metadata": True,
                "usage": True,
                "model_name": True,
                "provider_name": True,
                "provider_url": True,
                "provider_details": True,
                "provider_response_id": True,
                "finish_reason": True,
                "state": True,
                "parts": {
                    "__all__": {
                        "timestamp": True,
                        "part_kind": True,
                        "id": True,
                        "provider_name": True,
                        "provider_details": True,
                    }
                },
            }
        },
    )
