"""Minimal, deterministic per-run context construction."""

from dataclasses import dataclass

from qwen_harness.orchestration import TaskContract


CONTEXT_VERSION = "context.v1"


@dataclass(frozen=True)
class TaskContext:
    """Instructions and identity for one model-call context."""

    version: str
    instructions: str


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
