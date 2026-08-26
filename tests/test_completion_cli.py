from typing import Any, NoReturn

import pytest
from typer.testing import CliRunner

from qwen_harness.cli import app
from qwen_harness.harness import IncompleteTaskError
from qwen_harness.orchestration import TaskContract


runner = CliRunner()


class ContractRecordingHarness:
    def __init__(self) -> None:
        self.calls: list[tuple[str, TaskContract | None]] = []

    def chat(
        self,
        prompt: str,
        *,
        contract: TaskContract | None = None,
    ) -> str:
        self.calls.append((prompt, contract))
        return "completed"


def test_cli_passes_repeatable_required_tools_as_one_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = ContractRecordingHarness()
    monkeypatch.setattr("qwen_harness.cli.build_harness", lambda: harness)

    result = runner.invoke(
        app,
        [
            "chat",
            "--require-tool",
            "list_files",
            "--require-tool",
            "read_file",
            "Inspect the workspace",
        ],
    )

    assert result.exit_code == 0
    assert result.stdout == "completed\n"
    assert harness.calls == [
        (
            "Inspect the workspace",
            TaskContract(required_tools=frozenset({"list_files", "read_file"})),
        )
    ]


def test_cli_rejects_unknown_required_tool_without_starting_the_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_build(**kwargs: Any) -> NoReturn:
        raise AssertionError("the model must not start for an unknown tool")

    monkeypatch.setattr("qwen_harness.cli.build_harness", unexpected_build)

    result = runner.invoke(
        app,
        ["chat", "--require-tool", "delete_everything", "Do it"],
    )

    assert result.exit_code != 0
    assert result.stdout == ""
    assert "unknown required tool" in result.stderr.lower()
    assert "delete_everything" in result.stderr
    assert "Traceback" not in result.output


def test_cli_incomplete_task_is_nonzero_and_hides_the_model_excuse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    excuse = "I cannot do that, but I will claim success anyway."

    class IncompleteHarness:
        def chat(
            self,
            prompt: str,
            *,
            contract: TaskContract | None = None,
        ) -> NoReturn:
            raise IncompleteTaskError("Required tool did not complete.", evidence=())

    monkeypatch.setattr("qwen_harness.cli.build_harness", IncompleteHarness)

    result = runner.invoke(
        app,
        ["chat", "--require-tool", "list_files", excuse],
    )

    assert result.exit_code != 0
    assert result.stdout == ""
    assert "Error: The requested task was not completed." in result.stderr
    assert excuse not in result.output
    assert "Traceback" not in result.output


def test_verbose_cli_renders_verification_failure_reason_without_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class VerboseIncompleteHarness:
        def __init__(self, event_sink: Any) -> None:
            self.event_sink = event_sink

        def chat(
            self,
            prompt: str,
            *,
            contract: TaskContract | None = None,
        ) -> NoReturn:
            from datetime import datetime, timezone
            from types import MappingProxyType

            from qwen_harness.orchestration import RunEvent

            self.event_sink(
                RunEvent(
                    run_id="incomplete-run",
                    timestamp=datetime(2026, 8, 25, tzinfo=timezone.utc),
                    name="verification.failed",
                    data=MappingProxyType(
                        {
                            "reason": "required_tool_not_completed",
                            "content": "TOP-SECRET model output",
                            "arguments": {"path": "TOP-SECRET.txt"},
                        }
                    ),
                )
            )
            raise IncompleteTaskError("Required tool did not complete.", evidence=())

    def build_verbose_harness(*, event_sink: Any = None) -> VerboseIncompleteHarness:
        assert event_sink is not None
        return VerboseIncompleteHarness(event_sink)

    monkeypatch.setattr("qwen_harness.cli.build_harness", build_verbose_harness)

    result = runner.invoke(
        app,
        ["chat", "--verbose", "--require-tool", "list_files", "Inspect"],
    )

    assert result.exit_code != 0
    assert "verification.failed" in result.stderr
    assert "reason=required_tool_not_completed" in result.stderr
    assert "TOP-SECRET" not in result.output
    assert "arguments" not in result.output
    assert "content" not in result.output
