from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, Callable, NoReturn

import pytest
from pydantic_ai.exceptions import ModelHTTPError, UnexpectedModelBehavior
from typer.testing import CliRunner

from qwen_harness.cli import app
from qwen_harness.harness import (
    HarnessConfigurationError,
    HarnessExecutionError,
    HarnessTimeoutError,
    InvalidModelDecisionError,
)


runner = CliRunner()


class StubHarness:
    def __init__(self, output: str = "Qwen says hello") -> None:
        self.output = output
        self.prompts: list[str] = []

    def chat(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.output


def test_chat_command_prints_agent_output(monkeypatch: pytest.MonkeyPatch) -> None:
    harness = StubHarness()
    monkeypatch.setattr("qwen_harness.cli.build_harness", lambda: harness)

    result = runner.invoke(app, ["chat", "Hello"])

    assert result.exit_code == 0
    assert harness.prompts == ["Hello"]
    assert "Qwen says hello" in result.output


def test_chat_command_is_quiet_without_verbose_logging(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = StubHarness()
    monkeypatch.setattr("qwen_harness.cli.build_harness", lambda: harness)

    result = runner.invoke(app, ["chat", "Hello"])

    assert result.exit_code == 0
    assert result.stdout == "Qwen says hello\n"
    assert result.stderr == ""


def test_verbose_chat_logs_run_boundaries_to_stderr_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id = "run-test-123"

    class VerboseHarness:
        def __init__(self, event_sink: Callable[[Any], None]) -> None:
            self.event_sink = event_sink

        def chat(self, prompt: str) -> str:
            timestamp = datetime(2026, 8, 23, 10, 0, tzinfo=timezone.utc)
            for name, data in [
                ("run.started", {}),
                (
                    "state.transition",
                    {"from_state": "created", "to_state": "building_context"},
                ),
                ("model.call.started", {}),
                (
                    "tool.call.started",
                    {
                        "tool_name": "read_file",
                        "risk": "read",
                        "tool_call_id": "tool-call-123",
                    },
                ),
                (
                    "tool.call.completed",
                    {
                        "tool_name": "read_file",
                        "risk": "read",
                        "tool_call_id": "tool-call-123",
                        "elapsed_seconds": 0.1,
                    },
                ),
                ("model.call.completed", {"elapsed_seconds": 0.25}),
                ("verification.started", {}),
                ("verification.completed", {}),
                (
                    "state.transition",
                    {"from_state": "verifying", "to_state": "succeeded"},
                ),
                ("run.completed", {"elapsed_seconds": 0.5}),
            ]:
                self.event_sink(
                    SimpleNamespace(
                        run_id=run_id,
                        timestamp=timestamp,
                        name=name,
                        data=data,
                    )
                )
            return "Qwen says hello"

    def build_verbose_harness(
        *, event_sink: Callable[[Any], None] | None = None
    ) -> VerboseHarness:
        assert event_sink is not None
        return VerboseHarness(event_sink)

    monkeypatch.setattr("qwen_harness.cli.build_harness", build_verbose_harness)

    result = runner.invoke(app, ["chat", "--verbose", "Hello"])

    assert result.exit_code == 0
    assert result.stdout == "Qwen says hello\n"
    assert run_id in result.stderr
    assert "created -> building_context" in result.stderr
    assert "model.call.started" in result.stderr
    assert "read_file" in result.stderr
    assert "tool-call-123" in result.stderr
    assert "verification.started" in result.stderr
    assert "verifying -> succeeded" in result.stderr
    assert "elapsed=" in result.stderr
    assert "run.completed" in result.stderr


def test_verbose_failure_redacts_raw_exception_text_from_stderr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sensitive_error = (
        "connection failed\n"
        "Authorization: Bearer super-secret-token\x1b[31m"
    )

    class VerboseFailingHarness:
        def __init__(self, event_sink: Callable[[Any], None]) -> None:
            self.event_sink = event_sink

        def chat(self, prompt: str) -> NoReturn:
            self.event_sink(
                SimpleNamespace(
                    run_id="run-failed-123",
                    timestamp=datetime(
                        2026, 8, 23, 10, 0, tzinfo=timezone.utc
                    ),
                    name="run.failed",
                    data={
                        "error_category": "connection_error",
                        "message": "The model service is unavailable.",
                        "elapsed_seconds": 0.1,
                    },
                )
            )
            raise _execution_error_caused_by(ConnectionError(sensitive_error))

    def build_failing_harness(
        *, event_sink: Callable[[Any], None] | None = None
    ) -> VerboseFailingHarness:
        assert event_sink is not None
        return VerboseFailingHarness(event_sink)

    monkeypatch.setattr("qwen_harness.cli.build_harness", build_failing_harness)

    result = runner.invoke(app, ["chat", "--verbose", "Hello"])

    assert result.exit_code != 0
    assert result.stdout == ""
    assert "connection_error" in result.stderr
    assert "The model service is unavailable." in result.stderr
    assert "super-secret-token" not in result.stderr
    assert "Authorization" not in result.stderr
    assert "\x1b" not in result.stderr


@pytest.mark.parametrize("verbose", [False, True])
def test_invalid_model_decision_uses_a_fixed_safe_cli_message(
    monkeypatch: pytest.MonkeyPatch,
    verbose: bool,
) -> None:
    sensitive_error = (
        "invalid response: /think private reasoning\n"
        "Authorization: Bearer super-secret-token\x1b[31m"
    )

    class InvalidDecisionHarness:
        def chat(self, prompt: str) -> NoReturn:
            raise InvalidModelDecisionError(sensitive_error)

    def build_invalid_harness(
        *, event_sink: Callable[[Any], None] | None = None
    ) -> InvalidDecisionHarness:
        return InvalidDecisionHarness()

    monkeypatch.setattr("qwen_harness.cli.build_harness", build_invalid_harness)
    arguments = ["chat"]
    if verbose:
        arguments.append("--verbose")
    arguments.append("Hello")

    result = runner.invoke(app, arguments)

    assert result.exit_code != 0
    assert result.stdout == ""
    assert "Error: The model returned an invalid response." in result.stderr
    assert "private reasoning" not in result.stderr
    assert "super-secret-token" not in result.stderr
    assert "Authorization" not in result.stderr
    assert "\x1b" not in result.stderr


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (
            HarnessConfigurationError("invalid harness configuration"),
            "invalid harness configuration",
        ),
        (ConnectionError("cannot connect to Ollama"), "cannot connect to Ollama"),
        (HarnessTimeoutError("agent timed out"), "agent timed out"),
    ],
)
def test_chat_command_reports_expected_startup_errors_without_traceback(
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
    message: str,
) -> None:
    def fail_to_build() -> NoReturn:
        raise error

    monkeypatch.setattr("qwen_harness.cli.build_harness", fail_to_build)

    result = runner.invoke(app, ["chat", "Hello"])

    assert result.exit_code != 0
    assert message in result.output
    assert "Traceback" not in result.output


def test_chat_command_does_not_hide_unexpected_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_to_build() -> NoReturn:
        raise RuntimeError("programming defect")

    monkeypatch.setattr("qwen_harness.cli.build_harness", fail_to_build)

    result = runner.invoke(app, ["chat", "Hello"])

    assert isinstance(result.exception, RuntimeError)


def _execution_error_caused_by(cause: Exception) -> HarnessExecutionError:
    error = HarnessExecutionError(str(cause))
    error.__cause__ = cause
    return error


def test_chat_command_reports_wrapped_model_format_failure_without_traceback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cause = UnexpectedModelBehavior(
        "model output did not match the decision schema"
    )
    error = _execution_error_caused_by(cause)

    class InvalidOutputHarness:
        def chat(self, prompt: str) -> NoReturn:
            raise error

    monkeypatch.setattr("qwen_harness.cli.build_harness", InvalidOutputHarness)

    result = runner.invoke(app, ["chat", "Hello"])

    assert result.exit_code != 0
    assert "did not match the decision schema" in result.output
    assert "Traceback" not in result.output
    assert result.exception is not error


def test_chat_command_does_not_hide_wrapped_programming_defect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error = _execution_error_caused_by(RuntimeError("programming defect"))

    class DefectiveHarness:
        def chat(self, prompt: str) -> NoReturn:
            raise error

    monkeypatch.setattr("qwen_harness.cli.build_harness", DefectiveHarness)

    result = runner.invoke(app, ["chat", "Hello"])

    assert result.exception is error
    assert "Error:" not in result.output


def test_chat_command_explains_ollama_model_runner_crash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error = ModelHTTPError(
        status_code=500,
        model_name="qwen3.5:4b",
        body={
            "message": (
                "llama-server process has terminated: exit status 0xe06d7363: "
                "NTSTATUS 0xe06d7363"
            )
        },
    )

    class CrashingHarness:
        def chat(self, prompt: str) -> NoReturn:
            raise error

    monkeypatch.setattr("qwen_harness.cli.build_harness", CrashingHarness)

    result = runner.invoke(app, ["chat", "Hello"])

    assert result.exit_code != 0
    assert "Ollama's model runner crashed" in result.output
    assert "CPU-only" in result.output
    assert "Ollama logs" in result.output
    assert "status_code: 500" in result.output
    assert "NTSTATUS 0xe06d7363" in result.output
    assert "Traceback" not in result.output


def test_chat_command_does_not_label_an_unrelated_http_500_as_runner_crash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error = ModelHTTPError(
        status_code=500,
        model_name="qwen3.5:4b",
        body={"message": "internal server error"},
    )

    class FailingHarness:
        def chat(self, prompt: str) -> NoReturn:
            raise error

    monkeypatch.setattr("qwen_harness.cli.build_harness", FailingHarness)

    result = runner.invoke(app, ["chat", "Hello"])

    assert result.exit_code != 0
    assert "internal server error" in result.output
    assert "model runner crashed" not in result.output
    assert "Traceback" not in result.output


def test_openai_client_disables_hidden_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_client(**kwargs: object) -> object:
        captured.update(kwargs)
        return object()

    monkeypatch.setattr("openai.AsyncOpenAI", fake_client)

    from qwen_harness.cli import build_openai_client
    from qwen_harness.config import Settings

    build_openai_client(Settings(timeout_seconds=17))

    assert captured["max_retries"] == 0
    assert captured["timeout"] == 17


def test_stats_with_explicit_database_ignores_unrelated_invalid_context_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    database = tmp_path / "existing-metrics.sqlite3"
    output = tmp_path / "report.html"
    monkeypatch.setenv("HARNESS_CONTEXT_WINDOW_TOKENS", "100")
    monkeypatch.setenv("HARNESS_MAX_OUTPUT_TOKENS", "100")

    result = runner.invoke(
        app,
        [
            "stats",
            "--database",
            str(database),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0
    assert output.exists()
    assert "Traceback" not in result.output


def test_stats_without_database_reports_invalid_settings_without_traceback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    monkeypatch.setenv("HARNESS_CONTEXT_WINDOW_TOKENS", "100")
    monkeypatch.setenv("HARNESS_MAX_OUTPUT_TOKENS", "100")

    result = runner.invoke(
        app,
        ["stats", "--output", str(tmp_path / "report.html")],
    )

    assert result.exit_code != 0
    assert "Error:" in result.output
    assert "Traceback" not in result.output
