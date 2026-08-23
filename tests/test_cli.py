from typing import NoReturn

import pytest
from pydantic_ai.exceptions import ModelHTTPError, UnexpectedModelBehavior
from typer.testing import CliRunner

from qwen_harness.cli import app
from qwen_harness.harness import (
    HarnessConfigurationError,
    HarnessExecutionError,
    HarnessTimeoutError,
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
