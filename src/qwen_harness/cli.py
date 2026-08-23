"""Command-line entry point."""

from typing import Any, NoReturn

import httpx
import typer
from pydantic import ValidationError

from qwen_harness.config import Settings
from qwen_harness.harness import (
    Harness,
    HarnessConfigurationError,
    HarnessExecutionError,
    HarnessTimeoutError,
    EventSink,
    InvalidModelDecisionError,
    InvalidPromptError,
)
from qwen_harness.orchestration import RunEvent


app = typer.Typer(
    name="qwen-harness",
    no_args_is_help=True,
    help="Run a bounded Qwen agent through a local Ollama server.",
)


@app.callback()
def main() -> None:
    """Run the local Qwen harness."""


def build_openai_client(settings: Settings) -> Any:
    """Create a local client without automatic timeout retries."""
    from openai import AsyncOpenAI

    return AsyncOpenAI(
        base_url=settings.ollama_base_url,
        api_key="ollama",
        timeout=settings.timeout_seconds,
        max_retries=0,
    )


def build_harness(*, event_sink: EventSink | None = None) -> Harness:
    """Build the production harness from environment-backed settings."""
    from pydantic_ai import Agent
    from pydantic_ai.models.ollama import OllamaModel
    from pydantic_ai.providers.ollama import OllamaProvider

    try:
        settings = Settings()
    except ValidationError as exc:
        raise HarnessConfigurationError(str(exc)) from exc

    provider = OllamaProvider(openai_client=build_openai_client(settings))
    model = OllamaModel(settings.model, provider=provider)
    agent = Agent(
        model,
        instructions=(
            "You are a careful local assistant. Follow the user's request, "
            "state uncertainty plainly, and do not claim to have used tools "
            "that were not provided."
        ),
    )
    return Harness(
        agent=agent,
        max_steps=settings.max_steps,
        timeout_seconds=settings.timeout_seconds,
        max_output_tokens=settings.max_output_tokens,
        event_sink=event_sink,
    )


def _fail(message: str) -> NoReturn:
    typer.echo(f"Error: {message}", err=True)
    raise typer.Exit(code=1)


def _is_expected_runtime_error(exc: Exception) -> bool:
    if isinstance(exc, HarnessExecutionError):
        cause = exc.__cause__
        return isinstance(cause, Exception) and _is_expected_runtime_error(cause)

    if isinstance(
        exc,
        (
            HarnessConfigurationError,
            HarnessTimeoutError,
            InvalidModelDecisionError,
            InvalidPromptError,
            ValidationError,
            ConnectionError,
            httpx.HTTPError,
        ),
    ):
        return True

    from openai import APIConnectionError, APITimeoutError
    from pydantic_ai.exceptions import (
        ModelAPIError,
        UnexpectedModelBehavior,
        UsageLimitExceeded,
        UserError,
    )

    return isinstance(
        exc,
        (
            APIConnectionError,
            APITimeoutError,
            ModelAPIError,
            UnexpectedModelBehavior,
            UsageLimitExceeded,
            UserError,
        ),
    )


def _expected_error_message(exc: Exception) -> str:
    if isinstance(exc, InvalidModelDecisionError):
        return "The model returned an invalid response."

    if isinstance(exc, HarnessExecutionError):
        cause = exc.__cause__
        if isinstance(cause, ConnectionError):
            return "The model service is unavailable."

        from pydantic_ai.exceptions import UnexpectedModelBehavior

        if isinstance(cause, UnexpectedModelBehavior):
            return "model output did not match the decision schema"
        return "The model call failed."

    original = str(exc)
    if "llama-server process has terminated" in original.lower():
        return (
            "Ollama's model runner crashed. Use the CPU-only model and check "
            "the Ollama logs in %LOCALAPPDATA%\\Ollama. "
            f"Original error: {original}"
        )
    return original


def _render_verbose_event(event: RunEvent) -> None:
    """Render one sanitized event without mixing logs into the answer stream."""
    timestamp = event.timestamp.isoformat(timespec="milliseconds")
    if event.name == "state.transition":
        detail = f"{event.data['from_state']} -> {event.data['to_state']}"
    else:
        detail = event.name

    elapsed = event.data.get("elapsed_seconds")
    if isinstance(elapsed, (int, float)):
        detail = f"{detail} elapsed={elapsed:.3f}s"

    category = event.data.get("error_category")
    message = event.data.get("message")
    if category:
        detail = f"{detail} category={category}"
    if message:
        detail = f"{detail} message={message}"

    typer.echo(
        f"[trace] {timestamp} run={event.run_id} {detail}",
        err=True,
    )


@app.command()
def chat(
    prompt: str = typer.Argument(..., help="The message to send to Qwen."),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show sanitized run events on stderr (not private model reasoning).",
    ),
) -> None:
    """Send one prompt to Qwen and print its response."""
    try:
        harness = (
            build_harness(event_sink=_render_verbose_event)
            if verbose
            else build_harness()
        )
        typer.echo(harness.chat(prompt))
    except Exception as exc:
        if _is_expected_runtime_error(exc):
            _fail(_expected_error_message(exc))
        raise


if __name__ == "__main__":
    app()
