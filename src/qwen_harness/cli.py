"""Command-line entry point."""

from pathlib import Path
from typing import Any, NoReturn

import httpx
import typer
from pydantic import ValidationError

from qwen_harness.config import Settings
from qwen_harness.harness import (
    Harness,
    HarnessConfigurationError,
    ContextBudgetError,
    HarnessExecutionError,
    HarnessTimeoutError,
    IncompleteTaskError,
    EventSink,
    InvalidModelDecisionError,
    InvalidPromptError,
    InvalidTaskContractError,
)
from qwen_harness.metrics import SQLiteMetricsStore, generate_metrics_report
from qwen_harness.orchestration import RunEvent, TaskContract
from qwen_harness.tools import WORKSPACE_TOOL_NAMES, WorkspaceToolRegistry


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


def build_harness(
    *,
    event_sink: EventSink | None = None,
    workspace_root: Path | None = None,
) -> Harness:
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
    workspace_tools = WorkspaceToolRegistry(
        workspace_root or Path.cwd(),
        max_list_entries=settings.max_list_entries,
        max_file_bytes=settings.max_file_bytes,
        max_search_results=settings.max_search_results,
        max_search_files=settings.max_search_files,
        max_search_bytes=settings.max_search_bytes,
        max_line_characters=settings.max_line_characters,
        max_search_directories=settings.max_search_directories,
        tool_timeout_seconds=settings.tool_timeout_seconds,
        event_sink=event_sink,
    )
    agent = Agent(
        model,
        tools=list(workspace_tools.pydantic_ai_tools),
    )
    metrics_path = settings.metrics_database
    if not metrics_path.is_absolute():
        metrics_path = workspace_tools.workspace_root / metrics_path
    metrics_store = SQLiteMetricsStore(metrics_path)
    return Harness(
        agent=agent,
        max_steps=settings.max_steps,
        timeout_seconds=settings.timeout_seconds,
        max_output_tokens=settings.max_output_tokens,
        max_tool_calls=settings.max_tool_calls,
        event_sink=event_sink,
        workspace_root=workspace_tools.workspace_root,
        available_tools=frozenset(
            definition.name for definition in workspace_tools.definitions
        ),
        context_capacity=settings.context_window_tokens,
        context_overhead_tokens=settings.context_overhead_tokens,
        model_name=settings.model,
        metrics_sink=metrics_store.record,
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
            ContextBudgetError,
            HarnessTimeoutError,
            InvalidModelDecisionError,
            InvalidPromptError,
            InvalidTaskContractError,
            IncompleteTaskError,
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
    if isinstance(exc, IncompleteTaskError):
        return "The requested task was not completed."

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

    tool_name = event.data.get("tool_name")
    tool_call_id = event.data.get("tool_call_id")
    if tool_name:
        detail = f"{detail} tool={tool_name}"
    if tool_call_id:
        detail = f"{detail} call={tool_call_id}"

    elapsed = event.data.get("elapsed_seconds")
    if isinstance(elapsed, (int, float)):
        detail = f"{detail} elapsed={elapsed:.3f}s"

    category = event.data.get("error_category")
    message = event.data.get("message")
    reason = event.data.get("reason")
    if category:
        detail = f"{detail} category={category}"
    if message:
        detail = f"{detail} message={message}"
    if reason:
        detail = f"{detail} reason={reason}"

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
    require_tool: list[str] = typer.Option(
        [],
        "--require-tool",
        help="Require successful evidence for this tool; may be repeated.",
    ),
) -> None:
    """Send one prompt to Qwen and print its response."""
    try:
        unknown_tools = frozenset(require_tool) - WORKSPACE_TOOL_NAMES
        if unknown_tools:
            names = ", ".join(sorted(unknown_tools))
            raise InvalidTaskContractError(f"unknown required tool(s): {names}")
        harness = (
            build_harness(event_sink=_render_verbose_event)
            if verbose
            else build_harness()
        )
        if require_tool:
            contract = TaskContract(required_tools=frozenset(require_tool))
            typer.echo(harness.chat(prompt, contract=contract))
        else:
            typer.echo(harness.chat(prompt))
    except Exception as exc:
        if _is_expected_runtime_error(exc):
            _fail(_expected_error_message(exc))
        raise


@app.command()
def stats(
    database: Path | None = typer.Option(
        None,
        "--database",
        help="SQLite metrics database; defaults to harness configuration.",
    ),
    output: Path = typer.Option(
        Path("token-usage.html"),
        "--output",
        help="Self-contained HTML report to generate.",
    ),
    limit: int = typer.Option(
        50,
        "--limit",
        min=1,
        help="Maximum number of recent runs to chart.",
    ),
) -> None:
    """Generate local token and context-usage graphs without loading Qwen."""
    try:
        database_path = database or Settings().metrics_database
    except ValidationError as exc:
        _fail(str(exc))
    report = generate_metrics_report(
        SQLiteMetricsStore(database_path),
        output,
        limit=limit,
    )
    typer.echo(str(report.resolve()))


if __name__ == "__main__":
    app()
