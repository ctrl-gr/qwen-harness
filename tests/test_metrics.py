import asyncio
import sqlite3
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RequestUsage, RunUsage
from typer.testing import CliRunner

from qwen_harness.cli import app
from qwen_harness.harness import Harness
from qwen_harness.metrics import (
    RequestTokenUsage,
    RunMetrics,
    SQLiteMetricsStore,
    TokenUsage,
    collect_token_usage,
    generate_metrics_report,
)


@dataclass
class ProviderResult:
    output: str
    usage: RunUsage
    messages: list[ModelRequest | ModelResponse]

    def all_messages(self) -> list[ModelRequest | ModelResponse]:
        return self.messages


def _provider_result() -> ProviderResult:
    secret_prompt = "PRIVATE user prompt that metrics must not retain"
    secret_output = "PRIVATE model output that metrics must not retain"
    return ProviderResult(
        output="safe answer",
        usage=RunUsage(
            input_tokens=1_500,
            output_tokens=80,
            requests=2,
            tool_calls=1,
        ),
        messages=[
            ModelRequest(parts=[UserPromptPart(secret_prompt)]),
            ModelResponse(
                parts=[TextPart(secret_output)],
                usage=RequestUsage(input_tokens=600, output_tokens=30),
            ),
            ModelResponse(
                parts=[TextPart(secret_output)],
                usage=RequestUsage(input_tokens=900, output_tokens=50),
            ),
        ],
    )


def test_collects_provider_usage_and_distinguishes_total_from_peak_context() -> None:
    usage = collect_token_usage(_provider_result(), context_capacity=2_048)

    assert usage == TokenUsage(
        input_tokens=1_500,
        output_tokens=80,
        total_tokens=1_580,
        requests=2,
        tool_calls=1,
        peak_context_tokens=900,
        context_capacity=2_048,
        context_utilization=pytest.approx(900 / 2_048),
        per_request=(
            RequestTokenUsage(
                request_index=1,
                input_tokens=600,
                output_tokens=30,
                total_tokens=630,
            ),
            RequestTokenUsage(
                request_index=2,
                input_tokens=900,
                output_tokens=50,
                total_tokens=950,
            ),
        ),
    )
    serialized = repr(usage)
    assert "PRIVATE user prompt" not in serialized
    assert "PRIVATE model output" not in serialized


def test_usage_collection_does_not_estimate_tokens_for_legacy_results() -> None:
    result = type(
        "LegacyResult",
        (),
        {"output": "A very long response that must not be counted by characters."},
    )()

    assert collect_token_usage(result, context_capacity=2_048) is None


def test_usage_collection_does_not_call_current_pydantic_usage_property() -> None:
    result = asyncio.run(Agent(TestModel(custom_output_text="safe")).run("Hello"))

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        usage = collect_token_usage(result, context_capacity=2_048)

    assert usage is not None
    assert not [item for item in captured if "usage" in str(item.message)]


def test_usage_collection_excludes_response_usage_from_supplied_history() -> None:
    historical = ModelResponse(
        parts=[TextPart("old")],
        usage=RequestUsage(input_tokens=1_900, output_tokens=10),
    )
    current = ModelResponse(
        parts=[TextPart("new")],
        usage=RequestUsage(input_tokens=700, output_tokens=20),
    )

    class ResultWithHistory:
        output = "safe"
        usage = RunUsage(input_tokens=700, output_tokens=20, requests=1)

        def all_messages(self) -> list[ModelResponse]:
            return [historical, current]

        def new_messages(self) -> list[ModelResponse]:
            return [current]

    usage = collect_token_usage(ResultWithHistory(), context_capacity=2_048)

    assert usage is not None
    assert usage.peak_context_tokens == 700
    assert len(usage.per_request) == 1


def test_harness_remains_compatible_when_fake_result_has_no_usage() -> None:
    events: list[Any] = []

    class LegacyAgent:
        async def run(self, prompt: str, **kwargs: Any) -> Any:
            return type("LegacyResult", (), {"output": "legacy answer"})()

    harness = Harness(
        agent=LegacyAgent(),
        max_steps=2,
        timeout_seconds=20,
        max_output_tokens=64,
        context_capacity=2_048,
        event_sink=events.append,
    )

    run = harness.run("Hello")

    assert run.output == "legacy answer"
    assert run.token_usage is None
    assert "model.usage" not in [event.name for event in events]


def test_broken_optional_usage_does_not_change_a_successful_run() -> None:
    class BrokenUsageResult:
        output = "valid answer"

        @property
        def usage(self) -> Any:
            raise RuntimeError("provider usage is malformed")

    class BrokenUsageAgent:
        async def run(self, prompt: str, **kwargs: Any) -> BrokenUsageResult:
            return BrokenUsageResult()

    run = Harness(
        agent=BrokenUsageAgent(),
        max_steps=2,
        timeout_seconds=20,
        max_output_tokens=64,
    ).run("Hello")

    assert run.output == "valid answer"
    assert run.token_usage is None


class UsageAgent:
    async def run(self, prompt: str, **kwargs: Any) -> ProviderResult:
        return _provider_result()


def _usage_harness(**overrides: Any) -> Harness:
    arguments = {
        "agent": UsageAgent(),
        "max_steps": 4,
        "timeout_seconds": 20,
        "max_output_tokens": 128,
        "context_capacity": 2_048,
    }
    arguments.update(overrides)
    return Harness(**arguments)


def test_harness_exposes_sanitized_usage_and_emits_counts_only() -> None:
    events: list[Any] = []

    run = _usage_harness(event_sink=events.append).run("PRIVATE harness prompt")

    assert run.token_usage is not None
    assert run.token_usage.input_tokens == 1_500
    assert run.token_usage.peak_context_tokens == 900
    usage_event = next(event for event in events if event.name == "model.usage")
    assert usage_event.data == {
        "input_tokens": 1_500,
        "output_tokens": 80,
        "total_tokens": 1_580,
        "requests": 2,
        "tool_calls": 1,
        "peak_context_tokens": 900,
        "context_capacity": 2_048,
        "context_utilization": pytest.approx(900 / 2_048),
    }
    serialized = repr(usage_event)
    assert "PRIVATE" not in serialized
    assert "prompt" not in usage_event.data
    assert "output" not in usage_event.data


def test_metrics_sink_failure_does_not_change_a_successful_harness_run() -> None:
    def broken_metrics_sink(metrics: RunMetrics) -> None:
        raise OSError("metrics disk is unavailable")

    run = _usage_harness(metrics_sink=broken_metrics_sink).run("Hello")

    assert run.output == "safe answer"
    assert run.token_usage is not None


def _sample_metrics(
    *,
    run_id: str = "run-1",
    model: str = "qwen3.5:0.8b-cpu",
    requests: tuple[RequestTokenUsage, ...] | None = None,
) -> RunMetrics:
    request_rows = requests or (
        RequestTokenUsage(
            request_index=1,
            input_tokens=600,
            output_tokens=30,
            total_tokens=630,
        ),
        RequestTokenUsage(
            request_index=2,
            input_tokens=900,
            output_tokens=50,
            total_tokens=950,
        ),
    )
    usage = TokenUsage(
        input_tokens=1_500,
        output_tokens=80,
        total_tokens=1_580,
        requests=2,
        tool_calls=1,
        peak_context_tokens=900,
        context_capacity=2_048,
        context_utilization=900 / 2_048,
        per_request=request_rows,
    )
    return RunMetrics(
        run_id=run_id,
        timestamp=datetime(2026, 8, 29, 10, 30, tzinfo=timezone.utc),
        model=model,
        final_state="succeeded",
        elapsed_seconds=3.25,
        token_usage=usage,
    )


def test_sqlite_store_creates_parent_and_persists_sanitized_run_and_requests(
    tmp_path: Path,
) -> None:
    database = tmp_path / "nested" / "metrics.sqlite3"
    store = SQLiteMetricsStore(database)

    store.record(_sample_metrics())

    assert database.exists()
    assert store.runs() == (_sample_metrics(),)
    with sqlite3.connect(database) as connection:
        run_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(run_metrics)")
        }
        request_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(request_metrics)")
        }
        persisted = " ".join(
            str(value)
            for row in connection.execute("SELECT * FROM run_metrics")
            for value in row
        )

    assert run_columns == {
        "run_id",
        "timestamp",
        "model",
        "final_state",
        "elapsed_seconds",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "requests",
        "tool_calls",
        "peak_context_tokens",
        "context_capacity",
        "context_utilization",
    }
    assert request_columns == {
        "run_id",
        "request_index",
        "input_tokens",
        "output_tokens",
        "total_tokens",
    }
    assert "PRIVATE" not in persisted


def test_sqlite_store_rolls_back_run_when_a_request_row_fails(
    tmp_path: Path,
) -> None:
    database = tmp_path / "metrics.sqlite3"
    store = SQLiteMetricsStore(database)
    duplicate_requests = (
        RequestTokenUsage(
            request_index=1,
            input_tokens=10,
            output_tokens=2,
            total_tokens=12,
        ),
        RequestTokenUsage(
            request_index=1,
            input_tokens=20,
            output_tokens=3,
            total_tokens=23,
        ),
    )

    with pytest.raises(sqlite3.IntegrityError):
        store.record(_sample_metrics(requests=duplicate_requests))

    assert store.runs() == ()
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM request_metrics").fetchone() == (
            0,
        )


def test_report_is_self_contained_and_contains_both_token_graphs(
    tmp_path: Path,
) -> None:
    store = SQLiteMetricsStore(tmp_path / "metrics.sqlite3")
    store.record(_sample_metrics())
    report = tmp_path / "reports" / "tokens.html"

    generated = generate_metrics_report(store, report)

    assert generated == report
    html = report.read_text(encoding="utf-8")
    assert "<svg" in html
    assert "Input tokens" in html
    assert "Output tokens" in html
    assert "Context utilization" in html
    assert "Cumulative tokens" in html
    assert "Run latency and throughput" in html
    assert "tokens/second" in html
    assert "qwen3.5:0.8b-cpu" in html
    assert 'src="http' not in html
    assert 'href="http' not in html


def test_report_escapes_persisted_text_before_rendering_html(tmp_path: Path) -> None:
    store = SQLiteMetricsStore(tmp_path / "metrics.sqlite3")
    dangerous = '\"><script>alert("metrics")</script>'
    store.record(_sample_metrics(run_id=dangerous, model=dangerous))
    report = tmp_path / "tokens.html"

    generate_metrics_report(store, report)

    html = report.read_text(encoding="utf-8")
    assert dangerous not in html
    assert "&lt;script&gt;" in html


def test_limited_report_preserves_the_global_cumulative_token_total(
    tmp_path: Path,
) -> None:
    store = SQLiteMetricsStore(tmp_path / "metrics.sqlite3")
    for index in range(3):
        store.record(_sample_metrics(run_id=f"run-{index}"))
    report = tmp_path / "tokens.html"

    generate_metrics_report(store, report, limit=2)

    html = report.read_text(encoding="utf-8")
    assert "4740 cumulative tokens" in html
    assert "run-0:" not in html


def test_report_handles_an_empty_database(tmp_path: Path) -> None:
    store = SQLiteMetricsStore(tmp_path / "empty.sqlite3")
    report = tmp_path / "empty.html"

    generate_metrics_report(store, report)

    html = report.read_text(encoding="utf-8")
    assert "No token usage recorded yet" in html
    assert "Input tokens" in html
    assert "Context utilization" in html


def test_report_backs_up_an_existing_output_before_replacing_it(
    tmp_path: Path,
) -> None:
    store = SQLiteMetricsStore(tmp_path / "metrics.sqlite3")
    report = tmp_path / "tokens.html"
    report.write_text("previous report", encoding="utf-8")

    generate_metrics_report(store, report)

    assert report.read_text(encoding="utf-8") != "previous report"
    assert (tmp_path / "tokens.html.bak").read_text(encoding="utf-8") == (
        "previous report"
    )


def test_report_never_overwrites_an_existing_backup(tmp_path: Path) -> None:
    store = SQLiteMetricsStore(tmp_path / "metrics.sqlite3")
    report = tmp_path / "tokens.html"
    report.write_text("current report", encoding="utf-8")
    first_backup = tmp_path / "tokens.html.bak"
    first_backup.write_text("older backup", encoding="utf-8")

    generate_metrics_report(store, report)

    assert first_backup.read_text(encoding="utf-8") == "older backup"
    assert (tmp_path / "tokens.html.bak.1").read_text(encoding="utf-8") == (
        "current report"
    )


def test_stats_cli_generates_report_without_constructing_ollama(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def unexpected_model_construction(**kwargs: Any) -> None:
        raise AssertionError("stats must not construct the model harness")

    monkeypatch.setattr("qwen_harness.cli.build_harness", unexpected_model_construction)
    database = tmp_path / "metrics.sqlite3"
    report = tmp_path / "report.html"

    result = CliRunner().invoke(
        app,
        [
            "stats",
            "--database",
            str(database),
            "--output",
            str(report),
        ],
    )

    assert result.exit_code == 0
    assert report.exists()
    assert str(report) in result.stdout


def test_stats_cli_uses_configured_database_when_option_is_omitted(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    database = tmp_path / "configured.sqlite3"
    SQLiteMetricsStore(database).record(_sample_metrics())
    report = tmp_path / "configured-report.html"
    monkeypatch.setenv("HARNESS_METRICS_DATABASE", str(database))

    result = CliRunner().invoke(
        app,
        ["stats", "--output", str(report)],
    )

    assert result.exit_code == 0
    assert "qwen3.5:0.8b-cpu" in report.read_text(encoding="utf-8")
