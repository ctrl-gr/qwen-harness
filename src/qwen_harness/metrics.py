"""Sanitized token metrics, local persistence, and static reporting."""

from dataclasses import dataclass
from datetime import datetime
from html import escape
from pathlib import Path
import shutil
import sqlite3
from typing import Any
from uuid import uuid4


@dataclass(frozen=True)
class RequestTokenUsage:
    """Provider-reported token counts for one model request."""

    request_index: int
    input_tokens: int
    output_tokens: int
    total_tokens: int


@dataclass(frozen=True)
class TokenUsage:
    """Aggregate and peak context usage for one complete agent run."""

    input_tokens: int
    output_tokens: int
    total_tokens: int
    requests: int
    tool_calls: int
    peak_context_tokens: int | None
    context_capacity: int
    context_utilization: float | None
    per_request: tuple[RequestTokenUsage, ...]


@dataclass(frozen=True)
class RunMetrics:
    """Prompt-free metrics persisted for one terminal harness run."""

    run_id: str
    timestamp: datetime
    model: str
    final_state: str
    elapsed_seconds: float
    token_usage: TokenUsage


def collect_token_usage(result: Any, *, context_capacity: int) -> TokenUsage | None:
    """Read provider usage without estimating tokens from user or model text."""
    if context_capacity <= 0:
        raise ValueError("context_capacity must be greater than zero")
    try:
        return _collect_token_usage(result, context_capacity=context_capacity)
    except Exception:
        # Usage metadata is optional telemetry and must not change task outcome.
        return None


def _collect_token_usage(result: Any, *, context_capacity: int) -> TokenUsage | None:
    raw_usage = getattr(result, "usage", None)
    if raw_usage is None:
        return None
    required = ("input_tokens", "output_tokens", "requests", "tool_calls")
    if not all(hasattr(raw_usage, name) for name in required) and callable(raw_usage):
        raw_usage = raw_usage()
    if not all(hasattr(raw_usage, name) for name in required):
        return None

    per_request: list[RequestTokenUsage] = []
    messages_getter = getattr(result, "new_messages", None)
    if not callable(messages_getter):
        messages_getter = getattr(result, "all_messages", None)
    if callable(messages_getter):
        for message in messages_getter():
            if getattr(message, "kind", None) != "response":
                continue
            usage = getattr(message, "usage", None)
            if usage is None:
                continue
            input_tokens = int(getattr(usage, "input_tokens", 0))
            output_tokens = int(getattr(usage, "output_tokens", 0))
            per_request.append(
                RequestTokenUsage(
                    request_index=len(per_request) + 1,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=input_tokens + output_tokens,
                )
            )

    peak = max((item.input_tokens for item in per_request), default=None)
    input_tokens = int(raw_usage.input_tokens)
    output_tokens = int(raw_usage.output_tokens)
    return TokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        requests=int(raw_usage.requests),
        tool_calls=int(raw_usage.tool_calls),
        peak_context_tokens=peak,
        context_capacity=context_capacity,
        context_utilization=None if peak is None else peak / context_capacity,
        per_request=tuple(per_request),
    )


class SQLiteMetricsStore:
    """Persist sanitized run/request metrics in one local SQLite database."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path)
        connection.execute("PRAGMA foreign_keys = ON")
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS run_metrics (
                run_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                model TEXT NOT NULL,
                final_state TEXT NOT NULL,
                elapsed_seconds REAL NOT NULL,
                input_tokens INTEGER NOT NULL,
                output_tokens INTEGER NOT NULL,
                total_tokens INTEGER NOT NULL,
                requests INTEGER NOT NULL,
                tool_calls INTEGER NOT NULL,
                peak_context_tokens INTEGER,
                context_capacity INTEGER NOT NULL,
                context_utilization REAL
            );
            CREATE TABLE IF NOT EXISTS request_metrics (
                run_id TEXT NOT NULL,
                request_index INTEGER NOT NULL,
                input_tokens INTEGER NOT NULL,
                output_tokens INTEGER NOT NULL,
                total_tokens INTEGER NOT NULL,
                PRIMARY KEY (run_id, request_index),
                FOREIGN KEY (run_id) REFERENCES run_metrics(run_id) ON DELETE CASCADE
            );
            """
        )
        return connection

    def record(self, metrics: RunMetrics) -> None:
        """Atomically persist a run and each of its provider request rows."""
        usage = metrics.token_usage
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO run_metrics VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    metrics.run_id,
                    metrics.timestamp.isoformat(),
                    metrics.model,
                    metrics.final_state,
                    metrics.elapsed_seconds,
                    usage.input_tokens,
                    usage.output_tokens,
                    usage.total_tokens,
                    usage.requests,
                    usage.tool_calls,
                    usage.peak_context_tokens,
                    usage.context_capacity,
                    usage.context_utilization,
                ),
            )
            connection.executemany(
                """
                INSERT INTO request_metrics VALUES (?, ?, ?, ?, ?)
                """,
                (
                    (
                        metrics.run_id,
                        item.request_index,
                        item.input_tokens,
                        item.output_tokens,
                        item.total_tokens,
                    )
                    for item in usage.per_request
                ),
            )

    def runs(self, *, limit: int | None = None) -> tuple[RunMetrics, ...]:
        """Return recorded runs oldest-first, optionally keeping only the latest."""
        if limit is not None and limit <= 0:
            raise ValueError("limit must be greater than zero")
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM run_metrics ORDER BY timestamp, run_id"
            ).fetchall()
            if limit is not None:
                rows = rows[-limit:]
            result: list[RunMetrics] = []
            for row in rows:
                request_rows = connection.execute(
                    """
                    SELECT request_index, input_tokens, output_tokens, total_tokens
                    FROM request_metrics WHERE run_id = ? ORDER BY request_index
                    """,
                    (row[0],),
                ).fetchall()
                per_request = tuple(RequestTokenUsage(*item) for item in request_rows)
                usage = TokenUsage(
                    input_tokens=row[5],
                    output_tokens=row[6],
                    total_tokens=row[7],
                    requests=row[8],
                    tool_calls=row[9],
                    peak_context_tokens=row[10],
                    context_capacity=row[11],
                    context_utilization=row[12],
                    per_request=per_request,
                )
                result.append(
                    RunMetrics(
                        run_id=row[0],
                        timestamp=datetime.fromisoformat(row[1]),
                        model=row[2],
                        final_state=row[3],
                        elapsed_seconds=row[4],
                        token_usage=usage,
                    )
                )
        return tuple(result)


def generate_metrics_report(
    store: SQLiteMetricsStore,
    output_path: Path,
    *,
    limit: int = 50,
) -> Path:
    """Generate a dependency-free, self-contained HTML report with SVG charts."""
    all_runs = store.runs()
    runs = all_runs[-limit:]
    cumulative_baseline = sum(
        item.token_usage.total_tokens for item in all_runs[: -len(runs)]
    ) if runs else 0
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        _create_verified_backup(output_path)
    temporary = output_path.with_name(f".{output_path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(
            _report_html(runs, cumulative_baseline=cumulative_baseline),
            encoding="utf-8",
        )
        temporary.replace(output_path)
    finally:
        temporary.unlink(missing_ok=True)
    return output_path


def _create_verified_backup(path: Path) -> Path:
    index = 1
    while True:
        suffix = ".bak" if index == 1 else f".bak.{index - 1}"
        candidate = path.with_name(f"{path.name}{suffix}")
        try:
            with path.open("rb") as source, candidate.open("xb") as destination:
                shutil.copyfileobj(source, destination)
            shutil.copystat(path, candidate)
        except FileExistsError:
            index += 1
            continue
        except Exception:
            candidate.unlink(missing_ok=True)
            raise
        if candidate.read_bytes() != path.read_bytes():
            candidate.unlink(missing_ok=True)
            raise OSError("report backup verification failed")
        return candidate


def _report_html(
    runs: tuple[RunMetrics, ...], *, cumulative_baseline: int = 0
) -> str:
    empty = "" if runs else '<p class="empty">No token usage recorded yet.</p>'
    models = ", ".join(sorted({escape(item.model) for item in runs}))
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Qwen token usage</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:1100px;margin:2rem auto;padding:0 1rem;color:#18212f;background:#f7f9fc}}
.card{{background:white;border:1px solid #dce3ed;border-radius:12px;padding:1rem;margin:1rem 0}}
svg{{width:100%;height:auto}} .input{{fill:#4f7cff}} .output{{fill:#50b58b}} .line{{fill:none;stroke:#e06b4f;stroke-width:3}}
.cumulative{{fill:none;stroke:#4f7cff;stroke-width:3}} .latency{{fill:none;stroke:#805ad5;stroke-width:3}}
.grid{{stroke:#dce3ed;stroke-width:1}} text{{font-size:12px;fill:#526071}} .empty{{color:#526071}}
</style></head><body><h1>Qwen token usage</h1><p>{models}</p>{empty}
<section class="card"><h2>Input tokens and Output tokens</h2>{_token_svg(runs)}</section>
<section class="card"><h2>Context utilization</h2>{_context_svg(runs)}</section>
<section class="card"><h2>Cumulative tokens</h2>{_cumulative_svg(runs, initial_total=cumulative_baseline)}</section>
<section class="card"><h2>Run latency and throughput</h2>{_latency_svg(runs)}</section>
</body></html>"""


def _token_svg(runs: tuple[RunMetrics, ...]) -> str:
    width, height, baseline = 900, 300, 260
    maximum = max((item.token_usage.total_tokens for item in runs), default=1)
    bar_width = max(8, min(44, 700 // max(1, len(runs))))
    gap = max(4, (width - 80) // max(1, len(runs)))
    shapes: list[str] = []
    for index, run in enumerate(runs):
        safe_run_id = escape(run.run_id)
        x = 55 + index * gap
        input_height = 210 * run.token_usage.input_tokens / maximum
        output_height = 210 * run.token_usage.output_tokens / maximum
        shapes.append(
            f'<rect class="input" x="{x}" y="{baseline-input_height:.1f}" width="{bar_width}" height="{input_height:.1f}"><title>{safe_run_id}: {run.token_usage.input_tokens} input tokens</title></rect>'
        )
        shapes.append(
            f'<rect class="output" x="{x}" y="{baseline-input_height-output_height:.1f}" width="{bar_width}" height="{output_height:.1f}"><title>{safe_run_id}: {run.token_usage.output_tokens} output tokens</title></rect>'
        )
    return _svg_frame(width, height, shapes)


def _context_svg(runs: tuple[RunMetrics, ...]) -> str:
    width, height, baseline = 900, 300, 260
    points: list[str] = []
    circles: list[str] = []
    gap = (width - 100) / max(1, len(runs) - 1)
    for index, run in enumerate(runs):
        safe_run_id = escape(run.run_id)
        utilization = run.token_usage.context_utilization or 0.0
        x = 55 + index * gap
        y = baseline - min(utilization, 1.0) * 210
        points.append(f"{x:.1f},{y:.1f}")
        circles.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="#e06b4f"><title>{safe_run_id}: {utilization:.1%}</title></circle>'
        )
    shapes = [f'<polyline class="line" points="{" ".join(points)}"/>', *circles]
    return _svg_frame(width, height, shapes)


def _cumulative_svg(
    runs: tuple[RunMetrics, ...], *, initial_total: int = 0
) -> str:
    total = initial_total
    values: list[float] = []
    titles: list[str] = []
    for run in runs:
        total += run.token_usage.total_tokens
        values.append(float(total))
        titles.append(f"{escape(run.run_id)}: {total} cumulative tokens")
    return _series_svg(values, titles, css_class="cumulative", color="#4f7cff")


def _latency_svg(runs: tuple[RunMetrics, ...]) -> str:
    values = [run.elapsed_seconds for run in runs]
    titles: list[str] = []
    for run in runs:
        throughput = (
            run.token_usage.total_tokens / run.elapsed_seconds
            if run.elapsed_seconds > 0
            else 0.0
        )
        titles.append(
            f"{escape(run.run_id)}: {run.elapsed_seconds:.2f}s, "
            f"{throughput:.2f} whole-run tokens/second"
        )
    return _series_svg(values, titles, css_class="latency", color="#805ad5")


def _series_svg(
    values: list[float],
    titles: list[str],
    *,
    css_class: str,
    color: str,
) -> str:
    width, height, baseline = 900, 300, 260
    maximum = max(values, default=1.0) or 1.0
    gap = (width - 100) / max(1, len(values) - 1)
    points: list[str] = []
    circles: list[str] = []
    for index, value in enumerate(values):
        x = 55 + index * gap
        y = baseline - min(value / maximum, 1.0) * 210
        points.append(f"{x:.1f},{y:.1f}")
        circles.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}">'
            f"<title>{titles[index]}</title></circle>"
        )
    shapes = [
        f'<polyline class="{css_class}" points="{" ".join(points)}"/>',
        *circles,
    ]
    return _svg_frame(width, height, shapes)


def _svg_frame(width: int, height: int, shapes: list[str]) -> str:
    return (
        f'<svg viewBox="0 0 {width} {height}" role="img">'
        '<line class="grid" x1="45" y1="260" x2="875" y2="260"/>'
        + "".join(shapes)
        + "</svg>"
    )
