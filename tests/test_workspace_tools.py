import os
import subprocess
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel
from pydantic_ai import ModelRetry

from qwen_harness.tools import (
    FileTooLargeError,
    InvalidToolArgumentsError,
    ToolRisk,
    UnknownToolError,
    WorkspacePathError,
    WorkspaceToolError,
    WorkspaceToolRegistry,
)


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "README.md").write_bytes(b"Qwen harness\nSafe tools\n")
    source = root / "src"
    source.mkdir()
    (source / "agent.py").write_bytes(
        b"def run_agent():\n    return 'Qwen'\n"
    )
    return root


def test_registry_resolves_the_explicit_workspace_root(workspace: Path) -> None:
    registry = WorkspaceToolRegistry(workspace / "src" / "..")

    assert registry.workspace_root == workspace.resolve()


def test_registry_exposes_only_typed_read_tools(workspace: Path) -> None:
    registry = WorkspaceToolRegistry(workspace)

    assert {definition.name for definition in registry.definitions} == {
        "list_files",
        "read_file",
        "search_text",
    }
    for definition in registry.definitions:
        assert definition.risk is ToolRisk.READ
        assert issubclass(definition.input_model, BaseModel)
        assert issubclass(definition.output_model, BaseModel)


def test_list_files_returns_sorted_relative_entries(workspace: Path) -> None:
    registry = WorkspaceToolRegistry(workspace)

    result = registry.execute("list_files", {"path": "."})

    assert type(result).__name__ == "ListFilesOutput"
    assert [(entry.path, entry.kind) for entry in result.entries] == [
        ("README.md", "file"),
        ("src", "directory"),
    ]


def test_read_file_returns_typed_content_and_size(workspace: Path) -> None:
    registry = WorkspaceToolRegistry(workspace)

    result = registry.execute("read_file", {"path": "README.md"})

    assert type(result).__name__ == "ReadFileOutput"
    assert result.path == "README.md"
    assert result.content == "Qwen harness\nSafe tools\n"
    assert result.size_bytes == len((workspace / "README.md").read_bytes())


def test_search_text_returns_bounded_line_matches(workspace: Path) -> None:
    registry = WorkspaceToolRegistry(workspace, max_search_results=1)

    result = registry.execute(
        "search_text",
        {"query": "Qwen", "path": "."},
    )

    assert type(result).__name__ == "SearchTextOutput"
    assert len(result.matches) == 1
    assert result.matches[0].path == "README.md"
    assert result.matches[0].line_number == 1
    assert result.matches[0].line == "Qwen harness"
    assert result.truncated is True


@pytest.mark.parametrize(
    ("tool_name", "arguments"),
    [
        ("list_files", {"path": "../outside"}),
        ("read_file", {"path": "../secret.txt"}),
        ("search_text", {"query": "secret", "path": "../"}),
    ],
)
def test_tools_reject_parent_traversal(
    workspace: Path,
    tool_name: str,
    arguments: dict[str, str],
) -> None:
    registry = WorkspaceToolRegistry(workspace)

    with pytest.raises(WorkspacePathError, match="outside.*workspace|workspace.*outside"):
        registry.execute(tool_name, arguments)


@pytest.mark.parametrize("absolute_path", [r"C:\Windows", "/etc/passwd"])
def test_tools_reject_absolute_paths(
    workspace: Path,
    absolute_path: str,
) -> None:
    registry = WorkspaceToolRegistry(workspace)

    with pytest.raises(WorkspacePathError, match="relative|absolute"):
        registry.execute("read_file", {"path": absolute_path})


def test_read_file_rejects_files_over_the_byte_limit(workspace: Path) -> None:
    (workspace / "large.txt").write_text("12345", encoding="utf-8")
    registry = WorkspaceToolRegistry(workspace, max_file_bytes=4)

    with pytest.raises(FileTooLargeError, match=r"large\.txt.*4"):
        registry.execute("read_file", {"path": "large.txt"})


def test_registry_rejects_unknown_tools_without_dispatching(workspace: Path) -> None:
    registry = WorkspaceToolRegistry(workspace)

    with pytest.raises(UnknownToolError, match="delete_file"):
        registry.execute("delete_file", {"path": "README.md"})


@pytest.mark.parametrize(
    ("tool_name", "arguments"),
    [
        ("read_file", {}),
        ("read_file", {"path": "README.md", "surprise": True}),
        ("search_text", {"query": "", "path": "."}),
    ],
)
def test_registry_rejects_invalid_arguments_with_a_useful_error(
    workspace: Path,
    tool_name: str,
    arguments: dict[str, Any],
) -> None:
    registry = WorkspaceToolRegistry(workspace)

    with pytest.raises(InvalidToolArgumentsError, match=tool_name):
        registry.execute(tool_name, arguments)


@pytest.mark.parametrize(
    ("invoke", "sensitive_fragment"),
    [
        (
            lambda registry: registry.list_files(
                r"C:\private\TOP-SECRET-listing"
            ),
            "TOP-SECRET-listing",
        ),
        (
            lambda registry: registry.read_file(
                "../TOP-SECRET-document.txt"
            ),
            "TOP-SECRET-document.txt",
        ),
        (
            lambda registry: registry.search_text(
                "password",
                "../TOP-SECRET-search-root",
            ),
            "TOP-SECRET-search-root",
        ),
    ],
    ids=["list_files", "read_file", "search_text"],
)
def test_model_facing_tools_turn_path_errors_into_safe_retry_guidance(
    workspace: Path,
    invoke: Any,
    sensitive_fragment: str,
) -> None:
    registry = WorkspaceToolRegistry(workspace)

    with pytest.raises(ModelRetry) as captured:
        invoke(registry)

    message = str(captured.value)
    assert "Use '.' to refer to the workspace root." in message
    assert sensitive_fragment not in message
    assert str(workspace) not in message


def test_model_facing_read_file_turns_size_error_into_a_safe_retry(
    workspace: Path,
) -> None:
    sensitive_name = "TOP-SECRET-large.txt"
    (workspace / sensitive_name).write_text("12345", encoding="utf-8")
    registry = WorkspaceToolRegistry(workspace, max_file_bytes=4)

    with pytest.raises(ModelRetry) as captured:
        registry.read_file(sensitive_name)

    message = str(captured.value)
    assert "too large" in message.casefold()
    assert "smaller" in message.casefold()
    assert sensitive_name not in message


def test_registry_execute_keeps_typed_errors_for_deterministic_callers(
    workspace: Path,
) -> None:
    registry = WorkspaceToolRegistry(workspace)

    with pytest.raises(WorkspacePathError):
        registry.execute("read_file", {"path": "../outside.txt"})


def test_read_tool_rejects_a_reparse_point_that_escapes_workspace(
    workspace: Path,
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("private", encoding="utf-8")
    link = workspace / "escape"

    # Creating symbolic links needs elevation on many Windows machines. An
    # NTFS junction is an equivalent path-resolution escape for this contract.
    completed = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(link), str(outside)],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        pytest.skip("this filesystem cannot create a directory reparse point")

    try:
        registry = WorkspaceToolRegistry(workspace)

        with pytest.raises(
            WorkspacePathError,
            match="outside.*workspace|workspace.*outside",
        ):
            registry.execute("read_file", {"path": "escape/secret.txt"})
    finally:
        os.rmdir(link)


def test_build_harness_registers_workspace_tools_with_pydantic_ai(
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    events: list[Any] = []
    monkeypatch.setenv("HARNESS_MAX_TOOL_CALLS", "2")
    monkeypatch.setenv("HARNESS_MAX_LIST_ENTRIES", "3")
    monkeypatch.setenv("HARNESS_MAX_FILE_BYTES", "1000")
    monkeypatch.setenv("HARNESS_MAX_SEARCH_RESULTS", "4")
    monkeypatch.setenv("HARNESS_MAX_SEARCH_FILES", "5")
    monkeypatch.setenv("HARNESS_MAX_SEARCH_BYTES", "2000")
    monkeypatch.setenv("HARNESS_MAX_LINE_CHARACTERS", "80")
    monkeypatch.setenv("HARNESS_MAX_SEARCH_DIRECTORIES", "6")
    monkeypatch.setenv("HARNESS_TOOL_TIMEOUT_SECONDS", "1.5")

    class RecordingAgent:
        def __init__(self, model: Any, **kwargs: Any) -> None:
            captured.update(kwargs)

    monkeypatch.setattr("pydantic_ai.Agent", RecordingAgent)

    from qwen_harness.cli import build_harness

    harness = build_harness(workspace_root=workspace, event_sink=events.append)

    registered_tools = captured["tools"]
    assert {tool.__name__ for tool in registered_tools} == {
        "list_files",
        "read_file",
        "search_text",
    }
    assert harness.workspace_root == workspace.resolve()
    assert harness.max_tool_calls == 2

    registry = registered_tools[0].__self__
    assert registry.max_list_entries == 3
    assert registry.max_file_bytes == 1000
    assert registry.max_search_results == 4
    assert registry.max_search_files == 5
    assert registry.max_search_bytes == 2000
    assert registry.max_line_characters == 80
    assert registry.max_search_directories == 6
    assert registry.tool_timeout_seconds == 1.5

    registry.read_file("README.md")
    assert [event.name for event in events] == [
        "tool.call.started",
        "tool.call.completed",
    ]


@pytest.mark.parametrize(
    ("tool_name", "arguments"),
    [
        ("list_files", {"path": "missing"}),
        ("list_files", {"path": "README.md"}),
        ("read_file", {"path": "missing.txt"}),
        ("read_file", {"path": "src"}),
        ("search_text", {"query": "Qwen", "path": "missing"}),
    ],
)
def test_registry_translates_missing_and_wrong_kind_targets_to_typed_errors(
    workspace: Path,
    tool_name: str,
    arguments: dict[str, str],
) -> None:
    registry = WorkspaceToolRegistry(workspace)

    with pytest.raises(WorkspaceToolError):
        registry.execute(tool_name, arguments)


@pytest.mark.parametrize(
    "invoke",
    [
        lambda registry: registry.list_files("TOP-SECRET-missing"),
        lambda registry: registry.read_file("TOP-SECRET-missing.txt"),
        lambda registry: registry.search_text(
            "secret", "TOP-SECRET-missing"
        ),
    ],
    ids=["list_files", "read_file", "search_text"],
)
def test_model_facing_tools_turn_missing_targets_into_safe_retries(
    workspace: Path,
    invoke: Any,
) -> None:
    registry = WorkspaceToolRegistry(workspace)

    with pytest.raises(ModelRetry) as captured:
        invoke(registry)

    assert "TOP-SECRET" not in str(captured.value)
    assert "list_files" in str(captured.value)


def test_registry_translates_filesystem_permission_errors(
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = WorkspaceToolRegistry(workspace)

    def deny_access(*args: Any, **kwargs: Any) -> Any:
        raise PermissionError("TOP-SECRET operating-system detail")

    monkeypatch.setattr(Path, "open", deny_access)

    with pytest.raises(WorkspaceToolError) as captured:
        registry.execute("read_file", {"path": "README.md"})

    assert "TOP-SECRET" not in str(captured.value)


def test_model_facing_tool_sanitizes_filesystem_permission_errors(
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = WorkspaceToolRegistry(workspace)

    def deny_access(*args: Any, **kwargs: Any) -> Any:
        raise PermissionError("TOP-SECRET operating-system detail")

    monkeypatch.setattr(Path, "open", deny_access)

    with pytest.raises(ModelRetry) as captured:
        registry.read_file("README.md")

    assert "TOP-SECRET" not in str(captured.value)


def test_list_files_obeys_its_entry_cap_and_reports_truncation(
    workspace: Path,
) -> None:
    (workspace / "a.txt").write_text("a", encoding="utf-8")
    registry = WorkspaceToolRegistry(workspace, max_list_entries=2)

    result = registry.execute("list_files", {"path": "."})

    assert len(result.entries) == 2
    assert result.truncated is True


def test_list_files_scans_only_one_entry_beyond_its_cap(
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for index in range(5):
        (workspace / f"item-{index}.txt").write_bytes(b"x")
    original_iterdir = Path.iterdir
    scanned = 0

    def bounded_iterdir(path: Path) -> Any:
        nonlocal scanned
        entries = original_iterdir(path)
        if path != workspace:
            return entries

        def guarded() -> Any:
            nonlocal scanned
            for entry in entries:
                scanned += 1
                if scanned > 3:
                    raise AssertionError("list_files scanned beyond cap + 1")
                yield entry

        return guarded()

    monkeypatch.setattr(Path, "iterdir", bounded_iterdir)
    registry = WorkspaceToolRegistry(workspace, max_list_entries=2)

    result = registry.execute("list_files", {"path": "."})

    assert len(result.entries) == 2
    assert result.truncated is True
    assert scanned == 3


def test_search_obeys_visited_file_cap_and_reports_usage(
    workspace: Path,
) -> None:
    registry = WorkspaceToolRegistry(workspace, max_search_files=1)

    result = registry.execute("search_text", {"query": "Qwen", "path": "."})

    assert result.files_visited == 1
    assert result.truncated is True


def test_search_obeys_aggregate_byte_cap_and_reports_usage(
    workspace: Path,
) -> None:
    (workspace / "a.txt").write_text("Qwen", encoding="utf-8")
    (workspace / "b.txt").write_text("Qwen", encoding="utf-8")
    registry = WorkspaceToolRegistry(workspace, max_search_bytes=4)

    result = registry.execute("search_text", {"query": "Qwen", "path": "."})

    assert result.bytes_searched <= 4
    assert result.truncated is True
    assert [match.path for match in result.matches] == ["a.txt"]


def test_search_skips_an_oversized_file_and_continues_to_later_files(
    workspace: Path,
) -> None:
    (workspace / "a-oversized.txt").write_bytes(b"x" * 11)
    (workspace / "b-small.txt").write_bytes(b"needle")
    registry = WorkspaceToolRegistry(
        workspace,
        max_file_bytes=10,
        max_search_bytes=100,
    )

    result = registry.execute(
        "search_text", {"query": "needle", "path": "."}
    )

    assert result.truncated is True
    assert [match.path for match in result.matches] == ["b-small.txt"]
    assert result.files_visited >= 2


def test_search_continues_past_a_nonmatch_after_skipping_an_oversized_file(
    workspace: Path,
) -> None:
    (workspace / "a-oversized.txt").write_bytes(b"x" * 11)
    (workspace / "b-nonmatch.txt").write_bytes(b"nothing")
    (workspace / "c-match.txt").write_bytes(b"needle")
    registry = WorkspaceToolRegistry(
        workspace,
        max_file_bytes=10,
        max_search_bytes=100,
    )

    result = registry.execute(
        "search_text", {"query": "needle", "path": "."}
    )

    assert result.truncated is True
    assert [match.path for match in result.matches] == ["c-match.txt"]
    assert result.files_visited >= 3


def test_search_processes_all_bounded_candidates_after_traversal_truncation(
    workspace: Path,
) -> None:
    (workspace / "a-nonmatch.txt").write_bytes(b"nothing")
    (workspace / "b-match.txt").write_bytes(b"needle")
    registry = WorkspaceToolRegistry(workspace, max_search_files=2)

    result = registry.execute(
        "search_text", {"query": "needle", "path": "."}
    )

    assert result.truncated is True
    assert [match.path for match in result.matches] == ["b-match.txt"]
    assert result.files_visited == 2


def test_search_stops_at_the_directory_cap_and_reports_truncation(
    workspace: Path,
) -> None:
    nested = workspace / "a" / "b"
    nested.mkdir(parents=True)
    (nested / "match.txt").write_bytes(b"needle")
    registry = WorkspaceToolRegistry(workspace, max_search_directories=1)

    result = registry.execute(
        "search_text", {"query": "needle", "path": "."}
    )

    assert result.directories_visited == 1
    assert result.truncated is True
    assert result.matches == ()


def test_search_stops_at_its_deadline_and_reports_truncation(
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock_calls = 0

    def deterministic_clock() -> float:
        nonlocal clock_calls
        clock_calls += 1
        return 0.0 if clock_calls == 1 else 1.0

    monkeypatch.setattr("qwen_harness.tools.perf_counter", deterministic_clock)
    registry = WorkspaceToolRegistry(workspace, tool_timeout_seconds=0.5)

    result = registry.execute(
        "search_text", {"query": "Qwen", "path": "."}
    )

    assert result.truncated is True
    assert result.matches == ()


@pytest.mark.parametrize(
    ("limit_name", "value"),
    [
        ("max_search_directories", 0),
        ("tool_timeout_seconds", 0),
        ("tool_timeout_seconds", -0.1),
    ],
)
def test_registry_rejects_non_positive_directory_and_deadline_limits(
    workspace: Path,
    limit_name: str,
    value: int | float,
) -> None:
    with pytest.raises(ValueError, match=limit_name):
        WorkspaceToolRegistry(workspace, **{limit_name: value})


def test_search_caps_returned_line_length_and_marks_the_match(
    workspace: Path,
) -> None:
    (workspace / "long.txt").write_text("Qwen" + "x" * 100, encoding="utf-8")
    registry = WorkspaceToolRegistry(workspace, max_line_characters=10)

    result = registry.execute("search_text", {"query": "Qwen", "path": "."})
    match = next(item for item in result.matches if item.path == "long.txt")

    assert len(match.line) == 10
    assert match.line_truncated is True


def test_read_file_reports_actual_source_bytes_for_multibyte_utf8(
    workspace: Path,
) -> None:
    (workspace / "utf8.txt").write_bytes("é".encode("utf-8"))
    registry = WorkspaceToolRegistry(workspace)

    result = registry.execute("read_file", {"path": "utf8.txt"})

    assert result.content == "é"
    assert result.size_bytes == 2


def test_read_file_size_reports_source_bytes_when_newlines_are_normalized(
    workspace: Path,
) -> None:
    source = b"a\r\nb\r\n"
    (workspace / "windows-newlines.txt").write_bytes(source)
    registry = WorkspaceToolRegistry(workspace)

    result = registry.execute(
        "read_file", {"path": "windows-newlines.txt"}
    )

    assert result.content == "a\nb\n"
    assert result.size_bytes == len(source)


def test_read_file_rejects_invalid_utf8_as_a_typed_tool_error(
    workspace: Path,
) -> None:
    (workspace / "binary.dat").write_bytes(b"\xff\xfe\x00")
    registry = WorkspaceToolRegistry(workspace)

    with pytest.raises(WorkspaceToolError, match="binary|UTF-8|text"):
        registry.execute("read_file", {"path": "binary.dat"})


def test_read_file_checks_stream_bytes_instead_of_trusting_metadata(
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = (workspace / "changing.txt").resolve()
    target_key = os.path.normcase(os.path.abspath(os.fspath(target)))
    target.write_bytes(b"12345")
    registry = WorkspaceToolRegistry(workspace, max_file_bytes=4)
    original_stat = Path.stat

    def stale_stat(path: Path, *args: Any, **kwargs: Any) -> Any:
        result = original_stat(path, *args, **kwargs)
        path_key = os.path.normcase(os.path.abspath(os.fspath(path)))
        if path_key == target_key:
            class StaleSize:
                st_size = 1

                def __getattr__(self, name: str) -> Any:
                    return getattr(result, name)

            return StaleSize()
        return result

    monkeypatch.setattr(Path, "stat", stale_stat)

    with pytest.raises(FileTooLargeError):
        registry.execute("read_file", {"path": "changing.txt"})


def test_read_file_uses_one_bounded_binary_read(
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = WorkspaceToolRegistry(workspace, max_file_bytes=4)
    calls: list[tuple[str, int]] = []

    class RecordingFile:
        def __enter__(self) -> "RecordingFile":
            return self

        def __exit__(self, *args: Any) -> None:
            return None

        def read(self, size: int) -> bytes:
            calls.append(("read", size))
            return b"Qwen"

    def recording_open(path: Path, mode: str = "r", **kwargs: Any) -> RecordingFile:
        assert mode == "rb"
        assert not kwargs
        return RecordingFile()

    monkeypatch.setattr(Path, "open", recording_open)

    result = registry.execute("read_file", {"path": "README.md"})

    assert calls == [("read", 5)]
    assert result.content == "Qwen"
    assert result.size_bytes == 4


def test_search_does_not_follow_a_descendant_junction_outside_workspace(
    workspace: Path,
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside-search"
    outside.mkdir()
    (outside / "secret.txt").write_text(
        "TOP-SECRET-external-match", encoding="utf-8"
    )
    link = workspace / "linked-directory"
    completed = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(link), str(outside)],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        pytest.skip("this filesystem cannot create a directory reparse point")

    try:
        registry = WorkspaceToolRegistry(workspace)

        result = registry.execute(
            "search_text",
            {"query": "TOP-SECRET-external-match", "path": "."},
        )

        assert result.matches == ()
        assert "secret.txt" not in repr(result)
    finally:
        os.rmdir(link)


def test_search_does_not_read_a_descendant_file_symlink_outside_workspace(
    workspace: Path,
    tmp_path: Path,
) -> None:
    outside = tmp_path / "TOP-SECRET-outside.txt"
    outside.write_text("external-match", encoding="utf-8")
    link = workspace / "linked-file.txt"
    try:
        os.symlink(outside, link)
    except OSError:
        pytest.skip("this filesystem cannot create a file symlink")

    result = WorkspaceToolRegistry(workspace).execute(
        "search_text",
        {"query": "external-match", "path": "."},
    )

    assert result.matches == ()
