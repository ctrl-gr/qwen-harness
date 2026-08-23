"""Typed, read-only tools restricted to an explicit workspace."""

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import os
from itertools import islice
from pathlib import Path, PurePosixPath, PureWindowsPath
import stat
from time import perf_counter
from types import MappingProxyType
from typing import Any, Callable, Literal, Mapping
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from pydantic_ai import ModelRetry

from qwen_harness.observability import emit_current_run_event
from qwen_harness.orchestration import RunEvent


class WorkspaceToolError(Exception):
    """Base class for expected workspace-tool failures."""


class WorkspacePathError(WorkspaceToolError):
    """Raised when a requested path escapes the configured workspace."""


class FileTooLargeError(WorkspaceToolError):
    """Raised when a file exceeds the configured read limit."""


class UnknownToolError(WorkspaceToolError):
    """Raised when a tool name is not registered."""


class InvalidToolArgumentsError(WorkspaceToolError):
    """Raised when tool arguments do not match their typed contract."""


class WorkspaceTargetError(WorkspaceToolError):
    """Raised when a target is missing, inaccessible, or the wrong kind."""


class InvalidTextFileError(WorkspaceToolError):
    """Raised when a requested file is not valid UTF-8 text."""


class ToolRisk(str, Enum):
    """Permission level associated with a tool."""

    READ = "read"


class ToolInput(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ToolOutput(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ListFilesInput(ToolInput):
    path: str = "."


class FileEntry(ToolOutput):
    path: str
    kind: Literal["file", "directory"]


class ListFilesOutput(ToolOutput):
    entries: tuple[FileEntry, ...]
    truncated: bool = False


class ReadFileInput(ToolInput):
    path: str = Field(min_length=1)


class ReadFileOutput(ToolOutput):
    path: str
    content: str
    size_bytes: int = Field(ge=0)


class SearchTextInput(ToolInput):
    query: str = Field(min_length=1)
    path: str = "."


class SearchMatch(ToolOutput):
    path: str
    line_number: int = Field(gt=0)
    line: str
    line_truncated: bool = False


class SearchTextOutput(ToolOutput):
    matches: tuple[SearchMatch, ...]
    truncated: bool
    files_visited: int = Field(ge=0)
    bytes_searched: int = Field(ge=0)
    directories_visited: int = Field(ge=0)


@dataclass(frozen=True)
class ToolDefinition:
    """A validated tool contract and its local implementation."""

    name: str
    risk: ToolRisk
    input_model: type[ToolInput]
    output_model: type[ToolOutput]
    handler: Callable[[ToolInput], ToolOutput]


class WorkspaceToolRegistry:
    """Dispatch bounded read operations inside one resolved workspace root."""

    def __init__(
        self,
        root: Path,
        *,
        max_list_entries: int = 200,
        max_file_bytes: int = 4_000,
        max_search_results: int = 8,
        max_search_files: int = 500,
        max_search_bytes: int = 1_000_000,
        max_line_characters: int = 200,
        max_search_directories: int = 200,
        tool_timeout_seconds: float = 5.0,
        event_sink: Callable[[RunEvent], None] | None = None,
    ) -> None:
        limits = {
            "max_list_entries": max_list_entries,
            "max_file_bytes": max_file_bytes,
            "max_search_results": max_search_results,
            "max_search_files": max_search_files,
            "max_search_bytes": max_search_bytes,
            "max_line_characters": max_line_characters,
            "max_search_directories": max_search_directories,
            "tool_timeout_seconds": tool_timeout_seconds,
        }
        for name, value in limits.items():
            if value <= 0:
                raise ValueError(f"{name} must be greater than zero")

        workspace_root = root.resolve()
        if not workspace_root.is_dir():
            raise WorkspacePathError(
                f"workspace root is not a directory: {workspace_root}"
            )

        self.workspace_root = workspace_root
        self.max_list_entries = max_list_entries
        self.max_file_bytes = max_file_bytes
        self.max_search_results = max_search_results
        self.max_search_files = max_search_files
        self.max_search_bytes = max_search_bytes
        self.max_line_characters = max_line_characters
        self.max_search_directories = max_search_directories
        self.tool_timeout_seconds = tool_timeout_seconds
        self.event_sink = event_sink
        self._standalone_run_id = f"tool-session-{uuid4().hex}"
        self.definitions = (
            ToolDefinition(
                "list_files",
                ToolRisk.READ,
                ListFilesInput,
                ListFilesOutput,
                self._list_files,
            ),
            ToolDefinition(
                "read_file",
                ToolRisk.READ,
                ReadFileInput,
                ReadFileOutput,
                self._read_file,
            ),
            ToolDefinition(
                "search_text",
                ToolRisk.READ,
                SearchTextInput,
                SearchTextOutput,
                self._search_text,
            ),
        )
        self._definitions_by_name = {
            definition.name: definition for definition in self.definitions
        }

    @property
    def pydantic_ai_tools(self) -> tuple[Callable[..., ToolOutput], ...]:
        """Return typed callables suitable for PydanticAI registration."""
        return (self.list_files, self.read_file, self.search_text)

    def execute(self, name: str, arguments: dict[str, Any]) -> ToolOutput:
        """Validate and dispatch one registered read operation."""
        definition = self._definitions_by_name.get(name)
        if definition is None:
            raise UnknownToolError(f"unknown workspace tool: {name}")
        try:
            validated = definition.input_model.model_validate(arguments)
        except ValidationError as exc:
            raise InvalidToolArgumentsError(
                f"invalid arguments for {name}: {exc}"
            ) from exc
        result = definition.handler(validated)
        return definition.output_model.model_validate(result)

    def list_files(self, path: str = ".") -> ListFilesOutput:
        """List the immediate children of a workspace-relative directory."""
        return self._invoke_for_model(  # type: ignore[return-value]
            "list_files", {"path": path}
        )

    def read_file(self, path: str) -> ReadFileOutput:
        """Read one UTF-8 text file within the workspace size limit."""
        return self._invoke_for_model(  # type: ignore[return-value]
            "read_file", {"path": path}
        )

    def search_text(self, query: str, path: str = ".") -> SearchTextOutput:
        """Search bounded text-line matches under a workspace-relative path."""
        return self._invoke_for_model(  # type: ignore[return-value]
            "search_text", {"query": query, "path": path}
        )

    def _invoke_for_model(
        self, name: str, arguments: dict[str, Any]
    ) -> ToolOutput:
        started = perf_counter()
        common = {
            "tool_name": name,
            "risk": ToolRisk.READ.value,
            "tool_call_id": uuid4().hex,
        }
        self._emit("tool.call.started", common)
        try:
            result = self.execute(name, arguments)
        except WorkspaceToolError as exc:
            self._emit(
                "tool.call.failed",
                {
                    **common,
                    "elapsed_seconds": perf_counter() - started,
                    "error_category": "workspace_tool_error",
                },
            )
            raise ModelRetry(_safe_retry_message(exc)) from None
        self._emit(
            "tool.call.completed",
            {**common, "elapsed_seconds": perf_counter() - started},
        )
        return result

    def _emit(self, name: str, data: Mapping[str, Any]) -> None:
        if emit_current_run_event(name, data):
            return
        if self.event_sink is None:
            return
        try:
            self.event_sink(
                RunEvent(
                    run_id=self._standalone_run_id,
                    timestamp=datetime.now(timezone.utc),
                    name=name,
                    data=MappingProxyType(dict(data)),
                )
            )
        except Exception:
            pass

    def _resolve(self, relative_path: str) -> Path:
        raw_path = Path(relative_path)
        if (
            raw_path.is_absolute()
            or PureWindowsPath(relative_path).is_absolute()
            or PurePosixPath(relative_path).is_absolute()
        ):
            raise WorkspacePathError("workspace tool paths must be relative")

        try:
            resolved = (self.workspace_root / raw_path).resolve()
        except OSError as exc:
            raise WorkspaceTargetError("workspace target is inaccessible") from exc
        try:
            resolved.relative_to(self.workspace_root)
        except ValueError as exc:
            raise WorkspacePathError(
                f"requested path is outside the workspace: {relative_path}"
            ) from exc
        return resolved

    def _relative(self, path: Path) -> str:
        return path.relative_to(self.workspace_root).as_posix()

    def _target_kind(self, path: Path) -> Literal["file", "directory"]:
        try:
            mode = path.stat().st_mode
        except OSError as exc:
            raise WorkspaceTargetError("workspace target was not found") from exc
        if stat.S_ISREG(mode):
            return "file"
        if stat.S_ISDIR(mode):
            return "directory"
        raise WorkspaceTargetError("workspace target has an unsupported type")

    def _is_reparse_or_link(self, path: Path) -> bool:
        try:
            metadata = path.lstat()
        except OSError:
            return True
        attributes = getattr(metadata, "st_file_attributes", 0) or 0
        reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
        return path.is_symlink() or bool(attributes & reparse_flag)

    def _is_safe_descendant(self, path: Path) -> bool:
        if self._is_reparse_or_link(path):
            return False
        try:
            path.resolve().relative_to(self.workspace_root)
        except (OSError, ValueError):
            return False
        return True

    def _list_files(self, request: ToolInput) -> ListFilesOutput:
        assert isinstance(request, ListFilesInput)
        directory = self._resolve(request.path)
        if self._target_kind(directory) != "directory":
            raise WorkspaceTargetError("list_files requires a directory")
        try:
            scanned_children = list(
                islice(directory.iterdir(), self.max_list_entries + 1)
            )
        except OSError as exc:
            raise WorkspaceTargetError("workspace directory is inaccessible") from exc

        truncated = len(scanned_children) > self.max_list_entries
        children = sorted(
            scanned_children[: self.max_list_entries],
            key=lambda item: item.name.casefold(),
        )
        entries: list[FileEntry] = []
        for child in children:
            if not self._is_safe_descendant(child):
                continue
            try:
                kind = self._target_kind(child)
            except WorkspaceTargetError:
                continue
            entries.append(FileEntry(path=self._relative(child), kind=kind))
        return ListFilesOutput(entries=tuple(entries), truncated=truncated)

    def _read_file(self, request: ToolInput) -> ReadFileOutput:
        assert isinstance(request, ReadFileInput)
        path = self._resolve(request.path)
        if self._target_kind(path) != "file":
            raise WorkspaceTargetError("read_file requires a text file")
        if self._is_reparse_or_link(path) or not self._is_safe_descendant(path):
            raise WorkspacePathError("requested path is outside the workspace")
        try:
            with path.open("rb") as stream:
                source = stream.read(self.max_file_bytes + 1)
        except OSError as exc:
            raise WorkspaceTargetError("workspace file is inaccessible") from exc
        if len(source) > self.max_file_bytes:
            raise FileTooLargeError(
                f"{request.path} exceeds the {self.max_file_bytes}-byte read limit"
            )
        try:
            content = source.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
        except UnicodeDecodeError as exc:
            raise InvalidTextFileError(
                "requested file is not valid UTF-8 text or is binary"
            ) from exc
        return ReadFileOutput(
            path=self._relative(path),
            content=content,
            size_bytes=len(source),
        )

    def _search_text(self, request: ToolInput) -> SearchTextOutput:
        assert isinstance(request, SearchTextInput)
        target = self._resolve(request.path)
        target_kind = self._target_kind(target)
        deadline = perf_counter() + self.tool_timeout_seconds
        candidates, directories_visited, traversal_truncated = (
            self._search_candidates(target, target_kind, deadline)
        )
        matches: list[SearchMatch] = []
        truncated = traversal_truncated
        files_visited = 0
        bytes_searched = 0
        stop_search = False

        for candidate in candidates:
            if perf_counter() >= deadline:
                truncated = True
                break
            if files_visited >= self.max_search_files:
                truncated = True
                break
            files_visited += 1
            remaining_bytes = self.max_search_bytes - bytes_searched
            if remaining_bytes <= 0:
                truncated = True
                break
            try:
                with candidate.open("rb") as stream:
                    source = stream.read(min(self.max_file_bytes, remaining_bytes) + 1)
            except OSError:
                continue
            if len(source) > self.max_file_bytes:
                truncated = True
                continue
            if len(source) > remaining_bytes:
                truncated = True
                break
            bytes_searched += len(source)
            try:
                lines = source.decode("utf-8").splitlines()
            except UnicodeDecodeError:
                continue
            for line_number, line in enumerate(lines, start=1):
                if request.query not in line:
                    continue
                if len(matches) >= self.max_search_results:
                    truncated = True
                    stop_search = True
                    break
                matches.append(
                    SearchMatch(
                        path=self._relative(candidate),
                        line_number=line_number,
                        line=line[: self.max_line_characters],
                        line_truncated=len(line) > self.max_line_characters,
                    )
                )
            if stop_search:
                break

        return SearchTextOutput(
            matches=tuple(matches),
            truncated=truncated,
            files_visited=files_visited,
            bytes_searched=bytes_searched,
            directories_visited=directories_visited,
        )

    def _search_candidates(
        self,
        target: Path,
        target_kind: Literal["file", "directory"],
        deadline: float,
    ) -> tuple[list[Path], int, bool]:
        if target_kind == "file":
            candidates = [target] if self._is_safe_descendant(target) else []
            return candidates, 0, False

        candidates: list[Path] = []
        directories_visited = 0
        truncated = False
        try:
            for current, directory_names, file_names in os.walk(
                target, topdown=True, followlinks=False
            ):
                if perf_counter() >= deadline:
                    truncated = True
                    break
                if directories_visited >= self.max_search_directories:
                    truncated = True
                    break
                directories_visited += 1
                current_path = Path(current)
                safe_directories = []
                for name in sorted(directory_names, key=str.casefold):
                    directory = current_path / name
                    if self._is_safe_descendant(directory):
                        safe_directories.append(name)
                directory_names[:] = safe_directories
                if directories_visited >= self.max_search_directories:
                    if directory_names:
                        truncated = True
                    directory_names[:] = []

                for name in sorted(file_names, key=str.casefold):
                    candidate = current_path / name
                    if self._is_safe_descendant(candidate):
                        candidates.append(candidate)
                    if len(candidates) > self.max_search_files:
                        return candidates, directories_visited, True
        except OSError as exc:
            raise WorkspaceTargetError("workspace search target is inaccessible") from exc
        return candidates, directories_visited, truncated


def _safe_retry_message(exc: WorkspaceToolError) -> str:
    """Give the model actionable guidance without echoing sensitive arguments."""
    if isinstance(exc, WorkspacePathError):
        return (
            "Use '.' to refer to the workspace root. Only relative paths inside "
            "the workspace are allowed."
        )
    if isinstance(exc, FileTooLargeError):
        return "The requested file is too large; choose a smaller file."
    if isinstance(exc, WorkspaceTargetError):
        return (
            "The requested target is unavailable. Use list_files to choose an "
            "existing relative path, then retry."
        )
    if isinstance(exc, InvalidTextFileError):
        return "The requested file is not UTF-8 text; choose another file."
    return "The read-only tool arguments were invalid; use list_files and retry."
