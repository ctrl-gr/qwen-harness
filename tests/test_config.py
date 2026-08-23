from pathlib import Path

import pytest

from qwen_harness.config import Settings


def test_settings_use_local_qwen_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "OLLAMA_BASE_URL",
        "OLLAMA_MODEL",
        "HARNESS_MAX_STEPS",
        "HARNESS_TIMEOUT_SECONDS",
        "HARNESS_MAX_OUTPUT_TOKENS",
        "HARNESS_MAX_TOOL_CALLS",
        "HARNESS_MAX_LIST_ENTRIES",
        "HARNESS_MAX_FILE_BYTES",
        "HARNESS_MAX_SEARCH_RESULTS",
        "HARNESS_MAX_SEARCH_FILES",
        "HARNESS_MAX_SEARCH_BYTES",
        "HARNESS_MAX_LINE_CHARACTERS",
        "HARNESS_MAX_SEARCH_DIRECTORIES",
        "HARNESS_TOOL_TIMEOUT_SECONDS",
    ):
        monkeypatch.delenv(name, raising=False)

    settings = Settings()

    assert settings.ollama_base_url == "http://localhost:11434/v1"
    assert settings.model == "qwen3.5:0.8b-cpu"
    assert settings.max_steps > 0
    assert settings.timeout_seconds > 0
    assert settings.max_output_tokens == 512
    assert settings.max_tool_calls > 0
    assert settings.max_list_entries > 0
    assert settings.max_file_bytes > 0
    assert settings.max_search_results > 0
    assert settings.max_search_files > 0
    assert settings.max_search_bytes > 0
    assert settings.max_line_characters > 0
    assert settings.max_search_directories > 0
    assert settings.tool_timeout_seconds > 0
    assert settings.max_file_bytes <= 4_000
    assert settings.max_search_results <= 10
    assert settings.max_line_characters <= 200


def test_settings_read_ollama_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://ollama.test:11434/v1")
    monkeypatch.setenv("OLLAMA_MODEL", "qwen3.5:9b")
    monkeypatch.setenv("HARNESS_MAX_STEPS", "3")
    monkeypatch.setenv("HARNESS_TIMEOUT_SECONDS", "15")
    monkeypatch.setenv("HARNESS_MAX_OUTPUT_TOKENS", "128")
    monkeypatch.setenv("HARNESS_MAX_TOOL_CALLS", "2")
    monkeypatch.setenv("HARNESS_MAX_LIST_ENTRIES", "20")
    monkeypatch.setenv("HARNESS_MAX_FILE_BYTES", "1000")
    monkeypatch.setenv("HARNESS_MAX_SEARCH_RESULTS", "8")
    monkeypatch.setenv("HARNESS_MAX_SEARCH_FILES", "12")
    monkeypatch.setenv("HARNESS_MAX_SEARCH_BYTES", "2000")
    monkeypatch.setenv("HARNESS_MAX_LINE_CHARACTERS", "160")
    monkeypatch.setenv("HARNESS_MAX_SEARCH_DIRECTORIES", "24")
    monkeypatch.setenv("HARNESS_TOOL_TIMEOUT_SECONDS", "2.5")

    settings = Settings()

    assert settings.ollama_base_url == "http://ollama.test:11434/v1"
    assert settings.model == "qwen3.5:9b"
    assert settings.max_steps == 3
    assert settings.timeout_seconds == 15
    assert settings.max_output_tokens == 128
    assert settings.max_tool_calls == 2
    assert settings.max_list_entries == 20
    assert settings.max_file_bytes == 1000
    assert settings.max_search_results == 8
    assert settings.max_search_files == 12
    assert settings.max_search_bytes == 2000
    assert settings.max_line_characters == 160
    assert settings.max_search_directories == 24
    assert settings.tool_timeout_seconds == 2.5


def test_cpu_modelfile_uses_lightweight_qwen_with_cpu_only_limits() -> None:
    modelfile = (Path(__file__).parents[1] / "Modelfile.cpu").read_text(
        encoding="utf-8"
    )
    directives = {
        line.strip()
        for line in modelfile.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }

    assert "FROM qwen3.5:0.8b" in directives
    assert "PARAMETER num_gpu 0" in directives
    assert "PARAMETER num_ctx 2048" in directives


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_steps", 0),
        ("max_steps", -1),
        ("timeout_seconds", 0),
        ("timeout_seconds", -0.1),
        ("max_output_tokens", 0),
        ("max_tool_calls", 0),
        ("max_list_entries", 0),
        ("max_file_bytes", 0),
        ("max_search_results", 0),
        ("max_search_files", 0),
        ("max_search_bytes", 0),
        ("max_line_characters", 0),
        ("max_search_directories", 0),
        ("tool_timeout_seconds", 0),
    ],
)
def test_settings_reject_non_positive_execution_bounds(
    field: str, value: int | float
) -> None:
    with pytest.raises(ValueError):
        Settings(**{field: value})
