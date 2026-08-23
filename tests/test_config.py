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
    ):
        monkeypatch.delenv(name, raising=False)

    settings = Settings()

    assert settings.ollama_base_url == "http://localhost:11434/v1"
    assert settings.model == "qwen3.5:0.8b-cpu"
    assert settings.max_steps > 0
    assert settings.timeout_seconds > 0
    assert settings.max_output_tokens == 512


def test_settings_read_ollama_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://ollama.test:11434/v1")
    monkeypatch.setenv("OLLAMA_MODEL", "qwen3.5:9b")
    monkeypatch.setenv("HARNESS_MAX_STEPS", "3")
    monkeypatch.setenv("HARNESS_TIMEOUT_SECONDS", "15")
    monkeypatch.setenv("HARNESS_MAX_OUTPUT_TOKENS", "128")

    settings = Settings()

    assert settings.ollama_base_url == "http://ollama.test:11434/v1"
    assert settings.model == "qwen3.5:9b"
    assert settings.max_steps == 3
    assert settings.timeout_seconds == 15
    assert settings.max_output_tokens == 128


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
    ],
)
def test_settings_reject_non_positive_execution_bounds(
    field: str, value: int | float
) -> None:
    with pytest.raises(ValueError):
        Settings(**{field: value})
