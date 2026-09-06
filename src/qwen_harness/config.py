"""Validated runtime configuration."""

from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from qwen_harness.context import DEFAULT_CONTEXT_OVERHEAD_TOKENS


class Settings(BaseSettings):
    """Configuration read from explicit constructor values or the environment."""

    model_config = SettingsConfigDict(
        populate_by_name=True,
        extra="ignore",
    )

    ollama_base_url: str = Field(
        default="http://localhost:11434/v1",
        validation_alias="OLLAMA_BASE_URL",
        min_length=1,
    )
    model: str = Field(
        default="qwen3.5:0.8b-cpu",
        validation_alias="OLLAMA_MODEL",
        min_length=1,
    )
    max_steps: int = Field(
        default=4,
        validation_alias="HARNESS_MAX_STEPS",
        gt=0,
    )
    timeout_seconds: float = Field(
        default=120.0,
        validation_alias="HARNESS_TIMEOUT_SECONDS",
        gt=0,
    )
    max_output_tokens: int = Field(
        default=512,
        validation_alias="HARNESS_MAX_OUTPUT_TOKENS",
        gt=0,
    )
    context_window_tokens: int = Field(
        default=2_048,
        validation_alias="HARNESS_CONTEXT_WINDOW_TOKENS",
        gt=0,
    )
    context_overhead_tokens: int = Field(
        default=DEFAULT_CONTEXT_OVERHEAD_TOKENS,
        validation_alias="HARNESS_CONTEXT_OVERHEAD_TOKENS",
        gt=0,
    )
    metrics_database: Path = Field(
        default=Path(".qwen-harness/metrics.sqlite3"),
        validation_alias="HARNESS_METRICS_DATABASE",
    )
    max_tool_calls: int = Field(
        default=4,
        validation_alias="HARNESS_MAX_TOOL_CALLS",
        gt=0,
    )
    max_list_entries: int = Field(
        default=200,
        validation_alias="HARNESS_MAX_LIST_ENTRIES",
        gt=0,
    )
    max_file_bytes: int = Field(
        default=4_000,
        validation_alias="HARNESS_MAX_FILE_BYTES",
        gt=0,
    )
    max_search_results: int = Field(
        default=8,
        validation_alias="HARNESS_MAX_SEARCH_RESULTS",
        gt=0,
    )
    max_search_files: int = Field(
        default=500,
        validation_alias="HARNESS_MAX_SEARCH_FILES",
        gt=0,
    )
    max_search_bytes: int = Field(
        default=1_000_000,
        validation_alias="HARNESS_MAX_SEARCH_BYTES",
        gt=0,
    )
    max_line_characters: int = Field(
        default=200,
        validation_alias="HARNESS_MAX_LINE_CHARACTERS",
        gt=0,
    )
    max_search_directories: int = Field(
        default=200,
        validation_alias="HARNESS_MAX_SEARCH_DIRECTORIES",
        gt=0,
    )
    tool_timeout_seconds: float = Field(
        default=5.0,
        validation_alias="HARNESS_TOOL_TIMEOUT_SECONDS",
        gt=0,
    )

    @model_validator(mode="after")
    def validate_context_budget(self) -> "Settings":
        if (
            self.model == "qwen3.5:0.8b-cpu"
            and self.context_window_tokens != 2_048
        ):
            raise ValueError(
                "qwen3.5:0.8b-cpu context must match Modelfile.cpu num_ctx 2048"
            )
        if (
            self.max_output_tokens + self.context_overhead_tokens
            >= self.context_window_tokens
        ):
            raise ValueError(
                "output and overhead reserves must be smaller than context window"
            )
        return self
