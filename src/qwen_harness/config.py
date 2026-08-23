"""Validated runtime configuration."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


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
