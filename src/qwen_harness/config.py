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
        default="qwen3.5:4b-cpu",
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
