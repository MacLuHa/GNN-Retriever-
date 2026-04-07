from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from stage7_2_reranker import RerankerConfig

_ENV_FILE = Path(__file__).resolve().parent / ".env"


class ApiConfig(BaseSettings):
    """Настройки HTTP API."""

    model_config = SettingsConfigDict(
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    host: str = Field(default="0.0.0.0", validation_alias="RETRIEVER_API_HOST")
    port: int = Field(default=8010, validation_alias="RETRIEVER_API_PORT")
    retrieve_timeout_sec: float = Field(default=20.0, validation_alias="RETRIEVER_RETRIEVE_TIMEOUT_SEC")
    max_context_chars: int = Field(default=1200, validation_alias="RETRIEVER_MAX_CONTEXT_CHARS")
    log_level: str = Field(default="INFO", validation_alias="RETRIEVER_API_LOG_LEVEL")


@dataclass(frozen=True)
class AppConfig:
    """Сводная конфигурация API."""

    api: ApiConfig
    reranker: RerankerConfig


def load_config() -> AppConfig:
    """Загружает конфигурацию API."""
    return AppConfig(
        api=ApiConfig(),
        reranker=RerankerConfig(),
    )

