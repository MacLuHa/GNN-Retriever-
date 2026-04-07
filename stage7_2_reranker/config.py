from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_ENV_FILE = Path(__file__).resolve().parent / ".env"


class RerankerConfig(BaseSettings):
    """Настройки cross-encoder реранкера."""

    model_config = SettingsConfigDict(
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    enabled: bool = Field(default=False, validation_alias="RERANK_ENABLED")
    model_name: str = Field(default="BAAI/bge-reranker-v2-m3", validation_alias="RERANK_MODEL_NAME")
    top_n: int = Field(default=30, validation_alias="RERANK_TOP_N")
    batch_size: int = Field(default=8, validation_alias="RERANK_BATCH_SIZE")
    timeout_sec: float = Field(default=15.0, validation_alias="RERANK_TIMEOUT_SEC")
    max_length: int = Field(default=512, validation_alias="RERANK_MAX_LENGTH")
    device: str = Field(default="auto", validation_alias="RERANK_DEVICE")

