from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_ENV_FILE = Path(__file__).resolve().parent / ".env"


class KafkaConfig(BaseSettings):
    """Настройки Kafka для чтения и отправки событий."""

    model_config = SettingsConfigDict(
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    bootstrap_servers: str = Field(default="localhost:9092", validation_alias="KAFKA_BOOTSTRAP_SERVERS")
    vectorize_topic: str = Field(default="documents.vectorize", validation_alias="KAFKA_VECTORIZE_TOPIC")
    graph_topic: str = Field(default="documents.graph", validation_alias="KAFKA_GRAPH_TOPIC")
    consumer_group: str = Field(default="chunk-vectorizer", validation_alias="KAFKA_CONSUMER_GROUP")
    auto_offset_reset: str = Field(default="earliest", validation_alias="KAFKA_AUTO_OFFSET_RESET")


class OllamaConfig(BaseSettings):
    """Параметры подключения к Ollama."""

    model_config = SettingsConfigDict(
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )
    embedding_dim: int = Field(default=256, validation_alias="EMBEDDING_DIM")
    base_url: str = Field(default="http://localhost:11434", validation_alias="OLLAMA_BASE_URL")
    model_name: str = Field(default="qwen3-embedding:0.6b", validation_alias="OLLAMA_MODEL")
    timeout_sec: float = Field(default=30.0, validation_alias="OLLAMA_TIMEOUT_SEC")
    max_attempts: int = Field(default=3, validation_alias="OLLAMA_MAX_ATTEMPTS")


class QdrantConfig(BaseSettings):
    """Параметры подключения к Qdrant."""

    model_config = SettingsConfigDict(
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    url: str = Field(default="http://localhost:6333", validation_alias="QDRANT_URL")
    collection_name: str = Field(default="legal_chunks_vectors", validation_alias="QDRANT_COLLECTION")
    distance: str = Field(default="Cosine", validation_alias="QDRANT_DISTANCE")
    max_attempts: int = Field(default=3, validation_alias="QDRANT_MAX_ATTEMPTS")


class RuntimeConfig(BaseSettings):
    """Параметры рантайма сервиса."""

    model_config = SettingsConfigDict(
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")


@dataclass(frozen=True)
class AppConfig:
    """Сводная конфигурация приложения."""

    kafka: KafkaConfig
    ollama: OllamaConfig
    qdrant: QdrantConfig
    runtime: RuntimeConfig


def load_config() -> AppConfig:
    """Загружает конфигурацию сервиса."""
    return AppConfig(
        kafka=KafkaConfig(),
        ollama=OllamaConfig(),
        qdrant=QdrantConfig(),
        runtime=RuntimeConfig(),
    )
