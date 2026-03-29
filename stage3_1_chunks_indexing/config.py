from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_ENV_FILE = Path(__file__).resolve().parent / ".env"


class KafkaConsumerConfig(BaseSettings):
    """Настройки Kafka: топики, группа, смещения."""

    model_config = SettingsConfigDict(
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    bootstrap_servers: str = Field(default="localhost:9092", validation_alias="KAFKA_BOOTSTRAP_SERVERS")
    chunks_topic: str = Field(default="documents.chunks", validation_alias="KAFKA_CHUNKS_TOPIC")
    vectorize_topic: str = Field(default="documents.vectorize", validation_alias="KAFKA_VECTORIZE_TOPIC")
    consumer_group: str = Field(default="chunk-indexer", validation_alias="KAFKA_CONSUMER_GROUP")
    auto_offset_reset: str = Field(default="earliest", validation_alias="KAFKA_AUTO_OFFSET_RESET")


class ElasticsearchConfig(BaseSettings):
    """Подключение к Elasticsearch: URL и имя индекса."""

    model_config = SettingsConfigDict(
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    url: str = Field(default="http://localhost:9200", validation_alias="ELASTICSEARCH_URL")
    index_name: str = Field(default="legal_chunks", validation_alias="ELASTICSEARCH_INDEX")


class RuntimeConfig(BaseSettings):
    """Уровень логирования процесса."""

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

    kafka: KafkaConsumerConfig
    elasticsearch: ElasticsearchConfig
    runtime: RuntimeConfig


def load_config() -> AppConfig:
    """Загружает конфигурацию из переменных окружения и при наличии из .env."""
    return AppConfig(
        kafka=KafkaConsumerConfig(),
        elasticsearch=ElasticsearchConfig(),
        runtime=RuntimeConfig(),
    )
