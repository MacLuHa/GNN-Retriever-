from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_ENV_FILE = Path(__file__).resolve().parent / ".env"


class KafkaConfig(BaseSettings):
    """Kafka: чтение топика графа."""

    model_config = SettingsConfigDict(
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    bootstrap_servers: str = Field(default="localhost:9092", validation_alias="KAFKA_BOOTSTRAP_SERVERS")
    graph_topic: str = Field(default="documents.graph", validation_alias="KAFKA_GRAPH_TOPIC")
    graph_output_topic: str = Field(
        default="documents.graph.entities",
        validation_alias="KAFKA_GRAPH_OUTPUT_TOPIC",
    )
    graph_dlq_topic: str = Field(
        default="documents.graph-dlq",
        validation_alias="KAFKA_GRAPH_DLQ_TOPIC",
    )
    consumer_group: str = Field(default="graph-knowledge-base", validation_alias="KAFKA_CONSUMER_GROUP")
    auto_offset_reset: str = Field(default="earliest", validation_alias="KAFKA_AUTO_OFFSET_RESET")


class ElasticsearchConfig(BaseSettings):
    """Elasticsearch: загрузка текста чанка по es_doc_id."""

    model_config = SettingsConfigDict(
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    url: str = Field(default="http://localhost:9200", validation_alias="ELASTICSEARCH_URL")
    index_name: str = Field(default="legal_chunks", validation_alias="ELASTICSEARCH_INDEX")


class Neo4jConfig(BaseSettings):
    """Neo4j: хранение графа сущностей."""

    model_config = SettingsConfigDict(
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    uri: str = Field(default="bolt://localhost:7687", validation_alias="NEO4J_URI")
    user: str = Field(default="neo4j", validation_alias="NEO4J_USER")
    password: str = Field(default="password", validation_alias="NEO4J_PASSWORD")
    database: str = Field(default="neo4j", validation_alias="NEO4J_DATABASE")


class OllamaLlmConfig(BaseSettings):
    """Ollama: извлечение сущностей (чат-модель, не embedding)."""

    model_config = SettingsConfigDict(
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    base_url: str = Field(default="http://localhost:11434", validation_alias="OLLAMA_BASE_URL")
    model_name: str = Field(default="qwen3:14b", validation_alias="OLLAMA_LLM_MODEL")
    timeout_sec: float = Field(default=120.0, validation_alias="OLLAMA_LLM_TIMEOUT_SEC")
    max_attempts: int = Field(default=3, validation_alias="OLLAMA_LLM_MAX_ATTEMPTS")
    num_predict: int | None = Field(default=None, validation_alias="OLLAMA_LLM_NUM_PREDICT")


class OllamaEmbeddingConfig(BaseSettings):
    """Ollama: генерация embedding для сущностей."""

    model_config = SettingsConfigDict(
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    embedding_dim: int = Field(
        default=256,
        validation_alias=AliasChoices("ENTITY_EMBEDDING_DIM", "EMBEDDING_DIM"),
    )
    base_url: str = Field(
        default="http://localhost:11434",
        validation_alias=AliasChoices("ENTITY_OLLAMA_BASE_URL", "OLLAMA_BASE_URL"),
    )
    model_name: str = Field(
        default="qwen3-embedding:0.6b",
        validation_alias=AliasChoices("ENTITY_OLLAMA_MODEL", "OLLAMA_MODEL"),
    )
    timeout_sec: float = Field(default=30.0, validation_alias="ENTITY_OLLAMA_TIMEOUT_SEC")
    max_attempts: int = Field(default=3, validation_alias="ENTITY_OLLAMA_MAX_ATTEMPTS")


class QdrantConfig(BaseSettings):
    """Qdrant: коллекция векторов сущностей."""

    model_config = SettingsConfigDict(
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    url: str = Field(
        default="http://localhost:6333",
        validation_alias=AliasChoices("ENTITY_QDRANT_URL", "QDRANT_URL"),
    )
    collection_name: str = Field(
        default="legal_entities_vectors",
        validation_alias=AliasChoices("ENTITY_QDRANT_COLLECTION", "QDRANT_ENTITY_COLLECTION"),
    )
    distance: str = Field(
        default="Cosine",
        validation_alias=AliasChoices("ENTITY_QDRANT_DISTANCE", "QDRANT_DISTANCE"),
    )
    max_attempts: int = Field(default=3, validation_alias="ENTITY_QDRANT_MAX_ATTEMPTS")


class RuntimeConfig(BaseSettings):
    """Логирование."""

    model_config = SettingsConfigDict(
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")


class EntityNormalizationConfig(BaseSettings):
    """Нормализация имён сущностей (лемматизация simplemma)."""

    model_config = SettingsConfigDict(
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    lemma_lang: str = Field(default="en", validation_alias="ENTITY_LEMMATIZE_LANG")
    lemma_scope: str = Field(default="single", validation_alias="ENTITY_LEMMATIZE_SCOPE")


@dataclass(frozen=True)
class AppConfig:
    """Сводная конфигурация."""

    kafka: KafkaConfig
    elasticsearch: ElasticsearchConfig
    neo4j: Neo4jConfig
    ollama_llm: OllamaLlmConfig
    ollama_embedding: OllamaEmbeddingConfig
    qdrant: QdrantConfig
    entity_normalization: EntityNormalizationConfig
    runtime: RuntimeConfig


def load_config() -> AppConfig:
    """Загружает конфигурацию из окружения и .env."""
    return AppConfig(
        kafka=KafkaConfig(),
        elasticsearch=ElasticsearchConfig(),
        neo4j=Neo4jConfig(),
        ollama_llm=OllamaLlmConfig(),
        ollama_embedding=OllamaEmbeddingConfig(),
        qdrant=QdrantConfig(),
        entity_normalization=EntityNormalizationConfig(),
        runtime=RuntimeConfig(),
    )
