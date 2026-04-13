from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_ENV_FILE = Path(__file__).resolve().parent / ".env"


class ElasticsearchConfig(BaseSettings):
    """Elasticsearch для BM25 и текста чанков."""

    model_config = SettingsConfigDict(
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    url: str = Field(default="http://localhost:9200", validation_alias="ELASTICSEARCH_URL")
    index_name: str = Field(default="legal_chunks", validation_alias="ELASTICSEARCH_INDEX")


class QdrantConfig(BaseSettings):
    """Qdrant для seed-кандидатов."""

    model_config = SettingsConfigDict(
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    url: str = Field(default="http://localhost:6333", validation_alias="QDRANT_URL")
    collection_name: str = Field(default="legal_chunks_vectors", validation_alias="QDRANT_COLLECTION")
    max_attempts: int = Field(default=3, validation_alias="QDRANT_MAX_ATTEMPTS")


class Neo4jConfig(BaseSettings):
    """Neo4j для графовой части."""

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


class OllamaConfig(BaseSettings):
    """Ollama для эмбеддинга запроса."""

    model_config = SettingsConfigDict(
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    base_url: str = Field(default="http://localhost:11434", validation_alias="OLLAMA_BASE_URL")
    model_name: str = Field(default="qwen3-embedding:0.6b", validation_alias="OLLAMA_MODEL")
    timeout_sec: float = Field(default=30.0, validation_alias="OLLAMA_TIMEOUT_SEC")
    max_attempts: int = Field(default=3, validation_alias="OLLAMA_MAX_ATTEMPTS")
    embedding_dim: int = Field(default=256, validation_alias="EMBEDDING_DIM")


class SearchConfig(BaseSettings):
    """Параметры поиска и fusion."""

    model_config = SettingsConfigDict(
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    retriever_top_k: int = Field(default=5, validation_alias="RETRIEVER_TOP_K")
    bm25_top_k: int = Field(default=30, validation_alias="BM25_TOP_K")
    gnn_top_k: int = Field(default=30, validation_alias="GNN_TOP_K")
    bm25_weight: float = Field(default=0.5, validation_alias="BM25_WEIGHT")
    gnn_weight: float = Field(default=0.5, validation_alias="GNN_WEIGHT")
    min_score_threshold: float = Field(default=0.0, validation_alias="MIN_SCORE_THRESHOLD")


class GnnConfig(BaseSettings):
    """Параметры GNN-inference."""

    model_config = SettingsConfigDict(
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    model_path: str = Field(default="stage7_1_retriever/gnn_weights.json", validation_alias="GNN_MODEL_PATH")
    hops: int = Field(default=2, validation_alias="GNN_HOPS")
    batch_size: int = Field(default=64, validation_alias="GNN_BATCH_SIZE")
    use_neo4j_embeddings: bool = Field(default=True, validation_alias="GNN_USE_NEO4J_EMBEDDINGS")
    neo4j_gnn_vector_index: str = Field(
        default="chunk_gnn_embedding",
        validation_alias="GNN_NEO4J_VECTOR_INDEX",
    )

    @field_validator("neo4j_gnn_vector_index")
    @classmethod
    def _neo4j_vector_index_safe(cls, value: str) -> str:
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", value):
            raise ValueError("GNN_NEO4J_VECTOR_INDEX must match [A-Za-z][A-Za-z0-9_]*")
        return value


class RuntimeConfig(BaseSettings):
    """Параметры рантайма."""

    model_config = SettingsConfigDict(
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")


@dataclass(frozen=True)
class AppConfig:
    """Сводная конфигурация ретривера."""

    elasticsearch: ElasticsearchConfig
    qdrant: QdrantConfig
    neo4j: Neo4jConfig
    ollama: OllamaConfig
    search: SearchConfig
    gnn: GnnConfig
    runtime: RuntimeConfig


def load_config() -> AppConfig:
    """Загружает конфигурацию из окружения и .env."""
    return AppConfig(
        elasticsearch=ElasticsearchConfig(),
        qdrant=QdrantConfig(),
        neo4j=Neo4jConfig(),
        ollama=OllamaConfig(),
        search=SearchConfig(),
        gnn=GnnConfig(),
        runtime=RuntimeConfig(),
    )
