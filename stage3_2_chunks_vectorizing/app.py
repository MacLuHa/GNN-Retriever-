from __future__ import annotations

import json
from dataclasses import dataclass

from aiokafka import AIOKafkaProducer
from qdrant_client import AsyncQdrantClient

from shared.ollama_embedder import OllamaEmbedder

from .config import AppConfig, load_config
from .qdrant_store import QdrantStore
from .vectorize_service import ChunkVectorizeService


@dataclass(frozen=True)
class AppContext:
    """Контекст приложения и его зависимости."""

    config: AppConfig
    producer: AIOKafkaProducer
    qdrant_client: AsyncQdrantClient
    embedder: OllamaEmbedder
    qdrant_store: QdrantStore
    chunk_vectorize_service: ChunkVectorizeService


async def build_app(config: AppConfig | None = None) -> AppContext:
    """Собирает зависимости сервиса векторизации."""
    cfg = config or load_config()
    producer = AIOKafkaProducer(
        bootstrap_servers=cfg.kafka.bootstrap_servers,
        value_serializer=lambda v: json.dumps(v, ensure_ascii=False).encode("utf-8"),
    )
    await producer.start()

    qdrant_client = AsyncQdrantClient(url=cfg.qdrant.url)
    embedder = OllamaEmbedder(
        cfg.ollama,
        logger_name="stage3_2_chunks_vectorizing",
        log_prefix="Embedding",
        error_prefix="embedding",
    )
    qdrant_store = QdrantStore(qdrant_client, cfg.qdrant)
    service = ChunkVectorizeService(cfg, embedder, qdrant_store, producer)

    return AppContext(
        config=cfg,
        producer=producer,
        qdrant_client=qdrant_client,
        embedder=embedder,
        qdrant_store=qdrant_store,
        chunk_vectorize_service=service,
    )


async def shutdown_app(ctx: AppContext) -> None:
    """Закрывает все внешние соединения."""
    await ctx.producer.stop()
    await ctx.qdrant_client.close()
