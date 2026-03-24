from __future__ import annotations

import json
from dataclasses import dataclass

from aiokafka import AIOKafkaProducer
from elasticsearch import AsyncElasticsearch

from .config import AppConfig, load_config
from .es_indexer import ElasticsearchIndexer
from .index_service import ChunkIndexService


@dataclass(frozen=True)
class AppContext:
    """Контекст приложения: конфигурация и связанные зависимости воркера индексации."""

    config: AppConfig
    elasticsearch: AsyncElasticsearch
    indexer: ElasticsearchIndexer
    producer: AIOKafkaProducer
    chunk_index_service: ChunkIndexService


async def build_app(config: AppConfig | None = None) -> AppContext:
    """Собирает клиент ES, индексатор, Kafka producer и сервис индексации."""
    cfg = config or load_config()
    es = AsyncElasticsearch(hosts=[cfg.elasticsearch.url])
    indexer = ElasticsearchIndexer(es, cfg.elasticsearch.index_name)
    producer = AIOKafkaProducer(
        bootstrap_servers=cfg.kafka.bootstrap_servers,
        value_serializer=lambda v: json.dumps(v, ensure_ascii=False).encode("utf-8"),
    )
    await producer.start()
    chunk_index_service = ChunkIndexService(cfg, indexer, producer)
    return AppContext(
        config=cfg,
        elasticsearch=es,
        indexer=indexer,
        producer=producer,
        chunk_index_service=chunk_index_service,
    )


async def shutdown_app(ctx: AppContext) -> None:
    """Останавливает producer и закрывает соединение с Elasticsearch."""
    await ctx.producer.stop()
    await ctx.elasticsearch.close()
