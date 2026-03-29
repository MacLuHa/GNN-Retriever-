from __future__ import annotations

import json
from dataclasses import dataclass

from aiokafka import AIOKafkaProducer
from elasticsearch import AsyncElasticsearch

from .config import AppConfig, load_config
from .graph_service import GraphKnowledgeService
from .neo4j_store import Neo4jGraphStore
from .ollama_entity_extractor import OllamaEntityExtractor


@dataclass(frozen=True)
class AppContext:
    """Зависимости сервиса графа."""

    config: AppConfig
    es_client: AsyncElasticsearch
    producer: AIOKafkaProducer
    graph_store: Neo4jGraphStore
    service: GraphKnowledgeService


async def build_app(config: AppConfig | None = None) -> AppContext:
    cfg = config or load_config()
    producer = AIOKafkaProducer(
        bootstrap_servers=cfg.kafka.bootstrap_servers,
        value_serializer=lambda v: json.dumps(v, ensure_ascii=False).encode("utf-8"),
    )
    await producer.start()

    es_client = AsyncElasticsearch(hosts=[cfg.elasticsearch.url])
    graph_store = Neo4jGraphStore(cfg.neo4j)
    extractor = OllamaEntityExtractor(cfg.ollama_llm)
    service = GraphKnowledgeService(cfg, es_client, extractor, graph_store, producer)
    return AppContext(
        config=cfg,
        es_client=es_client,
        producer=producer,
        graph_store=graph_store,
        service=service,
    )


async def shutdown_app(ctx: AppContext) -> None:
    await ctx.producer.stop()
    await ctx.es_client.close()
    await ctx.graph_store.close()
