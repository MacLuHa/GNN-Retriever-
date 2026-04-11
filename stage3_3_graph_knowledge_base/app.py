from __future__ import annotations

import json
from dataclasses import dataclass

from aiokafka import AIOKafkaProducer
from elasticsearch import AsyncElasticsearch
from qdrant_client import AsyncQdrantClient

from shared.ollama_embedder import OllamaEmbedder

from .config import AppConfig, load_config
from .entity_embedding_service import EntityEmbeddingService
from .entity_qdrant_store import EntityQdrantStore
from .graph_service import GraphKnowledgeService
from .hybrid_extractor import HybridEntityExtractor
from .neo4j_store import Neo4jGraphStore
from .ner_entity_extractor import NerEntityExtractor
from .ollama_entity_extractor import OllamaEntityExtractor


@dataclass(frozen=True)
class AppContext:
    """Зависимости сервиса графа."""

    config: AppConfig
    es_client: AsyncElasticsearch
    producer: AIOKafkaProducer
    qdrant_client: AsyncQdrantClient
    graph_store: Neo4jGraphStore
    entity_embedding_service: EntityEmbeddingService
    service: GraphKnowledgeService


async def build_app(config: AppConfig | None = None) -> AppContext:
    cfg = config or load_config()
    producer = AIOKafkaProducer(
        bootstrap_servers=cfg.kafka.bootstrap_servers,
        value_serializer=lambda v: json.dumps(v, ensure_ascii=False).encode("utf-8"),
    )
    await producer.start()

    es_client = AsyncElasticsearch(hosts=[cfg.elasticsearch.url])
    qdrant_client = AsyncQdrantClient(url=cfg.qdrant.url)
    graph_store = Neo4jGraphStore(cfg.neo4j)
    llm_extractor = OllamaEntityExtractor(cfg.ollama_llm)
    ner_extractor = None
    if cfg.extraction.mode.strip().lower() == "ner_assisted":
        ner_extractor = NerEntityExtractor(cfg.ner)
    extractor = HybridEntityExtractor(
        cfg.extraction,
        llm_extractor,
        ner_extractor=ner_extractor,
    )
    entity_embedder = OllamaEmbedder(
        cfg.ollama_embedding,
        logger_name="stage3_3_graph_knowledge_base",
        log_prefix="Entity embedding",
        error_prefix="entity embedding",
    )
    entity_qdrant_store = EntityQdrantStore(qdrant_client, cfg.qdrant)
    entity_embedding_service = EntityEmbeddingService(
        graph_store,
        entity_embedder,
        entity_qdrant_store,
    )
    service = GraphKnowledgeService(
        cfg,
        es_client,
        extractor,
        graph_store,
        entity_embedding_service,
        producer,
    )
    return AppContext(
        config=cfg,
        es_client=es_client,
        producer=producer,
        qdrant_client=qdrant_client,
        graph_store=graph_store,
        entity_embedding_service=entity_embedding_service,
        service=service,
    )


async def shutdown_app(ctx: AppContext) -> None:
    await ctx.producer.stop()
    await ctx.es_client.close()
    close_method = getattr(ctx.qdrant_client, "close", None)
    if callable(close_method):
        await close_method()
    await ctx.graph_store.close()
