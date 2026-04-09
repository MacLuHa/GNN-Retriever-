from __future__ import annotations

from dataclasses import dataclass

from qdrant_client import AsyncQdrantClient

from shared.ollama_embedder import OllamaEmbedder

from ..config import AppConfig, load_config
from ..entity_embedding_service import EntityEmbeddingService
from ..entity_qdrant_store import EntityQdrantStore
from ..neo4j_store import Neo4jGraphStore
from .entity_embedding_backfill_service import EntityEmbeddingBackfillService


@dataclass(frozen=True)
class EntityEmbeddingAppContext:
    """Зависимости batch-прохода по embedding сущностей."""

    config: AppConfig
    qdrant_client: AsyncQdrantClient
    graph_store: Neo4jGraphStore
    service: EntityEmbeddingService
    backfill_service: EntityEmbeddingBackfillService


async def build_entity_embedding_app(
    config: AppConfig | None = None,
) -> EntityEmbeddingAppContext:
    cfg = config or load_config()
    qdrant_client = AsyncQdrantClient(url=cfg.qdrant.url)
    graph_store = Neo4jGraphStore(cfg.neo4j)
    embedder = OllamaEmbedder(
        cfg.ollama_embedding,
        logger_name="stage3_3_graph_knowledge_base",
        log_prefix="Entity embedding",
        error_prefix="entity embedding",
    )
    qdrant_store = EntityQdrantStore(qdrant_client, cfg.qdrant)
    service = EntityEmbeddingService(graph_store, embedder, qdrant_store)
    backfill_service = EntityEmbeddingBackfillService(graph_store, service)
    return EntityEmbeddingAppContext(
        config=cfg,
        qdrant_client=qdrant_client,
        graph_store=graph_store,
        service=service,
        backfill_service=backfill_service,
    )


async def shutdown_entity_embedding_app(ctx: EntityEmbeddingAppContext) -> None:
    close_method = getattr(ctx.qdrant_client, "close", None)
    if callable(close_method):
        await close_method()
    await ctx.graph_store.close()
