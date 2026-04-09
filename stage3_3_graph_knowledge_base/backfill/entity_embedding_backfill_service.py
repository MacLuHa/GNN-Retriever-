from __future__ import annotations

import logging

from ..entity_embedding_service import EntityEmbeddingService
from ..neo4j_store import Neo4jGraphStore

logger = logging.getLogger("stage3_3_graph_knowledge_base")


class EntityEmbeddingBackfillService:
    """Полный проход по всем сущностям для backfill entity embeddings."""

    def __init__(
        self,
        graph_store: Neo4jGraphStore,
        entity_embedding_service: EntityEmbeddingService,
    ) -> None:
        self._graph_store = graph_store
        self._entity_embedding_service = entity_embedding_service

    async def run(self) -> tuple[int, int]:
        """Возвращает количество обработанных сущностей и реально обновлённых ссылок."""
        total = 0
        updated = 0
        async for entity in self._graph_store.iter_entities():
            total += 1
            updated += await self._entity_embedding_service.upsert_entities([entity])

        logger.info("Entity embedding backfill finished total=%s updated=%s", total, updated)
        return total, updated
