from __future__ import annotations

import logging

from shared.ollama_embedder import OllamaEmbedder

from .models import GraphEntityNode
from .neo4j_store import Neo4jGraphStore
from .entity_qdrant_store import EntityQdrantStore

logger = logging.getLogger("stage3_3_graph_knowledge_base")


class EntityEmbeddingService:
    """Сохраняет entity embeddings в Qdrant и ссылки на них в Neo4j."""

    def __init__(
        self,
        graph_store: Neo4jGraphStore,
        embedder: OllamaEmbedder,
        qdrant_store: EntityQdrantStore,
    ) -> None:
        self._graph_store = graph_store
        self._embedder = embedder
        self._qdrant_store = qdrant_store

    async def upsert_entities(self, entities: list[GraphEntityNode]) -> int:
        """Записывает embedding для переданных сущностей и возвращает число обновлённых ссылок."""
        updated = 0
        for entity in entities:
            vector = await self._embedder.embed(entity.entity_name)
            embedding_id = await self._qdrant_store.upsert_entity_vector(
                entity.entity_id,
                entity.entity_name,
                vector,
            )
            if entity.embedding_id != embedding_id:
                await self._graph_store.set_entity_embedding_id(entity.entity_id, embedding_id)
                updated += 1
            logger.info(
                "Entity embedding stored entity_id=%s embedding_id=%s",
                entity.entity_id,
                embedding_id,
            )
        return updated
