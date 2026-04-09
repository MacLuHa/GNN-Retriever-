from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any

from neo4j import AsyncGraphDatabase, AsyncDriver

from .config import Neo4jConfig
from .entity_ids import make_entity_id, normalize_entity_name
from .models import EntityExtractionResult, GraphEntityNode, GraphMessage

logger = logging.getLogger("stage3_3_graph_knowledge_base")


class Neo4jGraphStore:
    """Запись Chunk (провенанс чанка), Entity, рёбер MENTIONS и RELATED_TO."""

    def __init__(self, config: Neo4jConfig) -> None:
        self._config = config
        self._driver: AsyncDriver = AsyncGraphDatabase.driver(
            config.uri,
            auth=(config.user, config.password),
        )

    async def close(self) -> None:
        await self._driver.close()

    async def iter_entities(self) -> AsyncIterator[GraphEntityNode]:
        """Итерирует по всем сущностям, у которых есть непустое имя."""
        async with self._driver.session(database=self._config.database) as session:
            result = await session.run(
                """
                MATCH (e:Entity)
                WHERE e.entity_name IS NOT NULL AND trim(e.entity_name) <> ""
                RETURN
                  e.entity_id AS entity_id,
                  e.entity_name AS entity_name,
                  e.embedding_id AS embedding_id
                ORDER BY e.entity_id
                """
            )
            async for record in result:
                yield GraphEntityNode(
                    entity_id=str(record["entity_id"]),
                    entity_name=str(record["entity_name"]),
                    embedding_id=(
                        str(record["embedding_id"])
                        if record["embedding_id"] is not None
                        else None
                    ),
                )
            await result.consume()

    async def set_entity_embedding_id(self, entity_id: str, embedding_id: str) -> None:
        """Сохраняет ссылку на вектор сущности в Neo4j."""
        async with self._driver.session(database=self._config.database) as session:
            result = await session.run(
                """
                MATCH (e:Entity {entity_id: $entity_id})
                SET e.embedding_id = $embedding_id
                RETURN e.entity_id AS entity_id
                """,
                entity_id=entity_id,
                embedding_id=embedding_id,
            )
            record = await result.single()
            if record is None:
                raise ValueError(f"Entity not found for entity_id={entity_id}")

    async def ensure_schema(self) -> None:
        """Уникальность chunk_id и entity_id для MERGE."""
        statements = (
            "CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS "
            "FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE",
            "CREATE CONSTRAINT entity_id_unique IF NOT EXISTS "
            "FOR (e:Entity) REQUIRE e.entity_id IS UNIQUE",
        )
        async with self._driver.session(database=self._config.database) as session:
            for q in statements:
                res = await session.run(q)
                await res.consume()
        logger.info("Neo4j schema ensured (Chunk.chunk_id, Entity.entity_id unique)")

    def _collect_rows(
        self,
        extraction: EntityExtractionResult,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Строки для UNWIND (Entity) и список рёбер RELATED_TO по id."""
        names: dict[str, str] = {}
        for ent in extraction.entities:
            key = normalize_entity_name(ent.name)
            if key:
                names[key] = ent.name.strip()
        for rel in extraction.relations:
            for raw in (rel.from_name, rel.to_name):
                n = normalize_entity_name(raw)
                if n and n not in names:
                    names[n] = raw.strip()

        rows: list[dict[str, Any]] = []
        key_to_id: dict[str, str] = {}
        for key, display in names.items():
            eid = make_entity_id(display)
            key_to_id[key] = eid
            rows.append({"id": eid, "name": display})

        rel_rows: list[dict[str, Any]] = []
        for rel in extraction.relations:
            fk = normalize_entity_name(rel.from_name)
            tk = normalize_entity_name(rel.to_name)
            if not fk or not tk:
                continue
            fid = key_to_id.get(fk)
            tid = key_to_id.get(tk)
            if fid is None or tid is None or fid == tid:
                continue
            rel_rows.append({"from_id": fid, "to_id": tid})

        return rows, rel_rows

    async def apply_extraction(
        self,
        extraction: EntityExtractionResult,
        *,
        message: GraphMessage,
    ) -> list[str]:
        """MERGE Chunk, Entity, (:Chunk)-[:MENTIONS]->(:Entity) и RELATED_TO между сущностями.

        Возвращает список ``entity_id`` для этого чанка (порядок соответствует обходу).
        """
        rows, rel_rows = self._collect_rows(extraction)
        if not rows:
            logger.info("No entities to write")
            return []

        async with self._driver.session(database=self._config.database) as session:
            res = await session.run(
                """
                MERGE (c:Chunk {chunk_id: $chunk_id})
                SET
                  c.doc_id = $doc_id,
                  c.version_id = $version_id,
                  c.es_doc_id = $es_doc_id,
                  c.embedding_id = $embedding_id
                WITH c
                UNWIND $rows AS row
                MERGE (e:Entity {entity_id: row.id})
                ON CREATE SET e.entity_name = row.name
                ON MATCH SET e.entity_name = row.name
                MERGE (c)-[:MENTIONS]->(e)
                """,
                chunk_id=message.chunk_id,
                doc_id=message.doc_id,
                version_id=message.version_id,
                es_doc_id=message.es_doc_id,
                embedding_id=message.embedding_id,
                rows=rows,
            )
            await res.consume()
            if rel_rows:
                res2 = await session.run(
                    """
                    UNWIND $rels AS rel
                    MATCH (a:Entity {entity_id: rel.from_id})
                    MATCH (b:Entity {entity_id: rel.to_id})
                    MERGE (a)-[:RELATED_TO]->(b)
                    """,
                    rels=rel_rows,
                )
                await res2.consume()

        return [str(r["id"]) for r in rows]
