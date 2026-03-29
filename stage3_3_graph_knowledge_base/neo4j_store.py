from __future__ import annotations

import logging
from typing import Any

from neo4j import AsyncGraphDatabase, AsyncDriver

from .config import Neo4jConfig
from .entity_ids import make_entity_id, normalize_entity_name
from .models import EntityExtractionResult

logger = logging.getLogger("stage3_3_graph_knowledge_base")


class Neo4jGraphStore:
    """Запись узлов Entity и рёбер RELATED_TO."""

    def __init__(self, config: Neo4jConfig) -> None:
        self._config = config
        self._driver: AsyncDriver = AsyncGraphDatabase.driver(
            config.uri,
            auth=(config.user, config.password),
        )

    async def close(self) -> None:
        await self._driver.close()

    async def ensure_schema(self) -> None:
        """Уникальность entity_id для MERGE."""
        q = (
            "CREATE CONSTRAINT entity_id_unique IF NOT EXISTS "
            "FOR (e:Entity) REQUIRE e.entity_id IS UNIQUE"
        )
        async with self._driver.session(database=self._config.database) as session:
            res = await session.run(q)
            await res.consume()
        logger.info("Neo4j schema ensured (Entity.entity_id unique)")

    def _collect_rows(
        self,
        extraction: EntityExtractionResult,
        embedding_id: str,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Строки для UNWIND и список рёбер по id."""
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
            rows.append(
                {
                    "id": eid,
                    "name": display,
                    "embedding_id": embedding_id,
                }
            )

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
        embedding_id: str,
    ) -> list[str]:
        """MERGE сущностей и рёбер RELATED_TO для одного чанка.

        Возвращает список ``entity_id`` записанных узлов (порядок соответствует обходу).
        """
        rows, rel_rows = self._collect_rows(extraction, embedding_id)
        if not rows:
            logger.info("No entities to write")
            return []

        async with self._driver.session(database=self._config.database) as session:
            res = await session.run(
                """
                UNWIND $rows AS row
                MERGE (e:Entity {entity_id: row.id})
                ON CREATE SET
                  e.entity_name = row.name,
                  e.embedding_id = row.embedding_id
                ON MATCH SET
                  e.embedding_id = row.embedding_id
                """,
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
