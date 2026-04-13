from __future__ import annotations

import asyncio
import json
import logging
import math
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from neo4j import AsyncDriver
from qdrant_client import AsyncQdrantClient, models

from .config import GnnConfig, Neo4jConfig, OllamaConfig, QdrantConfig
from .models import RetrievedChunk

logger = logging.getLogger("stage7_1_retriever")


@dataclass(frozen=True)
class QdrantSeed:
    """Seed-кандидат из векторного поиска."""

    chunk_id: str
    score: float
    vector: list[float]
    payload: dict[str, Any]


@dataclass(frozen=True)
class GnnWeights:
    """Весовые матрицы GNN."""

    w_self_1: list[list[float]]
    w_neigh_1: list[list[float]]
    b_1: list[float]
    w_self_2: list[list[float]]
    w_neigh_2: list[list[float]]
    b_2: list[float]
    query_projection: list[list[float]] | None
    scorer: list[float] | None
    scorer_bias: float


class GnnRetriever:
    """Графовый ретривер с inference GNN."""

    def __init__(
        self,
        qdrant_client: AsyncQdrantClient,
        qdrant_config: QdrantConfig,
        neo4j_driver: AsyncDriver,
        neo4j_config: Neo4jConfig,
        ollama_config: OllamaConfig,
        gnn_config: GnnConfig,
        default_top_k: int,
    ) -> None:
        self._qdrant = qdrant_client
        self._qdrant_config = qdrant_config
        self._neo4j_driver = neo4j_driver
        self._neo4j_config = neo4j_config
        self._ollama = ollama_config
        self._gnn = gnn_config
        self._default_top_k = default_top_k


    async def close(self) -> None:
        """Закрывает внешние ресурсы GNN-ретривера."""
        await self._neo4j_driver.close()
        close_method = getattr(self._qdrant, "close", None)
        if callable(close_method):
            maybe_awaitable = close_method()
            if asyncio.iscoroutine(maybe_awaitable):
                await maybe_awaitable

    async def retrieve(self, query_text: str, top_k: int | None = None) -> list[RetrievedChunk]:
        """Возвращает кандидаты через графовую модель."""
        query = query_text.strip()
        if not query:
            raise ValueError("Query text must not be empty")

        query_vector = await self._embed_query(query)
        size = max(1, top_k or self._default_top_k)
  
        return await self._retrieve_neo4j_gnn(query, query_vector, size)


    async def _retrieve_neo4j_gnn(self, query: str, query_vector: list[float], size: int) -> list[RetrievedChunk]:
        """Ранжирование по gnn_embedding из Neo4j (ноутбук) + vector index для seeds."""
        q_fit = _fit_vector(query_vector, self._ollama.embedding_dim)
        seeds = await self._fetch_neo4j_vector_seeds(q_fit, size)
        if not seeds:
            logger.info("Neo4j GNN retrieval has no vector-index seeds (check index ONLINE and gnn_embedding)")
            return []

        graph_map, _related_edges = await self._expand_graph({s.chunk_id for s in seeds})
        candidate_chunks = set(graph_map.keys()) | {s.chunk_id for s in seeds}
        chunk_vectors = await self._fetch_gnn_vectors_from_neo4j(candidate_chunks)
        if not chunk_vectors:
            logger.warning("Neo4j GNN retrieval: no gnn_embedding on candidate chunks")
            return []

        seed_index_scores = {s.chunk_id: s.score for s in seeds}
        ranked: list[RetrievedChunk] = []
        for chunk_id, raw_vec in chunk_vectors.items():
            chunk_fit = _fit_vector(raw_vec, self._ollama.embedding_dim)
            cosine = _cosine_similarity(chunk_fit, q_fit)
            score = cosine
            meta: dict[str, Any] = {
                "entity_count": len(graph_map.get(chunk_id, [])),
                "neo4j_gnn": True,
            }
            if chunk_id in seed_index_scores:
                idx_sc = seed_index_scores[chunk_id]
                meta["neo4j_vector_index_score"] = idx_sc
                meta["seed_score"] = idx_sc
            ranked.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    score=score,
                    source_scores={"gnn": score},
                    metadata=meta,
                )
            )

        ranked.sort(key=lambda item: item.score, reverse=True)
        logger.info("Neo4j GNN retrieval done query_len=%s results=%s", len(query), len(ranked[:size]))
        return ranked[:size]

    async def _fetch_neo4j_vector_seeds(self, query_vector: list[float], top_k: int) -> list[QdrantSeed]:
        """Top-k Chunk по векторному индексу Neo4j на gnn_embedding."""
        seeds: list[QdrantSeed] = []
        cypher = """
        CALL db.index.vector.queryNodes($index_name, $top_k, $embedding)
        YIELD node, score
        RETURN node.chunk_id AS chunk_id, score AS score, node.gnn_embedding AS gnn_embedding
        """
        try:
            async with self._neo4j_driver.session(database=self._neo4j_config.database) as session:
                result = await session.run(
                    cypher,
                    index_name=self._gnn.neo4j_gnn_vector_index,
                    top_k=top_k,
                    embedding=query_vector,
                )
                async for row in result:
                    raw_cid = row.get("chunk_id")
                    if raw_cid is None:
                        continue
                    chunk_id = str(raw_cid)
                    if not chunk_id:
                        continue
                    vec = _as_vector(row["gnn_embedding"])
                    score = float(row["score"] or 0.0)
                    if not vec:
                        continue
                    seeds.append(
                        QdrantSeed(
                            chunk_id=chunk_id,
                            score=score,
                            vector=vec,
                            payload={"source": "neo4j_vector_index"},
                        )
                    )
        except Exception as exc:  # noqa: BLE001
            msg = str(exc)
            logger.warning("Neo4j vector index query failed: %s", exc)
            if "no such vector schema index" in msg.lower():
                logger.warning(
                    "Создайте индекс в Neo4j 5.11+ (после записи Chunk.gnn_embedding): см. "
                    "stage7_1_retriever/neo4j/chunk_gnn_embedding_vector_index.cypher "
                    "или ячейку «Создать vector index» в notebooks/gnn_retrieval_query_test.ipynb. "
                    "Проверка: SHOW INDEXES. Имя индекса должно совпадать с GNN_NEO4J_VECTOR_INDEX=%r.",
                    self._gnn.neo4j_gnn_vector_index,
                )
        return seeds

    async def _fetch_gnn_vectors_from_neo4j(self, chunk_ids: set[str]) -> dict[str, list[float]]:
        """Читает c.gnn_embedding пакетами по списку chunk_id."""
        if not chunk_ids:
            return {}
        ids = list(chunk_ids)
        batch_sz = max(1, self._gnn.batch_size)
        vectors: dict[str, list[float]] = {}
        cypher = """
        UNWIND $chunk_ids AS cid
        MATCH (c:Chunk)
        WHERE c.chunk_id = cid AND c.gnn_embedding IS NOT NULL
        RETURN c.chunk_id AS chunk_id, c.gnn_embedding AS gnn_embedding
        """
        async with self._neo4j_driver.session(database=self._neo4j_config.database) as session:
            for offset in range(0, len(ids), batch_sz):
                batch = ids[offset : offset + batch_sz]
                result = await session.run(cypher, chunk_ids=batch)
                async for row in result:
                    raw_cid = row.get("chunk_id")
                    if raw_cid is None:
                        continue
                    cid = str(raw_cid)
                    vec = _as_vector(row["gnn_embedding"])
                    if cid and vec:
                        vectors[cid] = vec
        return vectors

    async def _embed_query(self, query_text: str) -> list[float]:
        """Строит embedding запроса через Ollama."""
        last_error: Exception | None = None
        for attempt in range(1, self._ollama.max_attempts + 1):
            try:
                timeout = httpx.Timeout(self._ollama.timeout_sec)
                async with httpx.AsyncClient(base_url=self._ollama.base_url, timeout=timeout) as client:
                    response = await client.post(
                        "/api/embed",
                        json={
                            "model": self._ollama.model_name,
                            "input": query_text,
                            "truncate": True,
                            "dimensions": self._ollama.embedding_dim,
                        },
                    )
                    response.raise_for_status()
                    payload = response.json()
                    embeddings = payload.get("embeddings")
                    if not embeddings or not isinstance(embeddings, list):
                        raise ValueError("Ollama returned empty embeddings")
                    vector = embeddings[0]
                    if not isinstance(vector, list) or not vector:
                        raise ValueError("Ollama embedding has invalid format")
                    return [float(value) for value in vector]
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                logger.warning(
                    "Query embedding failed attempt=%s/%s: %s",
                    attempt,
                    self._ollama.max_attempts,
                    exc,
                )
                if attempt < self._ollama.max_attempts:
                    await asyncio.sleep(min(2**attempt, 5))
        raise RuntimeError("Failed to create query embedding") from last_error

    async def _fetch_seed_candidates(self, query_vector: list[float], top_k: int) -> list[QdrantSeed]:
        """Получает начальные кандидаты из Qdrant."""
        try:
            response = await self._qdrant.query_points(
                collection_name=self._qdrant_config.collection_name,
                query=query_vector,
                with_payload=True,
                with_vectors=True,
                limit=top_k,
            )
            raw_points = getattr(response, "points", response)
        except Exception:
            raw_points = await self._qdrant.search(
                collection_name=self._qdrant_config.collection_name,
                query_vector=query_vector,
                with_payload=True,
                with_vectors=True,
                limit=top_k,
            )

        seeds: list[QdrantSeed] = []
        for point in raw_points:
            payload = _as_dict(getattr(point, "payload", {}) or {})
            chunk_id = str(payload.get("chunk_id") or "")
            if not chunk_id:
                continue
            vector_raw = getattr(point, "vector", None) or payload.get("vector") or []
            vector = _as_vector(vector_raw)
            if not vector:
                continue
            score = float(getattr(point, "score", 0.0) or 0.0)
            seeds.append(QdrantSeed(chunk_id=chunk_id, score=score, vector=vector, payload=payload))
        return seeds

    async def _expand_graph(self, seed_chunk_ids: set[str]) -> tuple[dict[str, list[str]], set[tuple[str, str]]]:
        """Расширяет seed-кандидаты по графу."""
        if not seed_chunk_ids:
            return {}, set()

        chunk_to_entities: dict[str, list[str]] = {}
        all_entities: set[str] = set()
        hops = max(0, self._gnn.hops)
        query = f"""
        MATCH (seed_chunk:Chunk)-[:MENTIONS]->(seed_entity:Entity)
        WHERE seed_chunk.chunk_id IN $chunk_ids
        MATCH path=(seed_entity)-[:RELATED_TO*0..{hops}]-(neighbor_entity:Entity)
        WITH collect(DISTINCT seed_entity) + collect(DISTINCT neighbor_entity) AS entities
        UNWIND entities AS ent
        MATCH (chunk:Chunk)-[:MENTIONS]->(ent)
        RETURN chunk.chunk_id AS chunk_id, collect(DISTINCT ent.entity_id) AS entity_ids
        """

        async with self._neo4j_driver.session(database=self._neo4j_config.database) as session:
            rows = await session.run(query, chunk_ids=list(seed_chunk_ids))
            async for row in rows:
                chunk_id = str(row["chunk_id"])
                entity_ids = [str(item) for item in row["entity_ids"] if item]
                if not entity_ids:
                    continue
                chunk_to_entities[chunk_id] = entity_ids
                all_entities.update(entity_ids)

            related_edges: set[tuple[str, str]] = set()
            if all_entities:
                rel_result = await session.run(
                    """
                    UNWIND $entity_ids AS entity_id
                    MATCH (source:Entity {entity_id: entity_id})-[:RELATED_TO]-(target:Entity)
                    WHERE target.entity_id IN $entity_ids
                    RETURN source.entity_id AS source_id, target.entity_id AS target_id
                    """,
                    entity_ids=list(all_entities),
                )
                async for row in rel_result:
                    source = str(row["source_id"])
                    target = str(row["target_id"])
                    if source and target and source != target:
                        related_edges.add((source, target))
            return chunk_to_entities, related_edges

    async def _fetch_chunk_vectors(self, chunk_ids: set[str]) -> dict[str, list[float]]:
        """Читает векторы чанков по chunk_id."""
        vectors: dict[str, list[float]] = {}
        for chunk_id in chunk_ids:
            try:
                points, _ = await self._qdrant.scroll(
                    collection_name=self._qdrant_config.collection_name,
                    scroll_filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="chunk_id",
                                match=models.MatchValue(value=chunk_id),
                            )
                        ]
                    ),
                    limit=1,
                    with_payload=True,
                    with_vectors=True,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Qdrant scroll failed chunk_id=%s: %s", chunk_id, exc)
                continue
            if not points:
                continue
            point = points[0]
            vector = _as_vector(getattr(point, "vector", None))
            if not vector:
                continue
            vectors[chunk_id] = vector
        return vectors

    def _build_node_vectors(
        self,
        chunk_to_entities: dict[str, list[str]],
        related_edges: set[tuple[str, str]],
        chunk_vectors: dict[str, list[float]],
        query_vector: list[float],
    ) -> dict[str, list[float]]:
        """Строит граф узлов и признаки узлов."""
        vector_dim = len(next(iter(chunk_vectors.values()))) if chunk_vectors else len(query_vector)
        adjacency: dict[str, set[str]] = defaultdict(set)
        node_features: dict[str, list[float]] = {}

        for chunk_id, vector in chunk_vectors.items():
            node_features[f"chunk:{chunk_id}"] = _fit_vector(vector, vector_dim)

        entity_to_chunks: dict[str, list[str]] = defaultdict(list)
        for chunk_id, entity_ids in chunk_to_entities.items():
            chunk_key = f"chunk:{chunk_id}"
            if chunk_key not in node_features:
                continue
            for entity_id in entity_ids:
                entity_key = f"entity:{entity_id}"
                adjacency[chunk_key].add(entity_key)
                adjacency[entity_key].add(chunk_key)
                entity_to_chunks[entity_id].append(chunk_id)

        for entity_id, chunk_ids in entity_to_chunks.items():
            vectors = [node_features[f"chunk:{chunk_id}"] for chunk_id in chunk_ids if f"chunk:{chunk_id}" in node_features]
            if vectors:
                node_features[f"entity:{entity_id}"] = _mean_vectors(vectors, vector_dim)
            else:
                node_features[f"entity:{entity_id}"] = _fit_vector(query_vector, vector_dim)

        for source, target in related_edges:
            source_key = f"entity:{source}"
            target_key = f"entity:{target}"
            if source_key not in node_features or target_key not in node_features:
                continue
            adjacency[source_key].add(target_key)
            adjacency[target_key].add(source_key)

        if not node_features:
            return {}

        return {
            node_id: _mean_vectors(
                [node_features[node_id]] + [node_features[neighbor] for neighbor in adjacency.get(node_id, set()) if neighbor in node_features],
                vector_dim,
            )
            for node_id in node_features
        }

    def _run_layer(
        self,
        node_vectors: dict[str, list[float]],
        w_self: list[list[float]],
        w_neigh: list[list[float]],
        bias: list[float],
    ) -> dict[str, list[float]]:
        """Запускает один слой message passing."""
        in_dim = len(next(iter(node_vectors.values())))
        out_dim = len(bias)
        if not _weights_shape_ok(w_self, in_dim, out_dim) or not _weights_shape_ok(w_neigh, in_dim, out_dim):
            raise ValueError("Invalid GNN weight dimensions")
        encoded: dict[str, list[float]] = {}
        for node_id, vector in node_vectors.items():
            self_part = _matmul_vec(vector, w_self)
            neigh_part = _matmul_vec(vector, w_neigh)
            encoded[node_id] = [_relu(self_part[i] + neigh_part[i] + bias[i]) for i in range(out_dim)]
        return encoded

    def _project_query(self, query_vector: list[float], output_dim: int) -> list[float]:
        """Проецирует вектор запроса в пространство модели."""
        if self._weights.query_projection is None:
            return _fit_vector(query_vector, output_dim)
        matrix = self._weights.query_projection
        in_dim = len(query_vector)
        if not _weights_shape_ok(matrix, in_dim, output_dim):
            raise ValueError("Invalid query projection dimensions")
        return _matmul_vec(query_vector, matrix)

    @staticmethod
    def _load_weights(path: Path) -> GnnWeights:
        """Загружает веса GNN из json."""
        if not path.exists():
            raise RuntimeError(
                f"GNN model file not found: {path}. Set GNN_MODEL_PATH to a valid json file"
            )
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid GNN model file format: {path}") from exc

        required = ("w_self_1", "w_neigh_1", "b_1", "w_self_2", "w_neigh_2", "b_2")
        missing = [key for key in required if key not in payload]
        if missing:
            raise RuntimeError(f"GNN model file is missing required keys: {', '.join(missing)}")

        return GnnWeights(
            w_self_1=_as_matrix(payload["w_self_1"]),
            w_neigh_1=_as_matrix(payload["w_neigh_1"]),
            b_1=_as_vector(payload["b_1"]),
            w_self_2=_as_matrix(payload["w_self_2"]),
            w_neigh_2=_as_matrix(payload["w_neigh_2"]),
            b_2=_as_vector(payload["b_2"]),
            query_projection=_as_matrix(payload["query_projection"]) if payload.get("query_projection") is not None else None,
            scorer=_as_vector(payload["scorer"]) if payload.get("scorer") is not None else None,
            scorer_bias=float(payload.get("scorer_bias", 0.0)),
        )


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


def _as_vector(value: Any) -> list[float]:
    if isinstance(value, dict):
        return []
    if not isinstance(value, Iterable) or isinstance(value, (str, bytes)):
        return []
    return [float(item) for item in value]


def _as_matrix(value: Any) -> list[list[float]]:
    if not isinstance(value, list):
        raise ValueError("Matrix must be a list")
    matrix = [_as_vector(row) for row in value]
    if not matrix:
        raise ValueError("Matrix must not be empty")
    width = len(matrix[0])
    if width == 0 or any(len(row) != width for row in matrix):
        raise ValueError("Matrix rows must have equal non-zero size")
    return matrix


def _fit_vector(vector: list[float], dim: int) -> list[float]:
    if len(vector) == dim:
        return list(vector)
    if len(vector) > dim:
        return list(vector[:dim])
    return list(vector) + [0.0] * (dim - len(vector))


def _mean_vectors(vectors: list[list[float]], dim: int) -> list[float]:
    if not vectors:
        return [0.0] * dim
    sums = [0.0] * dim
    for vector in vectors:
        fitted = _fit_vector(vector, dim)
        for idx, value in enumerate(fitted):
            sums[idx] += value
    return [value / len(vectors) for value in sums]


def _weights_shape_ok(matrix: list[list[float]], in_dim: int, out_dim: int) -> bool:
    return len(matrix) == in_dim and all(len(row) == out_dim for row in matrix)


def _matmul_vec(vector: list[float], matrix: list[list[float]]) -> list[float]:
    if not matrix:
        return []
    out_dim = len(matrix[0])
    output = [0.0] * out_dim
    for row_index, row in enumerate(matrix):
        value = vector[row_index] if row_index < len(vector) else 0.0
        for col_index in range(out_dim):
            output[col_index] += value * row[col_index]
    return output


def _dot(left: list[float], right: list[float]) -> float:
    size = min(len(left), len(right))
    return sum(left[idx] * right[idx] for idx in range(size))


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    dot = _dot(left, right)
    left_norm = math.sqrt(_dot(left, left))
    right_norm = math.sqrt(_dot(right, right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def _relu(value: float) -> float:
    return value if value > 0.0 else 0.0
