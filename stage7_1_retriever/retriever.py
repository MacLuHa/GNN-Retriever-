from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping
from typing import Any

from elasticsearch import AsyncElasticsearch
from neo4j import AsyncGraphDatabase
from qdrant_client import AsyncQdrantClient

from .bm25_retriever import Bm25Retriever
from .config import AppConfig, load_config
from .gnn_retriever import GnnRetriever
from .models import RetrievedChunk

logger = logging.getLogger("stage7_1_retriever")


class Retriever:
    """Фасад гибридного ретривера."""

    def __init__(
        self,
        config: AppConfig,
        elastic_client: AsyncElasticsearch,
        bm25_retriever: Bm25Retriever,
        gnn_retriever: GnnRetriever,
    ) -> None:
        self._config = config
        self._elastic = elastic_client
        self._bm25 = bm25_retriever
        self._gnn = gnn_retriever

    async def close(self) -> None:
        """Закрывает соединения ретривера."""
        await self._gnn.close()
        await self._elastic.close()

    async def retrieve(
        self,
        query_text: str,
        filters: Mapping[str, Any] | None = None,
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        """Выполняет гибридный поиск."""
        size = max(1, top_k or self._config.search.retriever_top_k)
        bm25_task = self._bm25.retrieve(query_text, filters=filters, top_k=self._config.search.bm25_top_k)
        gnn_task = self._gnn.retrieve(query_text, top_k=self._config.search.gnn_top_k)
        bm25_result, gnn_result = await asyncio.gather(bm25_task, gnn_task, return_exceptions=True)

        bm25_chunks = bm25_result if isinstance(bm25_result, list) else []
        gnn_chunks = gnn_result if isinstance(gnn_result, list) else []
        if isinstance(bm25_result, Exception):
            logger.exception("BM25 retriever failed: %s", bm25_result)
        if isinstance(gnn_result, Exception):
            logger.exception("GNN retriever failed: %s", gnn_result)

        fused = self._fuse_results(bm25_chunks, gnn_chunks)
        filtered = [chunk for chunk in fused if chunk.score >= self._config.search.min_score_threshold]
        top = filtered[:size]
        await self._enrich_text(top)

        logger.info(
            "Hybrid retrieval done query_len=%s bm25=%s gnn=%s returned=%s",
            len(query_text.strip()),
            len(bm25_chunks),
            len(gnn_chunks),
            len(top),
        )
        return top

    def _fuse_results(self, bm25: list[RetrievedChunk], gnn: list[RetrievedChunk]) -> list[RetrievedChunk]:
        bm25_norm = _normalize_scores(bm25)
        gnn_norm = _normalize_scores(gnn)
        merged: dict[str, RetrievedChunk] = {}

        for chunk in bm25:
            fused_score = bm25_norm.get(chunk.chunk_id, 0.0) * self._config.search.bm25_weight
            merged[chunk.chunk_id] = RetrievedChunk(
                chunk_id=chunk.chunk_id,
                score=fused_score,
                text=chunk.text,
                source_scores=dict(chunk.source_scores),
                metadata=dict(chunk.metadata),
            )
        for chunk in gnn:
            fused_score = gnn_norm.get(chunk.chunk_id, 0.0) * self._config.search.gnn_weight
            existing = merged.get(chunk.chunk_id)
            if existing is None:
                merged[chunk.chunk_id] = RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    score=fused_score,
                    text=chunk.text,
                    source_scores=dict(chunk.source_scores),
                    metadata=dict(chunk.metadata),
                )
                continue
            existing.score += fused_score
            existing.source_scores.update(chunk.source_scores)
            for key, value in chunk.metadata.items():
                existing.metadata.setdefault(key, value)
            if not existing.text and chunk.text:
                existing.text = chunk.text

        ranked = sorted(merged.values(), key=lambda item: item.score, reverse=True)
        return ranked

    async def _enrich_text(self, chunks: list[RetrievedChunk]) -> None:
        missing_ids = [chunk.chunk_id for chunk in chunks if not chunk.text]
        if not missing_ids:
            return

        response = await self._elastic.search(
            index=self._config.elasticsearch.index_name,
            body={
                "size": len(missing_ids),
                "query": {"terms": {"chunk_id": missing_ids}},
            },
        )
        hits = response.get("hits", {}).get("hits", [])
        text_by_chunk: dict[str, str] = {}
        for hit in hits:
            source = hit.get("_source", {})
            chunk_id = str(source.get("chunk_id") or "")
            if chunk_id:
                text_by_chunk[chunk_id] = str(source.get("text") or "")

        for chunk in chunks:
            if not chunk.text:
                chunk.text = text_by_chunk.get(chunk.chunk_id, "")


def build_retriever_from_env(config: AppConfig | None = None) -> Retriever:
    """Создаёт ретривер из окружения."""
    app_config = config or load_config()
    elastic = AsyncElasticsearch(app_config.elasticsearch.url)
    qdrant = AsyncQdrantClient(url=app_config.qdrant.url)
    neo4j_driver = AsyncGraphDatabase.driver(
        app_config.neo4j.uri,
        auth=(app_config.neo4j.user, app_config.neo4j.password),
    )
    bm25 = Bm25Retriever(
        client=elastic,
        index_name=app_config.elasticsearch.index_name,
        default_top_k=app_config.search.bm25_top_k,
    )
    gnn = GnnRetriever(
        qdrant_client=qdrant,
        qdrant_config=app_config.qdrant,
        neo4j_driver=neo4j_driver,
        neo4j_config=app_config.neo4j,
        ollama_config=app_config.ollama,
        gnn_config=app_config.gnn,
        default_top_k=app_config.search.gnn_top_k,
    )
    return Retriever(
        config=app_config,
        elastic_client=elastic,
        bm25_retriever=bm25,
        gnn_retriever=gnn,
    )


def _normalize_scores(chunks: list[RetrievedChunk]) -> dict[str, float]:
    if not chunks:
        return {}
    values = [chunk.score for chunk in chunks]
    min_score = min(values)
    max_score = max(values)
    if max_score == min_score:
        return {chunk.chunk_id: 1.0 for chunk in chunks}
    denominator = max_score - min_score
    return {chunk.chunk_id: (chunk.score - min_score) / denominator for chunk in chunks}
