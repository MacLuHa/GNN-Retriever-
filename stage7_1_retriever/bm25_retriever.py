from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from elasticsearch import AsyncElasticsearch

from .models import RetrievedChunk

logger = logging.getLogger("stage7_1_retriever")

_ALLOWED_FILTERS = {
    "chunk_id",
    "doc_id",
    "version_id",
    "section_id",
    "language",
    "jurisdiction",
    "source_type",
    "is_canonical",
}


class Bm25Retriever:
    """Лексический ретривер по индексу Elasticsearch."""

    def __init__(self, client: AsyncElasticsearch, index_name: str, default_top_k: int) -> None:
        self._client = client
        self._index_name = index_name
        self._default_top_k = default_top_k

    async def retrieve(
        self,
        query_text: str,
        filters: Mapping[str, Any] | None = None,
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        """Возвращает кандидаты BM25."""
        cleaned_query = query_text.strip()
        if not cleaned_query:
            raise ValueError("Query text must not be empty")

        size = max(1, top_k or self._default_top_k)
        filter_clause = self._build_filter_clause(filters or {})
        body: dict[str, Any] = {
            "size": size,
            "query": {
                "bool": {
                    "must": [
                        {
                            "multi_match": {
                                "query": cleaned_query,
                                "fields": ["text^2", "normalized_text"],
                            }
                        }
                    ],
                    "filter": filter_clause,
                }
            },
        }

        response = await self._client.search(index=self._index_name, body=body)
        hits = response.get("hits", {}).get("hits", [])
        results: list[RetrievedChunk] = []
        for hit in hits:
            source = hit.get("_source", {})
            chunk_id = str(source.get("chunk_id") or hit.get("_id") or "")
            if not chunk_id:
                continue

            metadata = {
                "doc_id": source.get("doc_id"),
                "version_id": source.get("version_id"),
                "language": source.get("language"),
                "jurisdiction": source.get("jurisdiction"),
                "is_canonical": source.get("is_canonical"),
                "source_type": source.get("source_type"),
                "es_doc_id": hit.get("_id"),
            }
            score = float(hit.get("_score") or 0.0)
            results.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    score=score,
                    text=str(source.get("text") or ""),
                    source_scores={"bm25": score},
                    metadata=metadata,
                )
            )
        logger.info("BM25 retrieval done query_len=%s results=%s", len(cleaned_query), len(results))
        return results

    @staticmethod
    def _build_filter_clause(filters: Mapping[str, Any]) -> list[dict[str, Any]]:
        clause: list[dict[str, Any]] = []
        for key, value in filters.items():
            if key not in _ALLOWED_FILTERS or value is None:
                continue
            if isinstance(value, (list, tuple, set)):
                values = [item for item in value if item is not None]
                if values:
                    clause.append({"terms": {key: values}})
                continue
            clause.append({"term": {key: value}})
        return clause
