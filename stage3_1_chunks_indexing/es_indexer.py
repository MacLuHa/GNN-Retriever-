from __future__ import annotations

from typing import Any

from elasticsearch import AsyncElasticsearch

# Шаблон индекса Elasticsearch: маппинг полей чанков.
INDEX_BODY: dict[str, Any] = {
    "settings": {"number_of_shards": 1, "number_of_replicas": 0},
    "mappings": {
        "properties": {
            "chunk_id": {"type": "keyword"},
            "doc_id": {"type": "keyword"},
            "version_id": {"type": "keyword"},
            "section_id": {"type": "keyword"},
            "text": {"type": "text", "similarity": "BM25"},
            "normalized_text": {"type": "text"},
            "language": {"type": "keyword"},
            "jurisdiction": {"type": "keyword"},
            "source_type": {"type": "keyword"},
            "is_canonical": {"type": "boolean"},
            "page": {"type": "integer"},
            "span_start": {"type": "integer"},
            "span_end": {"type": "integer"},
            "effective_date": {"type": "date"},
            "exact_hash": {"type": "keyword"},
            "entity_ids": {"type": "keyword"},
        }
    },
}


class ElasticsearchIndexer:
    """Асинхронное создание индекса и добавление документов."""

    def __init__(self, client: AsyncElasticsearch, index_name: str) -> None:
        """Клиент ES и имя индекса для записи."""
        self._client = client
        self._index_name = index_name

    @property
    def index_name(self) -> str:
        """Имя индекса в кластере."""
        return self._index_name

    async def ensure_index(self) -> None:
        """Создаёт индекс с маппингом, если его ещё нет."""
        exists = await self._client.indices.exists(index=self._index_name)
        if not exists:
            await self._client.indices.create(
                index=self._index_name,
                settings=INDEX_BODY["settings"],
                mappings=INDEX_BODY["mappings"],
            )

    async def index_document(self, document_id: str, body: dict[str, Any]) -> str:
        """Индексирует документ с явным _id; для чанков передают chunk_id, ответный _id тот же."""
        resp = await self._client.index(
            index=self._index_name,
            id=document_id,
            document=body,
            refresh=True,
        )
        es_doc_id = str(resp["_id"])
        return es_doc_id
