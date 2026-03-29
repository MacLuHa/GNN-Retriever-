from __future__ import annotations

import logging

from elasticsearch import AsyncElasticsearch
from elasticsearch.exceptions import NotFoundError

logger = logging.getLogger("stage3_3_graph_knowledge_base")


async def fetch_chunk_text(
    client: AsyncElasticsearch,
    *,
    index_name: str,
    es_doc_id: str,
) -> str | None:
    """Возвращает поле text чанка из Elasticsearch или None, если документ не найден."""
    try:
        resp = await client.get(index=index_name, id=es_doc_id)
    except NotFoundError:
        logger.warning("ES document not found index=%s id=%s", index_name, es_doc_id)
        return None
    source = resp.get("_source") or {}
    text = source.get("text")
    if not isinstance(text, str) or not text.strip():
        logger.warning("ES document has empty or missing text id=%s", es_doc_id)
        return None
    return text
