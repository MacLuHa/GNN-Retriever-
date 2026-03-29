from __future__ import annotations

from pydantic import BaseModel


class VectorizeMessage(BaseModel):
    """Входное сообщение из топика векторизации."""

    doc_id: str
    chunk_id: str
    version_id: str
    es_doc_id: str
    text: str


class GraphMessage(BaseModel):
    """Выходное сообщение для графового этапа."""

    doc_id: str
    chunk_id: str
    version_id: str
    es_doc_id: str
    embedding_id: str
