from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class GraphMessage(BaseModel):
    """Вход из топика documents.graph (после векторизации чанка)."""

    doc_id: str
    chunk_id: str
    version_id: str
    es_doc_id: str
    embedding_id: str


class GraphEntityOutputMessage(BaseModel):
    """Событие после обработки чанка: все сущности из этого чанка."""

    doc_id: str
    chunk_id: str
    version_id: str
    es_doc_id: str
    embedding_id: str
    entity_ids: list[str]


class GraphDlqMessage(BaseModel):
    """Сообщение в DLQ при ошибке обработки или отсутствии сущностей."""

    reason: str
    detail: str | None = None
    doc_id: str | None = None
    chunk_id: str | None = None
    version_id: str | None = None
    es_doc_id: str | None = None
    embedding_id: str | None = None
    original_payload: dict[str, Any] | None = None

    @classmethod
    def from_graph_message(
        cls,
        message: GraphMessage,
        *,
        reason: str,
        detail: str | None = None,
    ) -> GraphDlqMessage:
        return cls(
            reason=reason,
            detail=detail,
            doc_id=message.doc_id,
            chunk_id=message.chunk_id,
            version_id=message.version_id,
            es_doc_id=message.es_doc_id,
            embedding_id=message.embedding_id,
        )

    @classmethod
    def from_payload_dict(
        cls,
        payload: dict[str, Any],
        *,
        reason: str,
        detail: str | None = None,
    ) -> GraphDlqMessage:
        def _str_field(key: str) -> str | None:
            v = payload.get(key)
            return v if isinstance(v, str) else None

        return cls(
            reason=reason,
            detail=detail,
            doc_id=_str_field("doc_id"),
            chunk_id=_str_field("chunk_id"),
            version_id=_str_field("version_id"),
            es_doc_id=_str_field("es_doc_id"),
            embedding_id=_str_field("embedding_id"),
            original_payload=payload,
        )


class ExtractedEntity(BaseModel):
    """Сущность из ответа LLM."""

    name: str = Field(min_length=1)


class ExtractedRelation(BaseModel):
    """Связь между сущностями (направленная)."""

    model_config = ConfigDict(populate_by_name=True)

    from_name: str = Field(alias="from", min_length=1)
    to_name: str = Field(alias="to", min_length=1)


class EntityExtractionResult(BaseModel):
    """Структурированный ответ модели извлечения."""

    entities: list[ExtractedEntity] = Field(default_factory=list)
    relations: list[ExtractedRelation] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def normalize_entities(cls, data: Any) -> Any:
        """LLM иногда отдаёт ``entities`` как ``[\"a\", \"b\"]`` вместо ``[{\"name\": \"a\"}, ...]``."""
        if not isinstance(data, dict):
            return data
        raw = data.get("entities")
        if raw is None:
            return {**data, "entities": []}
        if not isinstance(raw, list):
            return data
        normalized: list[Any] = []
        for item in raw:
            if isinstance(item, str):
                s = item.strip()
                if s:
                    normalized.append({"name": s})
            elif isinstance(item, dict):
                normalized.append(item)
        return {**data, "entities": normalized}
