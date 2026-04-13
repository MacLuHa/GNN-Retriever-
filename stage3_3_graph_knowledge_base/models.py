from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .entity_ids import normalize_entity_name

_MIN_ENTITY_NAME_LEN = 3
_GENERIC_ENTITY_STOPWORDS = {
    "state",
    "system",
    "group",
    "people",
    "society",
    "government",
    "movement",
    "religion",
    "empire",
    "law",
}
_MAX_ENTITIES_PER_CHUNK = 24


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


class GraphEntityNode(BaseModel):
    """Сущность, считанная из Neo4j для последующей векторизации."""

    entity_id: str
    entity_name: str
    embedding_id: str | None = None


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
                if len(s) >= _MIN_ENTITY_NAME_LEN:
                    normalized.append({"name": s})
            elif isinstance(item, ExtractedEntity):
                normalized.append(item.model_dump())
            elif isinstance(item, dict):
                normalized.append(item)
        return {**data, "entities": normalized}

    @model_validator(mode="after")
    def drop_entities_shorter_than_min(self) -> EntityExtractionResult:
        """Имя сущности не короче ``_MIN_ENTITY_NAME_LEN`` символов (после strip)."""
        kept = [e for e in self.entities if len(e.name.strip()) >= _MIN_ENTITY_NAME_LEN]
        self.entities = kept
        return self

    @model_validator(mode="after")
    def filter_low_quality_entities(self) -> EntityExtractionResult:
        """Снижает шум в LLM-only extraction: защищает relation-backed entities и отбрасывает слабые singleton'ы."""
        relation_backed_keys = set()
        for rel in self.relations:
            relation_backed_keys.add(normalize_entity_name(rel.from_name))
            relation_backed_keys.add(normalize_entity_name(rel.to_name))
        relation_backed_keys.discard("")

        stronger_names = []
        for entity in self.entities:
            name = " ".join(entity.name.strip().split())
            if not name:
                continue
            key = normalize_entity_name(name)
            token_count = len(name.split())
            is_relation_backed = key in relation_backed_keys
            is_generic_singleton = token_count == 1 and key in _GENERIC_ENTITY_STOPWORDS

            if is_generic_singleton and not is_relation_backed:
                continue
            stronger_names.append(ExtractedEntity(name=name))

        normalized_names = [normalize_entity_name(entity.name) for entity in stronger_names]
        kept_entities: list[ExtractedEntity] = []
        for idx, entity in enumerate(stronger_names):
            key = normalized_names[idx]
            is_relation_backed = key in relation_backed_keys
            token_count = len(entity.name.split())

            # Убираем короткую общую сущность, если рядом есть более длинная, содержащая её как подстроку.
            shadowed_by_longer = any(
                idx != other_idx
                and key
                and key in normalized_names[other_idx]
                and len(normalized_names[other_idx]) > len(key) + 2
                for other_idx in range(len(stronger_names))
            )
            if shadowed_by_longer and token_count == 1 and not is_relation_backed:
                continue
            kept_entities.append(entity)

        prioritized = sorted(
            kept_entities,
            key=lambda entity: (
                normalize_entity_name(entity.name) not in relation_backed_keys,
                len(entity.name.split()) == 1,
                len(entity.name),
            ),
            reverse=False,
        )
        self.entities = prioritized[:_MAX_ENTITIES_PER_CHUNK]
        return self

    @model_validator(mode="after")
    def relations_only_between_listed_entities(self) -> EntityExtractionResult:
        """Оставляет только связи, у которых оба конца есть в ``entities`` (по нормализованному имени)."""
        keys = {normalize_entity_name(e.name) for e in self.entities}
        keys.discard("")
        filtered = [
            r
            for r in self.relations
            if normalize_entity_name(r.from_name) in keys and normalize_entity_name(r.to_name) in keys
        ]
        self.relations = filtered
        return self


class RelationExtractionResult(BaseModel):
    """Структурированный ответ модели извлечения только связей."""

    relations: list[ExtractedRelation] = Field(default_factory=list)
