from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from gliner import GLiNER

from .config import NerConfig
from .entity_ids import normalize_entity_name
from .models import ExtractedEntity

logger = logging.getLogger("stage3_3_graph_knowledge_base")


@dataclass(frozen=True)
class NerCandidate:
    """Кандидат сущности из NER до финальной дедупликации."""

    name: str
    source: str = "ner"


@dataclass(frozen=True)
class NerSpanSanity:
    """Простые признаки качества NER span outputs."""

    total: int
    hash_artifacts: int
    short_entities: int
    usable_entities: int


class NerEntityExtractor:
    """Извлечение candidate entities через GLiNER с кастомными KG label'ами."""

    def __init__(self, config: NerConfig) -> None:
        self._config = config
        allowed = [x.strip().lower() for x in config.allowed_groups.split(",") if x.strip()]
        self._allowed_groups = set(allowed)
        map_location = "cuda" if config.device >= 0 else "cpu"
        self._model = GLiNER.from_pretrained(
            config.model_name,
            map_location=map_location,
        )

    async def extract(self, text: str) -> list[NerCandidate]:
        if not text.strip():
            return []
        return await asyncio.to_thread(self._extract_sync, text)

    def _extract_sync(self, text: str) -> list[NerCandidate]:
        raw_items = self._model.predict_entities(
            text,
            labels=list(self._allowed_groups),
            threshold=self._config.min_score,
        )
        candidates: list[NerCandidate] = []
        seen: set[str] = set()

        for item in raw_items:
            if not isinstance(item, dict):
                continue
            group = str(item.get("label", "")).strip().lower()
            if self._allowed_groups and group not in self._allowed_groups:
                continue
            score = float(item.get("score", 0.0))
            if score < self._config.min_score:
                continue
            name = str(item.get("text", "")).strip()
            if not name:
                continue
            key = normalize_entity_name(name)
            if not key or key in seen:
                continue
            seen.add(key)
            candidates.append(NerCandidate(name=name))
            if len(candidates) >= self._config.max_entities:
                break

        logger.debug("GLiNER extracted candidates=%s labels=%s", len(candidates), sorted(self._allowed_groups))
        return candidates


def merge_entity_candidates(*candidate_lists: list[str]) -> list[ExtractedEntity]:
    """Объединяет entity-кандидаты по нормализованному имени, сохраняя первый display-name."""

    merged: dict[str, str] = {}
    for candidates in candidate_lists:
        for raw_name in candidates:
            name = raw_name.strip()
            if not name:
                continue
            key = normalize_entity_name(name)
            if not key or key in merged:
                continue
            merged[key] = name
    return [ExtractedEntity(name=name) for name in merged.values()]


def measure_ner_span_sanity(names: list[str]) -> NerSpanSanity:
    """Считает простые признаки шумных span outputs."""

    hash_artifacts = 0
    short_entities = 0
    usable_entities = 0

    for raw_name in names:
        name = raw_name.strip()
        if "##" in name:
            hash_artifacts += 1
        if len(name) <= 2:
            short_entities += 1
        if name and "##" not in name and len(name) > 2:
            usable_entities += 1

    return NerSpanSanity(
        total=len(names),
        hash_artifacts=hash_artifacts,
        short_entities=short_entities,
        usable_entities=usable_entities,
    )
