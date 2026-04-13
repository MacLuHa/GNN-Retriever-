from __future__ import annotations

import logging
from dataclasses import dataclass

from .config import ExtractionConfig
from .entity_ids import normalize_entity_name
from .models import EntityExtractionResult, ExtractedEntity
from .ner_entity_extractor import NerEntityExtractor, measure_ner_span_sanity, merge_entity_candidates
from .ollama_entity_extractor import OllamaEntityExtractor

logger = logging.getLogger("stage3_3_graph_knowledge_base")


@dataclass(frozen=True)
class ExtractionDiagnostics:
    """Диагностика одного прохода extraction для сравнения режимов."""

    mode: str
    llm_entities_raw: int
    ner_entities_raw: int
    merged_entities: int
    relations: int
    dropped_relations: int
    overlap_entities: int
    ner_hash_artifacts: int
    ner_short_entities: int
    ner_usable_entities: int


class HybridEntityExtractor:
    """Orchestration над LLM-only и NER-assisted режимами без изменения формата результата."""

    def __init__(
        self,
        config: ExtractionConfig,
        llm_extractor: OllamaEntityExtractor,
        ner_extractor: NerEntityExtractor | None = None,
    ) -> None:
        self._config = config
        self._llm_extractor = llm_extractor
        self._ner_extractor = ner_extractor

    async def extract(self, text: str) -> tuple[EntityExtractionResult, ExtractionDiagnostics]:
        mode = self._config.mode.strip().lower()
        if mode == "llm_only":
            return await self._extract_llm_only(text)
        if mode == "ner_assisted":
            return await self._extract_ner_assisted(text)
        if mode == "ner_only":
            return await self._extract_ner_only(text)

        logger.warning("Unknown extraction mode=%s, falling back to llm_only", mode)
        return await self._extract_llm_only(text)

    async def _extract_llm_only(self, text: str) -> tuple[EntityExtractionResult, ExtractionDiagnostics]:
        result = await self._llm_extractor.extract(text)
        diagnostics = ExtractionDiagnostics(
            mode="llm_only",
            llm_entities_raw=len(result.entities),
            ner_entities_raw=0,
            merged_entities=len(result.entities),
            relations=len(result.relations),
            dropped_relations=0,
            overlap_entities=0,
            ner_hash_artifacts=0,
            ner_short_entities=0,
            ner_usable_entities=0,
        )
        return result, diagnostics

    async def _extract_ner_assisted(self, text: str) -> tuple[EntityExtractionResult, ExtractionDiagnostics]:
        if self._ner_extractor is None:
            raise RuntimeError("NER-assisted mode requested but NerEntityExtractor is not configured")

        llm_result = await self._llm_extractor.extract(text)
        ner_candidates = await self._ner_extractor.extract(text)

        llm_names = [entity.name for entity in llm_result.entities]
        ner_names = [candidate.name for candidate in ner_candidates]
        ner_sanity = measure_ner_span_sanity(ner_names)
        merged_entities = merge_entity_candidates(llm_names, ner_names)
        constrained_entities = self._limit_entities_for_relation_prompt(merged_entities)

        relation_result = await self._llm_extractor.extract_relations(
            text,
            [entity.name for entity in constrained_entities],
        )
        raw_relations = len(relation_result.relations)
        final_result = EntityExtractionResult(
            entities=merged_entities,
            relations=relation_result.relations,
        )
        overlap_entities = len(
            {
                self._normalize_name(name)
                for name in llm_names
                if self._normalize_name(name)
            }
            & {
                self._normalize_name(name)
                for name in ner_names
                if self._normalize_name(name)
            }
        )
        diagnostics = ExtractionDiagnostics(
            mode="ner_assisted",
            llm_entities_raw=len(llm_result.entities),
            ner_entities_raw=len(ner_candidates),
            merged_entities=len(merged_entities),
            relations=len(final_result.relations),
            dropped_relations=max(raw_relations - len(final_result.relations), 0),
            overlap_entities=overlap_entities,
            ner_hash_artifacts=ner_sanity.hash_artifacts,
            ner_short_entities=ner_sanity.short_entities,
            ner_usable_entities=ner_sanity.usable_entities,
        )
        return final_result, diagnostics

    async def _extract_ner_only(self, text: str) -> tuple[EntityExtractionResult, ExtractionDiagnostics]:
        if self._ner_extractor is None:
            raise RuntimeError("NER-only mode requested but NerEntityExtractor is not configured")

        print("\ntext: ", text)

        ner_candidates = await self._ner_extractor.extract(text)
        ner_names = [candidate.name for candidate in ner_candidates]
        print("\nner_names: ", ner_names)
        ner_sanity = measure_ner_span_sanity(ner_names)
        print("\nner_sanity: ", ner_sanity)
        merged_entities = merge_entity_candidates(ner_names)
        print("\nmerged_entities: ", merged_entities)
        constrained_entities = self._limit_entities_for_relation_prompt(merged_entities)    
        print("\nconstrained_entities: ", constrained_entities)


        relation_result = await self._llm_extractor.extract_relations(
            text,
            [entity.name for entity in constrained_entities],
        )
        relation_backed_keys = {
            self._normalize_name(rel.from_name)
            for rel in relation_result.relations
            if self._normalize_name(rel.from_name)
        } | {
            self._normalize_name(rel.to_name)
            for rel in relation_result.relations
            if self._normalize_name(rel.to_name)
        }
        relation_backed_entities = [
            entity
            for entity in merged_entities
            if self._normalize_name(entity.name) in relation_backed_keys
        ]

        raw_relations = len(relation_result.relations)
        final_result = EntityExtractionResult(
            entities=relation_backed_entities,
            relations=relation_result.relations,
        )
        diagnostics = ExtractionDiagnostics(
            mode="ner_only",
            llm_entities_raw=0,
            ner_entities_raw=len(ner_candidates),
            merged_entities=len(merged_entities),
            relations=len(final_result.relations),
            dropped_relations=max(raw_relations - len(final_result.relations), 0),
            overlap_entities=0,
            ner_hash_artifacts=ner_sanity.hash_artifacts,
            ner_short_entities=ner_sanity.short_entities,
            ner_usable_entities=ner_sanity.usable_entities,
        )
        return final_result, diagnostics

    def _limit_entities_for_relation_prompt(self, entities: list[ExtractedEntity]) -> list[ExtractedEntity]:
        limit = self._config.relation_entity_limit
        if limit <= 0:
            return entities
        return entities[:limit]

    @staticmethod
    def _normalize_name(name: str) -> str:
        return normalize_entity_name(name)
