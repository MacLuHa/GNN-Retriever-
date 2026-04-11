from __future__ import annotations

import logging
from dataclasses import dataclass

from .config import ExtractionConfig
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
        if mode == "ner_assisted":
            return await self._extract_ner_assisted(text)
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
        #ner_sanity = measure_ner_span_sanity(ner_names)
        #merged_entities = merge_entity_candidates(ner_names, llm_names)

        print("\nner_names: ", ner_names)
        print("\nllm_names: ", llm_names)

        import time 
        time.sleep(1000)
