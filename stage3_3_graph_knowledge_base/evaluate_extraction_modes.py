from __future__ import annotations

import argparse
import asyncio
import json
import logging
from statistics import mean

from elasticsearch import AsyncElasticsearch

from .config import AppConfig, ExtractionConfig, load_config
from .entity_ids import set_entity_lemma_lang, set_entity_lemma_scope
from .es_chunk_fetcher import fetch_chunk_text
from .hybrid_extractor import HybridEntityExtractor
from .ner_entity_extractor import NerEntityExtractor
from .ollama_entity_extractor import OllamaEntityExtractor

logger = logging.getLogger("stage3_3_graph_knowledge_base")


async def _iter_chunk_ids(client: AsyncElasticsearch, cfg: AppConfig, limit: int) -> list[str]:
    response = await client.search(
        index=cfg.elasticsearch.index_name,
        size=limit,
        _source=False,
        sort=["_doc"],
        query={"match_all": {}},
    )
    hits = response.get("hits", {}).get("hits", [])
    return [str(hit["_id"]) for hit in hits if hit.get("_id") is not None]


async def _evaluate_mode(
    *,
    client: AsyncElasticsearch,
    cfg: AppConfig,
    chunk_ids: list[str],
    mode: str,
) -> dict[str, float]:
    extraction_cfg = ExtractionConfig(
        mode=mode,
        relation_entity_limit=cfg.extraction.relation_entity_limit,
        diagnostics_enabled=cfg.extraction.diagnostics_enabled,
        quality_filter_enabled=cfg.extraction.quality_filter_enabled,
        max_entities_per_chunk=cfg.extraction.max_entities_per_chunk,
        generic_entity_stopwords=cfg.extraction.generic_entity_stopwords,
    )
    extractor = HybridEntityExtractor(
        extraction_cfg,
        OllamaEntityExtractor(cfg.ollama_llm),
        ner_extractor=(
            NerEntityExtractor(cfg.ner)
            if mode.strip().lower() in {"ner_assisted", "ner_only"}
            else None
        ),
    )

    diagnostics_rows = []
    for es_doc_id in chunk_ids:
        text = await fetch_chunk_text(client, index_name=cfg.elasticsearch.index_name, es_doc_id=es_doc_id)
        if text is None:
            continue
        extraction, diagnostics = await extractor.extract(text)
        diagnostics_rows.append(
            {
                "entities": len(extraction.entities),
                "relations": len(extraction.relations),
                "llm_entities_raw": diagnostics.llm_entities_raw,
                "ner_entities_raw": diagnostics.ner_entities_raw,
                "merged_entities": diagnostics.merged_entities,
                "dropped_relations": diagnostics.dropped_relations,
                "overlap_entities": diagnostics.overlap_entities,
                "ner_hash_artifacts": diagnostics.ner_hash_artifacts,
                "ner_short_entities": diagnostics.ner_short_entities,
                "ner_usable_entities": diagnostics.ner_usable_entities,
            }
        )

    if not diagnostics_rows:
        return {"chunks": 0.0}

    summary = {
        "chunks": float(len(diagnostics_rows)),
        "avg_entities": mean(row["entities"] for row in diagnostics_rows),
        "avg_relations": mean(row["relations"] for row in diagnostics_rows),
        "avg_llm_entities_raw": mean(row["llm_entities_raw"] for row in diagnostics_rows),
        "avg_ner_entities_raw": mean(row["ner_entities_raw"] for row in diagnostics_rows),
        "avg_merged_entities": mean(row["merged_entities"] for row in diagnostics_rows),
        "avg_dropped_relations": mean(row["dropped_relations"] for row in diagnostics_rows),
        "avg_overlap_entities": mean(row["overlap_entities"] for row in diagnostics_rows),
        "avg_ner_hash_artifacts": mean(row["ner_hash_artifacts"] for row in diagnostics_rows),
        "avg_ner_short_entities": mean(row["ner_short_entities"] for row in diagnostics_rows),
        "avg_ner_usable_entities": mean(row["ner_usable_entities"] for row in diagnostics_rows),
    }
    return summary


async def _async_main(limit: int) -> None:
    cfg = load_config()
    set_entity_lemma_lang(cfg.entity_normalization.lemma_lang)
    set_entity_lemma_scope(cfg.entity_normalization.lemma_scope)

    client = AsyncElasticsearch(hosts=[cfg.elasticsearch.url])
    try:
        chunk_ids = await _iter_chunk_ids(client, cfg, limit)
        llm_only = await _evaluate_mode(client=client, cfg=cfg, chunk_ids=chunk_ids, mode="llm_only")
        ner_assisted = await _evaluate_mode(client=client, cfg=cfg, chunk_ids=chunk_ids, mode="ner_assisted")
        ner_only = await _evaluate_mode(client=client, cfg=cfg, chunk_ids=chunk_ids, mode="ner_only")
    finally:
        await client.close()

    print(
        json.dumps(
            {
                "sample_size": len(chunk_ids),
                "llm_only": llm_only,
                "ner_assisted": ner_assisted,
                "ner_only": ner_only,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare llm_only, ner_assisted and ner_only extraction on ES chunks.")
    parser.add_argument("--limit", type=int, default=20, help="How many Elasticsearch chunks to sample")
    args = parser.parse_args()

    cfg = load_config()
    logging.basicConfig(
        level=getattr(logging, cfg.runtime.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    asyncio.run(_async_main(args.limit))


if __name__ == "__main__":
    main()
