from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re

from aiokafka import AIOKafkaConsumer
from elasticsearch import AsyncElasticsearch
from neo4j import AsyncGraphDatabase
from neo4j.exceptions import ClientError

from .config import AppConfig, ExtractionConfig, Neo4jConfig, load_config
from .entity_ids import set_entity_lemma_lang, set_entity_lemma_scope
from .es_chunk_fetcher import fetch_chunk_text
from .hybrid_extractor import HybridEntityExtractor
from .models import GraphMessage
from .neo4j_store import Neo4jGraphStore
from .ner_entity_extractor import NerEntityExtractor
from .ollama_entity_extractor import OllamaEntityExtractor

logger = logging.getLogger("stage3_3_graph_knowledge_base.replay")

_DB_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]*$")


def _deserialize_value(raw: bytes | None):
    if raw is None:
        return None
    return json.loads(raw.decode("utf-8"))


def _build_replay_group_id(base_group_id: str, database: str, mode: str) -> str:
    safe_database = re.sub(r"[^A-Za-z0-9_.-]+", "-", database).strip("-") or "neo4j"
    safe_mode = re.sub(r"[^A-Za-z0-9_.-]+", "-", mode).strip("-") or "default"
    return f"{base_group_id}-replay-{safe_database}-{safe_mode}"


async def _ensure_database(cfg: AppConfig, database: str) -> None:
    if not _DB_NAME_RE.match(database):
        raise ValueError(f"Unsafe Neo4j database name: {database!r}")
    if database == "neo4j":
        return

    driver = AsyncGraphDatabase.driver(
        cfg.neo4j.uri,
        auth=(cfg.neo4j.user, cfg.neo4j.password),
    )
    try:
        try:
            async with driver.session(database="system") as session:
                result = await session.run(f"CREATE DATABASE `{database}` IF NOT EXISTS")
                await result.consume()
        except ClientError as exc:
            if "UnsupportedAdministrationCommand" in str(exc):
                raise RuntimeError(
                    "Current Neo4j instance does not support CREATE DATABASE. "
                    "Use a separate Neo4j instance with database=neo4j for isolated tests."
                ) from exc
            raise
    finally:
        await driver.close()


def _build_consumer(cfg: AppConfig, group_id: str) -> AIOKafkaConsumer:
    return AIOKafkaConsumer(
        cfg.kafka.graph_topic,
        bootstrap_servers=cfg.kafka.bootstrap_servers,
        group_id=group_id,
        enable_auto_commit=False,
        auto_offset_reset="earliest",
        value_deserializer=_deserialize_value,
    )


def _build_extraction_config(cfg: AppConfig, mode: str) -> ExtractionConfig:
    return ExtractionConfig(
        mode=mode,
        relation_entity_limit=cfg.extraction.relation_entity_limit,
        diagnostics_enabled=cfg.extraction.diagnostics_enabled,
        quality_filter_enabled=cfg.extraction.quality_filter_enabled,
        max_entities_per_chunk=cfg.extraction.max_entities_per_chunk,
        generic_entity_stopwords=cfg.extraction.generic_entity_stopwords,
    )


async def _build_extractor(cfg: AppConfig, mode: str) -> HybridEntityExtractor:
    extraction_cfg = _build_extraction_config(cfg, mode)
    llm_extractor = OllamaEntityExtractor(cfg.ollama_llm)
    ner_extractor = None
    if extraction_cfg.mode.strip().lower() in {"ner_assisted", "ner_only"}:
        ner_extractor = NerEntityExtractor(cfg.ner)
    return HybridEntityExtractor(extraction_cfg, llm_extractor, ner_extractor=ner_extractor)


async def replay_topic_to_database(
    *,
    cfg: AppConfig,
    database: str,
    limit: int,
    group_id: str,
    mode: str,
) -> tuple[int, int]:
    await _ensure_database(cfg, database)

    graph_cfg = Neo4jConfig(
        uri=cfg.neo4j.uri,
        user=cfg.neo4j.user,
        password=cfg.neo4j.password,
        database=database,
    )
    graph_store = Neo4jGraphStore(graph_cfg)
    extractor = await _build_extractor(cfg, mode)
    es_client = AsyncElasticsearch(hosts=[cfg.elasticsearch.url])
    consumer = _build_consumer(cfg, group_id)

    processed = 0
    written = 0
    seen = 0
    try:
        await consumer.start()
        await graph_store.ensure_schema()
        async for msg in consumer:
            payload = msg.value
            should_commit = False
            try:
                seen += 1
                if not isinstance(payload, dict):
                    logger.warning("Skip non-dict payload at offset=%s", msg.offset)
                    should_commit = True
                    continue

                message = GraphMessage.model_validate(payload)
                text = await fetch_chunk_text(
                    es_client,
                    index_name=cfg.elasticsearch.index_name,
                    es_doc_id=message.es_doc_id,
                )
                if text is None:
                    logger.warning("Skip message with missing text chunk_id=%s", message.chunk_id)
                    should_commit = True
                    continue

                extraction, diagnostics = await extractor.extract(text)
                if not extraction.entities:
                    logger.info("Skip empty extraction chunk_id=%s", message.chunk_id)
                    should_commit = True
                    continue

                entity_ids = await graph_store.apply_extraction(extraction, message=message)
                processed += 1
                written += len(entity_ids)
                should_commit = True

                if processed % 25 == 0:
                    logger.info(
                        (
                            "Replay progress db=%s group_id=%s processed=%s total_entities=%s "
                            "last_chunk=%s mode=%s entities=%s relations=%s"
                        ),
                        database,
                        group_id,
                        processed,
                        written,
                        message.chunk_id,
                        diagnostics.mode,
                        len(extraction.entities),
                        len(extraction.relations),
                    )
            except Exception as exc:
                logger.exception(
                    "Replay stopped without committing offset topic=%s partition=%s offset=%s",
                    msg.topic,
                    msg.partition,
                    msg.offset,
                )
                raise RuntimeError(
                    f"Replay failed at topic={msg.topic} partition={msg.partition} offset={msg.offset}"
                ) from exc
            finally:
                if should_commit:
                    await consumer.commit()
                    if limit > 0 and seen >= limit:
                        break
    finally:
        await consumer.stop()
        await es_client.close()
        await graph_store.close()

    return processed, written


async def _async_main(database: str, limit: int, mode: str, group_id: str | None) -> None:
    cfg = load_config()
    set_entity_lemma_lang(cfg.entity_normalization.lemma_lang)
    set_entity_lemma_scope(cfg.entity_normalization.lemma_scope)
    effective_group_id = group_id or _build_replay_group_id(cfg.kafka.consumer_group, database, mode)
    processed, written = await replay_topic_to_database(
        cfg=cfg,
        database=database,
        limit=limit,
        group_id=effective_group_id,
        mode=mode,
    )
    logger.info(
        "Replay finished db=%s mode=%s group_id=%s processed_messages=%s written_entities=%s",
        database,
        mode,
        effective_group_id,
        processed,
        written,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay Kafka topic documents.graph into a separate Neo4j database.")
    parser.add_argument("--database", default="graph_ner_only", help="Target Neo4j database name")
    parser.add_argument(
        "--mode",
        default="ner_only",
        choices=["llm_only", "ner_assisted", "ner_only"],
        help="Extraction mode to use during replay",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="How many topic messages to replay; 0 means replay the whole topic",
    )
    parser.add_argument(
        "--group-id",
        default=None,
        help="Consumer group id for resumable replay; defaults to a stable value based on database and mode",
    )
    args = parser.parse_args()

    cfg = load_config()
    logging.basicConfig(
        level=getattr(logging, cfg.runtime.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    asyncio.run(_async_main(args.database, args.limit, args.mode, args.group_id))


if __name__ == "__main__":
    main()
