from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import uuid
from collections.abc import AsyncIterator

from aiokafka import AIOKafkaConsumer
from elasticsearch import AsyncElasticsearch
from neo4j import AsyncGraphDatabase
from neo4j.exceptions import ClientError

from .config import AppConfig, Neo4jConfig, load_config
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


async def _iter_messages(cfg: AppConfig, group_id: str, limit: int) -> AsyncIterator[dict]:
    consumer = AIOKafkaConsumer(
        cfg.kafka.graph_topic,
        bootstrap_servers=cfg.kafka.bootstrap_servers,
        group_id=group_id,
        enable_auto_commit=False,
        auto_offset_reset="earliest",
        value_deserializer=_deserialize_value,
    )
    await consumer.start()
    seen = 0
    try:
        async for msg in consumer:
            if isinstance(msg.value, dict):
                yield msg.value
                seen += 1
                if seen >= limit:
                    break
    finally:
        await consumer.stop()


async def _build_extractor(cfg: AppConfig) -> HybridEntityExtractor:
    llm_extractor = OllamaEntityExtractor(cfg.ollama_llm)
    ner_extractor = None
    if cfg.extraction.mode.strip().lower() == "ner_assisted":
        ner_extractor = NerEntityExtractor(cfg.ner)
    return HybridEntityExtractor(cfg.extraction, llm_extractor, ner_extractor=ner_extractor)


async def replay_topic_to_database(
    *,
    cfg: AppConfig,
    database: str,
    limit: int,
    group_id: str,
) -> tuple[int, int]:
    await _ensure_database(cfg, database)

    graph_cfg = Neo4jConfig(
        uri=cfg.neo4j.uri,
        user=cfg.neo4j.user,
        password=cfg.neo4j.password,
        database=database,
    )
    graph_store = Neo4jGraphStore(graph_cfg)
    extractor = await _build_extractor(cfg)
    es_client = AsyncElasticsearch(hosts=[cfg.elasticsearch.url])

    processed = 0
    written = 0
    try:
        await graph_store.ensure_schema()
        async for payload in _iter_messages(cfg, group_id, limit):
            try:
                message = GraphMessage.model_validate(payload)
            except Exception as exc:
                logger.warning("Skip invalid payload: %s", exc)
                continue

            text = await fetch_chunk_text(
                es_client,
                index_name=cfg.elasticsearch.index_name,
                es_doc_id=message.es_doc_id,
            )
            if text is None:
                logger.warning("Skip message with missing text chunk_id=%s", message.chunk_id)
                continue

            try:
                extraction, diagnostics = await extractor.extract(text)
            except Exception as exc:
                logger.warning("Skip extraction failure chunk_id=%s: %s", message.chunk_id, exc)
                continue

            if not extraction.entities:
                logger.info("Skip empty extraction chunk_id=%s", message.chunk_id)
                continue

            entity_ids = await graph_store.apply_extraction(extraction, message=message)
            processed += 1
            written += len(entity_ids)

            if processed % 25 == 0:
                logger.info(
                    (
                        "Replay progress db=%s processed=%s total_entities=%s "
                        "last_chunk=%s mode=%s entities=%s relations=%s"
                    ),
                    database,
                    processed,
                    written,
                    message.chunk_id,
                    diagnostics.mode,
                    len(extraction.entities),
                    len(extraction.relations),
                )
    finally:
        await es_client.close()
        await graph_store.close()

    return processed, written


async def _async_main(database: str, limit: int) -> None:
    cfg = load_config()
    set_entity_lemma_lang(cfg.entity_normalization.lemma_lang)
    set_entity_lemma_scope(cfg.entity_normalization.lemma_scope)
    group_id = f"{cfg.kafka.consumer_group}-replay-{database}-{uuid.uuid4().hex[:8]}"
    processed, written = await replay_topic_to_database(
        cfg=cfg,
        database=database,
        limit=limit,
        group_id=group_id,
    )
    logger.info(
        "Replay finished db=%s processed_messages=%s written_entities=%s",
        database,
        processed,
        written,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay Kafka topic documents.graph into a separate Neo4j database.")
    parser.add_argument("--database", default="graph_test_replay", help="Target Neo4j database name")
    parser.add_argument("--limit", type=int, default=1000, help="How many topic messages to replay")
    args = parser.parse_args()

    cfg = load_config()
    logging.basicConfig(
        level=getattr(logging, cfg.runtime.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    asyncio.run(_async_main(args.database, args.limit))


if __name__ == "__main__":
    main()
