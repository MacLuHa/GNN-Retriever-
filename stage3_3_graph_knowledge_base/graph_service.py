from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Any

from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from aiokafka.errors import GroupCoordinatorNotAvailableError, NodeNotReadyError
from elasticsearch import AsyncElasticsearch

from .config import AppConfig
from .entity_embedding_service import EntityEmbeddingService
from .entity_ids import make_entity_id
from .es_chunk_fetcher import fetch_chunk_text
from .models import (
    EntityExtractionResult,
    GraphDlqMessage,
    GraphEntityNode,
    GraphEntityOutputMessage,
    GraphMessage,
)
from .neo4j_store import Neo4jGraphStore
from .ollama_entity_extractor import OllamaEntityExtractor

logger = logging.getLogger("stage3_3_graph_knowledge_base")

_KAFKA_START_MAX_ATTEMPTS = 60
_KAFKA_START_RETRY_SEC = 2.0

_DLQ_INVALID_PAYLOAD_TYPE = "invalid_payload_type"
_DLQ_VALIDATION_FAILED = "validation_failed"
_DLQ_CHUNK_TEXT_MISSING = "chunk_text_missing"
_DLQ_EXTRACTION_FAILED = "extraction_failed"
_DLQ_NO_ENTITIES_FOUND = "no_entities_found"
_DLQ_GRAPH_WRITE_FAILED = "graph_write_failed"


class GraphKnowledgeService:
    """Читает documents.graph, извлекает сущности, пишет Neo4j."""

    def __init__(
        self,
        config: AppConfig,
        es_client: AsyncElasticsearch,
        extractor: OllamaEntityExtractor,
        graph_store: Neo4jGraphStore,
        entity_embedding_service: EntityEmbeddingService,
        producer: AIOKafkaProducer,
    ) -> None:
        self._config = config
        self._es_client = es_client
        self._extractor = extractor
        self._graph_store = graph_store
        self._entity_embedding_service = entity_embedding_service
        self._producer = producer

    async def run_forever(self) -> None:
        await self._graph_store.ensure_schema()
        consumer = await self._start_consumer_with_retry()
        logger.info(
            "Listening topic=%s group=%s",
            self._config.kafka.graph_topic,
            self._config.kafka.consumer_group,
        )
        try:
            async for msg in consumer:
                await self._process_record(msg.value)
        finally:
            await consumer.stop()

    async def _start_consumer_with_retry(self) -> AIOKafkaConsumer:
        for attempt in range(1, _KAFKA_START_MAX_ATTEMPTS + 1):
            consumer = AIOKafkaConsumer(
                self._config.kafka.graph_topic,
                bootstrap_servers=self._config.kafka.bootstrap_servers,
                group_id=self._config.kafka.consumer_group,
                enable_auto_commit=True,
                auto_offset_reset=self._config.kafka.auto_offset_reset,
                value_deserializer=self._deserialize_value,
            )
            try:
                await consumer.start()
                return consumer
            except (GroupCoordinatorNotAvailableError, NodeNotReadyError) as exc:
                logger.warning(
                    "Kafka not ready for consumer group (attempt %s/%s): %s",
                    attempt,
                    _KAFKA_START_MAX_ATTEMPTS,
                    exc,
                )
                try:
                    await consumer.stop()
                except Exception:
                    pass
                if attempt == _KAFKA_START_MAX_ATTEMPTS:
                    raise
                await asyncio.sleep(_KAFKA_START_RETRY_SEC)
            except BaseException:
                try:
                    await consumer.stop()
                except Exception:
                    pass
                raise
        raise RuntimeError("Kafka consumer start failed after retries")

    @staticmethod
    def _deserialize_value(raw: bytes | None) -> Any:
        if raw is None:
            return None
        return json.loads(raw.decode("utf-8"))

    async def _process_record(self, payload: Any) -> None:
        if not isinstance(payload, dict):
            logger.error("Graph message must be JSON object, got %s", type(payload).__name__)
            await self._publish_dlq(
                GraphDlqMessage(
                    reason=_DLQ_INVALID_PAYLOAD_TYPE,
                    detail=f"expected JSON object, got {type(payload).__name__}",
                )
            )
            return
        try:
            message = GraphMessage.model_validate(payload)
        except Exception as exc:
            logger.exception("Invalid graph message payload")
            await self._publish_dlq(
                GraphDlqMessage.from_payload_dict(
                    payload,
                    reason=_DLQ_VALIDATION_FAILED,
                    detail=str(exc),
                )
            )
            return

        text = await fetch_chunk_text(
            self._es_client,
            index_name=self._config.elasticsearch.index_name,
            es_doc_id=message.es_doc_id,
        )
        print("\nText: ", text)
        if text is None:
            await self._publish_dlq(
                GraphDlqMessage.from_graph_message(
                    message,
                    reason=_DLQ_CHUNK_TEXT_MISSING,
                    detail="chunk text missing or empty in Elasticsearch",
                )
            )
            return

        try:
            extraction = await self._extractor.extract(text)
            print("\nExtraction: ", extraction)
        except Exception as exc:
            logger.exception("Entity extraction failed chunk_id=%s", message.chunk_id)
            await self._publish_dlq(
                GraphDlqMessage.from_graph_message(
                    message,
                    reason=_DLQ_EXTRACTION_FAILED,
                    detail=str(exc),
                )
            )
            return

        try:
            entity_ids = await self._graph_store.apply_extraction(
                extraction,
                message=message,
            )
        except Exception as exc:
            logger.exception("Neo4j write failed chunk_id=%s", message.chunk_id)
            await self._publish_dlq(
                GraphDlqMessage.from_graph_message(
                    message,
                    reason=_DLQ_GRAPH_WRITE_FAILED,
                    detail=str(exc),
                )
            )
            return

        if not entity_ids:
            await self._publish_dlq(
                GraphDlqMessage.from_graph_message(
                    message,
                    reason=_DLQ_NO_ENTITIES_FOUND,
                    detail="model returned no extractable entities for this chunk",
                )
            )
            return

        try:
            await self._entity_embedding_service.upsert_entities(
                self._build_entities_for_embedding(extraction, entity_ids)
            )
        except Exception as exc:
            logger.exception("Entity embedding write failed chunk_id=%s", message.chunk_id)
            await self._publish_dlq(
                GraphDlqMessage.from_graph_message(
                    message,
                    reason=_DLQ_GRAPH_WRITE_FAILED,
                    detail=f"entity embedding write failed: {exc}",
                )
            )
            return

        logger.info(
            "Graph updated chunk_id=%s entities=%s relations=%s",
            message.chunk_id,
            len(extraction.entities),
            len(extraction.relations),
        )
        await self._publish_chunk_entities_event(message, entity_ids)

    @staticmethod
    def _build_entities_for_embedding(
        extraction: EntityExtractionResult,
        entity_ids: list[str],
    ) -> list[GraphEntityNode]:
        """Собирает сущности текущего чанка для online-векторизации."""
        names_by_id: dict[str, str] = {}
        for ent in extraction.entities:
            key = ent.name.strip()
            if not key:
                continue
            # entity_ids формируются в том же порядке, что и уникальные сущности после нормализации.
            # Здесь оставляем display-name из extraction, чтобы embedding строился по человекочитаемому имени.
            entity_id = make_entity_id(key)
            names_by_id.setdefault(entity_id, key)

        return [
            GraphEntityNode(entity_id=entity_id, entity_name=names_by_id[entity_id])
            for entity_id in entity_ids
            if entity_id in names_by_id
        ]

    async def _publish_dlq(self, body: GraphDlqMessage) -> None:
        """Публикует запись в DLQ; при сбое отправки только логирует."""
        trace_id = str(uuid.uuid4())
        value = body.model_dump(mode="json", exclude_none=True)
        kafka_headers = [("trace_id", trace_id.encode("utf-8"))]
        try:
            await self._producer.send_and_wait(
                self._config.kafka.graph_dlq_topic,
                value,
                headers=kafka_headers,
            )
            logger.info(
                "Sent to DLQ topic=%s reason=%s chunk_id=%s",
                self._config.kafka.graph_dlq_topic,
                body.reason,
                body.chunk_id,
            )
        except Exception:
            logger.exception(
                "Failed to publish DLQ message topic=%s reason=%s",
                self._config.kafka.graph_dlq_topic,
                body.reason,
            )

    async def _publish_chunk_entities_event(
        self,
        message: GraphMessage,
        entity_ids: list[str],
    ) -> None:
        """Публикует одно событие в выходной топик со списком entity_id по чанку."""
        trace_id = str(uuid.uuid4())
        value = GraphEntityOutputMessage(
            doc_id=message.doc_id,
            chunk_id=message.chunk_id,
            version_id=message.version_id,
            es_doc_id=message.es_doc_id,
            embedding_id=message.embedding_id,
            entity_ids=entity_ids,
        ).model_dump(mode="json")
        kafka_headers = [("trace_id", trace_id.encode("utf-8"))]
        await self._producer.send_and_wait(
            self._config.kafka.graph_output_topic,
            value,
            headers=kafka_headers,
        )
