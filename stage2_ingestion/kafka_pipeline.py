from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any

from aiokafka import AIOKafkaProducer
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from .pipeline import process_document_message

logger = logging.getLogger("stage2_ingestion.kafka")


class KafkaSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    bootstrap_servers: str = Field(default="localhost:9092", validation_alias="KAFKA_BOOTSTRAP_SERVERS")
    chunks_topic: str = Field(default="documents.chunks", validation_alias="KAFKA_CHUNKS_TOPIC")


class ChunkingSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    chunk_size: int = Field(default=1024, validation_alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, validation_alias="CHUNK_OVERLAP")


class RuntimeSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")


class HfSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    hf_dataset_name: str | None = Field(default="wikimedia/wikipedia", validation_alias="HF_DATASET_NAME")
    hf_dataset_config_name: str | None = Field(default="20231101.en", validation_alias="HF_DATASET_CONFIG_NAME")
    hf_dataset_split: str = Field(default="train[:1%]", validation_alias="HF_DATASET_SPLIT")
    hf_dataset_streaming: bool = Field(default=False, validation_alias="HF_DATASET_STREAMING")
    hf_limit: int | None = Field(default=None, validation_alias="HF_LIMIT")
    hf_doc_id_field: str = Field(default="id", validation_alias="HF_DOC_ID_FIELD")
    hf_version_id_field: str = Field(default="version_id", validation_alias="HF_VERSION_ID_FIELD")
    hf_title_field: str = Field(default="title", validation_alias="HF_TITLE_FIELD")
    hf_source_type_field: str = Field(default="source_type", validation_alias="HF_SOURCE_TYPE_FIELD")
    hf_content_base64_field: str = Field(default="content_base64", validation_alias="HF_CONTENT_BASE64_FIELD")
    hf_text_field: str = Field(default="text", validation_alias="HF_TEXT_FIELD")
    hf_metadata_field: str = Field(default="metadata", validation_alias="HF_METADATA_FIELD")


@dataclass(frozen=True)
class Settings:
    kafka: KafkaSettings
    chunking: ChunkingSettings
    runtime: RuntimeSettings
    hf: HfSettings


def load_settings() -> Settings:
    return Settings(
        kafka=KafkaSettings(),
        chunking=ChunkingSettings(),
        runtime=RuntimeSettings(),
        hf=HfSettings(),
    )


def _normalize_metadata(metadata_raw: Any) -> dict[str, Any]:
    if isinstance(metadata_raw, dict):
        return metadata_raw
    if metadata_raw is None:
        return {}
    if isinstance(metadata_raw, str):
        try:
            parsed = json.loads(metadata_raw)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
        return {"raw_metadata": metadata_raw}
    return {"raw_metadata": metadata_raw}


def _build_payload_from_hf_record(record: dict[str, Any], index: int, cfg: Settings) -> dict[str, Any]:
    content_base64 = record.get(cfg.hf.hf_content_base64_field)
    text = record.get(cfg.hf.hf_text_field)
    source_type = record.get(cfg.hf.hf_source_type_field) or "text"

    if content_base64 is None and text is None:
        raise ValueError(
            f"HF record #{index} has neither '{cfg.hf.hf_content_base64_field}' nor '{cfg.hf.hf_text_field}'"
        )

    if text is not None and not isinstance(text, str):
        text = json.dumps(text, ensure_ascii=False)

    doc_id = str(record.get(cfg.hf.hf_doc_id_field) or record.get("id") or f"hf-{index}")
    version_id = str(record.get(cfg.hf.hf_version_id_field) or "v1")
    title = str(record.get(cfg.hf.hf_title_field) or record.get("title") or f"{doc_id}.txt")
    metadata = _normalize_metadata(record.get(cfg.hf.hf_metadata_field))
    if not metadata:
        fallback_meta = {}
        if record.get("url") is not None:
            fallback_meta["url"] = str(record.get("url"))
        if record.get("title") is not None:
            fallback_meta["title"] = str(record.get("title"))
        metadata = fallback_meta

    return {
        "doc_id": doc_id,
        "version_id": version_id,
        "title": title,
        "source_type": str(source_type),
        "content_base64": str(content_base64) if content_base64 is not None else None,
        "text": text,
        "metadata": metadata,
    }


async def _process_and_send(payload: dict[str, Any], producer: AIOKafkaProducer, cfg: Settings) -> int:
    chunks = process_document_message(
        payload,
        chunk_size=cfg.chunking.chunk_size,
        chunk_overlap=cfg.chunking.chunk_overlap,
    )
    for chunk in chunks:
        await producer.send_and_wait(cfg.kafka.chunks_topic, chunk.model_dump(mode="json"))

    logger.info(
        "Processed doc_id=%s version_id=%s -> chunks=%d",
        payload.get("doc_id"),
        payload.get("version_id"),
        len(chunks),
    )
    return len(chunks)


async def _run_hf_dataset(producer: AIOKafkaProducer, cfg: Settings) -> None:
    if not cfg.hf.hf_dataset_name:
        raise ValueError("HF_DATASET_NAME must be set")

    from datasets import load_dataset

    logger.info(
        "Hugging Face dataset mode enabled (dataset=%s, config=%s, split=%s)",
        cfg.hf.hf_dataset_name,
        cfg.hf.hf_dataset_config_name,
        cfg.hf.hf_dataset_split,
    )
    dataset = load_dataset(
        cfg.hf.hf_dataset_name,
        name=cfg.hf.hf_dataset_config_name,
        split=cfg.hf.hf_dataset_split,
        streaming=cfg.hf.hf_dataset_streaming,
    )

    processed = 0
    total_chunks = 0
    for idx, row in enumerate(dataset):
        if cfg.hf.hf_limit is not None and processed >= cfg.hf.hf_limit:
            break

        payload = _build_payload_from_hf_record(row, idx, cfg)
        chunks_count = await _process_and_send(payload, producer, cfg)
        processed += 1
        total_chunks += chunks_count

    logger.info("HF ingestion finished: documents=%d chunks=%d", processed, total_chunks)


async def run(cfg: Settings) -> None:
    producer = AIOKafkaProducer(
        bootstrap_servers=cfg.kafka.bootstrap_servers,
        value_serializer=lambda v: json.dumps(v, ensure_ascii=False).encode("utf-8"),
    )

    await producer.start()
    logger.info("Stage 2 ingestion started (output topic=%s)", cfg.kafka.chunks_topic)
    try:
        await _run_hf_dataset(producer, cfg)
    finally:
        await producer.stop()


def main() -> None:
    cfg = load_settings()
    logging.basicConfig(
        level=getattr(logging, cfg.runtime.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    asyncio.run(run(cfg))


if __name__ == "__main__":
    main()
