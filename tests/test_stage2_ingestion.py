from __future__ import annotations

import uuid

from stage2_ingestion.chunking import chunk_text
from stage2_ingestion.kafka_pipeline import (
    ChunkingSettings,
    HfSettings,
    KafkaSettings,
    RuntimeSettings,
    Settings,
    _build_payload_from_hf_record,
    _normalize_metadata,
)
from stage2_ingestion.pipeline import process_document_message


def test_chunk_text_with_overlap() -> None:
    source = "a" * 20
    chunks = chunk_text(source, chunk_size=8, chunk_overlap=0)

    assert chunks == [
        (0, 8, "aaaaaaaa"),
        (8, 16, "aaaaaaaa"),
        (16, 20, "aaaa"),
    ]


def test_process_document_message_keeps_provenance() -> None:
    payload = {
        "doc_id": "doc-1",
        "version_id": "v1",
        "title": "example-title",
        "source_type": "text",
        "text": "Параграф 1. Параграф 2.",
        "metadata": {"language": "ru"},
    }

    result = process_document_message(payload, chunk_size=12, chunk_overlap=0)

    assert len(result) == 2
    first = result[0]
    assert first.doc_id == "doc-1"
    assert first.version_id == "v1"
    assert first.title == "example-title"
    assert uuid.UUID(first.chunk_id).version == 7
    assert first.page == 1
    assert first.span_start == 0
    assert first.metadata == {"language": "ru"}


def test_build_payload_from_hf_record_with_text() -> None:
    cfg = Settings(
        kafka=KafkaSettings(bootstrap_servers="localhost:9092", chunks_topic="out"),
        chunking=ChunkingSettings(chunk_size=1024, chunk_overlap=200),
        runtime=RuntimeSettings(log_level="INFO"),
        hf=HfSettings(
            hf_dataset_name="ag_news",
            hf_dataset_split="train",
            hf_dataset_streaming=False,
            hf_limit=1,
            hf_doc_id_field="id",
            hf_version_id_field="version",
            hf_title_field="title",
            hf_source_type_field="src",
            hf_content_base64_field="b64",
            hf_text_field="text",
            hf_metadata_field="meta",
        ),
    )

    row = {
        "id": "42",
        "version": "v7",
        "title": "sample-title",
        "url": "https://example.org/doc/42",
        "src": "text",
        "text": "hello",
        "meta": "{\"a\": 1}",
    }
    payload = _build_payload_from_hf_record(row, 0, cfg)

    assert payload["doc_id"] == "42"
    assert payload["version_id"] == "v7"
    assert payload["title"] == "sample-title"
    assert payload["source_type"] == "text"
    assert payload["text"] == "hello"
    assert payload["content_base64"] is None
    assert payload["metadata"] == {"a": 1}


def test_build_payload_from_hf_record_adds_wikipedia_fallback_metadata() -> None:
    cfg = Settings(
        kafka=KafkaSettings(bootstrap_servers="localhost:9092", chunks_topic="out"),
        chunking=ChunkingSettings(chunk_size=1024, chunk_overlap=200),
        runtime=RuntimeSettings(log_level="INFO"),
        hf=HfSettings(hf_metadata_field="missing_meta", hf_text_field="text"),
    )
    row = {"id": "12", "title": "Anarchism", "url": "https://en.wikipedia.org/wiki/Anarchism", "text": "abc"}
    payload = _build_payload_from_hf_record(row, 0, cfg)
    assert payload["metadata"] == {"url": "https://en.wikipedia.org/wiki/Anarchism", "title": "Anarchism"}


def test_normalize_metadata() -> None:
    assert _normalize_metadata(None) == {}
    assert _normalize_metadata({"k": "v"}) == {"k": "v"}
    assert _normalize_metadata('{"k":"v"}') == {"k": "v"}
    assert _normalize_metadata("raw") == {"raw_metadata": "raw"}
