"""
title: Retriever Pipeline
author: local
version: 0.1.0
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import json
import urllib.error
import urllib.request
from typing import Any

logger = logging.getLogger("openwebui.retriever_pipeline")


class Pipeline:
    """Pipeline Open WebUI с HTTP retrieval."""

    def __init__(self) -> None:
        self.type = "filter"
        self.name = "retriever_pipeline"
        self.pipelines = ["*"]
        self.priority = 10
        self.retriever_api_url = os.getenv("RETRIEVER_API_URL", "http://retriever-api:8010")
        self.retriever_timeout_sec = float(os.getenv("RETRIEVER_TIMEOUT_SEC", "10"))
        self.retriever_top_k = int(os.getenv("RETRIEVER_TOP_K", "5"))
        self.context_max_chars = int(os.getenv("RETRIEVER_MAX_CONTEXT_CHARS", "1200"))
        self.context_max_chunks = int(os.getenv("RETRIEVER_MAX_CONTEXT_CHUNKS", "5"))

    async def inlet(self, body: dict[str, Any], user: dict[str, Any] | None = None) -> dict[str, Any]:
        """Добавляет retrieval context в system message."""
        query = _extract_user_query(body)
        if not query:
            logger.warning("No user query found for retrieval")
            return body

        start = time.perf_counter()
        context = await asyncio.to_thread(self._fetch_context, query)
        latency_ms = int((time.perf_counter() - start) * 1000)
        logger.info("Retriever pipeline latency_ms=%s query_len=%s", latency_ms, len(query))

        if not context:
            # Fallback: keep prompt untouched when retriever is unavailable or empty.
            return body

        system_prefix = (
            "Use the retrieved context first. "
            "If context is insufficient, explicitly say so.\n\n"
            f"{context}"
        )
        messages = body.get("messages", [])
        if messages and messages[0].get("role") == "system":
            messages[0]["content"] = f"{system_prefix}\n\n{messages[0].get('content', '')}"
        else:
            messages.insert(0, {"role": "system", "content": system_prefix})
        body["messages"] = messages
        return body

    def _fetch_context(self, query: str) -> str:
        payload = {"query": query, "top_k": self.retriever_top_k, "filters": {}}
        try:
            request = urllib.request.Request(
                f"{self.retriever_api_url}/retrieve",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=self.retriever_timeout_sec) as response:
                data = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError) as exc:
            logger.warning("Retriever API call failed: %s", exc)
            return ""

        results = data.get("results", [])
        if not results:
            logger.info("Retriever returned empty context")
            return ""

        lines: list[str] = []
        for index, item in enumerate(results[: self.context_max_chunks], start=1):
            text = str(item.get("text", "")).strip()
            if len(text) > self.context_max_chars:
                text = text[: self.context_max_chars]
            score = item.get("score", 0.0)
            chunk_id = item.get("chunk_id", "")
            lines.append(f"Context #{index} (chunk_id={chunk_id}, score={score:.4f})\n{text}")
        return "\n\n".join(lines)


def _extract_user_query(body: dict[str, Any]) -> str:
    messages = body.get("messages")
    if not isinstance(messages, list):
        return ""
    for message in reversed(messages):
        if message.get("role") == "user":
            return str(message.get("content") or "").strip()
    return ""

