from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

import httpx

from .config import OllamaLlmConfig
from .models import EntityExtractionResult

logger = logging.getLogger("stage3_3_graph_knowledge_base")

_SYSTEM = (
    "You extract entities and directed relationships from the given text only. "
    "Output a single JSON object. No markdown, no commentary, no external knowledge."
)

_USER_TEMPLATE = """From the text below, extract (1) entities and (2) directed relationship pairs for a knowledge graph.
Use only information stated in the text; do not infer facts from general knowledge.

Return exactly one JSON object with this shape (keys and nesting must match):
{{"entities": [{{"name": "..."}}], "relations": [{{"from": "...", "to": "..."}}]}}

Entities:
- Include named entities, technical terms, organizations, laws, locations, and other concepts that matter for understanding the excerpt.
- Prefer specific, unambiguous names; avoid vague labels unless the text uses them.
- Cover useful levels of detail when the text supports both general and specific concepts; merge near-duplicates into one canonical name used consistently below.

Relations:
- Each relation is directed: "from" and "to" must be entity names that also appear in "entities" (same spelling).
- Add a pair when the text clearly links two entities (dependency, association, part-of, causal, or other stated link). Set direction so it matches how the text describes the link (source "from" toward target "to").
- Include as many distinct, text-supported relations as reasonable; use "relations": [] if none.

Text:
---
{text}
---
"""


def _parse_json_object(raw: str) -> dict[str, Any]:
    """Достаёт JSON из ответа модели (в т.ч. обёрнутого в ```json)."""
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if fence:
        text = fence.group(1).strip()
    return json.loads(text)


class OllamaEntityExtractor:
    """Извлечение сущностей и связей через Ollama /api/chat."""

    def __init__(self, config: OllamaLlmConfig) -> None:
        self._config = config

    async def extract(self, text: str) -> EntityExtractionResult:
        """Возвращает структурированное извлечение или пустой результат при ошибке."""
        if not text.strip():
            raise ValueError("Chunk text must not be empty")

        user_content = _USER_TEMPLATE.format(text=text)
        last_error: Exception | None = None
        for attempt in range(1, self._config.max_attempts + 1):
            try:
                timeout = httpx.Timeout(self._config.timeout_sec)
                async with httpx.AsyncClient(base_url=self._config.base_url, timeout=timeout) as client:
                    response = await client.post(
                        "/api/chat",
                        json={
                            "model": self._config.model_name,
                            "messages": [
                                {"role": "system", "content": _SYSTEM},
                                {"role": "user", "content": user_content},
                            ],
                            "format": "json",
                            "stream": False,
                        },
                    )
                    response.raise_for_status()
                    payload = response.json()
                    raw_content = (payload.get("message") or {}).get("content")
                    if not isinstance(raw_content, str):
                        raise ValueError("Ollama returned no message content")
                    data = _parse_json_object(raw_content)
                    return EntityExtractionResult.model_validate(data)
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "Entity extraction failed attempt=%s/%s: %s",
                    attempt,
                    self._config.max_attempts,
                    exc,
                )
                if attempt < self._config.max_attempts:
                    await asyncio.sleep(min(2**attempt, 5))

        raise RuntimeError("Failed to extract entities from Ollama") from last_error
