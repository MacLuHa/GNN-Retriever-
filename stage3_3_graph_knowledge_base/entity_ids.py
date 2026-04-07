"""Идентификаторы сущностей: нормализация имени и стабильный ``entity_id``.

Лемматизация (если включена) приводит слова к словарной форме — например,
``organizations`` → ``organization``, что уменьшает дубликаты узлов из-за числа/формы слова.

**Стемминг** (для сравнения): грубое отсечение суффиксов по правилам (Porter и т.д.),
часто даёт несуществующие основы. Лемматизация опирается на словарь и обычно даёт
норму слова, но зависит от языка и словаря.
"""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Final

from simplemma import lemmatize

logger = logging.getLogger("stage3_3_graph_knowledge_base")

_default_lemma_lang: str | None = "en"
# "single" — лемматизация только однословных имён (сохраняет «United States»); «all» — каждый токен.
_default_lemma_scope: str = "single"

_TOKEN_RE: Final[re.Pattern[str]] = re.compile(r"[\w'-]+", re.UNICODE)


def set_entity_lemma_lang(lang: str | None) -> None:
    """Задаёт язык лемматизации для ``normalize_entity_name`` (вызывать при старте из конфига)."""
    global _default_lemma_lang
    if lang is None or not str(lang).strip():
        _default_lemma_lang = None
    else:
        _default_lemma_lang = str(lang).strip().lower()


def set_entity_lemma_scope(scope: str | None) -> None:
    """``single`` | ``all`` — см. модульный docstring."""
    global _default_lemma_scope
    s = (scope or "single").strip().lower()
    _default_lemma_scope = s if s in ("single", "all") else "single"


def _lemmatize_token(token: str, lang: str) -> str:
    if not any(c.isalpha() for c in token):
        return token
    try:
        lemma = lemmatize(token, lang=lang)
    except Exception as exc:
        logger.debug("simplemma skip token=%r: %s", token, exc)
        return token
    return lemma if lemma else token


def _lemmatize_fragment(fragment: str, lang: str) -> str:
    """Лемматизирует буквенные токены; пунктуация между ними сохраняется."""
    parts: list[str] = []
    pos = 0
    for m in _TOKEN_RE.finditer(fragment):
        parts.append(fragment[pos : m.start()])
        parts.append(_lemmatize_token(m.group(0), lang))
        pos = m.end()
    parts.append(fragment[pos:])
    return "".join(parts)


def normalize_entity_name(
    name: str,
    *,
    lemma_lang: str | None = None,
    lemma_scope: str | None = None,
) -> str:
    """Нормализация имени для стабильного ``entity_id``: пробелы, lower, опционально лемматизация."""
    base = " ".join(name.strip().split()).lower()
    if not base:
        return base
    lang = lemma_lang if lemma_lang is not None else _default_lemma_lang
    if not lang:
        return base
    scope = (lemma_scope or _default_lemma_scope).strip().lower()
    if scope not in ("single", "all"):
        scope = "single"
    words = base.split()
    if scope == "single" and len(words) != 1:
        return base
    return " ".join(_lemmatize_fragment(fr, lang) for fr in words)


def make_entity_id(name: str) -> str:
    """Детерминированный entity_id из нормализованного имени."""
    key = normalize_entity_name(name)
    return hashlib.sha256(key.encode("utf-8")).hexdigest()
