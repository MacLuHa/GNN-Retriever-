from __future__ import annotations

import hashlib


def normalize_entity_name(name: str) -> str:
    """Нормализация имени для стабильного entity_id."""
    return " ".join(name.strip().split()).lower()


def make_entity_id(name: str) -> str:
    """Детерминированный entity_id из нормализованного имени."""
    key = normalize_entity_name(name)
    return hashlib.sha256(key.encode("utf-8")).hexdigest()
