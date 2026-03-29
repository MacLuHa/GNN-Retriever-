from __future__ import annotations

import re
import unicodedata

WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Normalize input text for stable chunking and metadata offsets."""
    normalized = unicodedata.normalize("NFKC", text)
    normalized = WHITESPACE_RE.sub(" ", normalized).strip()
    return normalized
