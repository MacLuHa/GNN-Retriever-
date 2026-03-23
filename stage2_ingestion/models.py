from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator


class DocumentMessage(BaseModel):
    doc_id: str
    version_id: str
    title: str
    source_type: str = Field(description="pdf/docx/html/text")
    content_base64: str | None = None
    text: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _alias_filename_to_title(cls, data: Any) -> Any:
        if isinstance(data, dict) and "title" not in data and "filename" in data:
            data = dict(data)
            data["title"] = data["filename"]
        return data

    @model_validator(mode="after")
    def _validate_content(self) -> "DocumentMessage":
        if self.content_base64 is None and self.text is None:
            raise ValueError("One of 'content_base64' or 'text' must be provided")
        return self


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    version_id: str
    title: str
    page: int | None
    span_start: int
    span_end: int
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
