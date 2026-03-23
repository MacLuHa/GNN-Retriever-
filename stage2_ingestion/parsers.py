from __future__ import annotations

from io import BytesIO

class UnsupportedDocumentTypeError(ValueError):
    pass


def parse_pdf(raw_bytes: bytes) -> list[tuple[int, str]]:
    from pypdf import PdfReader

    reader = PdfReader(BytesIO(raw_bytes))
    pages: list[tuple[int, str]] = []
    for idx, page in enumerate(reader.pages, start=1):
        pages.append((idx, page.extract_text() or ""))
    return pages


def parse_docx(raw_bytes: bytes) -> list[tuple[int, str]]:
    from docx import Document as DocxDocument

    doc = DocxDocument(BytesIO(raw_bytes))
    text = "\n".join(p.text for p in doc.paragraphs if p.text)
    return [(1, text)]


def parse_html(raw_bytes: bytes) -> list[tuple[int, str]]:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(raw_bytes, "lxml")
    text = soup.get_text(separator="\n")
    return [(1, text)]


def parse_text(raw_bytes: bytes) -> list[tuple[int, str]]:
    return [(1, raw_bytes.decode("utf-8", errors="replace"))]


def parse_document(raw_bytes: bytes, source_type: str) -> list[tuple[int, str]]:
    source = source_type.lower()
    if source == "pdf":
        return parse_pdf(raw_bytes)
    if source == "docx":
        return parse_docx(raw_bytes)
    if source == "html":
        return parse_html(raw_bytes)
    if source == "text":
        return parse_text(raw_bytes)

    raise UnsupportedDocumentTypeError(
        f"Unsupported source_type='{source_type}'. Expected one of: pdf, docx, html, text"
    )
