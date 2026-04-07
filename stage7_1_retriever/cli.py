from __future__ import annotations

import argparse
import asyncio
import logging
from typing import Any

from .config import load_config
from .retriever import build_retriever_from_env


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hybrid BM25 + GNN retriever demo")
    parser.add_argument("--query", required=True, help="Query text")
    parser.add_argument("--top-k", type=int, default=None, help="Override top-k")
    parser.add_argument(
        "--filter",
        action="append",
        default=[],
        help="Filter in key=value format. Can be used multiple times.",
    )
    return parser


def _parse_filters(raw_filters: list[str]) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for item in raw_filters:
        if "=" not in item:
            continue
        key, raw_value = item.split("=", 1)
        key = key.strip()
        value = raw_value.strip()
        if not key:
            continue
        lowered = value.lower()
        if lowered in {"true", "false"}:
            parsed[key] = lowered == "true"
            continue
        try:
            parsed[key] = int(value)
            continue
        except ValueError:
            pass
        try:
            parsed[key] = float(value)
            continue
        except ValueError:
            pass
        parsed[key] = value
    return parsed


async def _run_cli() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    config = load_config()
    logging.basicConfig(
        level=config.runtime.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    retriever = build_retriever_from_env(config)
    filters = _parse_filters(args.filter)

    try:
        results = await retriever.retrieve(query_text=args.query, filters=filters, top_k=args.top_k)
    finally:
        await retriever.close()

    if not results:
        print("No results")
        return 0

    for index, chunk in enumerate(results, start=1):
        short_text = " ".join(chunk.text.strip().split())[:180]
        sources = ",".join(sorted(chunk.source_scores.keys()))
        print(
            f"{index}. chunk_id={chunk.chunk_id} "
            f"score={chunk.score:.4f} sources={sources} "
            f"text={short_text}"
        )
    return 0


def main() -> None:
    """Запускает CLI ретривера."""
    raise SystemExit(asyncio.run(_run_cli()))


if __name__ == "__main__":
    main()
