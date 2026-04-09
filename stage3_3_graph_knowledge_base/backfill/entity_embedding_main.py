from __future__ import annotations

import asyncio
import logging

from ..config import load_config
from ..entity_ids import set_entity_lemma_lang, set_entity_lemma_scope
from .entity_embedding_app import (
    EntityEmbeddingAppContext,
    build_entity_embedding_app,
    shutdown_entity_embedding_app,
)


async def _async_main() -> None:
    cfg = load_config()
    set_entity_lemma_lang(cfg.entity_normalization.lemma_lang)
    set_entity_lemma_scope(cfg.entity_normalization.lemma_scope)
    ctx: EntityEmbeddingAppContext = await build_entity_embedding_app(cfg)
    try:
        total, updated = await ctx.backfill_service.run()
        logging.getLogger("stage3_3_graph_knowledge_base").info(
            "Entity embedding backfill done total=%s updated=%s collection=%s",
            total,
            updated,
            cfg.qdrant.collection_name,
        )
    finally:
        await shutdown_entity_embedding_app(ctx)


def main() -> None:
    cfg = load_config()
    set_entity_lemma_lang(cfg.entity_normalization.lemma_lang)
    set_entity_lemma_scope(cfg.entity_normalization.lemma_scope)
    logging.basicConfig(
        level=getattr(logging, cfg.runtime.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    asyncio.run(_async_main())


if __name__ == "__main__":
    main()
