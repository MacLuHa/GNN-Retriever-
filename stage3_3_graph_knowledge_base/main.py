from __future__ import annotations

import asyncio
import logging

from .app import AppContext, build_app, shutdown_app
from .config import load_config
from .entity_ids import set_entity_lemma_lang, set_entity_lemma_scope


async def _async_main() -> None:
    cfg = load_config()
    set_entity_lemma_lang(cfg.entity_normalization.lemma_lang)
    set_entity_lemma_scope(cfg.entity_normalization.lemma_scope)
    ctx: AppContext = await build_app(cfg)
    try:
        await ctx.service.run_forever()
    finally:
        await shutdown_app(ctx)


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
