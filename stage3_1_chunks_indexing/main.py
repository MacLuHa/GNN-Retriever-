from __future__ import annotations

import asyncio
import logging

from .app import AppContext, build_app, shutdown_app
from .config import load_config


async def _async_main() -> None:
    """Точка входа asyncio: поднимает контекст и запускает цикл обработки."""
    cfg = load_config()
    ctx: AppContext = await build_app(cfg)
    try:
        await ctx.chunk_index_service.run_forever()
    finally:
        await shutdown_app(ctx)


def main() -> None:
    """Точка входа CLI: логирование и asyncio.run."""
    cfg = load_config()
    logging.basicConfig(
        level=getattr(logging, cfg.runtime.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    asyncio.run(_async_main())


if __name__ == "__main__":
    main()
