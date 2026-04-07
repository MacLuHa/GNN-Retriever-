from __future__ import annotations

import uvicorn

from .app import app
from .config import load_config


def main() -> None:
    """Запускает retriever HTTP API."""
    config = load_config()
    uvicorn.run(
        app,
        host=config.api.host,
        port=config.api.port,
        log_level=config.api.log_level.lower(),
    )


if __name__ == "__main__":
    main()

