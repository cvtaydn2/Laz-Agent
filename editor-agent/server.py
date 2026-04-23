from __future__ import annotations

import uvicorn

from agent_core.config import Settings


def main() -> None:
    settings = Settings.load()
    uvicorn.run(
        "agent_core.server.api:app",
        host=settings.server_host,
        port=settings.server_port,
        reload=False,
    )


if __name__ == "__main__":
    main()
