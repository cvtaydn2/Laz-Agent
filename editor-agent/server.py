from __future__ import annotations

import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure the .env next to this file is always loaded, regardless of the
# working directory the user started the server from.
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_DOTENV = _HERE / ".env"

from dotenv import load_dotenv
if _DOTENV.exists():
    load_dotenv(dotenv_path=_DOTENV, override=True)
    print(f"[server] Loaded .env from {_DOTENV}")
else:
    print(f"[server] WARNING: .env not found at {_DOTENV}", file=sys.stderr)

# Change cwd to the project root so find_dotenv() also works inside the app
os.chdir(_HERE)

import uvicorn
from agent_core.config import Settings


def main() -> None:
    # Clear the lru_cache so Settings.load() picks up the freshly loaded env
    Settings.load.cache_clear()
    settings = Settings.load()

    if not settings.nvidia_api_key:
        print("[server] WARNING: NVIDIA_API_KEY is not set. Requests will fail.", file=sys.stderr)
    else:
        print(f"[server] NVIDIA_API_KEY configured. Model: {settings.nvidia_model}")
        print(f"[server] Timeout: {settings.timeout_seconds}s  MaxTokens: {settings.max_completion_tokens}")

    uvicorn.run(
        "agent_core.server.api:app",
        host=settings.server_host,
        port=settings.server_port,
        reload=False,
    )


if __name__ == "__main__":
    main()
