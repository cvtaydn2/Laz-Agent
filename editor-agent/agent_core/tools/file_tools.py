from __future__ import annotations

from pathlib import Path


def normalize_workspace_path(path: Path) -> Path:
    return path.expanduser().resolve()
