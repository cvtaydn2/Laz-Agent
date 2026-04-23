from __future__ import annotations

from pathlib import Path


def normalize_workspace_path(path: Path) -> Path:
    """Resolve and expand a workspace path to its absolute form."""
    return path.expanduser().resolve()


def resolve_workspace(workspace: str) -> Path:
    """
    Resolve a workspace string to an absolute Path.
    Raises ValueError if the path does not exist or is not a directory.
    """
    path = normalize_workspace_path(Path(workspace))
    if not path.exists():
        raise ValueError(f"Workspace does not exist: {path}")
    if not path.is_dir():
        raise ValueError(f"Workspace must be a directory: {path}")
    return path
