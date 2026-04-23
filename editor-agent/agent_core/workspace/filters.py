from __future__ import annotations

from pathlib import Path

from agent_core.config import Settings


def is_ignored_directory(path: Path, settings: Settings) -> bool:
    return path.name.lower() in {item.lower() for item in settings.ignored_directories}


def is_allowed_file(path: Path, settings: Settings) -> bool:
    name = path.name.lower()
    if name == ".env.example":
        return True
    return path.suffix.lower() in settings.allowed_extensions


def is_probably_binary(raw_bytes: bytes) -> bool:
    if not raw_bytes:
        return False
    if b"\x00" in raw_bytes:
        return True
    sample = raw_bytes[:1024]
    non_text = sum(byte < 9 or (13 < byte < 32) for byte in sample)
    return non_text / max(1, len(sample)) > 0.3
