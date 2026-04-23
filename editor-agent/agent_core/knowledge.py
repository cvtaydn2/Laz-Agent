from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from pydantic import BaseModel, Field, ValidationError
from datetime import datetime

logger = logging.getLogger(__name__)


class KnowledgeEntry(BaseModel):
    pattern: str
    solution_summary: str
    success_count: int = 1
    last_applied: datetime = Field(default_factory=datetime.now)
    files_involved: List[str] = Field(default_factory=list)


class KnowledgeBase(BaseModel):
    entries: Dict[str, KnowledgeEntry] = Field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> "KnowledgeBase":
        """Synchronous load. For use in non-async contexts only.

        In async contexts prefer ``async_load()`` to avoid blocking the event loop.
        """
        if not path.exists():
            return cls()
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return cls.model_validate(data)
        except (FileNotFoundError, json.JSONDecodeError, ValidationError) as exc:
            logger.warning("Failed to load knowledge base from %s: %s", path, exc)
            return cls()

    @classmethod
    async def async_load(cls, path: Path) -> "KnowledgeBase":
        """Async load — runs file I/O in a thread so the event loop is never blocked.

        Returns an empty KnowledgeBase on any error (missing file, invalid JSON,
        validation failure, permission error) and logs a WARNING.
        """
        try:
            data = await asyncio.to_thread(_read_json_file, path)
            return cls.model_validate(data)
        except (FileNotFoundError, json.JSONDecodeError, ValidationError, OSError, ValueError) as exc:
            logger.warning("Failed to async_load knowledge base from %s: %s", path, exc)
            return cls()

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.model_dump(mode="json"), f, indent=2)

    def learn(self, error_pattern: str, solution: str, files: List[str]):
        if error_pattern in self.entries:
            entry = self.entries[error_pattern]
            entry.success_count += 1
            entry.last_applied = datetime.now()
            entry.solution_summary = solution
            # Merge files
            entry.files_involved = list(set(entry.files_involved + files))
        else:
            self.entries[error_pattern] = KnowledgeEntry(
                pattern=error_pattern,
                solution_summary=solution,
                files_involved=files
            )

    def query(self, query_text: str) -> List[KnowledgeEntry]:
        # Simple keyword matching for now
        results = []
        q = query_text.lower()
        for entry in self.entries.values():
            if q in entry.pattern.lower() or q in entry.solution_summary.lower():
                results.append(entry)
        return sorted(results, key=lambda x: x.success_count, reverse=True)


def _read_json_file(path: Path) -> dict:
    """Thread-safe synchronous helper used by async_load."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
