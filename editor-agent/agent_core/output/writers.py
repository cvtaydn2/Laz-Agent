from __future__ import annotations

import json
import asyncio
from pathlib import Path

import aiofiles

from agent_core.config import Settings
from agent_core.models import ApplyLogRecord, PatchProposal, SessionRecord


class SessionWriter:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def write(self, session: SessionRecord) -> Path:
        target = self.settings.session_dir / f"{session.session_id}.json"
        async with aiofiles.open(target, mode="w", encoding="utf-8") as f:
            await f.write(json.dumps(session.model_dump(mode="json"), indent=2, ensure_ascii=False))
        return target

    async def read(self, session_id: str) -> SessionRecord | None:
        target = self.settings.session_dir / f"{session_id}.json"
        if not target.exists():
            return None
        async with aiofiles.open(target, mode="r", encoding="utf-8") as f:
            data = json.loads(await f.read())
        return SessionRecord(**data)

    def prune_old_sessions(self, max_age_days: int = 7) -> int:
        import time
        count = 0
        now = time.time()
        max_age_seconds = max_age_days * 86400
        
        if not self.settings.session_dir.exists():
            return 0
            
        for file in self.settings.session_dir.glob("*.json"):
            if now - file.stat().st_mtime > max_age_seconds:
                try:
                    file.unlink()
                    count += 1
                except OSError:
                    continue
        return count


class PatchProposalWriter:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def write(self, proposal: PatchProposal) -> Path:
        target = self.settings.patches_dir / f"{proposal.session_id}.json"
        async with aiofiles.open(target, mode="w", encoding="utf-8") as f:
            await f.write(json.dumps(proposal.model_dump(mode="json"), indent=2, ensure_ascii=False))
        return target


class ApplyLogWriter:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def write(self, apply_log: ApplyLogRecord) -> Path:
        target = self.settings.logs_dir / f"{apply_log.session_id}.apply.json"
        async with aiofiles.open(target, mode="w", encoding="utf-8") as f:
            await f.write(json.dumps(apply_log.model_dump(mode="json"), indent=2, ensure_ascii=False))
        return target
