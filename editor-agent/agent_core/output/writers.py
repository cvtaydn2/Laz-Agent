from __future__ import annotations

import json
from pathlib import Path

from agent_core.config import Settings
from agent_core.models import ApplyLogRecord, PatchProposal, SessionRecord


class SessionWriter:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def write(self, session: SessionRecord) -> Path:
        target = self.settings.session_dir / f"{session.session_id}.json"
        target.write_text(
            json.dumps(session.model_dump(mode="json"), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return target


class PatchProposalWriter:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def write(self, proposal: PatchProposal) -> Path:
        target = self.settings.patches_dir / f"{proposal.session_id}.json"
        target.write_text(
            json.dumps(proposal.model_dump(mode="json"), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return target


class ApplyLogWriter:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def write(self, apply_log: ApplyLogRecord) -> Path:
        target = self.settings.logs_dir / f"{apply_log.session_id}.apply.json"
        target.write_text(
            json.dumps(apply_log.model_dump(mode="json"), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return target
