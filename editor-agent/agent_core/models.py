from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class AgentMode(str, Enum):
    ANALYZE = "analyze"
    ASK = "ask"
    SUGGEST = "suggest"
    PATCH_PREVIEW = "patch_preview"
    APPLY = "apply"


class FileScanResult(BaseModel):
    path: str
    relative_path: str
    extension: str
    size_bytes: int


class FileContext(BaseModel):
    path: str
    relative_path: str
    content: str
    size_bytes: int
    score: float = 0.0
    reason: str = ""


class RankedFile(BaseModel):
    path: str
    relative_path: str
    extension: str
    size_bytes: int
    score: float
    reason: str


class WorkspaceSummary(BaseModel):
    root_path: str
    total_files_scanned: int
    included_files: int
    skipped_files: int
    top_extensions: dict[str, int]
    sampled_files: list[str]
    notes: list[str] = Field(default_factory=list)


class ParsedAnswer(BaseModel):
    summary: str
    findings: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)
    commands_to_consider: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    affected_files: list[str] = Field(default_factory=list)
    proposed_changes: list[str] = Field(default_factory=list)
    next_steps: list[str] = Field(default_factory=list)
    file_operations: list["ProposedFileOperation"] = Field(default_factory=list)
    raw_text: str = ""
    parse_strategy: str = "text"


class ProposedFileOperation(BaseModel):
    path: str
    action: str
    content: str


class PatchProposal(BaseModel):
    session_id: str
    created_at: datetime
    workspace_path: str
    request: str | None = None
    summary: str
    risks: list[str] = Field(default_factory=list)
    affected_files: list[str] = Field(default_factory=list)
    proposed_changes: list[str] = Field(default_factory=list)
    next_steps: list[str] = Field(default_factory=list)
    file_operations: list[ProposedFileOperation] = Field(default_factory=list)


class AppliedFileRecord(BaseModel):
    path: str
    action: str
    backup_path: str
    existed_before: bool


class ApplyLogRecord(BaseModel):
    session_id: str
    created_at: datetime
    workspace_path: str
    confirmed: bool
    success: bool
    rollback_performed: bool = False
    request: str | None = None
    files_written: list[AppliedFileRecord] = Field(default_factory=list)
    error: str | None = None


class SessionRecord(BaseModel):
    session_id: str
    created_at: datetime
    mode: AgentMode
    workspace_path: str
    prompt: str
    user_input: str | None = None
    workspace_summary: WorkspaceSummary
    ranked_files: list[RankedFile]
    selected_context: list[FileContext]
    raw_response: str
    parsed_response: ParsedAnswer
    patch_proposal_path: str | None = None
    apply_log_path: str | None = None
    confirmed: bool = False


class HealthStatus(BaseModel):
    ok: bool
    base_url: str
    model: str
    api_key_configured: bool
    session_dir: str
    timeout_seconds: float


class PromptBundle(BaseModel):
    system_prompt: str
    user_prompt: str


class ChatMessage(BaseModel):
    role: str
    content: str


class ModelResponse(BaseModel):
    content: str
    usage: dict[str, Any] = Field(default_factory=dict)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def build_session_id(mode: AgentMode) -> str:
    timestamp = utc_now().strftime("%Y%m%d-%H%M%S")
    return f"{mode.value}-{timestamp}"


def normalize_path(path: Path) -> str:
    return str(path.resolve())
