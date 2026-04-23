from __future__ import annotations

from pydantic import BaseModel, Field

from agent_core.models import HealthStatus, SessionRecord


class WorkspaceRequest(BaseModel):
    workspace: str = Field(..., description="Absolute or relative path to a workspace directory.")


class AskRequest(WorkspaceRequest):
    question: str = Field(..., min_length=1)


class SuggestRequest(WorkspaceRequest):
    request: str = Field(..., min_length=1)


class PatchPreviewRequest(WorkspaceRequest):
    request: str = Field(..., min_length=1)


class SessionResponse(BaseModel):
    ok: bool = True
    session: SessionRecord


class HealthResponse(BaseModel):
    ok: bool
    health: HealthStatus
