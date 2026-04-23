from __future__ import annotations

from fastapi import FastAPI

from agent_core.models import AgentMode
from agent_core.server.schemas import (
    AskRequest,
    HealthResponse,
    PatchPreviewRequest,
    SessionResponse,
    SuggestRequest,
    WorkspaceRequest,
)
from agent_core.server.service import build_health_status, run_agent

app = FastAPI(title="Editor Agent API", version="0.1.0")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    status = build_health_status()
    return HealthResponse(ok=status.ok, health=status)


@app.post("/analyze", response_model=SessionResponse)
def analyze(request: WorkspaceRequest) -> SessionResponse:
    session = run_agent(AgentMode.ANALYZE, request.workspace, None)
    return SessionResponse(session=session)


@app.post("/ask", response_model=SessionResponse)
def ask(request: AskRequest) -> SessionResponse:
    session = run_agent(AgentMode.ASK, request.workspace, request.question)
    return SessionResponse(session=session)


@app.post("/suggest", response_model=SessionResponse)
def suggest(request: SuggestRequest) -> SessionResponse:
    session = run_agent(AgentMode.SUGGEST, request.workspace, request.request)
    return SessionResponse(session=session)


@app.post("/patch-preview", response_model=SessionResponse)
def patch_preview(request: PatchPreviewRequest) -> SessionResponse:
    session = run_agent(AgentMode.PATCH_PREVIEW, request.workspace, request.request)
    return SessionResponse(session=session)
