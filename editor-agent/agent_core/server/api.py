from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse

from agent_core.config import ensure_environment_ready
from agent_core.models import AgentMode
from agent_core.nvidia_client import NvidiaInferenceError
from agent_core.server.openai_adapter import (
    build_openai_error,
    build_openai_fallback_response,
    build_openai_models_response,
    build_openai_response,
    build_openai_streaming_chunks,
    extract_changed_files,
    extract_diff,
    extract_request_mode,
    extract_request_workspace,
    session_to_plain_text,
    extract_user_message,
    validate_openai_mode,
)
from agent_core.server.openai_schemas import OpenAIChatCompletionRequest, OpenAIModelsResponse
from agent_core.server.schemas import (
    AskRequest,
    HealthResponse,
    PatchPreviewRequest,
    SessionResponse,
    SuggestRequest,
    WorkspaceRequest,
)
from agent_core.server.service import build_health_status, run_agent


@asynccontextmanager
async def lifespan(_: FastAPI):
    ensure_environment_ready()
    yield


app = FastAPI(title="Editor Agent API", version="0.1.0", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    if request.url.path.startswith("/v1/"):
        return JSONResponse(
            status_code=422,
            content=build_openai_error(
                message="Invalid request body.",
                error_type="invalid_request_error",
                code="validation_error",
            ),
        )
    return JSONResponse(status_code=422, content={"detail": exc.errors()})


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    if request.url.path.startswith("/v1/"):
        message = exc.detail if isinstance(exc.detail, str) else "Request failed."
        return JSONResponse(
            status_code=exc.status_code,
            content=build_openai_error(
                message=message,
                error_type="invalid_request_error" if exc.status_code < 500 else "server_error",
            ),
        )
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


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


@app.get("/v1/models", response_model=OpenAIModelsResponse)
def list_models() -> OpenAIModelsResponse:
    return build_openai_models_response()


@app.post("/v1/chat/completions")
def chat_completions(request: OpenAIChatCompletionRequest):
    try:
        workspace = extract_request_workspace(request)
        if not workspace:
            raise HTTPException(
                status_code=400,
                detail="A workspace path is required under extra_body.workspace, metadata.workspace, or top-level workspace.",
            )

        mode_value = validate_openai_mode(extract_request_mode(request))
        user_message = extract_user_message(request.messages)
        if not user_message:
            raise HTTPException(
                status_code=400,
                detail="At least one non-empty user message is required.",
            )

        session = run_agent(
            AgentMode(mode_value),
            workspace,
            user_message,
            temperature_override=request.temperature,
            max_tokens_override=request.max_tokens,
            changed_files=extract_changed_files(request),
            diff_text=extract_diff(request),
        )
        if request.stream:
            content = session_to_plain_text(session)
            created = int(datetime.now(timezone.utc).timestamp())
            completion_id = f"chatcmpl-{session.session_id}"

            async def event_stream():
                for chunk in build_openai_streaming_chunks(
                    model=request.model,
                    content=content,
                    completion_id=completion_id,
                    created=created,
                ):
                    yield chunk

            return StreamingResponse(
                event_stream(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        return build_openai_response(session=session, requested_model=request.model)
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except NvidiaInferenceError as exc:
        if request.stream:
            created = int(datetime.now(timezone.utc).timestamp())
            async def error_event_stream():
                for chunk in build_openai_streaming_chunks(
                    model=request.model,
                    content=exc.user_message,
                    completion_id=f"chatcmpl-fallback-{created}",
                    created=created,
                ):
                    yield chunk

            return StreamingResponse(
                error_event_stream(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        return build_openai_fallback_response(
            model=request.model,
            content=exc.user_message,
        )
    except Exception:
        if request.stream:
            created = int(datetime.now(timezone.utc).timestamp())
            async def generic_error_stream():
                for chunk in build_openai_streaming_chunks(
                    model=request.model,
                    content="The local agent could not complete the request. Please retry with a smaller request or a narrower workspace context.",
                    completion_id=f"chatcmpl-fallback-{created}",
                    created=created,
                ):
                    yield chunk

            return StreamingResponse(
                generic_error_stream(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        return build_openai_fallback_response(
            model=request.model,
            content="The local agent could not complete the request. Please retry with a smaller request or a narrower workspace context.",
        )
