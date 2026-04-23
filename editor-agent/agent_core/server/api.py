from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse

from agent_core.config import ensure_environment_ready
from agent_core.models import AgentMode
from agent_core.llm.nvidia import NvidiaInferenceError
from agent_core.server.openai_adapter import (
    build_openai_error,
    build_openai_fallback_response,
    build_openai_models_response,
    build_openai_response,
    format_openai_stream_chunk,
    extract_changed_files,
    extract_diff,
    extract_request_mode,
    extract_request_workspace,
    session_to_plain_text,
    extract_user_message,
    validate_openai_mode,
    extract_preferred_files,
)
from agent_core.server.metrics import HTTP_REQUESTS_TOTAL
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from fastapi import Response
from agent_core.server.openai_schemas import OpenAIChatCompletionRequest, OpenAIModelsResponse
from agent_core.server.schemas import (
    AskRequest,
    HealthResponse,
    PatchPreviewRequest,
    SessionResponse,
    SuggestRequest,
    WorkspaceRequest,
)
from agent_core.server.service import build_health_status, run_agent, stream_agent
from agent_core.output.writers import SessionWriter
from agent_core.config import Settings


@asynccontextmanager
async def lifespan(_: FastAPI):
    ensure_environment_ready()
    # Prune old sessions on startup to prevent disk bloat
    settings = Settings.load()
    writer = SessionWriter(settings)
    pruned_count = writer.prune_old_sessions(max_age_days=7)
    if pruned_count > 0:
        print(f"INFO:     Pruned {pruned_count} old session(s).")
    yield


app = FastAPI(title="Editor Agent API", version="0.1.0", lifespan=lifespan)


@app.middleware("http")
async def track_metrics_middleware(request: Request, call_next):
    method = request.method
    endpoint = request.url.path
    response = await call_next(request)
    status = response.status_code
    
    # Exclude /metrics from tracking to avoid noise
    if endpoint != "/metrics":
        HTTP_REQUESTS_TOTAL.labels(method=method, endpoint=endpoint, status=status).inc()
        
    return response


@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


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
async def analyze(request: WorkspaceRequest) -> SessionResponse:
    session = await run_agent(AgentMode.ANALYZE, request.workspace, None)
    return SessionResponse(session=session)


@app.post("/ask", response_model=SessionResponse)
async def ask(request: AskRequest) -> SessionResponse:
    session = await run_agent(AgentMode.ASK, request.workspace, request.question)
    return SessionResponse(session=session)


@app.post("/suggest", response_model=SessionResponse)
async def suggest(request: SuggestRequest) -> SessionResponse:
    session = await run_agent(AgentMode.SUGGEST, request.workspace, request.request)
    return SessionResponse(session=session)


@app.post("/patch-preview", response_model=SessionResponse)
async def patch_preview(request: PatchPreviewRequest) -> SessionResponse:
    session = await run_agent(AgentMode.PATCH_PREVIEW, request.workspace, request.request)
    return SessionResponse(session=session)


@app.get("/v1/models", response_model=OpenAIModelsResponse)
def list_models() -> OpenAIModelsResponse:
    return build_openai_models_response()


@app.post("/v1/chat/completions")
async def chat_completions(request: OpenAIChatCompletionRequest, raw_request: Request):
    try:
        # Log raw body to see what Continue is actually sending
        try:
            body = await raw_request.json()
            print(f"DEBUG:    Raw Body Keys: {list(body.keys())}")
            if "extraBody" in body:
                 print(f"DEBUG:    Found extraBody: {list(body['extraBody'].keys())}")
            if "metadata" in body:
                 print(f"DEBUG:    Found metadata: {list(body['metadata'].keys()) if isinstance(body['metadata'], dict) else body['metadata']}")
        except:
            pass
        workspace = extract_request_workspace(request)
        print(f"DEBUG:    Workspace extracted: {workspace}")
        if not workspace:
            raise HTTPException(
                status_code=400,
                detail="A workspace path is required under extra_body.workspace, metadata.workspace, or top-level workspace.",
            )

        req_mode = extract_request_mode(request)
        print(f"DEBUG:    Raw mode from request: {req_mode}")
        mode_value = validate_openai_mode(req_mode)
        print(f"DEBUG:    Validated mode: {mode_value}")
        
        user_message = extract_user_message(request.messages)
        print(f"DEBUG:    Extracted user message (len={len(user_message) if user_message else 0}): {user_message[:50]}...")
        
        # Log extra_body and metadata for debugging
        print(f"DEBUG:    extra_body: {request.extra_body}")
        print(f"DEBUG:    metadata: {request.metadata}")
        print(f"DEBUG:    workspace_fallback: {Settings.load().default_workspace}")
        
        if not user_message:
            raise HTTPException(
                status_code=400,
                detail="At least one non-empty user message is required.",
            )

        print(f"DEBUG:    Calling run_agent...")
        session = await run_agent(
            AgentMode(mode_value),
            workspace,
            user_message,
            temperature_override=request.temperature,
            max_tokens_override=request.max_tokens,
            changed_files=extract_changed_files(request),
            diff_text=extract_diff(request),
            preferred_files=extract_preferred_files(request),
        )
        print(f"DEBUG:    run_agent returned successfully.")
        if request.stream:
            created = int(datetime.now(timezone.utc).timestamp())
            completion_id = f"chatcmpl-stream-{created}"

            async def event_stream():
                try:
                    async for text_chunk in stream_agent(
                        AgentMode(mode_value),
                        workspace,
                        user_message,
                        temperature_override=request.temperature,
                        max_tokens_override=request.max_tokens,
                        changed_files=extract_changed_files(request),
                        diff_text=extract_diff(request),
                        preferred_files=extract_preferred_files(request),
                    ):
                        yield format_openai_stream_chunk(
                            model=request.model,
                            content=text_chunk,
                            completion_id=completion_id,
                            created=created,
                        )
                    
                    yield format_openai_stream_chunk(
                        model=request.model,
                        content=None,
                        completion_id=completion_id,
                        created=created,
                        finish_reason="stop",
                    )
                    yield "data: [DONE]\n\n"
                except Exception as exc:
                    yield format_openai_stream_chunk(
                        model=request.model,
                        content=f"\n\n[Streaming Error]: {str(exc)}",
                        completion_id=completion_id,
                        created=created,
                        finish_reason="error",
                    )
                    yield "data: [DONE]\n\n"

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
                yield format_openai_stream_chunk(
                    model=request.model,
                    content=exc.user_message,
                    completion_id=f"chatcmpl-error-{created}",
                    created=created,
                    finish_reason="stop"
                )
                yield "data: [DONE]\n\n"

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
                yield format_openai_stream_chunk(
                    model=request.model,
                    content="The local agent could not complete the request. Please retry with a smaller request or a narrower workspace context.",
                    completion_id=f"chatcmpl-error-{created}",
                    created=created,
                    finish_reason="stop"
                )
                yield "data: [DONE]\n\n"

            return StreamingResponse(
                generic_error_stream(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        return build_openai_fallback_response(
            model=request.model,
            content="The local agent could not complete the request. Please retry with a smaller request or a narrower workspace context.",
        )
