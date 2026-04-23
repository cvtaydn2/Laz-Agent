from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse

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
    extract_last_user_text,
    normalize_openai_messages,
    validate_openai_mode,
    extract_preferred_files,
    extract_tools_payload,
    extract_tool_choice_payload,
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
from agent_core.logger import configure_logger as _configure_logger


def _get_logger():
    return _configure_logger(Settings.load().logs_dir / "editor-agent.log")


def _check_proxy_auth(raw_request: Request) -> None:
    """Optional inbound bearer auth.  Only enforced when PROXY_API_KEY is set."""
    settings = Settings.load()
    required_key = getattr(settings, "proxy_api_key", None)
    if not required_key:
        return
    auth = raw_request.headers.get("authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token.")
    token = auth.removeprefix("Bearer ").strip()
    if token != required_key:
        raise HTTPException(status_code=401, detail="Invalid bearer token.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load settings and record env readiness — do NOT crash if key is missing.
    # /health will report ok=False so operators can detect the problem without
    # the server being completely unavailable.
    settings = Settings.load()
    app.state.env_ready = bool(settings.nvidia_api_key)
    app.state.env_error = (
        None if app.state.env_ready else "NVIDIA_API_KEY is not configured."
    )
    if not app.state.env_ready:
        _get_logger().warning("Startup warning: %s", app.state.env_error)

    writer = SessionWriter(settings)
    pruned_count = writer.prune_old_sessions(max_age_days=7)
    if pruned_count > 0:
        _get_logger().info("Pruned %d old session(s) on startup.", pruned_count)
    yield


app = FastAPI(title="Editor Agent API", version="0.1.0", lifespan=lifespan)


@app.middleware("http")
async def track_metrics_middleware(request: Request, call_next):
    method = request.method
    endpoint = request.url.path
    response = await call_next(request)
    status = response.status_code
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
def health(request: Request) -> HealthResponse:
    status = build_health_status()
    # Reflect startup env check in the health response
    env_ready = getattr(request.app.state, "env_ready", True)
    if not env_ready:
        status.ok = False
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
    logger = _get_logger()
    try:
        # Optional proxy auth
        _check_proxy_auth(raw_request)

        try:
            body = await raw_request.json()
            logger.debug("Raw Body Keys: %s", list(body.keys()))
            if "extraBody" in body:
                logger.debug("Found extraBody: %s", list(body["extraBody"].keys()))
            if "metadata" in body:
                logger.debug(
                    "Found metadata: %s",
                    list(body["metadata"].keys()) if isinstance(body["metadata"], dict) else body["metadata"],
                )
        except Exception:
            pass

        workspace = extract_request_workspace(request)
        logger.debug("Workspace extracted: %s", workspace)
        if not workspace:
            raise HTTPException(
                status_code=400,
                detail=(
                    "A workspace path is required under extra_body.workspace, "
                    "metadata.workspace, or top-level workspace."
                ),
            )

        req_mode = extract_request_mode(request)
        mode_value = validate_openai_mode(req_mode)
        logger.debug("Validated mode: %s", mode_value)

        # Normalise messages — preserves role semantics
        normalized_messages = normalize_openai_messages(request.messages)
        last_user_text = extract_last_user_text(request.messages)
        logger.debug(
            "Extracted user text (len=%d): %s...",
            len(last_user_text) if last_user_text else 0,
            (last_user_text or "")[:60],
        )

        if not normalized_messages or not last_user_text:
            raise HTTPException(
                status_code=400,
                detail="At least one non-empty user message is required.",
            )

        # Serialise tool payloads for the provider
        tools_payload = extract_tools_payload(request)
        tool_choice_payload = extract_tool_choice_payload(request)

        # ----------------------------------------------------------------
        # STREAM PATH — true token-by-token SSE via stream_agent()
        # ----------------------------------------------------------------
        if request.stream:
            created = int(datetime.now(timezone.utc).timestamp())
            completion_id = f"chatcmpl-stream-{created}"

            async def event_stream():
                try:
                    logger.debug("Calling stream_agent (true token streaming)...")
                    async for chunk in stream_agent(
                        AgentMode(mode_value),
                        workspace,
                        last_user_text,
                        temperature_override=request.temperature,
                        max_tokens_override=request.max_tokens,
                        changed_files=extract_changed_files(request),
                        diff_text=extract_diff(request),
                        preferred_files=extract_preferred_files(request),
                        tools=tools_payload,
                        tool_choice=tool_choice_payload,
                    ):
                        yield format_openai_stream_chunk(
                            model=request.model,
                            completion_id=completion_id,
                            created=created,
                            delta_content=chunk.get("content"),
                            delta_tool_calls=chunk.get("tool_calls") or None,
                            finish_reason=chunk.get("finish_reason"),
                        )
                    yield "data: [DONE]\n\n"
                except NvidiaInferenceError as exc:
                    yield format_openai_stream_chunk(
                        model=request.model,
                        completion_id=completion_id,
                        created=created,
                        delta_content=exc.user_message,
                        finish_reason="stop",
                    )
                    yield "data: [DONE]\n\n"
                except Exception as exc:
                    logger.exception("Unhandled error in stream_agent")
                    yield format_openai_stream_chunk(
                        model=request.model,
                        completion_id=completion_id,
                        created=created,
                        delta_content=f"The local agent could not complete the request: {exc}",
                        finish_reason="stop",
                    )
                    yield "data: [DONE]\n\n"

            return StreamingResponse(
                event_stream(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        # ----------------------------------------------------------------
        # NON-STREAM PATH
        # ----------------------------------------------------------------
        logger.debug("Calling run_agent...")
        session = await run_agent(
            AgentMode(mode_value),
            workspace,
            last_user_text,
            temperature_override=request.temperature,
            max_tokens_override=request.max_tokens,
            changed_files=extract_changed_files(request),
            diff_text=extract_diff(request),
            preferred_files=extract_preferred_files(request),
        )
        logger.debug("run_agent returned successfully.")
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
                    completion_id=f"chatcmpl-error-{created}",
                    created=created,
                    delta_content=exc.user_message,
                    finish_reason="stop",
                )
                yield "data: [DONE]\n\n"

            return StreamingResponse(
                error_event_stream(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        return build_openai_fallback_response(model=request.model, content=exc.user_message)
    except Exception:
        logger.exception("Unhandled error in chat_completions")
        if request.stream:
            created = int(datetime.now(timezone.utc).timestamp())

            async def generic_error_stream():
                yield format_openai_stream_chunk(
                    model=request.model,
                    completion_id=f"chatcmpl-error-{created}",
                    created=created,
                    delta_content=(
                        "The local agent could not complete the request. "
                        "Please retry with a smaller request or a narrower workspace context."
                    ),
                    finish_reason="stop",
                )
                yield "data: [DONE]\n\n"

            return StreamingResponse(
                generic_error_stream(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        return build_openai_fallback_response(
            model=request.model,
            content=(
                "The local agent could not complete the request. "
                "Please retry with a smaller request or a narrower workspace context."
            ),
        )
