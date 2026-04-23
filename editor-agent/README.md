# Editor Agent

`editor-agent` is a local, CLI-first coding agent backend that exposes a stable OpenAI-compatible proxy for editor clients such as Continue.

The external proxy model id remains:

- `laz-agent`

The default NVIDIA backend model is now:

- `moonshotai/kimi-k2-instruct`

The project is optimized for:

- stable code analysis
- stable code review
- non-streaming OpenAI-compatible requests
- smaller, predictable workspace context
- safe failure handling when the backend is slow or unavailable

## Features

- CLI commands for `health`, `analyze`, `ask`, `suggest`, `patch-preview`, and `apply`
- FastAPI local server with custom endpoints and OpenAI-compatible endpoints
- non-streaming `GET /v1/models`
- non-streaming `POST /v1/chat/completions`
- review mode with structured output
- deterministic `.env` loading from the current working directory
- fail-fast startup if `NVIDIA_API_KEY` is missing
- bounded retries and timeout handling for NVIDIA requests
- safe fallback responses when backend inference times out

## Project Layout

```text
editor-agent/
  .env.example
  .gitignore
  README.md
  requirements.txt
  main.py
  server.py
  agent_core/
    __init__.py
    config.py
    models.py
    nvidia_client.py
    prompts.py
    logger.py
    workspace/
      __init__.py
      scanner.py
      filters.py
      reader.py
      ranker.py
    agent/
      __init__.py
      apply_mode.py
      orchestrator.py
      patch_preview.py
      planner.py
      review_verifier.py
      response_parser.py
      suggester.py
    tools/
      __init__.py
      apply_tools.py
      file_tools.py
      patch_tools.py
      command_tools.py
    output/
      __init__.py
      formatter.py
      writers.py
    server/
      __init__.py
      api.py
      openai_adapter.py
      openai_schemas.py
      schemas.py
      service.py
  tests/
    test_env_loading.py
    test_openai_api.py
    test_openai_compat.py
  state/
    sessions/
    logs/
    patches/
    backups/
```

## Requirements

- Windows PowerShell or compatible shell
- Python 3.11+
- NVIDIA API key in `NVIDIA_API_KEY`

## PowerShell Setup

```powershell
Set-Location C:\Users\Cevat\Documents\Github\Laz-Agent\editor-agent
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

## Environment Setup

1. Create `.env` in the project root.
2. Add your NVIDIA API key:

```env
NVIDIA_API_KEY=nvapi-...
```

3. Start the server only from the project root:

```powershell
Set-Location C:\Users\Cevat\Documents\Github\Laz-Agent\editor-agent
python .\server.py
```

Environment loading is explicit:

- `.env` is loaded with `find_dotenv(usecwd=True)`
- loaded path is printed at startup
- startup fails immediately if `NVIDIA_API_KEY` is missing

## Stable Defaults

Default backend configuration:

- `NVIDIA_MODEL=moonshotai/kimi-k2-instruct`
- `AGENT_TEMPERATURE=0.1`
- `AGENT_TIMEOUT_SECONDS=18`
- `AGENT_MAX_FILE_BYTES=120000`
- `AGENT_MAX_CHARS_PER_FILE=2500`
- `AGENT_MAX_CONTEXT_CHARS=12000`
- `AGENT_TOP_K_FILES=5`
- `AGENT_MAX_COMPLETION_TOKENS=300`
- `AGENT_DEFAULT_WORKSPACE=<server start directory by default>`

These remain configurable via environment variables.

## Running The Server

```powershell
Set-Location C:\Users\Cevat\Documents\Github\Laz-Agent\editor-agent
.\.venv\Scripts\Activate.ps1
python .\server.py
```

Or:

```powershell
uvicorn agent_core.server.api:app --host 127.0.0.1 --port 8000
```

## Continue Configuration

Use the local OpenAI-compatible proxy:

- `apiBase`: `http://127.0.0.1:8000/v1`
- `model`: `laz-agent`

Example Continue-style request body:

```json
{
  "model": "laz-agent",
  "messages": [
    {"role": "system", "content": "You are a stable code assistant."},
    {"role": "user", "content": "What does this project do?"}
  ],
  "extra_body": {
    "workspace": "C:/Users/Cevat/Documents/Github/Laz-Agent/editor-agent",
    "mode": "ask"
  }
}
```

## OpenAI-Compatible Endpoints

- `GET /v1/models`
- `POST /v1/chat/completions`

Rules:

- non-streaming only
- `workspace` is preferred
- external model id stays `laz-agent`
- internal NVIDIA backend model can be changed independently
- `stream=true` is supported with minimal OpenAI-style SSE

Supported `extra_body` fields:

- `workspace`
- `mode`
- `changed_files`
- `diff`

Workspace resolution order:

1. `extra_body.workspace`
2. `extraBody.workspace`
3. `metadata.workspace`
4. top-level `workspace`
5. `AGENT_DEFAULT_WORKSPACE`
6. server process current working directory

Allowed modes:

- `ask`
- `analyze`
- `suggest`
- `patch-preview`
- `review`

`apply` is not exposed through the OpenAI-compatible endpoint.

## Examples

### Model List

PowerShell:

```powershell
Invoke-RestMethod -Method Get -Uri http://127.0.0.1:8000/v1/models
```

curl:

```bash
curl http://127.0.0.1:8000/v1/models
```

### Ask Request

PowerShell:

```powershell
$body = @{
  model = "laz-agent"
  messages = @(
    @{
      role = "system"
      content = "You are a stable code assistant."
    },
    @{
      role = "user"
      content = "What does this project do?"
    }
  )
  extra_body = @{
    workspace = "C:/Users/Cevat/Documents/Github/Laz-Agent/editor-agent"
    mode = "ask"
  }
} | ConvertTo-Json -Depth 6

Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:8000/v1/chat/completions `
  -ContentType "application/json" `
  -Body $body
```

### Review Request

```json
{
  "model": "laz-agent",
  "messages": [
    {"role": "system", "content": "You are a stable code review engine."},
    {"role": "user", "content": "Review the recent changes for correctness and risk."}
  ],
  "extra_body": {
    "workspace": "C:/Users/Cevat/Documents/Github/Laz-Agent/editor-agent",
    "mode": "review",
    "changed_files": ["agent_core/server/api.py", "agent_core/nvidia_client.py"]
  }
}
```

### Review Response Shape

Review responses are returned inside `choices[0].message.content` as structured JSON text:

```json
{
  "summary": "Review completed with one verified finding.",
  "findings": [
    {
      "title": "Timeout fallback message can leak backend details",
      "severity": "medium",
      "file": "agent_core/server/api.py",
      "evidence": "build_openai_fallback_response",
      "issue": "An unstable fallback would reduce Continue compatibility.",
      "suggested_fix": "Return a fixed safe fallback message."
    }
  ],
  "risks": "Moderate reliability risk under backend instability.",
  "next_steps": "Fix the verified finding and re-run the compatibility tests."
}
```

## Timeout And Failure Behavior

The proxy does not stream and does not wait indefinitely.

Current behavior:

- bounded request timeout
- explicit connect/read/write/pool timeouts
- single-attempt backend requests for faster failure
- capped backoff between retries
- safe fallback response when the NVIDIA backend times out
- general `ask` requests can skip workspace scanning when the prompt does not appear repo-aware
- trivial greeting-style `ask` requests can return from a local fast-path without calling the backend model

If inference times out, the response remains OpenAI-compatible and the assistant message contains:

`The model backend timed out while generating a response. Please retry with a smaller request or a narrower workspace context.`

If responses are slow:

- reduce `changed_files`
- send a smaller diff
- narrow the workspace
- lower prompt complexity
- keep review scope to the files you actually changed

## Configuration

- `NVIDIA_API_KEY`: required for model calls
- `NVIDIA_BASE_URL`: defaults to `https://integrate.api.nvidia.com/v1`
- `NVIDIA_MODEL`: defaults to `moonshotai/kimi-k2-instruct`
- `AGENT_TEMPERATURE`: defaults to `0.1`
- `AGENT_TIMEOUT_SECONDS`: defaults to `18`
- `AGENT_MAX_FILE_BYTES`: defaults to `120000`
- `AGENT_MAX_CHARS_PER_FILE`: defaults to `2500`
- `AGENT_MAX_CONTEXT_CHARS`: defaults to `12000`
- `AGENT_TOP_K_FILES`: defaults to `5`
- `AGENT_MAX_COMPLETION_TOKENS`: defaults to `300`
- `AGENT_SERVER_HOST`: defaults to `127.0.0.1`
- `AGENT_SERVER_PORT`: defaults to `8000`

## Validation

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_env_loading
.\.venv\Scripts\python.exe -m unittest tests.test_openai_compat
.\.venv\Scripts\python.exe -m unittest tests.test_openai_api
```
