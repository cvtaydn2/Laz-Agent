# Editor Agent

`editor-agent` is a reusable, local, CLI-first coding agent backend that uses NVIDIA's OpenAI-compatible API with the free model endpoint `minimaxai/minimax-m2.7`.

This project currently includes Phase 1, Phase 2, and Phase 3 safe capabilities:

- no automatic file editing
- patch preview only, with no automatic patch apply
- no automatic command execution
- no destructive operations
- session outputs are saved locally under `state/sessions`
- patch proposals are saved locally under `state/patches`
- apply mode requires an explicit `--confirm` flag
- backups are created under `state/backups` before each file write
- failed apply runs rollback written files from backup

## Features

- `health` checks local config and endpoint readiness inputs
- `analyze` scans a workspace and summarizes what it contains
- `ask` asks a question about a workspace
- `suggest` asks for safe, non-executing suggestions to move a project forward
- `patch-preview` generates file-by-file proposed changes without modifying files
- `apply` can write explicitly proposed file changes only when `--confirm` is provided
- safe file scanning with ignore rules and extension allowlist
- UTF-8-safe file reads with binary detection
- relevant file ranking based on the task and project signals
- structured prompts and structured terminal output

## Project Layout

```text
editor-agent/
  .env.example
  .gitignore
  README.md
  requirements.txt
  main.py
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
      suggester.py
      response_parser.py
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

Then edit `.env` and set `NVIDIA_API_KEY`.

## Commands

### Health

```powershell
python .\main.py health
```

### Analyze a project

```powershell
python .\main.py analyze .
```

### Ask a question about a project

```powershell
python .\main.py ask . "What does this project do?"
```

### Get suggestions

```powershell
python .\main.py suggest . "Find what is missing to run this project"
```

### Generate a patch preview

```powershell
python .\main.py patch-preview . "Add missing environment setup docs"
```

### Generate and apply an approved patch

Preview only, no file writes:

```powershell
python .\main.py apply . "Implement the approved patch"
```

Apply with explicit confirmation:

```powershell
python .\main.py apply . "Implement the approved patch" --confirm
```

## Configuration

The app reads environment variables from `.env` when available.

- `NVIDIA_API_KEY`: required for model calls
- `NVIDIA_BASE_URL`: defaults to `https://integrate.api.nvidia.com/v1`
- `NVIDIA_MODEL`: defaults to `minimaxai/minimax-m2.7`
- `AGENT_TEMPERATURE`: defaults to `0.2`
- `AGENT_TIMEOUT_SECONDS`: defaults to `60`
- `AGENT_MAX_FILE_BYTES`: max bytes per file read
- `AGENT_MAX_CONTEXT_CHARS`: total context budget passed to the model
- `AGENT_TOP_K_FILES`: how many files to send as high-priority context

## Safety Notes

- The agent does not modify project files in Phase 1.
- The agent does not modify project files in Phase 2 patch preview mode.
- The agent does not execute shell commands.
- The agent only suggests commands in text form.
- Patch previews are proposals only and are never applied automatically.
- Apply mode writes files only when `--confirm` is present.
- Apply mode never deletes files or directories automatically.
- Apply mode creates backups before each file write and rolls back on failure.
- In this safe version, confirmed apply only updates existing files. New-file creation stays in preview mode until a later phase.
- Large files, binary files, ignored directories, and disallowed extensions are skipped.

## Session Output

Each `analyze`, `ask`, `suggest`, `patch-preview`, and `apply` run saves a JSON session file under `state/sessions/`.

Each `patch-preview` and `apply` run also saves a dedicated patch proposal JSON file under `state/patches/`.

Confirmed `apply` runs also save an apply log JSON file under `state/logs/`.

The saved data includes:

- command mode
- target workspace
- question or goal
- ranked files
- workspace summary
- raw model output
- parsed response
- timestamps

Patch proposal files include:

- summary
- risks
- affected_files
- proposed_changes
- next_steps
- source session id

Apply log files include:

- session id
- workspace path
- whether confirmation was provided
- files written
- backup locations
- rollback status
- timestamps

## Notes

- Health checks do not make a paid-provider call. They only validate local configuration and endpoint setup inputs.
- If you want to test a live model response, use `analyze`, `ask`, `suggest`, `patch-preview`, or `apply` after setting `NVIDIA_API_KEY`.
