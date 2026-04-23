# AGENTS.md

## Goal
Build a local, CLI-first coding agent that uses NVIDIA's OpenAI-compatible endpoint with the model `minimaxai/minimax-m2.7`.

## Constraints
- Do not use any paid provider by default.
- Use NVIDIA endpoint only:
  - Base URL: `https://integrate.api.nvidia.com/v1`
  - Model: `minimaxai/minimax-m2.7`
  - API key env var: `NVIDIA_API_KEY`
- The first version must be safe:
  - no automatic file overwrite
  - no automatic destructive commands
  - no delete/remove operations
  - no automatic command execution without explicit approval
- Build backend first, editor integration second.

## Product shape
We are building a mini coding agent for use in other projects.

Phase 1:
- Python CLI app
- analyze / ask / suggest commands
- scans workspace
- ranks relevant files
- reads safe text/code files
- sends selected context to NVIDIA model
- prints structured response
- saves session outputs

Phase 2:
- patch preview generation
- apply mode with backups and rollback
- local server wrapper for editor integration

Phase 3:
- VS Code extension that calls the local agent backend

## Tech stack
- Python 3.11+
- typer
- httpx
- pydantic
- python-dotenv
- rich
- tenacity

## File safety rules
Ignore:
- .git
- node_modules
- dist
- build
- .next
- coverage
- .venv
- venv
- __pycache__

Allowed extensions:
- .py
- .js
- .ts
- .tsx
- .jsx
- .json
- .md
- .txt
- .yml
- .yaml
- .html
- .css
- .env.example

## Output rules
Always:
1. explain the plan first
2. then create files
3. keep code modular
4. add basic error handling
5. keep Windows PowerShell compatibility in mind
6. use UTF-8-safe file reading
7. never claim something was tested if it was not actually tested

## Delivery rules
When asked to implement:
- first show the file tree
- then generate all files completely
- do not leave placeholders unless explicitly marked
- include README and usage examples
