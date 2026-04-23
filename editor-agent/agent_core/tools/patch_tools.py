from __future__ import annotations

from agent_core.models import ParsedAnswer, PatchProposal


def patch_support_status() -> str:
    return "Patch preview generation is available. Patch apply mode is still disabled."


def build_patch_proposal(
    session_id: str,
    created_at,
    workspace_path: str,
    request: str | None,
    parsed: ParsedAnswer,
) -> PatchProposal:
    return PatchProposal(
        session_id=session_id,
        created_at=created_at,
        workspace_path=workspace_path,
        request=request,
        summary=parsed.summary,
        risks=parsed.risks,
        affected_files=parsed.affected_files,
        proposed_changes=parsed.proposed_changes,
        next_steps=parsed.next_steps,
    )
