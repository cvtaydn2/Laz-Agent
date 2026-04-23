from __future__ import annotations

from textwrap import dedent

from agent_core.models import AgentMode, FileContext, PromptBundle, WorkspaceSummary


SYSTEM_PROMPT = dedent(
    """
    You are a safe local coding agent.
    Follow these rules strictly:
    - Do not suggest destructive actions as defaults.
    - Do not assume commands were run.
    - If project information is incomplete, say so clearly.
    - Prefer concrete observations from the provided files.
    - Keep suggestions non-destructive and approval-based.
    - Use the exact headings requested by the task.
    - Use plain text.
    """
).strip()


def build_prompt(
    mode: AgentMode,
    workspace_summary: WorkspaceSummary,
    selected_files: list[FileContext],
    user_input: str | None,
) -> PromptBundle:
    header = [
        f"MODE: {mode.value}",
        f"WORKSPACE: {workspace_summary.root_path}",
        f"TOTAL_SCANNED: {workspace_summary.total_files_scanned}",
        f"INCLUDED_FILES: {workspace_summary.included_files}",
        f"SKIPPED_FILES: {workspace_summary.skipped_files}",
        f"TOP_EXTENSIONS: {workspace_summary.top_extensions}",
        f"SAMPLED_FILES: {workspace_summary.sampled_files}",
    ]

    if workspace_summary.notes:
        header.append(f"NOTES: {workspace_summary.notes}")

    if user_input:
        header.append(f"USER_REQUEST: {user_input}")

    context_blocks: list[str] = []
    for file_context in selected_files:
        context_blocks.append(
            dedent(
                f"""
                FILE: {file_context.relative_path}
                SCORE: {file_context.score}
                REASON: {file_context.reason}
                CONTENT:
                {file_context.content}
                """
            ).strip()
        )

    task_instruction = _task_instruction(mode, user_input)
    user_prompt = "\n\n".join(["\n".join(header), task_instruction, *context_blocks])
    return PromptBundle(system_prompt=SYSTEM_PROMPT, user_prompt=user_prompt)


def _task_instruction(mode: AgentMode, user_input: str | None) -> str:
    if mode == AgentMode.ANALYZE:
        return (
            "OUTPUT_HEADINGS:\n"
            "SUMMARY:\n"
            "FINDINGS:\n"
            "SUGGESTIONS:\n"
            "COMMANDS_TO_CONSIDER:\n"
            "RISKS:\n\n"
            "TASK: Analyze the workspace and explain what the project appears to do, "
            "how it is structured, and any obvious missing information."
        )
    if mode == AgentMode.ASK:
        return (
            "OUTPUT_HEADINGS:\n"
            "SUMMARY:\n"
            "FINDINGS:\n"
            "SUGGESTIONS:\n"
            "COMMANDS_TO_CONSIDER:\n"
            "RISKS:\n\n"
            f"TASK: Answer this workspace question using the provided files: {user_input}"
        )
    if mode == AgentMode.SUGGEST:
        return (
            "OUTPUT_HEADINGS:\n"
            "SUMMARY:\n"
            "FINDINGS:\n"
            "SUGGESTIONS:\n"
            "COMMANDS_TO_CONSIDER:\n"
            "RISKS:\n\n"
            "TASK: Provide safe suggestions for the request below. Suggest commands only as text, "
            "and explain why they matter before listing them.\n"
            f"REQUEST: {user_input}"
        )
    return (
        "OUTPUT_HEADINGS:\n"
        "SUMMARY:\n"
        "RISKS:\n"
        "AFFECTED_FILES:\n"
        "PROPOSED_CHANGES:\n"
        "NEXT_STEPS:\n\n"
        "TASK: Produce a patch preview proposal only. Do not claim any file was modified. "
        "Describe file-by-file changes that should be made, why they are needed, and keep the plan safe and reversible.\n"
        "For AFFECTED_FILES, list one repository-relative path per bullet.\n"
        "For PROPOSED_CHANGES, include the target file path at the start of each bullet when possible.\n"
        f"REQUEST: {user_input}"
    )
