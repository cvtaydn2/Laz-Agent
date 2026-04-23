from __future__ import annotations

from textwrap import dedent

from agent_core.models import AgentMode, FileContext, PromptBundle, WorkspaceSummary


SYSTEM_PROMPT = """
Sen **Laz-Agent**, dünya klasmanında bir Senior Yazılım Mimarı ve asenkron sistemler uzmanısın. 
NVIDIA'nın en güçlü modelleriyle (Minimax-M2.7 ve Moonshot-Kimi) donatıldın.

### Görevin ve Tavrın:
1. **Acımasız Gerçekçilik**: Projedeki "kötü kod kokularını" (code smells), anti-pattern'leri ve potansiyel performans darboğazlarını çekinmeden yüzeye çıkar. 
2. **Yüzeysellikten Kaçın**: "Analiz tamamlandı" gibi jenerik cümleler kurma. Bunun yerine spesifik dosya ve satırlara atıfta bulunarak derin teknik analiz yap.
3. **Mimar Gözüyle Bak**: Sadece kodu değil, mimari bütünlüğü, asenkron güvenliğini ve hata yönetimini de denetle.
4. **Kesinlik**: Eğer bir dosyada açık bir sorun görmüyorsan, projenin diğer kısımlarıyla olan bağlamını sorgula.
5. **Dil ve Ton**: Kullanıcıyla Türkçe, teknik olarak çok donanımlı, proaktif ve çözüm odaklı bir dille konuş.

### Yanıt Stratejin:
- Her analizde en az 3 spesifik "İyileştirme Fırsatı" (Technical Debt) tanımla.
- Kod önerilerinde her zaman performans ve güvenliği ön planda tut.
- Jenerik özetler yerine, o anki kodun karakteristiğine odaklanan özgün cümleler kur.
"""

ASK_SYSTEM_PROMPT = dedent(
    """
    You are Laz-Agent, a stable local coding assistant using Minimax-M2.7.
    Answer briefly and directly.
    If asked about your identity, confirm you are Laz-Agent.
    Use the provided workspace context only when it is relevant.
    You are capable of scanning directories provided by the user if they specify an absolute path.
    Use plain text.
    """
).strip()


def build_prompt(
    mode: AgentMode,
    workspace_summary: WorkspaceSummary,
    selected_files: list[FileContext],
    user_input: str | None,
    changed_files: list[str] | None = None,
    diff_text: str | None = None,
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
    if changed_files:
        header.append(f"CHANGED_FILES: {changed_files}")
    if diff_text:
        header.append(f"DIFF_PROVIDED: yes")

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

    prompt_parts = ["\n".join(header), _task_instruction(mode, user_input)]
    if diff_text:
        prompt_parts.append(f"DIFF:\n{diff_text}")
    prompt_parts.extend(context_blocks)
    user_prompt = "\n\n".join(prompt_parts)
    system_prompt = ASK_SYSTEM_PROMPT if mode == AgentMode.ASK else SYSTEM_PROMPT
    return PromptBundle(system_prompt=system_prompt, user_prompt=user_prompt)


def _task_instruction(mode: AgentMode, user_input: str | None) -> str:
    if mode == AgentMode.ANALYZE:
        return (
            "OUTPUT_HEADINGS:\n"
            "summary:\n"
            "findings:\n"
            "suggestions:\n"
            "commands_to_consider:\n"
            "risks:\n\n"
            "TASK: Perform a deep-dive architectural audit of the workspace. "
            "Identify technical debt, security risks, and violations of clean code principles. "
            "Focus specifically on asynchronous safety, error handling, and modularity. "
            "Avoid generic summaries; be specific and critical."
        )
    if mode == AgentMode.ASK:
        return (
            "OUTPUT_HEADINGS:\n"
            "summary:\n\n"
            f"TASK: Answer this request briefly and directly: {user_input}"
        )
    if mode == AgentMode.SUGGEST:
        return (
            "OUTPUT_HEADINGS:\n"
            "summary:\n"
            "findings:\n"
            "suggestions:\n"
            "commands_to_consider:\n"
            "risks:\n\n"
            "TASK: Provide safe suggestions for the request below. Suggest commands only as text, "
            "and explain why they matter before listing them.\n"
            f"REQUEST: {user_input}"
        )
    if mode == AgentMode.PATCH_PREVIEW:
        return (
            "OUTPUT_HEADINGS:\n"
            "summary:\n"
            "risks:\n"
            "affected_files:\n"
            "proposed_changes:\n"
            "next_steps:\n\n"
            "TASK: Produce a patch preview proposal only. Do not claim any file was modified. "
            "Describe file-by-file changes that should be made, why they are needed, and keep the plan safe and reversible.\n"
            "For affected_files, list one repository-relative path per bullet.\n"
            "For proposed_changes, include the target file path at the start of each bullet when possible.\n"
            f"REQUEST: {user_input}"
        )
    if mode == AgentMode.REVIEW:
        return (
            "OUTPUT_FORMAT_JSON:\n"
            "{\n"
            '  "summary": "...",\n'
            '  "findings": [\n'
            "    {\n"
            '      "title": "...",\n'
            '      "severity": "low|medium|high",\n'
            '      "file": "...",\n'
            '      "evidence": "...",\n'
            '      "issue": "...",\n'
            '      "suggested_fix": "..."\n'
            "    }\n"
            "  ],\n"
            '  "risks": "...",\n'
            '  "next_steps": "..."\n'
            "}\n\n"
            "TASK: Act as a stable code review engine. Prefer fewer high-confidence findings over many weak findings. "
            "If a possible issue is uncertain, omit it. Base findings on the provided files and diff only.\n"
            "If a diff is provided, review the diff first. If changed files are provided, focus on those files. "
            "If neither is provided, perform a general repository review.\n"
            f"REVIEW_REQUEST: {user_input or 'Review the provided code context.'}"
        )
    return (
        "OUTPUT_HEADINGS:\n"
        "summary:\n"
        "risks:\n"
        "affected_files:\n"
        "proposed_changes:\n"
        "next_steps:\n"
        "file_operations:\n\n"
        "TASK: Produce an apply-ready patch proposal. Do not claim any file was modified yet. "
        "You can propose update, create, or delete actions. Be extremely careful with delete actions.\n"
        "For affected_files, list one repository-relative path per bullet.\n"
        "For proposed_changes, include the target file path at the start of each bullet when possible.\n"
        "For file_operations, emit one or more blocks in exactly this format:\n"
        "BEGIN_FILE\n"
        "PATH: relative/path.ext\n"
        "ACTION: update|create|delete\n"
        "CONTENT:\n"
        "<full file content here or empty for delete>\n"
        "END_FILE\n"
        "Only include files that should actually be modified.\n"
        f"REQUEST: {user_input}"
    )
