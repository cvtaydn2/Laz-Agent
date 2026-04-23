from __future__ import annotations

from agent_core.models import AgentMode


class AgentPlanner:
    @staticmethod
    def plan_label(mode: AgentMode) -> str:
        if mode == AgentMode.ANALYZE:
            return "Analyze"
        if mode == AgentMode.ASK:
            return "Ask"
        return "Suggest"
