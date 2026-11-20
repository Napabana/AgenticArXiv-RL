"""AgenticArxiv Agent using OpenAI-compatible structured tool messages."""

from typing import Any, Dict, List, Optional, Tuple

from agents.base_agent import BaseAgent
from agents.prompt_templates import NATIVE_TOOL_SYSTEM_PROMPT
from tools.tool_registry import registry
from utils.llm_client import parse_tool_response


class NativeToolCallingAgent(BaseAgent):
    """Run the shared Agent loop with native API requests and tool feedback."""

    agent_type = "native_tool_calling"
    stop_sequences = ()

    def run(
        self,
        task: str,
        agent_model: Optional[str] = None,
        session_id: str = "default",
        initial_history: str = "",
    ) -> Dict[str, Any]:
        self._conversation: List[Dict[str, Any]] = []
        self._terminal_error = ""
        return super().run(task, agent_model, session_id, initial_history)

    def discover_tools(self) -> List[Dict[str, Any]]:
        return registry.list_tools()

    def format_history(self, steps: list) -> str:
        return ""

    def build_messages(
        self, task: str, tools_description: str, history_text: str,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        user_content = f"当前任务：{task}"
        if history_text:
            user_content += f"\n\n请求前的会话上下文：\n{history_text}"
        return [
            {"role": "system", "content": NATIVE_TOOL_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            *self._conversation,
        ], {
            "tools": [{"type": "function", "function": tool} for tool in self.discover_tools()],
            "tool_choice": "auto",
        }

    def parse_response(self, raw_response: Dict) -> Tuple[str, Optional[Dict[str, Any]]]:
        choices = raw_response.get("choices")
        choice = choices[0] if isinstance(choices, list) and choices else {}
        try:
            text, call, action = parse_tool_response(choice)
        except ValueError as exc:
            self._terminal_error = f"无法解析工具调用响应: {exc}"
            return "", None

        if action is None:
            return text.strip(), None
        tool = registry.get_tool(action["name"])
        if tool is None:
            self._terminal_error = f"未知工具: {action['name']}"
            return text, None
        valid, reason = self._validate_routed_action(action, tool)
        if not valid:
            self._terminal_error = f"Schema 校验失败: {reason}"
            return text, None

        self._conversation.append({
            "role": "assistant", "content": text or None, "tool_calls": [call],
        })
        return text, action

    def _terminal_step(self, thought: str) -> Dict[str, str]:
        if self._terminal_error:
            return {"thought": thought, "action": "ERROR", "observation": self._terminal_error}
        return {"thought": thought, "action": "FINISH", "observation": thought}

    def invoke_tool(self, tool_name: str, args: Dict[str, Any]) -> Any:
        return registry.execute_tool(tool_name, args)

    def _execute_with_side_effects(self, action_dict: Dict[str, Any]) -> str:
        observation = super()._execute_with_side_effects(action_dict)
        call_id = self._conversation[-1]["tool_calls"][0]["id"]
        self._conversation.append({
            "role": "tool", "tool_call_id": call_id, "content": observation,
        })
        return observation
