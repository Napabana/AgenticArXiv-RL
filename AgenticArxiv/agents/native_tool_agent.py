"""AgenticArxiv Agent using OpenAI-compatible structured tool messages."""

import json
from typing import Any, Dict, List, Optional, Tuple

from agents.base_agent import BaseAgent, PAPER_SEARCH_ACTIONS
from agents.prompt_templates import NATIVE_TOOL_SYSTEM_PROMPT
from tools.tool_registry import registry
from utils.llm_client import parse_tool_response


class NativeToolCallingAgent(BaseAgent):
    """Run the shared Agent loop with native API requests and tool feedback."""

    agent_type = "native_tool_calling"
    stop_sequences = ()

    def _reset_native_state(self, session_id: str) -> None:
        self._translation_state = {}
        self._pending_cache_checks = set()
        self._setup_session = session_id

    def apply_setup(self, setup: list, session_id: str) -> list[dict]:
        """Execute historical actions using the same tool path and retain receipts."""
        self._reset_native_state(session_id)
        self.session_id = session_id
        receipts = []
        for step in setup:
            name = step["name"]
            args = registry.normalize_arguments(name, step.get("args") or {})
            observation = self._execute_native_tool({"name": name, "args": args})
            result = self._last_tool_result
            if name == "translate_arxiv_pdf" and result and not self._last_tool_issue:
                paper_id = result["paper_id"]
                has_recorded_translation = getattr(self.env, "has_recorded_translation", None)
                if result["status"] == "PENDING" and has_recorded_translation and has_recorded_translation(paper_id):
                    # Historical setup can restore a translation already present in this snapshot.
                    self._translation_state[paper_id] = "READY"
                    result = {**result, "status": "READY", "source": "snapshot"}
                    observation = self._format_native_result(name, result)
            receipt = {"name": name, "args": args, "observation": observation,
                       "tool_issue": self._last_tool_issue}
            receipts.append(receipt)
            if self._last_tool_issue:
                break
        return receipts

    def setup_context(self, receipts: list[dict]) -> str:
        """Describe recorded setup and the current session, without inferring success."""
        if not receipts:
            return ""
        lines = [f'- {receipt["name"]}：{receipt["observation"]}' for receipt in receipts]
        papers = self.side_effects.get_last_papers(self.session_id)
        lines.append(f"- 当前候选列表有 {len(papers)} 篇。")
        store = getattr(self.side_effects, "store", None)
        active = store.get_last_active_paper_id(self.session_id) if store is not None else None
        lines.append(f"- 最近操作的论文是 {active}，可用 null 引用。" if active
                     else "- 当前没有最近操作的论文。")
        return "\n".join(lines)

    def run(
        self,
        task: str,
        agent_model: Optional[str] = None,
        session_id: str = "default",
        initial_history: str = "",
    ) -> Dict[str, Any]:
        if getattr(self, "_setup_session", None) != session_id:
            self._reset_native_state(session_id)
        self._setup_session = None
        self._conversation: List[Dict[str, Any]] = []
        self._terminal_error = ""
        self._parse_failed = False
        return super().run(task, agent_model, session_id, initial_history)

    def discover_tools(self) -> List[Dict[str, Any]]:
        return [registry.model_tool(tool["name"]) for tool in registry.list_tools()]

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
            self._parse_failed = True
            return "", None

        if action is None:
            return text.strip(), None
        tool = registry.model_tool(action["name"])
        if tool is None:
            self._terminal_error = f"未知工具: {action['name']}"
            return text, None
        raw_args = action["args"]
        action["args"] = registry.normalize_arguments(action["name"], raw_args)
        if raw_args != action["args"]:
            action["raw_args"] = raw_args
            call["function"]["arguments"] = json.dumps(action["args"], ensure_ascii=False)
        valid, reason = self._validate_routed_action(action, tool)
        if not valid:
            self._terminal_error = f"Schema 校验失败: {reason}"
            return text, None

        self._conversation.append({
            "role": "assistant", "content": text or None, "tool_calls": [call],
        })
        return text, action

    def _terminal_step(self, thought: str) -> Dict[str, Any]:
        if self._terminal_error:
            return {"thought": thought, "action": "ERROR", "observation": self._terminal_error,
                    "parse_failed": self._parse_failed}
        return {"thought": thought, "action": "FINISH", "observation": thought}

    def invoke_tool(self, tool_name: str, args: Dict[str, Any]) -> Any:
        return registry.execute_tool(tool_name, args)

    def _execute_with_side_effects(self, action_dict: Dict[str, Any]) -> str:
        observation = self._execute_native_tool(action_dict)
        call_id = self._conversation[-1]["tool_calls"][0]["id"]
        self._conversation.append({
            "role": "tool", "tool_call_id": call_id, "content": observation,
        })
        return observation

    def _execute_native_tool(self, action: Dict[str, Any]) -> str:
        name = action["name"]
        args = action["args"]
        store = getattr(self.side_effects, "store", None)
        ref = args.get("ref")
        paper_id = None
        if name in {"translate_arxiv_pdf", "get_translated_content", "get_paper_cache_status"} and store is not None:
            paper = store.resolve_paper(self.session_id, ref)
            paper_id = paper.id if paper else (
                store.get_last_active_paper_id(self.session_id) if ref is None else None
            )
        if paper_id and self._translation_state.get(paper_id) == "PENDING":
            live_asset = store.get_translate_asset(paper_id) if self.env is None else None
            if live_asset and live_asset.status in {"READY", "FAILED"}:
                self._translation_state[paper_id] = live_asset.status
            elif name != "get_paper_cache_status" or paper_id in self._pending_cache_checks:
                self._last_tool_result = None
                self._last_tool_issue = "pending_guard"
                return f"工具执行失败: 论文 {paper_id} 的翻译处于 PENDING，译文尚未就绪，请勿重提交或轮询。"
            else:
                self._pending_cache_checks.add(paper_id)
        if name == "get_translated_content" and paper_id and getattr(self.env, "mode", None) == "replay":
            if self._translation_state.get(paper_id) != "READY":
                self._last_tool_result = None
                self._last_tool_issue = "translation_not_ready"
                return f"工具执行失败: 论文 {paper_id} 没有可读取的 READY 译文。"
        observation = super()._execute_with_side_effects(action)
        result = self._last_tool_result
        if self._last_tool_issue:
            return observation
        if name == "translate_arxiv_pdf":
            self._translation_state[result["paper_id"]] = result["status"]
            self._pending_cache_checks.discard(result["paper_id"])
        if name == "get_paper_cache_status" and isinstance(result, dict):
            status = self._translation_state.get(result.get("paper_id"))
            if status:
                result = {**result, "translated_ready": status == "READY", "translate": {"status": status}}
                self._last_tool_result = result
        if isinstance(result, (dict, list)):
            return self._format_native_result(name, result)
        return observation

    @staticmethod
    def _format_native_result(name: str, result: Any) -> str:
        if name in PAPER_SEARCH_ACTIONS and isinstance(result, list):
            return json.dumps({
                "paper_count": len(result),
                "papers": [{"ref": index, "paper_id": paper["id"], "title": paper["title"]}
                           for index, paper in enumerate(result, 1)],
            }, ensure_ascii=False)

        def portable(value):
            if isinstance(value, dict):
                return {key: portable(item) for key, item in value.items()
                        if key != "session_id" and not key.endswith("_path")}
            if isinstance(value, list):
                return [portable(item) for item in value]
            return value

        return json.dumps(portable(result), ensure_ascii=False, default=str)
