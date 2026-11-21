# AgenticArxiv/tools/tool_registry.py
from copy import deepcopy
from typing import Dict, Any, Callable, List, Optional
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


PAPER_REF_SCHEMA = {
    "description": (
        "当前会话候选列表的 1-based 序号、完整 arXiv ID（保留版本后缀）或标题片段。"
        "省略或 null 指最近操作的论文；没有最近操作对象时无法解析。"
    ),
    "anyOf": [{"type": "integer", "minimum": 1}, {"type": "string", "minLength": 1}, {"type": "null"}],
    "default": None,
}

_HARNESS_ARGS = {"session_id", "paper_id", "pdf_url", "input_pdf_path"}


class ToolRegistry:
    """
    工具注册表，管理所有可用的工具
    """

    def __init__(self):
        self._tools: Dict[str, Dict[str, Any]] = {}

    def register_tool(
        self,
        name: str,
        description: str,
        parameter_schema: Dict[str, Any],
        func: Callable,
    ) -> None:
        """
        注册一个工具

        Args:
            name: 工具名称（唯一标识）
            description: 工具描述, 说明工具的功能
            parameter_schema: 参数模式, 遵循JSON Schema规范
            func: 工具函数
        """
        self._tools[name] = {
            "name": name,
            "description": description,
            "parameters": parameter_schema,
            "func": func,
        }

    def get_tool(self, name: str) -> Optional[Dict[str, Any]]:
        """获取指定名称的工具"""
        return self._tools.get(name)

    def list_tools(self) -> List[Dict[str, Any]]:
        """列出所有可用工具"""
        return [
            {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["parameters"],
            }
            for tool in self._tools.values()
        ]

    def model_tool(self, name: str) -> Optional[Dict[str, Any]]:
        """Build the Native schema without internal execution overrides."""
        tool = self.get_tool(name)
        if tool is None:
            return None
        schema = deepcopy(tool["parameters"])
        properties = schema.get("properties", {})
        for key in _HARNESS_ARGS:
            properties.pop(key, None)
        schema["required"] = [key for key in schema.get("required", []) if key not in _HARNESS_ARGS]
        schema["additionalProperties"] = False
        if name == "translate_arxiv_pdf":
            from config import settings
            properties["service"]["default"] = settings.pdf2zh_service
            properties["threads"]["default"] = settings.pdf2zh_threads
        return {"name": name, "description": tool["description"], "parameters": schema}

    @staticmethod
    def normalize_arguments(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize arXiv category aliases before validating the canonical enum."""
        args = dict(arguments)
        if name == "get_recently_submitted_cs_papers" and isinstance(args.get("aspect"), str):
            args["aspect"] = args["aspect"].strip().upper().removeprefix("CS.")
        return args

    def effective_arguments(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Expand model-visible defaults for scoring and canonical expert calls."""
        tool = self.model_tool(name)
        properties = tool["parameters"]["properties"] if tool else {}
        args = {key: spec["default"] for key, spec in properties.items() if "default" in spec}
        args.update(self.normalize_arguments(name, arguments))
        args.pop("session_id", None)
        if name == "get_recently_submitted_cs_papers":
            if not args.get("output_path") or args.get("save_to_file") is False:
                args["output_path"] = None
        if name == "get_translated_content" and args.get("page") is None:
            args["page"] = 1
        if name == "summarize_paper":
            from tools.paper_summary_tool import bucket_max_words, normalize_style
            if "style" in args and "max_words" in args:
                try:
                    style = normalize_style(args["style"])
                    max_words = bucket_max_words(args["max_words"])
                except ValueError:
                    pass  # Preserve invalid inputs for mismatching in historical traces.
                else:
                    args.update(style=style, max_words=max_words)
        return args

    def execute_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        """
        执行指定工具

        Args:
            name: 工具名称
            arguments: 工具参数

        Returns:
            工具执行结果

        Raises:
            ValueError: 如果工具不存在或参数无效
        """
        if name not in self._tools:
            raise ValueError(f"工具 '{name}' 未注册")

        tool = self._tools[name]
        try:
            # 将JSON参数传递给工具函数
            return tool["func"](**arguments)
        except TypeError as e:
            raise ValueError(f"工具参数错误: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"工具执行失败: {str(e)}")


# 创建全局工具注册表实例
registry = ToolRegistry()
