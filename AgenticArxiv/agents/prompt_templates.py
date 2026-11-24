# AgenticArxiv/agents/prompt_templates.py

from typing import Any, Dict, Mapping, Sequence

REACT_PROMPT_TEMPLATE = """你是一个AI研究助手,可以获取最新的arXiv计算机科学论文。你有以下工具可以使用：

{tools_description}

当前任务：{task}
请按照ReAct框架的格式思考和行动:
Thought: 分析当前情况和下一步需要做什么
Action: {{"name":"工具名称","args":{{参数对象}}}}
Observation: 工具执行的结果
当你认为任务已经完成时，使用以下格式结束：
Thought: 任务已完成
Action: FINISH
如果任务因为缺少必要上下文、参数非法或工具能力不支持而无法执行，不要编造
参数或调用注定失败的工具；请在 Thought 中明确说明无法执行的原因，再用
Action: FINISH 结束。此时不要声称任务已经完成。
注意：只能使用上面列出的工具。每次只能执行一个动作。
强约束：
- Action 后必须是“严格 JSON”(双引号、true/false/null, 小写)
- 禁止 Python 风格 True/False/None
- 禁止尾随逗号、注释、以及任何额外文本
关于“指代上一条操作的论文”：
- 当用户没有给出明确的论文序号/ID/标题，但你判断用户是在指代“最近一次操作过的那篇论文”（例如上下文中的指代性表达），
  则在调用需要论文引用的工具时，把 args 里的 ref 设置为 null(JSON null), 由工具自动定位最近操作的论文。
- 如果用户给出了明确序号/ID/标题，则正常传 ref。
关于异步任务（翻译）：
- translate_arxiv_pdf 是异步任务，调用后会立即返回 task_id 和 PENDING 状态。
- 翻译进度由前端通过 SSE 实时推送，你不需要也不应该轮询检查翻译状态。
- 调用 translate_arxiv_pdf 之后，直接 FINISH 即可，不要再调用 get_paper_cache_status 去查看翻译是否完成。
正确示例：
Action: {{"name":"translate_arxiv_pdf","args":{{"ref":2,"session_id":"demo1","force":false,"service":"bing","threads":4,"keep_dual":false}}}}
Action: {{"name":"download_arxiv_pdf","args":{{"ref":null,"session_id":"demo1","force":false}}}}
现在开始执行任务：
{history}
"""

def get_react_prompt(task: str, tools_description: str, history: str = "") -> str:
    """生成ReAct提示词"""
    return REACT_PROMPT_TEMPLATE.format(
        task=task,
        tools_description=tools_description,
        history=history
    )


NATIVE_TOOL_SYSTEM_PROMPT = """你是一个 arXiv 计算机科学论文研究助手。

根据用户目标和当前会话状态使用工具。每轮最多调用一个工具。优先利用已有会话上下文和历史结果。

调用工具前可以用一句简短自然的话说明已知状态和下一步动作。以用户请求、会话状态和工具结果为依据。不要输出冗长的内部推理或隐藏思考标签。

工具名称、参数含义、类型、合法取值和默认行为以工具定义为准。只调用提供的工具。

### 用户约束与任务完成规则

1. 遵守用户约束：用户明确提出的禁止条件、范围限制、参数要求和步骤顺序必须遵守。工具默认行为与用户限制冲突时，选择能够满足限制的合法参数；只执行与用户目标相关的操作。
2. 跟踪任务进度：根据用户请求、历史交互和真实工具 Observation，判断已完成、待完成和受阻的步骤。按用户要求及步骤依赖继续执行；只有全部必需步骤完成或存在明确阻塞时，才能给出最终回答。区分 PENDING 和 READY，不重复提交或无意义轮询。若用户要求且后续步骤不依赖待就绪的结果，继续执行；若依赖该结果，说明尚未就绪。
3. 处理阻塞与部分完成：缺少必要上下文、引用无效或工具能力不支持时，说明具体原因，不猜测或编造。若其他独立子目标仍可执行且不违反用户约束，应继续完成，并分别汇报完成情况和阻塞原因。工具出错后，根据错误修正可修正的调用，不原样重试。
4. 先判断用户要求的操作能否由当前工具完成。若工具缺少完成该操作所需的能力，直接说明哪一步做不了。在这种情况下，不要用无关工具代替，也不要索取即使补充了仍无法完成操作的信息。

最终回答直接交付用户要求的结果和必要状态。每个结果汇报一次，回答完即结束。"""


def get_routed_argument_prompt(
    task: str,
    selected_tool: str,
    tool_description: str,
    history: str = "",
) -> str:
    """在训练分布内的 ReAct 模板中固定路由器已经选择的工具。

    小型 GRPO checkpoint 对全新的“参数生成器”模板容易直接输出 FINISH。
    保留原模板，只把不可更改的工具决定放进任务块，可最大程度减少 prompt 漂移。
    """
    constrained_task = (
        f"[系统路由决定] 外部路由器已经做出不可更改的工具决定：{selected_tool}。\n"
        f"本轮必须调用 {selected_tool}，你只负责生成该工具的 args；"
        "不能改用其他工具，也不能在本轮输出 FINISH。\n"
        "参数必须符合下方唯一工具 schema；明确 ID、ref 序号或 null 必须原样保留。\n"
        f"[原始用户任务] {task}"
    )
    return get_react_prompt(
        task=constrained_task,
        tools_description=tool_description,
        history=history,
    )


def build_visible_setup_context(task: Mapping[str, Any]) -> str:
    """把任务开始前的 ``setup`` 转成模型可见的会话状态。

    ``setup`` 是 benchmark/RL 框架为了还原既有会话而执行的动作，不是
    当前策略应该生成的专家步骤。模型若完全看不到这些状态，同一句
    “把刚才那篇论文翻译一下”在空会话和已有活跃论文的会话中会拥有
    相反的最优动作，形成不可学习的观测混叠。

    这里只描述 setup 执行后必然成立的状态，不包含 task_id、``steps``、
    expected_tools 或奖励，因此不会把标准答案泄漏给策略。
    """
    setup: Sequence[Mapping[str, Any]] = task.get("setup") or []
    lines = ["[任务开始前的会话状态]"]
    if not setup:
        lines.append(
            "- 当前会话没有既有论文列表，也没有最近操作的论文；"
            "ref=null 暂时无法解析。"
        )
        return "\n".join(lines)

    has_paper_list = False
    last_active_ref = None
    for action in setup:
        name = str(action.get("name") or "")
        args: Dict[str, Any] = dict(action.get("args") or {})
        if name == "get_recently_submitted_cs_papers":
            count = int(args.get("max_results", 50))
            aspect = str(args.get("aspect", "*"))
            days = int(args.get("days", 7))
            has_paper_list = True
            lines.append(
                f"- 此前已检索最近 {days} 天的 cs.{aspect} 论文，"
                f"当前候选列表最多有 {count} 篇，可直接按 ref 序号引用。"
            )
        elif name == "search_arxiv_papers":
            count = int(args.get("max_results", 10))
            query = str(args.get("query") or "").strip()
            has_paper_list = True
            lines.append(
                f"- 此前已按关键词 {query!r} 检索论文，"
                f"当前候选列表最多有 {count} 篇，可直接按 ref 序号引用。"
            )
        elif name == "download_arxiv_pdf":
            last_active_ref = args.get("ref")
            lines.append(
                f"- 此前已下载 ref={last_active_ref!r} 的论文；"
                "它现在是最近操作的论文，ref=null 会指向它。"
            )
        elif name == "translate_arxiv_pdf":
            last_active_ref = args.get("paper_id", args.get("ref"))
            lines.append(
                f"- 此前已翻译 ref={last_active_ref!r} 的论文；"
                "它现在是最近操作的论文，ref=null 会指向它。"
            )
        elif name == "get_paper_cache_status":
            last_active_ref = args.get("paper_id", args.get("ref"))
            lines.append(
                f"- 此前已查看 ref={last_active_ref!r} 的缓存状态；"
                "它现在是最近操作的论文，ref=null 会指向它。"
            )
        # 解读类工具（T2/T3/T4）同样会写 last_active，漏掉它们的后果和上面
        # 三个一样：会话状态块不提，ref=null 的指代就无从解释。
        elif name == "get_paper_content":
            last_active_ref = args.get("ref")
            section = args.get("section")
            scope = f"{section!r} 章节" if section else "标题与摘要"
            lines.append(
                f"- 此前已读取 ref={last_active_ref!r} 的{scope}；"
                "它现在是最近操作的论文，ref=null 会指向它。"
            )
        elif name == "summarize_paper":
            last_active_ref = args.get("ref")
            style = args.get("style") or "tldr"
            max_words = args.get("max_words")
            budget = f"（{max_words} 词以内）" if max_words else ""
            lines.append(
                f"- 此前已用 {style} 风格{budget}总结 ref={last_active_ref!r} 的论文；"
                "它现在是最近操作的论文，ref=null 会指向它。"
            )
        elif name == "extract_paper_figures":
            last_active_ref = args.get("ref")
            lines.append(
                f"- 此前已抽取 ref={last_active_ref!r} 的图表；"
                "它现在是最近操作的论文，ref=null 会指向它。"
            )

    if not has_paper_list:
        lines.append("- 当前会话没有由先前检索建立的论文候选列表。")
    lines.append("以上是已存在的会话状态，不属于当前任务动作，请勿重复执行。")
    return "\n".join(lines)

def format_tool_description(tools) -> str:
    """格式化工具描述 - 更详细的版本"""
    if not tools:
        return "当前没有可用工具"
    
    descriptions = []
    for tool in tools:
        name = tool.get('name', '未知工具')
        desc = tool.get('description', '无描述')
        
        params_info = ""
        if 'parameters' in tool and 'properties' in tool['parameters']:
            params = []
            for param_name, param_spec in tool['parameters']['properties'].items():
                param_type = param_spec.get('type', 'unknown')
                param_desc = param_spec.get('description', '')
                default_val = param_spec.get('default', '无默认值')
                
                if 'enum' in param_spec:
                    enum_vals = param_spec['enum']
                    if len(enum_vals) > 5:
                        param_info = f"{param_name} ({param_type}): {param_desc}, 可选值: {enum_vals[:3]}...等{len(enum_vals)}个值"
                    else:
                        param_info = f"{param_name} ({param_type}): {param_desc}, 可选值: {enum_vals}"
                else:
                    param_info = f"{param_name} ({param_type}): {param_desc}, 默认: {default_val}"
                
                params.append(f"    {param_info}")
            
            if params:
                params_info = "\n" + "\n".join(params)
        
        tool_desc = f"- {name}: {desc}"
        if params_info:
            tool_desc += "\n  参数:"
            tool_desc += params_info
        
        descriptions.append(tool_desc)
    
    return "\n\n".join(descriptions)
