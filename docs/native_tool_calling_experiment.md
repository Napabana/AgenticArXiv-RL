# ArxivFetch 原生工具调用实验设计

## 目标与范围

用同一个 `Qwen/Qwen3.5-2B` 检查点评估原始 ReAct 文本协议和原生工具调用协议。两组均不加载项目的 SFT 或 GRPO Adapter。本阶段交付 OpenAI 兼容 API 实现，通过本地离线快照和远程 `Qwen/Qwen3.5-4B` 检查 API、Agent 和 Benchmark 的接线。正式 2B 对照等服务器可用后进行。Qwen 原生聊天模板使用 XML 风格的调用和结果标签，兼容 API 服务负责将生成结果转换为 `message.tool_calls`。 [Qwen3.5-2B 模型卡](https://huggingface.co/Qwen/Qwen3.5-2B#serving-qwen35) · [官方聊天模板](https://huggingface.co/Qwen/Qwen3.5-2B/blob/main/chat_template.jinja)

## 请求与多轮执行

- 原始 ReAct 组沿用当前提示词、解析和执行路径。原生组从同一个工具注册表生成 `tools`，使用 `tool_choice=auto`，保留任务、可见 setup、论文指代和异步翻译规则，提示每轮最多一个工具调用。
- API 边界解析第一条候选消息的 `tool_calls`、调用 ID、工具名和 JSON 参数对象。既有工具参数校验检查类型、必填字段和额外字段。未知工具、参数错误、残缺调用、多调用和截断分别记录。末轮正常文本回复映射为内部 `FINISH`，最终文字供不可行任务评分。
- 检索工具的 `max_results` 是可选字段。4B 冒烟中 API 曾省略明确要求的篇数上限，调用通过 Schema 校验后使用了工具默认值 50。原生提示明确把用户给出的篇数和时间窗映射到 `max_results` 和 `days`，保留 Schema 定义及严格参数评分。
- Agent 保存已执行调用的 assistant 消息和带对应 `tool_call_id` 的工具结果消息。两组共用 Tool Dispatcher、会话注入、翻译副作用和迭代预算。工具执行错误作为 Observation 回填后，模型可以在剩余预算内继续决策。
- 裸 XML、空输出或异常结束由协议错误路径处理。服务必须能返回结构化调用。后续若在服务器直接使用 Transformers 推理，需要另接 Qwen 原生输出解码器。

## Benchmark 与诊断

- 用 `--agents regex native_tool_calling --offline` 调用同一个 Runner。任务定义、语义标准答案、Snapshot、Reward 公式与权重沿用现有代码。对照模式记录原始 API 候选及统一轨迹。
- 原始 ReAct 可能把解析失败当成 `FINISH`。实验只在计分轨迹标记可确认的误记，保留真实运行的终止信息。原有解析降级仍可执行，原生组的格式错误无法触发 ReAct 降级。
- 汇总原始格式解析率、Schema 合规率、工具序列准确率、参数准确率、执行错误率、Strict Success、调用次数、token 和延迟。格式以完成的模型响应为分母，Schema 以可解析调用为分母，执行错误以实际调用为分母。Strict Success 包括所有预定运行，API 故障单列。用原始轨迹核验纯格式问题，无法判断模型意图时保留未知。
- GPU 正式对照固定模型与 tokenizer revision、快照 SHA256、解码参数、工具定义和预算，使用 `v7_86` 的 dev、IID 和 OOD 共 31 项任务，每项运行三次。配对键为 `(task_id, trial)`，报告双向变化和各切分分母。API 生成种子需要在服务器确认，4B 接口 smoke 用于验证接线。

## 验收

本地以固定 API 响应验证两次连续执行、Observation 回填、最终回答、异常调用和共用评分，再使用指定 API 端点跑一组小规模离线任务。真实 API 结果记录请求模型、返回模型和实际执行轨迹。模型是否接受 Schema、是否生成结构化调用，以及 Benchmark 是否正确计分均以可观测结果判定。

在仓库根目录使用已配置的 API 环境运行以下冒烟测试。硅基流动端点的 `--no-thinking` 会为两组同时传递顶层 `enable_thinking=false`。 [硅基流动 Chat Completions 文档](https://docs.siliconflow.cn/docs/api/chat-completions-post)

```sh
python -m AgenticArxiv.benchmark.run_benchmark --backend api --offline \
  --agents regex native_tool_calling --model Qwen/Qwen3.5-4B \
  --snapshot data/mock_arxiv_snapshot.json --task-ids search_01 composite_01 \
  --repeat 1 --no-thinking --output outputs/native_tool_calling_smoke
```

`protocol_comparison.json` 记录协议指标、配对 Strict Success 变化和快照 SHA256，`traces.jsonl` 记录每次运行的原始 API 候选、工具轨迹和 Reward 分解。

### 2026-11-20 接口验证

硅基流动返回的模型 ID 是 `Qwen/Qwen3.5-4B`。关闭 thinking 后，扩展任务 `state_dl_then_cache` 在两组均通过 Strict Success。原生组执行下载和缓存检查，两个工具结果指向同一论文，随后正常结束。该下载由快照环境的离线桩执行。

`search_01` 与 `composite_01` 的一轮配对中，ReAct 严格成功 2/2，原生组 0/2。原生响应解析 7/7、Schema 合规 6/6。检索调用遗漏 `max_results` 并采用了默认值。复合任务随后重复提交快照不支持的关键词查询，达到轮次上限。该小规模试验验证 API 与打分接线，2B 的配对效果仍需在目标服务器测量。
