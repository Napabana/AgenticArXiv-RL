# Qwen3.5-2B 原生 Tool Calling 留出评测

2026-11-21，未加载 Adapter 的 `Qwen/Qwen3.5-2B` 使用原生 Tool Calling 在 31 个留出任务上各跑三次，严格成功 **60/93（64.5%）**。三个采样种子下通过的都是同一批 20 个任务，另有 11 个任务三遍均失败。历史同任务 ReAct Base 的严格成功率为 38.7%，ReAct + SFT 为 74.2%，ReAct + SFT + GRPO 为 77.4%。[本次汇总数据](qwen35_native_tool_calling_benchmark_20261121.json) · [历史评测](qwen35_sft_grpo_training_analysis_20261118.md#图-6同协议留出-benchmark)

## 评测设置

原生组使用实验提交 `0ce41ace8cb8b177d9b7d837cae25c5c64645419`，在单张 RTX 4090 上用 BF16 加载 `Qwen/Qwen3.5-2B`。本地 API 基于 Transformers 5.16.1，按 Qwen 聊天模板将生成的工具调用转成 `message.tool_calls`，并将真实工具结果以 `role=tool` 回填。工具执行使用离线 `MockArxivEnv` 快照。

任务来自 `v7_86` 的 Dev、IID 和 OOD，严格成功沿用现有评分器。三遍使用请求序列种子 42、1042、2042，保持 `temperature=0.1`、关闭 thinking、每次生成最多 1,000 token。四组的任务 ID 和每题运行次数一致。历史三组运行 ReAct，本次只运行原生组，推理接入和随机流有差异。[实验设计](../docs/native_tool_calling_experiment.md) · [任务切分](../data/splits/v7_86.json)

## 严格成功对比

| 方案 | Dev | IID | OOD | 合计 | 三遍均通过的任务 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 历史 ReAct Base | 12/24 | 19/57 | 5/12 | 36/93（38.7%） | 3/31 |
| **原生 Tool Calling Base** | **15/24** | **39/57** | **6/12** | **60/93（64.5%）** | **20/31** |
| 历史 ReAct + SFT | 12/24 | 45/57 | 12/12 | 69/93（74.2%） | 23/31 |
| 历史 ReAct + SFT + GRPO | 12/24 | 48/57 | 12/12 | 72/93（77.4%） | 24/31 |

相对历史 ReAct Base，原生组多成功 24 次，其中 IID 贡献 20 次。按各任务三遍的成功次数比较，原生组在 18 个任务上更高、8 个持平、5 个更低。与历史 SFT、GRPO 相比，原生组分别少成功 9 次和 12 次，差距主要出现在 IID 和 OOD。相对历史 Base 的 25.8 个百分点属于跨运行观察差值，协议本身的效果仍需在匹配的推理设置下测量。

| 指标 | 历史 ReAct Base | 原生 Tool Calling Base | 历史 ReAct + SFT | 历史 ReAct + SFT + GRPO |
| --- | ---: | ---: | ---: | ---: |
| 工具序列准确率 | 50/93（53.8%） | 66/93（71.0%） | 79/93（84.9%） | 81/93（87.1%） |
| 平均总 Token／任务 | 8,505 | 7,366 | 6,982 | 6,998 |

## 原生工具调用失败解析

- **模型续写了工具结果。** 12 轮先输出完整工具调用，又自行生成 `<tool_response>`。本次 API 解析器要求调用后的输出也符合调用协议，因此拒绝整轮。其中 8 轮的首个调用符合当步预期的工具和参数，例如 `analyze_ai5_ref4_desc` 生成了 `extract_paper_figures(ref=4)`。该计数只核对当步调用，后续任务结果仍待实测。
- **生成预算和每轮调用限制。** 6 轮达到生成上限，3 轮返回了多个调用。`state_cache_two` 的两项缓存查询同轮返回，触发了每轮一个调用的限制。
- **Schema 与任务约束。** 3 轮传入 `aspect="cs.CL"`，工具定义要求 `"CL"`。还有 9 条轨迹正常结束但严格评分失败，包括 `constraint_search_no_file` 漏传分类参数，以及 `state_dl_then_cache` 重做了 setup 已完成的检索。

格式解析通过 **211/232（90.9%）**，Schema 合规 **139/142（97.9%）**，实际工具执行失败 **3/139（2.2%）**，API 请求错误为 0。格式解析以模型响应轮次计数，Schema 合规以可解析调用计数，执行失败以实际执行调用计数。[计数与失败样本索引](qwen35_native_tool_calling_benchmark_20261121.json)
