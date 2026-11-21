# 原生 Tool Calling 的 arXiv ID 解码与专项复测

在纯预训练 `Qwen/Qwen3.5-2B-Base` 的全量运行中，两次合法的 XML 调用因解码器把 `2608.14528` 转成浮点数而未通过 Schema。提交 `d37c93b` 修复了类型推断。专项复测中，**15/15 个调用通过 Schema，严格成功仍为 0/3**。 [专项数据](qwen35_native_arxiv_id_parser_retest_20261121.json) · [全量报告](qwen35_pretrained_base_benchmark_20261121.md)

## 原因与修复

服务器上重新渲染的 Base 聊天模板包含全部 10 个工具。工具名、说明、参数 Schema 与注册表逐项一致。Dev 的 `ref_ctrl_id_download` 在试次 1 和 2（从 0 开始计数）生成了正确闭合的 XML，其第二轮调用包含 `<parameter=ref>2608.14528</parameter>`。工具 Schema 允许整数序号、字符串引用和 `null`。旧解码器对该联合类型直接尝试 `json.loads`，产生浮点数，随后的校验报 `schema:invalid_value:ref`。这两轮的格式解析均已通过。

新版在数字字面值被解出为浮点数、字段又允许字符串且不接受 `number` 时，保留原始字符串。由此保留 ID 的原始位数和末尾零。整数序号、`null`、带版本后缀的字符串引用和工具执行继续按原契约处理。本地定向回归 126 项及 8 个子测试通过。服务器用两条历史原始输出重放，均得到字符串引用并通过 Schema。 [解码实现](../AgenticArxiv/utils/llm_client.py) · [回归测试](../AgenticArxiv/tests/test_native_tool_calling.py)

## 单题 GPU 专项

任务要求直接调用 `download_arxiv_pdf(ref="2608.14528v1")`。用同一模型、RTX 4090、BF16、同一 Mock 快照和基于任务与试次的种子重新生成 3 次。旧列取自全量中的同一道题，新列是独立专项运行。专项于北京时间 11:12:08 至 11:13:34 完成。

| 指标 | 全量中的同题，`6cbf8f9` | ID 修复后专项，`d37c93b` |
| --- | ---: | ---: |
| 严格成功 | 0/3 | 0/3 |
| 格式解析 | 9/9 | 15/15 |
| Schema 合规 | 7/9 | 15/15 |
| 实际执行的调用 | 7 | 15 |
| 工具执行错误 | 7/7 | 15/15 |
| 终止状态 | 2 次 `ERROR`，1 次 `FORCE_STOP` | 3 次 `FORCE_STOP` |

试次 1 和 2 的第二轮调用现在通过校验并执行，整个专项共有 8 次无版本引用以字符串形式通过校验。模型仍然先发起未命中快照的关键词搜索，随后用缺少 `v1` 的 ID 查缓存或下载，最终耗尽轮数。修复消除了类型层面的拦截，任务成功仍取决于模型调用指定工具并保留完整 ID。全量的 33/93 对应 `6cbf8f9`，专项的 0/3 对应 `d37c93b`。

## 历史结果核查

- **后训练版 `Qwen/Qwen3.5-2B` 原生评测**。检查生成边界修复前的 232 轮和修复后的 273 轮原始输出，均未发现无版本十进制 XML `ref` 或解码后的浮点引用。历史的 21 次格式失败由 12 次续写工具响应、6 次生成触顶和 3 次单轮多调用构成。 [历史原生评测](qwen35_native_tool_calling_benchmark_20261121.md) · [边界修复后评测](qwen35_native_tool_calling_fixed_benchmark_20261121.md)
- **ReAct Base、SFT、GRPO**。每组核查 93 条持久化轨迹。ReAct Base 的动作中浮点 `ref` 为 0 次，SFT 和 GRPO 各有 9 次，分别集中于 `ref_ctrl_id_download` 的 3 次试验。历史 ReAct 证据为持久化动作和工具执行记录，解析路径是 `Action` JSON 的 `json.loads`。这条路径保留 JSON 字符串的类型。 [历史训练评测](qwen35_sft_grpo_training_analysis_20261118.md#图-6同协议留出-benchmark)

## 复现命令

使用 `d37c93b` 的仓库根目录和与全量相同的环境变量，另设独立输出目录。 [运行环境与路径变量](qwen35_pretrained_base_benchmark_20261121.md#复现命令)

```bash
assets="$OUT/ref-check-assets"
PDF_RAW_PATH="$assets/pdf_raw" PDF_TRANSLATED_PATH="$assets/pdf_translated" \
PDF_TRANSLATED_LOG_PATH="$assets/pdf_logs" PDF_FIGURES_PATH="$assets/pdf_figures" \
PDF_CACHE_PATH="$assets/pdf_cache" TRANSLATE_CACHE_PATH="$assets/translate_cache" \
env -u VLM_ADAPTER_PATH -u AA_EVAL_GREEDY \
  python -u -m AgenticArxiv.benchmark.run_benchmark \
  --backend transformers --model "$MODEL_DIR" --local-device cuda --local-dtype bfloat16 \
  --agents native_tool_calling --offline --snapshot "$SNAPSHOT" --task-set expanded \
  --task-ids ref_ctrl_id_download --repeat 3 --seed 42 --no-thinking \
  --save-traces --output "$OUT/ref-check"
```
