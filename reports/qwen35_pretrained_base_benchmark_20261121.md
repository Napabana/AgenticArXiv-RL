# Qwen3.5-2B-Base 原生 Tool Calling 全量评测

2026-11-21，纯预训练权重 `Qwen/Qwen3.5-2B-Base` 在 31 个任务上各运行 3 次，严格成功 **33/93（35.5%）**，格式解析 **290/292（99.3%）**。全量使用加入聊天轮结束符的提交 `6cbf8f9`。后续的 ID 解码修复在提交 `d37c93b` 上做了独立专项复测。 [全量指标与逐次结果](qwen35_pretrained_base_benchmark_20261121.json) · [ID 专项报告](qwen35_native_arxiv_id_parser_retest_20261121.md)

## 全量设置

模型从 ModelScope 下载。在单张 RTX 4090 上，用 PyTorch 2.8.0+cu128 和 Transformers 5.16.1 直接以 BF16 加载，Adapter 为空。评测使用 `native_tool_calling`、默认 zero-shot 提示、温度 0.1 和关闭 thinking 的聊天模板。每个任务和试次从基础种子 42 派生随机种子。工具执行沿用离线 `MockArxivEnv`、`v7_86` 任务切分和原严格评分器。权重、快照和本地原始结果压缩包的 SHA-256 收录在汇总 JSON 中。 [官方权重](https://modelscope.cn/models/Qwen/Qwen3.5-2B-Base) · [任务切分](../data/splits/v7_86.json)

全量运行于北京时间 10:38:06 至 11:11:25，总耗时 **33 分 19 秒**。三个切分正常退出，请求异常和 benchmark 运行异常均为 0。

## 指标

| 切分 | 严格成功 | 格式解析 | Schema 合规 | 工具序列准确 | 参数准确度 | 工具执行错误 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Dev | 7/24 | 68/68 | 48/53 | 10/24 | 44.8% | 16/48 |
| IID | 20/57 | 172/173 | 126/137 | 22/57 | 40.4% | 41/126 |
| OOD | 6/12 | 50/51 | 43/44 | 7/12 | 75.0% | 11/43 |
| **合计** | **33/93（35.5%）** | **290/292（99.3%）** | **217/234（92.7%）** | **39/93（41.9%）** | **46.0%** | **68/217（31.3%）** |

格式解析按模型响应轮次计数，Schema 合规按可解析的工具调用计数，工具执行错误按实际执行的调用计数。参数准确度是逐任务参数分的平均值，包含部分得分。严格成功要求任务正常结束，工具序列、参数、引用和终止语义符合标准，且无解析和工具执行错误。31 个任务中，8 个三遍全过，16 个三遍均未过，另外 7 个有部分试次通过。平均每次执行 **2.33** 个工具调用、使用 **9,675** 个总 token，LLM 耗时 **20.97 秒**。

## 生成边界与剩余失败

三道题的配对试跑使用相同的 Base 权重、任务、快照和种子。旧版 `d423b89` 和边界修复版 `6cbf8f9` 各运行 3 次，均为严格成功 0/3、格式解析 10/10、Schema 合规 8/9。新版记录了 9 次调用边界和 1 次聊天轮结束。同一次搜索的最终答案正文相同，并在 `<|im_end|>` 处结束，少生成了尾部 2 个 token。

- 全量 292 轮中，234 轮停在工具调用结束边界，56 轮停在聊天轮结束符，2 轮达到 token 上限。后两轮分别发生在 `opt_service` 和 `multi_ai5_dl1_tr`，模型在收到翻译提交结果后反复输出 `PENDING` 状态。
- 17 次 Schema 拒绝包括 9 次分类枚举值错误、2 次传入该工具不认识的 `aspect` 字段、3 次调用未知工具、1 次 `question` 枚举值错误，以及 2 次由 XML 解码造成的引用类型错误。后两次在专项报告中逐条核对。 [ID 专项报告](qwen35_native_arxiv_id_parser_retest_20261121.md)
- 严格成功和格式解析的差距主要出现在模型选工具、传参和执行阶段。例如简单检索前先查不存在的缓存，或超出会话列表范围后反复调用缓存工具。历史后训练版在相同任务上的原生严格成功是 72/93。两次运行使用不同的推理代码提交，结果可作历史参考。 [历史后训练版原生评测](qwen35_native_tool_calling_fixed_benchmark_20261121.md)

## 复现命令

在对应代码提交的仓库根目录运行。将 `MODEL_DIR`、`SNAPSHOT`、`OUT` 指向本地权重目录、校验过的快照和独立输出目录。激活上文的 Python 环境。

```bash
export STORE_BACKEND=memory TOOL_ROUTER=policy HF_HUB_OFFLINE=1
for split in dev iid_test ood_test; do
  assets="$OUT/$split-assets"
  PDF_RAW_PATH="$assets/pdf_raw" PDF_TRANSLATED_PATH="$assets/pdf_translated" \
  PDF_TRANSLATED_LOG_PATH="$assets/pdf_logs" PDF_FIGURES_PATH="$assets/pdf_figures" \
  PDF_CACHE_PATH="$assets/pdf_cache" TRANSLATE_CACHE_PATH="$assets/translate_cache" \
  env -u VLM_ADAPTER_PATH -u AA_EVAL_GREEDY \
    python -u -m AgenticArxiv.benchmark.run_benchmark \
    --backend transformers --model "$MODEL_DIR" --local-device cuda --local-dtype bfloat16 \
    --agents native_tool_calling --offline --snapshot "$SNAPSHOT" --task-set expanded \
    --split "data/splits/v7_86.json:$split" --repeat 3 --seed 42 --no-thinking \
    --save-traces --output "$OUT/$split"
done
```
