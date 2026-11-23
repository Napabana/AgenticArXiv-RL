# Native GRPO GPU 启动前审计

状态为 `LOCAL_PASS`。实现基于 `feat/qwen35-native-grpo` 的 `ecab507a11ed7813cc28a3fbad93377eddeba8ad`，从 2026-11-21 的 Qwen3.5-2B-Base Native SFT Adapter 起训。

## 代码与环境

- Native Rollout 使用官方 Qwen 模板及 10 个 Native Tool Schema，每条轨迹的 MockArxivEnv、MemoryStore 和 PDF 桩目录独立。Setup 进入初始上下文，真实工具回包通过 `tool_call_id` 回填。每轮保留实际生成的 Assistant token，后续模板与工具反馈的 loss mask 为零。
- 五维 Reward 保持 `1 / 3 / 2 / 1 / 3`，课程步数为 0。Prefix-tree 使用 `progress`、权重 0.5、β 1.0，组内最终 Reward 有差异时优先使用 Reward Credit。整组含环境错误时排除 policy loss，含截断时关闭 Step Credit。
- WSL2 Ubuntu 22.04，Python 3.11.15，PyTorch 2.15.0+cpu。独立环境为 `~/.venvs/arxivfetch-native-grpo-cpu`，固定 Transformers 5.22.2、TRL 1.17.1、PEFT 0.21.2、Accelerate 1.15.0、Datasets 5.2.0。
- 从本分支根目录执行 `python -m pytest AgenticArxiv/tests/ -q --tb=short`，1014 通过、7 跳过、701 个子测试通过。基线复现的五个历史失败来自三处测试夹具，修复了临时文件读取时机、JEV 的冻结快照路径和 BenchmarkRunner 的初始化。
- CPU 集成测试使用微型 Qwen3.5，完成一条真实 TRL 优化步，验证工具和 padding mask、冻结 KL 参考 Adapter、有限 loss、LoRA 指纹变化和 Step Credit 守恒。两个 Bash 启动器通过 `bash -n`，变更的 Python 文件通过语法检查，实际 Qwen tokenizer 通过停止边界预检。

## 任务和数据

[候选清单](../data/splits/native_grpo_qwen35_candidates_v1.json) 含 18 个候选，8 个来自先前的 Train 模板，10 个按最新 Native SFT 的策略错误派生。31 个 Dev、IID 和 OOD 任务用于泄漏审计。检查任务 ID、规范化请求，以及 Setup、实际生效参数和终止条件的组合签名。新变体同时与冻结的 188 条 SFT 任务目录核对。每条专家轨迹均离线执行并达到严格成功、Reward 1.0 和 `phi=1.0`。PDF 下载以 Mock 回放的 READY 桩验证，会话缓存和译文就绪也由回放状态校验。

| 冻结资产 | SHA256 |
| --- | --- |
| Native SFT Adapter | `789af8cf28960892a1fde034d32f9b8ddf230e8450552be70d15024c964ed6c8` |
| Adapter 配置 | `95666744bf5e4034466c0ff539acdfabb14c147b29f6d7f6f89ad4a253381a5c` |
| SFT Train Manifest（冻结归档） | `34196a2639203ca290a43c1012daa1f82f4621a4f8beb2203c597713d013bf70` |
| Snapshot | `d97035db326ca11935e882fd2aca7fe932e9e76cf1bc8a21078be7bec1ae8794` |
| Base tokenizer.json | `fe000e3ed39ed12b8d2481d527d44f93c65d37e87645d2dcc80d1bf9d50d2927` |
| Native System Prompt | `31674f2183d642a9035ecf8360c138135c5ef4645904017eadf4ec7bdf4abf05` |
| 10 个 Tool Schema | `01ed14f7f8a041ca9752475c1bf1513996c375c5a32cadf2514089fb04401ae0` |
| Qwen Chat Template | `273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80` |
| Probe 候选清单 | `bbe1cb5bdf69dca8dd573ca7f06ea01979d604c121f7034ebdad96e5af54ed25` |

最长专家轨迹包含 4 次工具调用和 1 次最终回答。最长完整上下文为 4,798 tokens，最长 completion 为 1,378 tokens，最长 Assistant turn 为 637 tokens。实际训练设置为 5 轮、completion 1,792 tokens、每轮生成上限 704 tokens。

## GPU 阶段

首选上次实验使用的单张 RTX 4090（24 GB）。上次 312 步 ReAct GRPO 实测约 169 分钟，G=4、单卡 batch 4、completion 上限 1,024 tokens。本轮 Native completion 上限为 1,792 tokens，含工具反馈。24 GB 起步使用微批次 1、累积 4，整组采样继续使用 G=4。上机以 Smoke 的峰值显存和吞吐核验。

历史 SFT 机器的 Base 路径为 `/root/sj-tmp/models/Qwen3.5-2B-Base`，Snapshot 路径为 `/root/sj-tmp/AgenticArXiv-RL/data/mock_arxiv_snapshot.json`。GPU 预检运行时核对实际 Base 文件和 Adapter 路径、SHA256、Torch、CUDA 和驱动。Base 缓存不齐时按 [已验证的 ModelScope 文件清单](native_sft_final_20261121/base_model_modelscope_files.json) 恢复。

Probe 首轮 72 组、288 条轨迹。只有首轮 1/4 组有信号的边缘候选补采到 8 组。Probe 通过后跑 4 步 Smoke，再从冻结 SFT Adapter 起跑 100 步训练，每 25 步保存。使用单进程、G=4、每个优化步 4 条轨迹、学习率 `5e-6`、KL β `0.04`、seed 42 和 `step_credit=progress`。四个检查点先测 Dev，再在选定检查点与冻结 SFT Adapter 上完成 93 试次的同协议配对 Benchmark。

Base、Adapter 和 Snapshot 在实例本地已有缓存时，预计连续占用 **3–5 小时**，包含环境校验、Probe、Smoke、正式训练、选模和配对评测。同机历史 Native SFT 单次 93 试次约 36 分钟，旧 GRPO 每步约 32.5 秒。实际租用时间随本轮 Native 轨迹长度和实例吞吐变化，Smoke 后重新估算。GPU 所有阶段当前为 `NOT_RUN`。

在目标机的独立 GRPO Python 环境中，从分支根目录执行以下命令。由流水线按顺序推进，结果路径应是新目录。

```bash
GRPO_MICRO_BATCH=1 PYTHON_BIN="$GRPO_PYTHON" bash scripts/run_qwen35_native_grpo_pipeline.sh \
  "$BASE_MODEL" "$NATIVE_SFT_ADAPTER" "$SNAPSHOT" \
  data/splits/native_grpo_qwen35_candidates_v1.json "$RUN_DIR"
```
