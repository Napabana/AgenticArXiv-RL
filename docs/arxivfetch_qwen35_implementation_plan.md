# ArxivFetch-RL：Qwen3.5-2B 适配与 SFT→GRPO 复现计划

> 本文是本轮开发分支的实施计划与验收记录。目标是验证 Qwen3.5-2B 能否在单张 RTX 4090 24GB 上，接入本仓库现有 arXiv 多轮工具 Agent，并完成 SFT→GRPO；**不预设成功，不把上游已有能力写成本 Fork 的原创实现**。
>
> 分支：`feat/arxivfetch-qwen35-2b`。基础分支：`feat/qwen3-1p7b-posttrain`。本分支继承此前本地推理 `chat_template_kwargs` 转发修复与回归测试。

## 1. 本轮目标

1. 保留原有 arXiv 任务环境：论文搜索、元数据读取、下载/缓存、翻译/阅读等工具及 MockArxivEnv。
2. 将策略模型首选设为 `Qwen/Qwen3.5-2B`，优先尝试 BF16 LoRA；不默认使用 QLoRA。
3. 先完成架构/依赖/Tokenizer/Chat Template/LoRA target 的兼容性检查，再接入 SFT，最后接入本仓库自定义多轮 GRPO rollout。
4. 用同一套离线快照、任务切分、ReAct parser 和解码协议对比 Base、SFT、GRPO。
5. 只有前三阶段跑通后，才考虑增加安全扩展（恶意 Tool Observation、越权工具请求、安全奖励与 ASR 评测）。

## 2. 模型与实现事实

### 2.1 官方模型事实（外部文档，实施时再次核对版本）

- 模型 ID：`Qwen/Qwen3.5-2B`；模型卡声明约 2B 参数、262,144 原生上下文、图文输入能力，使用混合 Gated DeltaNet / Gated Attention 架构。
- Transformers 文档区分文本生成类 `Qwen3_5ForCausalLM` 与完整多模态类 `Qwen3_5ForConditionalGeneration`；本项目的 arXiv 任务只输入文本和工具 Observation，优先评估纯文本模型路径，避免无必要地加载视觉塔。
- Qwen3.5 需要较新的 Transformers 实现。仓库当前依赖锁定可能不包含该架构，必须以安装版本实际 `AutoConfig` / `AutoModelForCausalLM` 映射测试为准。
- 不预设 Qwen3.5 可直接用旧 Qwen/Qwen2.5 的 4-bit QLoRA 配方。第一条训练路径优先验证 BF16 LoRA；若 24GB 显存不够，再测更短序列、micro-batch=1、gradient checkpointing，最后才考虑其他量化后端。
- 不能把“模型可推理”或“TRL 官方有该模型示例”等同于“本仓库的自定义多轮 GRPO 已兼容”。

参考：
- https://huggingface.co/Qwen/Qwen3.5-2B
- https://huggingface.co/docs/transformers/main/model_doc/qwen3_5
- https://huggingface.co/docs/trl/grpo_trainer

### 2.2 当前仓库的潜在适配点（需逐项实测）

| 位置 | 当前实现/风险 | 本轮处理 |
|---|---|---|
| `rl/train_sft.py` | 默认 `AutoModelForCausalLM` + 固定 QLoRA target modules | 检查文本模型 AutoClass、BF16 LoRA、目标模块命中和反向传播 |
| `rl/train_grpo.py::_load_policy` | QLoRA 路径按 CausalLM 加载；VLM 路径走 ConditionalGeneration + BF16 LoRA | 为 Qwen3.5 选择文本模型类；验证 PEFT 包装后 TRL Trainer 能取 logits、logprobs 和梯度 |
| `rl/grpo_reward.py` | 自定义多轮 rollout，逐轮调用 `trainer._generate_single_turn`，工具 Observation 追加到历史 | 验证 hybrid cache / generation kwargs / 多轮状态连续性；确认 env tokens 不计入策略 loss |
| `utils/llm_client.py` | 依据 config 判断 VLM，并在 VLM/文本路径间分流 | 明确 Qwen3.5 text-only 与 multimodal checkpoint 的加载策略，避免误加载视觉路径 |
| `rl/train_vlm_sft.py` | 已有视觉模型 BF16 LoRA 及动态 target resolver | 仅复用模块发现思路，不假设它已兼容 Qwen3.5 |
| Canary / Stage Verifier / Benchmark | 需要加载训练后的 Adapter 并重放 ReAct 对话 | 统一 tokenizer、chat template、stop token 和 Adapter base revision |

## 3. 执行阶段与验收标准

### P0 — 架构与依赖探测（本轮首先执行）

- [ ] 固定基础环境版本：Python、PyTorch、CUDA、Transformers、TRL、PEFT、Accelerate、bitsandbytes。
- [ ] 读取 Qwen3.5-2B config：model_type、architectures、text_config、vision_config、layer_types、tie_word_embeddings。
- [ ] 验证 tokenizer / processor 的 chat template；确认工具对话和非 Thinking 设置如何传递。
- [ ] 验证 `AutoModelForCausalLM` 能否加载文本路径；如需显式 `Qwen3_5ForCausalLM` 或构造 text config，记录依据与实现方式。
- [ ] 若在单卡加载完整 ConditionalGeneration 会引入无用视觉塔，优先采用官方支持的 text-only 模型类；禁止自行删除 config 字段或修改权重结构来“骗过”AutoClass。
- [ ] 实际加载 BF16 模型并打印参数量、模块名称、显存峰值；确定 LoRA target modules，确认 trainable params > 0。
- [ ] 产出 `artifacts/qwen35_compatibility_report.json`，记录版本、加载类、模块命中、显存和错误信息。

**P0 通过条件：** 单卡 4090 上模型可加载、文本生成可用、BF16 LoRA 注入成功，且一次 forward/backward/optimizer step 成功。仅 tokenizer/config 加载成功不算通过。

### P1 — SFT 最小训练与正式训练

- [ ] 确认 SFT 样本中的 assistant target、tool action 与 FINISH 格式符合 tokenizer 的 chat template。
- [ ] 先跑 1–3 step smoke：micro-batch=1、短 max_length、gradient checkpointing；验证 loss 有限、梯度非空、Adapter 可保存和重新加载。
- [ ] 跑完整 SFT，记录 seed、数据 manifest/hash、训练参数、耗时、峰值显存、训练曲线。
- [ ] 评测 Base 与 SFT：Strict Success、Tool Accuracy、Argument Accuracy、False Finish、Parse Error、平均 Token 数。
- [ ] 抽样人工检查至少 20 条失败轨迹，按错误工具、错误参数、未调用工具、过早结束、格式错误分类。

**P1 通过条件：** 可恢复 Adapter + 完整训练日志 + Base/SFT 同协议评测报告。

### P2 — 自定义多轮 GRPO 兼容与训练

- [ ] 用 SFT Adapter 执行 frozen-policy probe（学习率为 0），确认每个 prompt 的 4 条 rollout 能完成多轮工具交互。
- [ ] 检查 GRPO 的 policy log-prob 只覆盖模型生成 token；工具 Observation token 的 env_mask 为 0。
- [ ] 检查 reward 组内方差、zero-std 比例、reward hacking、过早 FINISH 与 parse failure。
- [ ] 从 train-only 候选任务生成 Qwen3.5 专属 GRPO split；禁止使用 dev/IID/OOD 选择训练任务。
- [ ] 先跑 1 次 optimizer update，再跑 10–30 step smoke；通过后才扩大到 60+ steps。
- [ ] 记录完整 rollout traces、reward components、KL、梯度范数、显存峰值与 checkpoint。

**P2 通过条件：** 完成至少一次有效参数更新，loss/梯度有限，Adapter 能保存重载，Rollout 与 reward trace 可复查。单纯生成轨迹不算 GRPO 训练成功。

### P3 — 对照评测与安全扩展

- [ ] 固定同一 Benchmark split、offline snapshot、seed、decode config，比较 Base / SFT / GRPO。
- [ ] 分别报告 dev、IID、OOD 的样本分母；现有 OOD 任务数量有限，不夸大泛化结论。
- [ ] 做至少一组 reward ablation（Outcome-only vs 多维 RLVR），控制 rollout 与 optimizer budget。
- [ ] 可选安全扩展：合成恶意 Tool Observation、工具权限越界请求、可信指令/不可信观察边界；增加 GRPO+Safety 训练臂。
- [ ] 安全评测报告 Attack Success Rate 与 benign task success 的变化，检查是否通过过早拒答/提前 FINISH 虚假降低 ASR。

## 4. 4090 24GB 资源策略

初始实验配置（仅作为 smoke 起点，不作为保证）：

- BF16 base model + LoRA；rank 8 或 16；micro-batch=1。
- SFT max_length 先设 1024/2048；通过长度统计后再提高。
- gradient checkpointing；关闭 KV cache（训练时）。
- GRPO：先 1 prompt × 4 generations、2 turns、短 completion；通过后再增加并发和上下文。
- 每阶段记录 `torch.cuda.max_memory_allocated()` 与 `max_memory_reserved()`。
- 若显存不足，按顺序减少 sequence length、completion length、generation 数、rollout batch；不能直接假定 gradient accumulation 能降低 rollout KV cache 峰值。
- 只有在 BF16 LoRA 路径稳定后，才评估兼容 Qwen3.5 的量化方案；不要默认套用旧 QLoRA 脚本。

## 5. 简历成果的真实性边界

最终可写入简历的内容必须能由本 Fork 的 commit、训练 config、数据 manifest、模型 Adapter、训练日志、rollout traces 和评测报告追溯。

在实验完成前：
- 不写“已在 RTX 4090 完成训练”；
- 不填写成功率提升、ASR 降低或显存数字；
- 不把上游已有的 arXiv 环境、10 个工具、多轮 GRPO trainer、五维 reward 写成本人从零设计；
- 将本 Fork 的真实增量限定为：Qwen3.5 模型适配、训练协议一致性、Qwen3.5 专属 probe/split、复现实验，以及后续新增的安全训练与评测。

## 6. 本轮开发任务列表

1. 新增模型兼容性探测脚本（本 PR 首个代码改动）。
2. 运行静态检查并补单测；记录当前仓库依赖与 Qwen3.5 所需依赖差异。
3. 根据探测结果决定是否修改 `train_sft.py`、`train_grpo.py` 和 `llm_client.py`；不做无证据的大规模重构。
4. 用真实 GPU 结果更新本文件的 P0/P1/P2 状态和报告链接。
