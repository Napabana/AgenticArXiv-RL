# ArxivFetch-RL：Qwen3.5-2B 后训练适配与实验计划

> 状态：**实施中；本文件只记录目标、设计和验收标准，不代表训练已完成。**
>
> 开发分支：`feat/arxivfetch-qwen35-2b`。基于此前模型模板修复分支继续开发；不直接修改 `main`。
>
> 项目展示名：**ArxivFetch-RL**（面向 arXiv 论文检索、下载与处理的轻量 Agent 后训练）。
>
> 主模型：`Qwen/Qwen3.5-2B`。首选文本专用 `Qwen3_5ForCausalLM`，不是完整视觉语言 `Qwen3_5ForConditionalGeneration`。Agent 环境只使用文本工具，不需要加载视觉塔。

## 1. 本次改进目标

本次目标不是重写上游 Agent 或训练框架，而是在保持 ArxivFetch 工具环境、任务定义、奖励与多轮交互语义的前提下，增加 Qwen3.5-2B 的可复现后训练支持。

1. **模型适配**：用 Transformers 的 `Qwen3_5ForCausalLM` 加载 Qwen3.5-2B 文本主干，复用 AutoTokenizer；明确跳过视觉塔，避免将纯文本 Agent 误识别为必须加载 Processor 的 VLM。
2. **SFT**：在单张 RTX 4090 24GB 上优先尝试 BF16 LoRA，不沿用 Qwen3.5 的 4-bit QLoRA 默认路线。支持训练前数据检查、短步数冒烟、adapter 保存/重载和阶段验证。
3. **GRPO**：在同一 Qwen3.5 文本主干上加载 SFT Adapter，继续复用上游自定义多轮 `rollout_func`、`env_mask`、离线 MockArxiv 环境和可验证奖励；确认真实的反向传播与参数更新，而不只检查生成是否成功。
4. **输入一致性**：SFT、GRPO rollout、Canary、StageVerifier 和本地 Benchmark 统一使用 Qwen3.5 chat template 与 `enable_thinking=False`；多轮对话必须满足 prefix-preserving 要求。
5. **可复现评测**：固定模型 revision、依赖版本、数据切分、seed、解码参数和 Mock Snapshot；比较 Base / SFT / GRPO 的严格任务成功率、工具/参数准确率、False Finish、Token Cost 和 Reward 分量。
6. **后续独立贡献**：主训练链路稳定后，再新增恶意工具 Observation 注入与越权工具调用的对抗评测及安全奖励消融。该项不阻塞第一版 SFT/GRPO 复现。

## 2. 模型与架构决策

### 2.1 为什么使用文本专用模型类

Qwen3.5-2B 仓库的顶层配置为 `model_type=qwen3_5`，`architectures=[Qwen3_5ForConditionalGeneration]`，并同时包含 `text_config` 与 `vision_config`。但当前 arXiv Agent 的工具输入/输出都是文本。

Transformers 文档明确区分：
- 文本生成：`Qwen3_5ForCausalLM` + `AutoTokenizer`
- 图像/视频输入：`Qwen3_5ForConditionalGeneration` + `AutoProcessor`

因此，本项目应选择前者，避免无意义地加载视觉编码器。不能只依赖 `AutoModelForCausalLM` 根据顶层 `architectures` 自动推断；模型加载器需要显式识别 Qwen3.5 并选择文本类。

### 2.2 混合注意力与 LoRA 目标

Qwen3.5-2B 文本主干包含 24 层，按 3:1 混合 Gated DeltaNet 线性注意力与 Full Attention。LoRA 目标不能只沿用旧模型的 `q_proj/k_proj/v_proj/o_proj/gate_proj/up_proj/down_proj`，否则会漏掉线性注意力层的投影。

第一版用运行时模块扫描，覆盖文本主干中所有合适的 `nn.Linear`，排除 `lm_head` 和 embedding；记录实际命中的完整模块名及数量。至少应覆盖：
- Full Attention：`q_proj/k_proj/v_proj/o_proj`
- MLP：`gate_proj/up_proj/down_proj`
- Gated DeltaNet：`in_proj_qkv/in_proj_z/in_proj_b/in_proj_a/out_proj`

不对 Conv1d、Norm、Embedding 或 LM Head 注入 LoRA。若模块扫描结果与预期不符，立即失败并打印命中清单，不允许静默漏配。

### 2.3 量化与显存

Qwen3.5 第一版使用 **BF16 LoRA**：
- 不启用 bitsandbytes 4-bit QLoRA；
- 不进行全参数微调；
- 单卡 `device_map={"": 0}`；
- 起步 `r=16`、`alpha=32`、`dropout=0.05`；
- Gradient Checkpointing 开启；
- 首次 smoke 将 max sequence length 控制在 1024–2048，batch size=1。

Unsloth 官方 Qwen3.5 Fine-tuning Guide 给出 2B BF16 LoRA 约 5GB 的参考显存，并明确不推荐 Qwen3.5 4-bit QLoRA。该数字不是本项目端到端 GRPO 显存保证；GRPO 还包含 rollout、log-prob、reference policy、激活和优化器开销，必须在 4090 上实测。

## 3. 依赖策略

Qwen3.5 要求 Transformers v5。上游现有依赖下限 `transformers>=4.45.0`、`trl>=0.28.0` 过于宽松，不能保证 Qwen3.5 与自定义 rollout API 兼容。

实施时：
1. 使用 Python >=3.11。TRL 1.17.1 的包元数据要求 Python >=3.11。
2. 使用 CUDA 匹配的 PyTorch >=2.6.0。TRL 1.17.1 的运行依赖要求 torch>=2.6.0；先在 AutoDL 镜像中核对 torch/CUDA 组合，不要让安装训练依赖时意外替换现有 CUDA wheel。
3. 先建立独立环境，不污染用户其他项目；在首次成功运行前不宣称环境已验证。
4. Transformers 至少使用包含 Qwen3.5 文本模型实现的 v5 版本；正式环境记录确切版本。
5. TRL 选择保留 `GRPOTrainer(rollout_func=...)` API 的版本；先检查当前构造函数、`env_mask` 和自定义 Rollout 返回结构，再冻结版本。
6. PEFT 版本必须支持 Transformers v5 和 Qwen3.5 文本模型；使用已验证的精确版本。
7. Gated DeltaNet 的 `causal_conv1d` / `flash-linear-attention` 属于性能依赖。缺少时 Transformers 可能回退到更慢、显存更高的 PyTorch 实现；首轮先验证功能，再决定是否安装加速依赖。
8. 不在本次无差别升级所有依赖。使用独立 requirements 文件固定直接依赖；首次环境验证通过后，额外保存完整 pip freeze（含传递依赖）作为实验锁文件，再决定是否合并到主 requirements。

## 4. 代码改造范围

| 文件 | 本次改进 |
|---|---|
| 新增 `AgenticArxiv/rl/model_loading.py` | 模型架构识别、Qwen3.5 文本类加载、BF16 dtype/device map、LoRA 目标模块扫描 |
| `AgenticArxiv/rl/train_sft.py` | 新增显式 `bf16_lora` 模式；Qwen3.5 使用文本模型类；保存训练方法、目标模块、版本和显存 Manifest |
| `AgenticArxiv/rl/train_grpo.py` | 新增 BF16 LoRA GRPO；从 Base 加载文本模型并附加 SFT Adapter；支持恢复可训练 Adapter |
| `AgenticArxiv/rl/grpo_reward.py` | 多轮 Rollout 的 tokenizer 模板显式关闭 Thinking；保留工具 observation 的 env mask 语义 |
| `AgenticArxiv/utils/llm_client.py` | Qwen3.5 识别为文本策略，加载 `Qwen3_5ForCausalLM` + AutoTokenizer；不误走 VLM Processor |
| `AgenticArxiv/rl/canary.py`、`stage_verifier.py` | 与训练和 Benchmark 使用一致的模板参数 |
| `AgenticArxiv/tests/` | 模型类选择、LoRA target、Adapter 加载、模板参数和 rollout 接口回归测试 |
| 新增 `requirements-qwen35.txt` | 固定直接训练依赖版本；首次环境验证后再生成完整 freeze 锁文件 |

### 4.1 训练接口约束

新增显式 `--bf16_lora`，与现有 `--qlora` 互斥：
- 旧模型：保持现有 QLoRA 默认行为。
- Qwen3.5：必须使用 `--bf16_lora`；若误传 `--qlora`，给出明确错误，不静默改训练方法。
- `--no-qlora` 仍表示原有全参数路径，不能被误解为 BF16 LoRA。

SFT 和 GRPO 两个脚本都必须在 Manifest 中记录 `training_method=bf16_lora`、模型类、LoRA 目标模块、参数量、dtype、seed、依赖版本和 peak VRAM。

### 4.2 GRPO Adapter 约束

GRPO 继续训练 SFT Adapter 时：
1. 从 Adapter 配置解析并校验 base model；
2. 用 BF16 加载 Qwen3.5 文本模型；
3. 用 `PeftModel.from_pretrained(..., is_trainable=True)` 加载 SFT Adapter；
4. 确认仅 LoRA 参数可训练；
5. 当 beta > 0 且传入已训练 PEFT Adapter 时，TRL 1.17.1 应复制 default Adapter 为冻结的 ref Adapter；这样 KL reference 才是初始 SFT policy，而不是未经过 SFT 的 base model。PEFT 版本必须 >=0.20.0。训练器初始化后检查 ref 是否存在、是否与 default 权重逐张量相同、是否冻结，以及 default 是否仍为 active policy。
6. 训练一步后确认 Adapter 权重发生变化、非 LoRA 参数未变化。

## 5. 分阶段实施与验收

### P0：模型加载与环境兼容

- [ ] 新建统一模型加载模块，识别 `model_type=qwen3_5`。
- [ ] 用 `Qwen3_5ForCausalLM` + `AutoTokenizer` 做本地/Hub 加载。
- [ ] 确认不加载 vision tower；统计模型参数与 CUDA 显存。
- [ ] 扫描并验证完整 LoRA target modules。
- [ ] 确定 Transformers / TRL / PEFT / PyTorch 兼容版本并锁定。
- [ ] 单元测试覆盖模型选择和 target modules。

**验收：** Qwen3.5-2B 能以 BF16 文本模型加载到单张 4090，完成一次文本生成；LoRA target 覆盖 Gated DeltaNet、Full Attention 和 MLP。

### P1：BF16 LoRA SFT

- [ ] 复用原始 SFT 数据和 Manifest；检查样本、长度与数据 hash。
- [ ] 验证 SFTTrainer 默认模板与显式 `enable_thinking=False` 输出一致（不假设 SFTConfig 支持该参数）。
- [ ] 运行 1–3 step smoke，确认 loss 有限、反向传播成功。
- [ ] 保存 Adapter、Tokenizer、Manifest；重新加载 Adapter 并生成合法 ReAct Action。
- [ ] 再执行 30-step smoke，检查 peak VRAM、step time 和训练曲线。
- [ ] 通过后才运行正式 SFT。

**验收：** Adapter 可独立保存/重载；训练参数确实更新；StageVerifier 可解析输出。

### P2：BF16 LoRA GRPO

- [ ] 从 SFT Adapter 恢复 Qwen3.5 文本模型。
- [ ] 对多轮 Rollout 显式传入 `enable_thinking=False`。
- [ ] 验证 TRL 自定义 `rollout_func`、返回的 `prompt_ids/completion_ids/logprobs/env_mask` 结构兼容。
- [ ] 单次 rollout 中执行至少一次真实工具交互，并验证 observation mask=0、policy token mask=1。
- [ ] 执行 1 optimizer-step GRPO smoke，确认 reward、advantage、grad_norm 有限且 LoRA 权重变化。
- [ ] 记录 reference adapter/KL 行为；确认没有重复初始化或错误 reference policy。
- [ ] 再做 10–30 step smoke，评估显存和训练速度。
- [ ] 通过后才运行正式 GRPO 与冻结策略奖励方差探测。

**验收：** 不只是生成成功；必须证明一次 optimizer step 后 LoRA Adapter 权重变化，且非 LoRA 参数保持冻结。

### P3：统一评测和实验报告

- [ ] Base / SFT / GRPO 使用相同任务切分、Mock Snapshot、seed 和解码配置。
- [ ] 在 dev / IID / OOD 上报告 strict success、tool accuracy、argument accuracy、false finish、平均 token 和 reward 组成。
- [ ] 所有指标报告样本数与分母；OOD 任务数量少时不得夸大泛化结论。
- [ ] 至少分析 20 条失败轨迹并归类。

### P4：安全扩展（第一版训练链路稳定后）

- [ ] 构建合成恶意 Observation 注入与越权工具调用任务。
- [ ] 设计安全约束奖励或拒绝/终止策略。
- [ ] 对比 GRPO 与 GRPO+Safety 的攻击成功率及正常任务成功率。
- [ ] 使用独立攻击模板留出集，避免训练/测试泄漏。

### AutoDL RTX 4090 首轮执行命令

以下命令从仓库根目录执行，只用于 smoke，不代表正式训练配置已经验证。

**前置检查：**
- 使用 Python >=3.11 和 CUDA 匹配的 PyTorch >=2.6.0。
- 先安装仓库基础依赖，再安装根目录的 `requirements-qwen35.txt`。
- 默认 SFT 数据为 `data/sft/sft_v3_train_mix.jsonl`，并要求同名 `.manifest.json`。如果该文件不存在，只有在原始数据、parametric 数据及各自 manifest 都已准备好时，才运行 `python scripts/build_sft_train_mix.py`。
- 默认 GRPO 离线快照为 `data/mock_arxiv_snapshot.json`；缺失时运行 `python -m AgenticArxiv.rl.build_snapshot`。

**1. 先做不加载模型的 SFT 数据预检：**

```bash
python -m AgenticArxiv.rl.train_sft --model Qwen/Qwen3.5-2B --no-qlora --bf16_lora --inspect_only --max_length 1024
```

该命令检查数据 manifest、样本长度和 Qwen3.5 Chat Template，不加载模型、不开始训练。

**2. SFT 单步 smoke：**

```bash
python -m AgenticArxiv.rl.train_sft --model Qwen/Qwen3.5-2B --no-qlora --bf16_lora --max_steps 1 --batch_size 1 --grad_accum 1 --max_length 1024 --output_dir outputs/qwen35_sft_smoke --no-verify
```

通过条件：完成至少一个 optimizer step、loss 有限、Adapter fingerprint 变化、Adapter 文件和 Manifest 成功保存。

**3. GRPO 单步 smoke：**

```bash
python -m AgenticArxiv.rl.train_grpo --model outputs/qwen35_sft_smoke/final --no-qlora --bf16_lora --max_steps 1 --batch_size 4 --grad_accum 1 --num_generations 4 --max_completion_length 256 --max_turns 4 --output_dir outputs/qwen35_grpo_smoke --canary_steps 0 --no-verify --save_rollout_traces
```

这里 `batch_size=4`、`num_generations=4` 对应一个 prompt 的 4 条 rollout；不要把 batch_size 改成 1，否则无法满足 GRPO 的 generation group 约束。检查输出中的 reference adapter 校验信息、optimizer step、Adapter fingerprint、reward 方差和 rollout traces。

**4. 保存完整环境信息：**

```bash
python -m pip freeze > outputs/qwen35_environment.freeze.txt
```

smoke 通过后再扩大训练步数、任务集和 completion budget。若轨迹频繁被截断，先检查 `rollout_traces.jsonl` 中 Observation 与 FINISH 是否因总 completion budget 不足而丢失，不要直接增加训练步数。

---

## 6. 风险与回退条件

| 风险 | 判定信号 | 处理 |
|---|---|---|
| Transformers/TRL API 不兼容 | 自定义 rollout 初始化失败或 API 参数缺失 | 锁定兼容版本，不先重写整个 GRPO trainer |
| Qwen3.5 模型类/配置不兼容 | 文本模型类不能从官方 checkpoint 正确加载 | 先复现官方最小加载代码；不退回 VLM 全模型作为默认 |
| LoRA 漏掉 Gated DeltaNet | target list 缺少 `linear_attn.in_proj_*` | 动态扫描并断言覆盖，不沿用旧固定 target list |
| 4090 显存不足 | SFT/GRPO OOM | 降低 sequence length、generation 数和 rollout batch；保留有效 batch 设计，不先降模型 |
| GDN kernel 构建失败 | causal-conv1d / FLA 编译或 CUDA ABI 报错 | 首轮测试 PyTorch fallback；若慢/高显存，再按 PyTorch/CUDA 版本安装兼容 kernel |
| GRPO 没有有效学习信号 | 零方差、梯度为零、Adapter 无变化 | 检查 reward 设计、采样温度、任务难度和有效组比例，不增加训练步数掩盖问题 |

如果 Qwen3.5 在完成 P0 适配后仍无法在单张 4090 上跑通 1-step GRPO，则暂停该模型路线，保留 Qwen3.5 SFT 结果和故障证据，评估 Qwen3-1.7B 回退；不得把推理可用等同于 GRPO 可训练。

## 7. 结果与简历口径

所有数字均为**待实测**。没有真实训练日志、Manifest、Checkpoint 和固定协议下的 Benchmark 报告，不写提升百分比。

第一版完成后，简历只写实际做到的：
- Qwen3.5-2B 文本模型适配与 BF16 LoRA SFT；
- 自定义多轮工具 Rollout 的 GRPO 训练与可验证奖励；
- 单卡 4090 的真实训练配置、显存和实验对照。

安全奖励和 ASR 改善只在 P4 完成后写入。

## 8. 参考资料

- [Qwen3.5-2B 官方模型卡](https://huggingface.co/Qwen/Qwen3.5-2B)
- [Transformers Qwen3.5 文档：文本模型与多模态模型类](https://huggingface.co/docs/transformers/main/model_doc/qwen3_5)
- [Unsloth Qwen3.5 Fine-tuning Guide：BF16 LoRA、显存参考与 QLoRA 注意事项](https://unsloth.ai/docs/models/qwen3.5/fine-tune)
- [TRL GRPOTrainer 文档：Qwen3.5 支持列表与自定义 Rollout](https://huggingface.co/docs/trl/main/en/grpo_trainer)
- [TRL v1.17.1 GRPOTrainer 源码：预训练 PEFT Adapter 的 ref 副本与 rollout_func 实现](https://github.com/huggingface/trl/blob/v1.17.1/trl/trainer/grpo_trainer.py)
- [PEFT v0.21.2 源码：add_adapter 不自动激活新 Adapter](https://github.com/huggingface/peft/blob/v0.21.2/src/peft/peft_model.py)

---


## 9. 本分支实施记录（2026-11-13）

### 已提交到本分支的代码

- [x] 新增 `rl/model_loading.py`：按 `model_type` 显式选择 `Qwen3_5ForCausalLM`，支持从 PEFT adapter 配置解析基座模型，并提供文本模型/Tokenizer 加载接口。
- [x] 新增 Qwen3.5 混合注意力 LoRA 目标扫描：覆盖 Gated DeltaNet、Full Attention 和 MLP 的 Linear 投影；缺少任一必需模块时失败退出。
- [x] SFT 新增显式 `--bf16_lora`，与 `--qlora` 互斥；Qwen3.5 不允许误走 QLoRA 或全参数微调分支。
- [x] GRPO 新增 BF16 LoRA 策略加载，可从 SFT Adapter 恢复并保持仅 LoRA 参数可训练。
- [x] GRPO 初始化后校验 TRL 创建的冻结 ref Adapter：要求其与初始 default/SFT Adapter 逐张量相同、保持冻结，并确保 default 仍是 active policy；beta=0 时不要求 reference Adapter。
- [x] GRPO 多轮生成适配 TRL 0.29.x 与 1.x 的 `_generate_single_turn` 参数差异及 `max_completion_length` 属性迁移；TRL 1.x 对已经展开的每条 prompt 显式生成 1 条 completion。
- [x] 统一非 Thinking 模式：GRPO、Canary、StageVerifier 和本地推理显式传入 `enable_thinking=False`；SFT 通过预检断言 TRL 默认模板与显式关闭 Thinking 的输出一致。
- [x] StageVerifier 与本地 Benchmark 支持加载 PEFT Adapter 目录，而不是把 adapter 目录误当成完整模型权重。
- [x] 新增 `requirements-qwen35.txt`，固定 Transformers 5.22.2 / TRL 1.17.1 / PEFT 0.21.2 / Accelerate 1.15.0 / Datasets 5.2.0 等直接依赖；不覆盖 AutoDL 已安装的 CUDA PyTorch。完整传递依赖 freeze 待首次环境验证后生成。
- [x] 新增 `tests/test_model_loading.py`，覆盖模型识别、adapter 基座解析、混合注意力 LoRA 覆盖、非 Thinking 模板、TRL rollout API/长度预算兼容和 adapter fingerprint。

### 尚未验证

- [ ] GitHub Actions 全量 CI 通过。
- [ ] AutoDL RTX 4090 上 Qwen3.5-2B 文本类加载和生成。
- [ ] BF16 LoRA SFT 1–3 step smoke、Adapter 保存/重载。
- [ ] 多轮 GRPO 至少 1 次 optimizer step，确认 LoRA 权重变化且非 LoRA 参数不变。
- [ ] 实测 peak VRAM、step time、训练曲线和 Base/SFT/GRPO 指标。

**重要：** 当前提交只表示代码适配已实现，不表示 Qwen3.5 已在本项目上成功训练。所有训练与性能指标仍待实测。


## 10. 本轮代码审查与修正（2026-11-14）

### 已核对的上游接口事实

- [x] 核对 Qwen3.5-2B 固定模型 revision：`15852e8c16360a2fea060d615a32b45270f8a8fc` 可解析；官方 Transformers 文档明确提供 `Qwen3_5ForCausalLM` 文本加载路径。
- [x] 核对 Qwen3.5-2B 的 chat template：`enable_thinking` 未定义或显式为 false 时，generation prompt 都写入空的 think 区块；因此当前 SFT 的默认模板一致性预检有明确依据。
- [x] 核对 TRL 1.17.1：训练 sampler 已按 `num_generations` 重复数据行；`rollout_func` 收到的是这些展开后的行。自定义 Rollout 内每行调用 `_generate_single_turn(..., num_generations=1)` 是正确的。
- [x] 核对 TRL 1.17.1 的 `env_mask`：1 表示策略生成 token，0 表示环境 token；loss mask 会与 completion mask 相乘，当前 Observation mask 方向正确。
- [x] 核对 TRL 1.17.1 + PEFT >=0.20.0：当传入已有 PEFT Adapter 且 beta 非零时，TRL 会创建名为 `ref` 的 Adapter，并复制初始 `default` 权重；PEFT 新增的非活动 Adapter 保持冻结。

### 本轮新增保护

- [x] 新增 `assert_grpo_reference_adapter`：训练器初始化后检查 `default/ref` 均存在、`default` 仍为 active policy、ref 与初始 policy 权重逐张量相同、ref 不可训练。
- [x] 仅对 Qwen3.5 BF16 LoRA GRPO 启动时执行上述检查；beta 非零而 reference adapter 不符合预期时立即失败，不允许静默地用 base model 代替 SFT reference。
- [x] 新增 7 个单元测试覆盖 beta=0、reference 缺失、权重不一致、reference 未冻结、active adapter 错误，以及 revision 只绑定 Qwen3.5-2B。
- [x] 固定 revision 仅对 Qwen/Qwen3.5-2B 生效，避免将 2B 仓库 commit hash 错用到其他 Qwen3.5 checkpoint。
- [x] 明确 Qwen3.5 专用环境要求 Python >=3.11、PyTorch >=2.6.0；先安装 CUDA 匹配的 torch，再安装固定直接依赖。
- [x] 扩充 SFT / GRPO Manifest 的版本记录，覆盖 torch、transformers、trl、accelerate、datasets、huggingface_hub、tokenizers、safetensors，以及实际使用时的 peft / bitsandbytes。

### 尚未验证

- [ ] GitHub Actions 当前没有返回任何 workflow run；本轮尚无 CI 通过证据。
- [ ] 本地执行环境没有安装 Transformers / TRL / PEFT / Datasets，且当前容器无法连接 GitHub，因此不能在此环境执行完整仓库测试。
- [ ] AutoDL RTX 4090 的模型加载、SFT smoke、GRPO optimizer-step 和显存数据仍待实测。

**本轮结论：** 代码审查确认了 TRL 1.17.1 自定义 Rollout 和 reference adapter 的接口契约，并增加了 reference adapter 的启动期硬校验；这仍然只是代码级兼容性修正，不代表已完成训练验证。
