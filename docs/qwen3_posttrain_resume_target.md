> **已被新方案取代。** 当前项目目标、模型选择和实施状态以 [docs/arxivfetch_qwen35_2b_upgrade_plan.md](arxivfetch_qwen35_2b_upgrade_plan.md) 为准。本文件保留此前 Qwen3-1.7B 方案的历史记录，不再作为执行依据。

# ArxivFetch-RL：以可验证简历成果为目标的 arXiv Agent 后训练与安全评测

> 状态：**目标设计，尚未完成训练或实验**。所有结果均须来自本 Fork 的实际运行产物；禁止将上游作者结果或预期指标当成本人成果。
>
> 基准：Fork `Napabana/AgenticArXiv-RL`，上游 `Algorineko/AgenticArXiv-RL`，起点 commit `df7a4b69673cc72bc6dd4f88c5fdc8ad0d778563`，许可证 MIT。开发分支 `feat/qwen3-1p7b-posttrain`。

## 一、最终简历目标（各阶段完成后才允许兑现）

> 项目名 `ArxivFetch-RL` 为本 Fork 拟用的展示名。GitHub 仓库 URL 目前仍为 `Napabana/AgenticArXiv-RL`，尚未执行远程仓库重命名。

> Qwen3.5-2B 的 4090 GRPO 训练**尚无本 Fork 实测**：官方模型是带视觉塔的 `Qwen3_5ForConditionalGeneration`，上游 SFT/GRPO 默认 CausalLM 或 VLM BF16 路径不直接兼容。TRL 支持该模型并不等于现有多轮 `rollout_func` 自动兼容。Unsloth 文档支持 Qwen3.5-2B BF16 LoRA，并不推荐 QLoRA；需先通过兼容性、训练和显存冒烟。

**ArxivFetch-RL：面向 arXiv 论文检索、下载及阅读工具的轻量 Agent 后训练与安全评测**（2026.11—至今）  
**技术栈（目标）：** Python、PyTorch、Transformers、TRL、PEFT、BF16 LoRA、SFT、GRPO、RLVR

- **小模型后训练（待实测）：** 基于开源 AgenticArXiv-RL 二次开发，优先验证 Qwen3.5-2B（失败回退 Qwen3-1.7B）；统一训练、Rollout、推理与阶段验证的 ReAct 协议，尝试在单张 RTX 4090 24GB 上完成 BF16 LoRA SFT→GRPO，并记录显存、训练时间和可恢复权重。
- **多轮策略优化：** 复用上游多轮工具交互与离线环境，针对新模型重新执行冻结策略奖励方差探测、筛选有学习信号的 GRPO 任务，分析零方差、错误工具、无效参数与过早 FINISH 对有效更新的影响。
- **Agent 安全对齐扩展（待实现）：** 新增恶意工具 Observation 注入与越权工具调用的可复现实验；设计安全约束奖励/终止判据，在控制正常任务成功率的前提下减少攻击成功率。
- **可复现对照（待实测）：** 在预先冻结的 train/dev/IID/OOD 和安全留出集上对比 Base / SFT / GRPO / GRPO+Safety，报告 Strict Success、Tool/Argument Accuracy、False Finish、Attack Success Rate、Token Cost；通过奖励消融解释性能变化。**指标：待实测，禁止填写臆测数字。**

完成第 1～2 条即可申请 Agent/LLM 算法方向的入门岗位；第 3～4 条形成与已有 Forge Agent 的安全/策略方向差异化。

## 二、上游与本 Fork 的贡献边界

| 组件 | 上游现有能力 | 本 Fork 的增量 / 验收 |
| --- | --- | --- |
| Agent 环境 | 10 个 arXiv 工具、ReAct、MockArxivEnv 和轨迹隔离 | 不改动工具语义；验证所选小模型下的格式兼容与重放一致性 |
| 数据 | TaskSpec、确定性专家样本、参数化扩增、Manifest | 从冻结 train 切分重新生成并记录 hash、统计样本来源；不把扩增样本视作独立任务 |
| SFT | TRL SFTTrainer、completion-only loss、QLoRA | 优先验证 Qwen3.5-2B BF16 LoRA，统一 non-thinking 训练/验证协议、4090 训练配方与真实产物；不可假定原 QLoRA 脚本可直接运行 |
| GRPO | 多轮 rollout_func、env_mask、五维 RLVR、RewardVarianceGuard | 新模型专属 frozen-policy probe、任务筛选、对照实验、错误轨迹分析 |
| 评测 | strict/pass^k、工具/参数、false finish；版本化切分 | 固定种子与解码，报告完整分母、跨模型一致性与失败原因 |
| 安全研究 | 上游有工具允许列表与失败惩罚 | **新增**可信指令与不可信 Observation 边界、恶意工具输出数据集、安全训练臂与攻击评测；同时测 benign utility |

禁止对外声称“独立设计了 10 工具环境、首创 GRPO 多轮训练器或原有五维奖励”，除非真正重写或有独立贡献证明。

## 三、目标模型与资源上限

- Policy（优先验证）：`Qwen/Qwen3.5-2B`，备选 `Qwen/Qwen3-1.7B`；固定 base revision，训练与推理共享 tokenizer 配置。
- 硬件：单 RTX 4090 24GB；Qwen3.5 先选 **BF16 LoRA**（r=16、gradient checkpointing），不沿用原项目默认 4-bit QLoRA；先通过兼容性/显存冒烟才开始正式训练。
- 第一阶段只验证 Qwen3.5-2B；如适配成本/显存过高，则回退 Qwen3-1.7B。Qwen3.5-4B、OPD、DAPO、GSPO 留作可选扩展。
- SFT 和 GRPO 必须与评测采用同一非 Thinking chat template；不能只在 Benchmark 使用 `--no-thinking`。
- 版本锁定须记录：torch / transformers / trl / peft / bitsandbytes / CUDA、上游及本 Fork commit。
- 训练前做 VRAM smoke；GRPO 初测应使用 1 prompt × 4 generations、短 context、2 turns；若超显存，先减少长度并检查 VLM 加载方式/显存占用，不直接照搬作者不同模型/硬件上的 Batch 配方。

## 四、数据和对照协议

- 上游当前 `data/splits/v7_86.json`：**86 个任务**，其中 train=55 / dev=8 / iid_test=19 / ood_test=4。使用版本化固定切分，不能用 holdout 挑选超参数。
- 上游 `v3_81.json`：81 个任务，train=51 / dev=8 / iid_test=18 / ood_test=4。初期复现可先沿用 v3，正式报告必须写明版本；不跨版本比较百分比。
- SFT 样本来自 train + train 派生任务。保持 `source_task_id` 和 Manifest；避免近重复模板、标题或任务实例泄漏。
- GRPO task selection：在**本次所选策略的 SFT** 权重上执行 `lr=0` frozen rollout（建议每任务 8 组 × 4 generations），只对 train 候选任务统计组内 reward 方差；筛出的任务写入新的带模型版本标识的 split 文件。
- Model arms：Base → SFT → GRPO；安全扩展后再加 GRPO+Safety 和奖励消融（同等 rollout/optimizer step 预算）。
- Evaluation：固定离线快照、ReAct parser、解码参数、seed/repeat、task split；分别披露 strict（成功）和 pass^k（可靠性）。由于现有 OOD 仅 4 题，必须报告分母并避免夸大泛化结论。
- 训练后安全评测（**待开发**）：建议至少 40 个攻击案例，按工具输出注入 / 身份冒充 / 越权工具 / 数据泄露模拟等攻击族分组。使用合成的无敏感信息诱饵，区分训练攻击样本与留出攻击模板，不执行真实越权行为。
- 核心指标：strict success rate、tool accuracy、argument accuracy、false finish rate、attack success rate (ASR)、benign success rate、mean tokens、peak GPU memory、reward variance、parse error rate。所有增幅同时注明基线、分母、随机种子与不确定性。

## 五、工作顺序与验收条件

### P0：模型适配 + 静态单测

- [ ] 统一 Qwen3.5 non-thinking chat template：SFT / GRPO / canary / stage verifier / local inference；校验 Qwen3.5 multimodal loader、hybrid backbone 和 LoRA target modules。
- [ ] 测试 ReAct JSON/FINISH 格式，确保工具输出不会被误解析为新指令。
- [ ] 检查 SFT 保存的 LoRA adapter 在 GRPO 与 Benchmark 中可被正确加载。
- [ ] 完整跑已有 parser/reward/rollout/unit tests，检查 upstream 行为不回归。

### P1：SFT 可复现

- [ ] 校验快照、train split、SFT JSONL、manifest 与 hash。
- [ ] SFT 30-step smoke（显存与损失有限、adapter 保存成功）。
- [ ] SFT 正式训练，记录 seed/config/日志/权重；评测 Base 与 SFT。
- [ ] 分析至少 20 条失败轨迹并分类：未调用工具/错误工具/参数错误/提前 FINISH/格式错误。

### P2：GRPO 训练与归因

- [ ] Frozen-policy probe：禁止更新参数，只统计所选模型 SFT 的任务组信息率。
- [ ] 从 train 生成新 GRPO split；不使用 dev/IID/OOD 选择训练任务。
- [ ] 30-step GRPO smoke；再 60-step 正式 run，记录 reward/variance/rollout 审计与 checkpoint。
- [ ] 对 Base/SFT/GRPO 在相同 dev/IID/OOD 测试协议下复评；如提升不显著，不在简历写“显著提高”。
- [ ] 至少 1 组消融：仅 Outcome 奖励 vs 五维 RLVR（保持任务、采样预算、seed 一致）。

### P3：独立安全贡献（简历差异化）

- [ ] 定义可信 User/System 指令与不可信工具 Observation 边界；固定工具权限策略。
- [ ] 实现带 ground truth 的恶意 Observation / 越权调用对抗任务集和确定性评测，不使用实际敏感数据。
- [ ] 引入安全奖励或不可补偿的越权惩罚，设计 GRPO+Safety 对照。
- [ ] 评测 ASR 降低及正常任务 success 变化，并复核是否存在过早 FINISH 的 reward hacking。
- [ ] 保存 seed、输入 payload hash、模型 revision、trace、奖励明细和实验报告。

### 完成门槛

1. 上述所有简历事实能够由 GitHub commit、config、训练 manifest、模型权重与离线评测报告追溯。
2. 训练数据与留出数据无泄漏；模型对照只改变明确列出的实验变量。
3. 算法指标有真实改善再填写百分比；若某一方向没有改善，保留实验失败分析而非虚构增益。
4. 所有独立修改在 Fork 单独分支开发；经测试再考虑合入 `main`。

## 六、首个实施任务

从 P0 开始；先验证 Qwen3.5-2B 模型类/processor/非 Thinking 模板兼容以及多轮 GRPO 接口，再跑 4090 上的 1-step SFT/GRPO 显存冒烟；失败则回退至兼容性更稳妥的 Qwen3-1.7B。未经实测不得在 README 或简历记录性能数字。
