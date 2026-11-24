# GRPO Credit v2 · RTX 4090 启动说明

2026-11-24，分支 `feat/grpo-credit-v2`。Phase A 使用本地 CPU。E1–E4 的 Strict Success、真实训练暴露和新显存峰值待 GPU 实测。复用同资产的 E0（18/24）和 Native SFT（20/24）；四个新实验都从 Native SFT Adapter 独立初始化。

## 代码与训练协议

| 审计对象 | 已核实的行为与本轮处理 |
|---|---|
| `reward.py`、`grpo_reward.py` | 五维打分与历史分组处理保留。v2 记录五维分量，唯一终局奖励取自既有 `benchmark.metrics.is_strict_success`。 |
| `step_credit.py`、`step_credit_trainer.py` | 保留历史 Prefix-tree 路径。v2 复用有序工具参数和论文身份判定，独立计算 episode、DFA 和 exact-anchor 三路信用。 |
| `native_grpo_rollout.py`、`env.py`、`store_memory.py`、`side_effects.py` | 同一 Native Agent 执行与重放，隔离每条轨迹。Anchor 纳入实际 Assistant token、工具反馈、任务状态、论文顺序与当前引用、PDF 缓存、翻译及 PENDING 状态。排除会话生命周期信息；按原模板核对实际反馈 token。 |
| `task_spec.py`、`metrics.py` | 从冻结的 TaskSpec 得到有序 DFA，保留参数默认值、引用解析、blocked 原因和终止契约。独立 Benchmark 前缀判定用于审计。 |
| TRL 1.17.1 | 单进程每个 optimizer step 对应同一 prompt 的四条轨迹。在 microbatch 拆分前按真实生成 token ID 核对行序、completion 长度与 tool mask。原始 tool mask 在 padding 位置可为 1，因此另外检查 completion mask，再屏蔽观察、padding、无效样本与受保护的错误后缀。 |
| 数据、日志与恢复 | 固定 25 组顺序，按任务组采样，P0 / 多步骤 / 控制为 13 / 7 / 5。逐组保存任务、四条原始轨迹、三路信用、每步最终 Advantage 和 mask。每个 Pilot 以 5 步为间隔保存检查点。 |

E1 仅使用组内标准化的 strict 0/1。E2 对同一 DFA 状态的可比较决策计分，保护首错前缀并屏蔽后缀。E3 只在同一 exact anchor 内以 strict return 做局部比较。E4 等权合并单独归一化的 DFA 和 anchor 信号，再按 0.5/0.5 合并 episode 与 local。单个基础设施无效样本退出归一化和梯度；若同组至少两条无效，保存该组后中止配置。无有效对照的组保留零局部信号。额外可验证的开机前判断是同一 Dev 上 strict、状态/约束、缓存及 PENDING 的表现；见下方留出规则。

方法来源为 GiGPO 的重复状态比较、GLARE 的确定性任务进度、首错保护和 IRC 的方向校准。实现针对本工作的 Native 工具执行，使用已有 rollout，无需额外 critic。[GiGPO](https://proceedings.neurips.cc/paper_files/paper/2025/hash/420c9f777c0b4f78d515e53cf74d58b2-Abstract-Conference.html) · [作者实现](https://github.com/langfengQ/verl-agent) · [GLARE](https://proceedings.mlr.press/v306/yan26k.html) · [Save the Good Prefix](https://aclanthology.org/2026.findings-acl.1767/) · [IRC](https://arxiv.org/abs/2604.02869)

## 固定资产

| 资产 | SHA-256 |
|---|---|
| Qwen3.5-2B-Base 五文件清单 | `7e1bfb7d6a4cf9f0266e4cea2a7d5d752f5f5ce1f6fce727d9fe6c5b4335a619` |
| Base 权重，4,548,221,488 bytes | `928acbf11878c32185bbd863514d191769285065ab9ea14fbfe431303f5fdf2d` |
| Native SFT Adapter | `789af8cf28960892a1fde034d32f9b8ddf230e8450552be70d15024c964ed6c8` |
| Native SFT Train Manifest | `34196a2639203ca290a43c1012daa1f82f4621a4f8beb2203c597713d013bf70` |
| Snapshot | `d97035db326ca11935e882fd2aca7fe932e9e76cf1bc8a21078be7bec1ae8794` |
| 候选 v2 | `87750f1ff966f001255b222a0d74869f01210f771d22885c56d032cefa61d939` |
| Train v2，18 个任务 | `3e55a419d4fe5d5c91410043926fb1d13624f3f4b32cf0b70da95cce53a63a39` |
| v4_86，固定 Dev/IID/OOD 的文件 | `0a176b0f67b8b61d4f84e27a40071ab81a5be3d2040c2a58aa695432f5a6eacc` |
| 仅 Dev 的有序 8 个 ID，紧凑 UTF-8 JSON 数组 | `e4074d69e29427ba1a5df49431c2753b6e77a1593d0c38a28e69d4a432e012b1` |
| System Prompt | `31674f2183d642a9035ecf8360c138135c5ef4645904017eadf4ec7bdf4abf05` |
| Tools | `01ed14f7f8a041ca9752475c1bf1513996c375c5a32cadf2514089fb04401ae0` |
| Chat Template | `273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80` |
| 历史 E0 checkpoint-25 | `73a9a3721aa57d658417968488f6e79a6c74e4e9f3f1e0c6af6607bd9fc8d940` |
| 历史 Probe / E0 原始轨迹 | `d8173d493233f7c0b12199246bab2ee7e5833761f3a4be32767fc56a10afc0ba` / `c3e81cdfd8d3ab53e9f56068a2378603ad4ea498a6a2eba473ecd2ea10dce4b5` |

在本地核对权重、模型配置、索引和 tokenizer 的实际文件，核对 SFT Adapter、Snapshot 与冻结训练数据。重建了 33 个候选的 gold trace，逐一校验 Train 父任务、文本、Setup 和动作签名，并与 31 个留出任务和冻结的 SFT catalog 对照。E0 保存的训练 Split 与当前 Train v2 一致，其 checkpoint 和原始 24 条 Dev 的哈希、严格成功计数均可复核。E0 沿用历史抽样顺序；新四组共享本轮分层顺序，E0 对比同时包含采样差异。

## CPU 验证与信号覆盖

- 本轮相关回归 45 项通过。完整 AgenticArxiv 套件 1,049 项通过，另有 704 个子测试通过，6 项跳过，0 失败。3 项依赖可选 TensorBoard，3 项需手动联网或提供模型 API 凭据。4 条警告来自既有 PyTorch/Pydantic 调用。
- 统一启动入口 CPU dry-run 通过全部 4 个 Smoke 配置和 4 个 Pilot 配置，并核对了已归档 SFT 的 Dev/IID/OOD 原始试次文件。

在 Python 3.11、CPU PyTorch、Transformers 5.22.2、TRL 1.17.1、PEFT 0.21.2 环境完成相关 pytest、完整测试、compileall、统一入口的八次配置 dry-run、Shell 语法检查和离线审计。CPU 微型 Qwen + LoRA 在原版 TRL 上实际执行反传和参数更新，覆盖旧路径和 E1–E4，验证 G=4、KL、真实 token mask；E4 从第 1 步检查点恢复后读到第 2 个任务组。

重放了冻结策略的 168 组（672 条）以及 E0 的 25 组（100 条）。下表的 Probe 对照数仅计入 Train v2 中该弱点对应任务的组；不同弱点可重叠。E0 栏取自正式训练原始样本。

| P0 弱点 | Train 任务 | Probe 目标组 | 有错误/成功对照的组 | E0 抽到目标组 | E0 观察到错误动作 |
|---|---:|---:|---:|---:|---:|
| 全 CS 搜索范围 | 3 | 12 | 3 | 3 | 0 |
| 漏查第二篇缓存 | 8 | 40 | 27 | 11 | 9 |
| PENDING 处理 | 2 | 8 | 0 | 3 | 2 |
| 多步骤提前结束 | 12 | 56 | 25 | 16 | 7 |

**截断按决策判定。** Probe 中 14 条截断轨迹共包含 56 个决策。8 个未完成的 Assistant turn 退出局部比较，其余 48 个保留，包括 13 个已验证正确决策、11 个可定位的首错和 24 个错误后缀。DFA 对照仅使用可验证决策，Anchor 按实际到达的状态与既有 strict return 分组，E2/E4 继续屏蔽错误后缀。达到轮数上限前完成的工具调用保留原始 span，轨迹在实际结束处停止。调用有明确执行失败时，DFA 优先记录执行错误；缺失回执按基础设施异常屏蔽，回执缺少论文身份记为未解析，仅在实际解析到另一篇论文时判 `wrong_ref`。

**保护前后使用相同的样本和对照分母。** 下表取同一批 168 组冻结 Probe。前缀保留指失败轨迹中正确前缀获得非负局部信用。首错受抑的分母仅含该通道有对照信号的首错。方向冲突包括正确动作得负分和首错得正分。四个字段均保存在逐决策轨迹中，并单独汇总符号、均值、方差和按 token 加权的贡献。

| E4 信号 | 前缀保留 | 首错受抑 | 首错获正分 | 正确动作获负分 | 方向冲突 |
|---|---:|---:|---:|---:|---:|
| 原始 DFA，`raw_dfa_adv` | 173/173 | 110/110 | 0/223 | 0/1293 | 0/1516 |
| 原始 Anchor，`raw_anchor_adv` | 97/173 | 63/63 | 0/223 | 76/1293 | 76/1516 |
| 两路等权合并，`raw_combined_local` | 98/173 | 110/110 | 0/223 | 75/1293 | 75/1516 |
| 保护及 mask 生效，`effective_local` | 173/173 | 110/110 | 0/223 | 0/1293 | 0/1516 |

DFA 输入本身包含任务契约标签，DFA 的符号符合率衡量实现与契约的一致性。E4 的保护消除了原始混合信用中的 75 个负向正确决策；保护后的 173/173 衡量规则执行。原始 Anchor 的有效覆盖为 343/1699，同状态正确动作的相对优势在 119/141 对中高于首错，这是与可验证标签的经验排序一致率。该回放没有估计动作的因果效果。独立 Benchmark 前缀检查仍与 DFA 的首错位置吻合 223/223。E1 有 102/168 组 strict reward 零方差。

**PENDING 维持 `no_signal`。** Train v2 对应任务在 Probe 中采样 8 组，有效目标对照为 0 组。冻结 Probe 的实际更新 token 为 0；离线应用 E4 时，PENDING 目标位置有 446 个最终 Advantage 非零的 token，局部信用非零的 token 为 0。历史 E0 正式训练实际采到 3 个目标组；按其归档奖励、旧 Prefix-tree、原生 mask 和 Assistant span 重建，PENDING 正确终止位置有 270 个非零 Advantage token，错误位置有 258 个，共 528 个。这些计数描述参与策略损失的 token，不代表 PENDING 专项梯度或能力改善。E1–E4 的实际训练计数待最终 GPU 检查后实测。

具体例子为 `chain_ro5_dl_three`、Probe 组 3、seed 53、generation 0。E4 对第 0 步正确前缀给 0，第 1 步首错给约 −1.50，第 2、3 步均屏蔽。先前报告的 Credit Accuracy 40% 属于旧口径。早先 100 步日志的 51 个无 step-credit 组已经包含 15 个截断组。

完整分子分母、通道统计、无信号原因和动作分布见 [Weakness Bank](weakness_bank.json)、[Credit 对齐](credit_alignment_audit.json)、[Anchor 审计](anchor_state_audit.json) 和 [Reward 校准](reward_calibration_audit.json)。逐轨迹重评分保存在本地 `outputs/credit_v2_cpu/{probe,e0}_decisions.jsonl`。稀疏奖励、PENDING 对照缺失及 24 个 Dev 试次限制了结论强度。

## 单卡执行

使用单张 RTX 4090 24 GB。在已同步本分支的仓库根目录执行以下命令；沿用前次实例的模型、Adapter、Snapshot、Python 与 SFT 基线目录。迁移磁盘时可通过 `BASE_MODEL`、`SFT_ADAPTER`、`SNAPSHOT`、`PYTHON_BIN` 和 `SFT_BENCHMARK` 指定路径。

```bash
bash scripts/run_grpo_credit_v2_ablation.sh --out /root/sj-tmp/grpo-credit-v2-run1
```

GPU 入口先检查设备、BF16、冻结依赖、磁盘余量和 SHA，然后依次执行 E1–E4 的 4 步 Smoke，再依 E1 → E2 → E3 → E4 执行各自 25 步 Pilot；每个 Pilot 结束即做 24 次 Dev。四种配置使用相同的 2B LoRA 起点、seed 42、G=4、batch 1、累积 4、BF16、梯度检查点、temperature 1、LR 5e-6、KL beta 0.04 以及 5 turns / 1792 tokens / 每轮 704 tokens。各阶段独立进程和输出目录。中断时原命令增加 `--resume-run`；单配置复核增加 `--experiment E2` 等。默认拒绝覆盖往期结果。

| RTX 4090 单卡预算 | 估计 |
|---|---:|
| 每种配置的 Smoke / 训练 / Dev | 5–8 / 25–30 / 8–10 分钟 |
| 四种配置含模型加载、落盘、清理 | 约 3–4 小时 |
| 单个候选满足门槛后追加 IID/OOD（69 次） | 约 25–40 分钟 |
| 显存 | 参考旧 25 / 100 步峰值已分配显存 6.01 / 6.11 GiB，计划 6–9 GiB 已分配，另外预留 CUDA 缓存空间 |

为落实任务书的“没有明显的状态/约束退化”，预先检查 Dev strict 至少 20/24、条件任务至少 5/6，并要求缓存和 PENDING 指标的分母覆盖及成功率至少达到对应 SFT 水平。满足者按 strict、条件成功数排序，平局按 E1–E4 顺序，只评测一个候选的 IID/OOD。每个实验记录峰值显存和耗时。GPU 结束后需核对 Adapter SHA、配对原始记录并拉回本地，再通知关闭实例。
