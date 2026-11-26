# E2-Repair 72 步训练就绪报告

## 实验边界

本轮从 Native SFT v2 的原始 Adapter 独立起跑。训练、开发和最终留出分别为 24、8、12 题。旧 Sealed 只用于历史诊断。新留出集在训练前固定，模型在 Dev 评测完成、四组权重和配置写入冻结记录后各评一次。同一预定试次的完整记录按 SHA 复用，训练和选参均无权使用留出结果。

## Snapshot 和 Strict

原 Snapshot 实际包含 419 个不同 paper_id，20 篇有详细记录。未增加或改写快照。覆盖矩阵将任务的检索参数、每个可见候选位置、目标工具及具体参数与真实执行回执关联，646 行对应 113 个不同执行条件，全部通过。部分训练题必须缩短可见候选列表才能使所有候选都支持所需操作。模型仍可生成其他合法参数；未录制的合法调用被记作环境缺项。

44 条专家轨迹各通过两次真实 Native Agent + Mock Strict 回放。另从每题的合法节点中反向选取可用节点，再完整执行一次，44/44 通过，其中 9 题走了真实不同的顺序。Strict 仍核对工具、执行成功、参数、当前列表的论文引用、次数、终止语义和状态依赖。新留出的操作目标与本轮 Train/Dev 分离，对已用 Challenge、生产题及 SFT v2 目录未发现完全相同的任务文本或行为签名。任务仍使用相同的工具族，此检查不能证明语义完全独立。

额外负例实际访问缺少记录的 method 段。同组另一条操作访问已有的 abstract 段。环境失效轨迹的 Strict 为 null，另三条保留各自的原始结果，整组用于策略和 KL 的有效 Token 为零。环境无效、可评分失败、预算审查未决三者单独记录。

详见 [覆盖矩阵](coverage_matrix.jsonl)、[数据审计](data_audit.json)、[原始 Gold 回放](gold.jsonl.gz) 和 [依赖审计](dependency_audit.json)。

## 截断与预算

| 旧实验 | 轨迹 | 单轮生成上限 | 总长度在生成中耗尽 | 剩余长度放不下真实工具反馈 | 达轮数且存在已核实的重复 | 其他轮数耗尽 | 总截断 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Frozen Probe | 768 | 22 | 1 | 63 | 35 | 80 | 201 |
| E2 八题 | 96 | 2 | 0 | 2 | 5 | 6 | 15 |
| E2 全题 | 96 | 7 | 0 | 5 | 5 | 12 | 29 |

每条轨迹均存有实际长度、用过的轮数、遗漏反馈长度和首错信息。重复动作消耗了轮次，无法从旧轨迹推断删去重复后一定会完成。对完整正确前缀遭预算截断的轨迹，保留可验证的前缀，终局记未知；若此前已有不可逆的 Strict 错误，仍可确定失败。按此规则旧三组各有 4、0、2 条预算未决。

历史 completion 加当轮原始反馈的最大实测长度为 27,708 Token。Gold 最长 completion 为 27,564，最长助手单轮为 239。新上限为单轮 1,536、总计 31,232，每题轮数为专家工具数加一再乘二（8–16）。新策略可能走出更长的路径。因此先用真实 SFT v2 在 24 道训练题各新采 G=4；若可评分组出现仍正确的前缀因预算耗尽而终止，或所有组均无 Strict 成败对照且无双向 DFA Token，就停在 Probe 并保存证据。

详见 [逐条轨迹](historical_trajectories.jsonl)、[截断统计](truncation.json) 和 [逐题预算](../../data/e2_repair/budget.json)。原始轨迹可由 [192 个 Probe 组的 SHA](../agent_challenge_final/frozen_probe/group_sha256.json)、[八题 E2 归档](../agent_challenge_final/dfa_transfer/archives.json)和[全题 E2 归档](../agent_challenge_final/dfa_all24/archives.json)核对。

## DFA 信用定位

下表在同一批旧的 24 题训练样本上重新计分。历史诊断标签可重叠，分母包括被整组排除的环境无效组。

| 旧诊断类别 | 标记位置 | 旧局部负向 | 修复后局部负向 | 环境无效组内的位置 |
|---|---:|---:|---:|---:|
| 重复调用 | 19 | 0 | 6 | 0 |
| PENDING | 7 | 0 | 7 | 0 |
| Tool Parse | 45 | 5 | 28 | 4 |
| Reference / 参数 | 53 | 20 | 25 | 3 |

重复需匹配工具、有效参数和真实环境状态，并排除任务要求的再次调用。PENDING 区分读取未就绪译文、重复提交及对另一篇论文的独立工作。Parse 核对助手生成内容，截断和缺失 Observation 有独立处理。Reference 用当时列表解析 ref 并核对真实回执。守卫要有本组其他实际轨迹提供可区分的动作，才产生局部负向优势；没有采到证据时保留零。正确前缀受保护，Observation 和传输填充不参与策略或 KL 损失。

176 个历史有效 Probe 组中，151 组有非零局部信用，包括 110 个任务文本、参数、起始设置和目标与本轮一致的组。总计 85,704 个非零有效优势 Token，含终局项。这是信用重放证据。新策略是否改进须以配对结果判定。各标记位置保留原始与修复后的信用、Span、环境摘要和零信号原因。详见 [信用覆盖](credit_coverage.json) 和 [逐位置明细](credit_positions.jsonl)。

## 模型、训练与评测

| 项目 | 冻结值 |
|---|---|
| Base 权重 SHA-256 | `928acbf11878c32185bbd863514d191769285065ab9ea14fbfe431303f5fdf2d` |
| Native SFT v2 Adapter SHA-256 | `f0c33a4d32a175bcaf1e83c55231a0108a84859cb748c318db49988a52f7911f` |
| 实际生产提示词 SHA-256 | `31674f2183d642a9035ecf8360c138135c5ef4645904017eadf4ec7bdf4abf05` |
| Snapshot SHA-256 | `d97035db326ca11935e882fd2aca7fe932e9e76cf1bc8a21078be7bec1ae8794` |
| E2 | Strict Outcome + DFA + First-error Protection，G=4，λ=0.5，KL β=0.04，初始 LR=5e-6，TRL 1.17.1 dapo |
| 更新与保存 | 每题每轮一个新组，三轮顺序独立打乱。共 72 次优化、288 条在线新轨迹，分别在 24、48、72 步保存 |
| 配对评测 | SFT v2 与三个 checkpoint 同题同试次。每模型 Dev 8×3、新留出 12×3。生产集三个 checkpoint 各 93 次，同已核验的 SFT v2 83/93 配对 |

完整文件 SHA、数据 SHA 和采样顺序见[冻结清单](../../data/e2_repair/freeze.json)、[基础资产](assets.json)和[训练配置](../../data/e2_repair/train_config.json)。输出包含逐步 KL、Grad Norm、Loss、Strict Reward、LR、正负 DFA Token、预算及环境缺项。最终严格成功、修复及回退、子目标覆盖、前缀、引用、重复、PENDING 和解析错误均有明确分母。

## CPU 验收记录

完整回归为 1,108 项通过、6 项跳过，另有 704 个子测试通过，耗时 167.51 秒。针对原 Credit v2、四个修复守卫、边界条件、真实列表切换、助手 Token Span 和 Challenge 采样规则的回归为 46/46 通过。JUnit 将子测试计入 tests 总数，两个试验集合有重叠。详见[测试回执与 SHA](cpu_tests.json)。

```bash
python -m pytest -q AgenticArxiv/tests scripts/agent_challenge_final/test_probe.py scripts/agent_challenge_final/test_dfa_transfer.py
python -m pytest -q AgenticArxiv/tests/test_grpo_credit_v2.py AgenticArxiv/tests/test_e2_repair.py scripts/agent_challenge_final/test_probe.py scripts/agent_challenge_final/test_dfa_transfer.py
```

## GPU 启动

以下命令在冻结版本的仓库根目录运行，模型和虚拟环境路径取自已有实例。新实例以实际路径替换。使用全新输出目录，测试与正式训练从同一个 SFT v2 起点分别加载。

```bash
export PYTHON=/root/sj-tmp/grpo-credit-56517-bfaaf59/grpo-env/bin/python
export BASE_MODEL=/root/sj-tmp/models/Qwen3.5-2B-Base
export SFT_ADAPTER=/root/sj-tmp/native-sft-v2-56780/run/formal/adapter
export REPAIR_OUT=/root/sj-tmp/e2-repair72/run
export EVAL_PARALLEL=1 PRODUCTION_PARALLEL=1
bash scripts/agent_challenge_final/repair_pipeline.sh
```

一张 24 GB RTX 4090 即可先做冻结 Probe。旧 24 步训练实测约 45 和 53 分钟，峰值分配 7.2 GiB；新长上下文的峰值和吞吐须在实机测量。整个 Probe、训练、四模型新评测和生产回归预留 **7–11 小时**，其中 72 步训练约 **2.5–4 小时**。显存充足时，评测并发可设为 2；同一个优化器的 72 步仍按顺序执行。以远端完成回执、本地拉回并核验三个 checkpoint 和原始数据为关机条件。
