# GRPO Credit Assignment v2：Codex 实施任务书与消融实验规范

> 本文件是 Codex 在本分支的执行规格。目标是在不改变 Base Model、Native SFT 起点、Mock Snapshot 和 Strict Success 评测定义的前提下，重构 Probe-aware task sampling、可验证的步骤级信用分配和 Reward/Advantage 对齐，并完成可复现的消融实验。
>
> **Codex 执行原则：**
> - 自主完成所有 CPU 可完成的审计、设计落地、代码实现、单元测试、离线评估和启动脚本。
> - 不要反复要求用户确认本文已明确的设计决策。
> - 不要在 CPU 阶段启动 GPU 训练，也不要假定 GPU 已经可用。
> - CPU 阶段全部验收通过后，生成 GPU readiness report，明确告诉用户需要启动 24GB RTX 4090；随后暂停，等待用户启动 GPU 并让 Codex 继续。
> - 不要覆盖旧 Prefix-tree 实现。新实现应可通过配置切换，旧实验可复现。
> - 不要为了得到更高成功率而修改 Benchmark 的 Strict Success 定义、评测任务或评测口径。

**工作分支：** `feat/grpo-credit-v2`  
**基线分支：** `feat/qwen35-native-grpo`  
**基线提交：** `aeb5e3b60ddf08a4a5f827d08e7fad283b65354b`  
**项目：** Qwen3.5-2B Native Tool Calling GRPO / AgenticArXiv-RL

---

## 0. 本轮研究目标与非目标

本轮要解决三个已由真实实验暴露的问题：

1. **Probe 与训练任务分布脱节。** Probe 已经发现全 CS 检索范围错误、漏查第二篇缓存等弱点，但正式训练未能稳定采样到这些错误对照。训练任务筛选必须由 Weakness Bank 驱动，并记录目标弱点在正式训练中的真实 exposure。
2. **现有 Prefix-tree Credit 的步骤归因不可靠。** 当前 40% Credit Accuracy 只是步骤正负方向与专家标签的一致率，不是因果归因准确率；Action Prefix 也不能无条件等同于完整 Agent State。需要引入 TaskSpec-derived DFA、First-error Protection 和 GiGPO 风格 Anchor-state Grouping。
3. **优化 Reward 与最终 Benchmark 目标不一致。** Benchmark 以 Strict Success 为最终目标，旧五维加权 Reward 允许错误维度被其他维度补偿。新消融中统一使用 Strict Outcome Reward；过程监督作为独立的 Advantage/Credit 通道，不再把启发式分数混入一个可补偿的总 Reward。

本轮不做：
- 不训练额外 PRM / Reward Model。
- 不引入 LLM-as-a-Judge 直接给每一步打分。
- 不做 ProVer 式大规模反事实续跑；只预留接口和后续实验设计。
- 不同时改变模型、Tokenizer、Prompt、Tools Schema、Snapshot、数据集和训练超参数。
- 不把“更多工具调用”“更长轨迹”本身当作优化目标。
- 不在 Dev / IID / OOD 上训练，也不把留出任务原文或精确参数搬进 Train。

---

## 1. 已有实验事实：必须作为本轮设计依据

先读取并复核以下文件，不要仅凭本任务书的摘要做假设：

- `reports/native_grpo_20261123/native_grpo_report.md`
- `reports/native_grpo_20261123/native_grpo_report.json`
- `reports/native_grpo_20261123/paired_case_audit.json`
- `reports/native_grpo_20261123/training_signal_audit.json`
- `reports/native_grpo_followup_20261124/experiment.md`
- `reports/native_grpo_followup_20261124/paired_dev.json`
- `reports/native_grpo_followup_20261124/pilot_summary.json`
- `reports/native_grpo_followup_20261124/probe_summary.json`
- `reports/native_grpo_followup_20261124/prior_policy_reward_audit.json`

### 1.1 已有 100-step GRPO 结果

2026-11-23 的 Native SFT → Native GRPO + Prefix-tree：

- Native SFT：72/93 Strict Success（77.4%）。
- GRPO：65/93（69.9%），下降 7.5 个百分点。
- Dev：20/24 → 17/24。
- IID：44/57 → 40/57。
- OOD：8/12 → 8/12。
- 配对结果：1 次 fail-to-pass、8 次 pass-to-fail。
- 多步骤完整执行：79.5% → 71.8%。
- 状态与约束任务严格成功：73.3% → 33.3%。
- 额外重复调用：12 → 0；工具错误：15 → 3。
- Probe 共 88 组、352 条轨迹；正式训练 100 组、400 条轨迹。
- 正式训练中 51/100 组没有有效 Step Credit，另有 15 组因截断关闭 Step Credit。
- 报告中的 Credit Coverage 为 29.4%，Credit Accuracy 为 40.0%；守恒误差接近 0 只证明数值守恒，不证明归因正确。

两个主要回退：
- `constraint_search_no_file`：3 次将全 CS 的 aspect="*" 错写成 "CL"。
- `state_cache_two`：3 次只检查第一篇缓存便 FINISH，漏掉第二篇。

### 1.2 已有 25-step Follow-up Pilot

2026-11-24 的扩充数据短程实验：

- 使用同一 Native SFT Adapter。
- 18 个任务进入 Train Split，25 optimizer steps，100 条正式训练轨迹。
- Dev Strict Success：SFT 20/24，旧 Prefix-tree Pilot 18/24。
- 多步骤完整执行：5/6 → 6/6。
- 缓存检查：2/3 → 3/3。
- 状态与约束任务：5/6 → 3/6。
- 论文引用：11.5/12 → 12/12。
- PENDING 提交后正确结束：2/2 → 0/2。
- 3 次全 CS 检索仍然把 aspect="*" 改为 "CL"。
- 本轮在 25 步后按预先设定的标准停止，没有扩大到 100 步。

该 Pilot 是本轮 E0 的优先基线。若其 Train Split SHA、SFT Adapter SHA、Snapshot SHA、Prompt/Tools Schema SHA 与本轮固定资产完全一致，可直接复用其结果；若任何一项不一致，必须重跑 E0，禁止把不同数据/环境的结果伪装成同条件消融。

### 1.3 训练资产身份

以仓库中的 manifest 和本地实际文件双重校验：

- Base：Qwen3.5-2B-Base（沿用现有 Native GRPO 配置）。
- Native SFT Adapter SHA256：`789af8cf28960892a1fde034d32f9b8ddf230e8450552be70d15024c964ed6c8`
- SFT Train Manifest SHA256：`34196a2639203ca290a43c1012daa1f82f4621a4f8beb2203c597713d013bf70`
- Mock Snapshot SHA256：`d97035db326ca11935e882fd2aca7fe932e9e76cf1bc8a21078be7bec1ae8794`
- 当前固定候选/训练资产：`data/splits/native_grpo_qwen35_candidates_v2.json`、`data/splits/native_grpo_qwen35_train_v2.json`。
- Train v2 当前 18 个任务 ID 必须从文件读取，不得手写复制；包括搜索范围、双论文缓存、条件下载/缓存、PENDING/翻译、引用错误和 blocked 控制任务。
- 旧 100-step GRPO Adapter 和本轮 25-step Pilot Adapter 都不能作为新实验起点。所有新实验从同一个 Native SFT Adapter 加载。

---

## 2. Phase A：构建 Weakness Bank，并让 Probe 真正影响训练

### 2.1 新增 Weakness Bank

新增一个确定性、可审计的弱点记录格式，建议保存为 JSON：

`reports/grpo_credit_v2/weakness_bank.json`

至少包含：

- `weakness_id`
- `task_family`
- `error_type`
- `evidence_source`
- `pass_to_fail_count`
- `fail_to_pass_count`
- `probe_groups`
- `informative_groups`
- `train_task_ids`
- `train_exposure_groups`
- `last_observed_checkpoint`
- `priority`
- `status`（uncovered / covered / improved / unresolved / no_signal）

初始弱点优先级必须来自已有报告，不要凭主观猜测：

| 弱点 | 已有证据 | 优先级 |
|---|---|---|
| 全 CS 搜索范围被缩小为单一子领域 | 100-step 配对评测 3 次回退；25-step Dev 3 次回退 | P0 |
| 多论文任务漏查第二篇缓存 | 100-step 配对评测 3 次回退 | P0 |
| PENDING 状态被错误处理或提前终止 | 25-step Dev 的 PENDING 指标 2/2 → 0/2 | P0 |
| 多步骤任务提前 FINISH | 已有多步骤完整执行回退和错误轨迹 | P0 |
| 引用/参数错误 | 旧报告中存在参数和论文身份错误 | P1 |
| 重复调用 | 旧 GRPO 已改善，保留监控，不再作为主要训练目标 | P2 |
| 工具执行错误 / 解析错误 | 旧 GRPO 已改善，保留监控 | P2 |

Weakness Bank 的来源包括：
- 上述已提交的配对评测和 Probe 汇总；
- 本轮 GPU 训练时新增的 rollout audit；
- 每轮 Dev 的 paired case audit。

**禁止数据泄漏：** 可以使用 Dev/OOD 的错误类型来确定“要研究哪种能力”，但不得将 Dev/IID/OOD 的原始请求、精确参数、Setup、论文 ID 或标准动作序列放入 Train。新增训练任务必须从 Train 父任务 TaskSpec 派生，并做文本、参数、Setup、动作签名的 holdout leakage audit。

### 2.2 Probe 规则

优先复用 2026-11-24 已有的 Frozen SFT Probe 和 Train v2 Split，不要无理由重跑 168 组 GPU Probe。

如果本地原始 Probe 轨迹存在：
- 重新计算新 DFA / Credit 诊断；
- 保留每个 task/group/generation 的原始轨迹和 seed；
- 统计每类弱点的可训练组覆盖。

如果只有已提交的 Probe 汇总：
- 使用汇总构建 Weakness Bank；
- 不得伪造逐轨迹指标；
- 当前固定 Train v2 Split 可继续用于消融。

只有在确认 Train v2 缺少 P0 弱点的训练任务时，才允许从 Train 父任务派生补充变体。若补充任务导致 Train Split 变化，**E0 也必须在新 Split 上重跑 25 steps**，保证 E0–E4 可比。不能用旧 Split 的 E0 与新 Split 的 E1–E4 做无条件结论。

### 2.3 Weakness-aware Group Sampling

实现按“任务组”而不是单条 completion 采样的分层采样器。每个 task prompt 仍由 GRPO 展开为 G=4 条 generation；不得把同一组四条轨迹拆散。

初始采样配比：
- 50%：P0 弱点任务（全 CS 范围、双论文缓存、PENDING、提前结束）。
- 30%：多步骤组合任务。
- 20%：普通控制任务和能力边界任务。

以上是本轮固定的初始采样配比，不要在消融中动态调整。实现时：
- 如果某个类别任务数量不足，按类别内可用任务归一化，不重复伪造任务；
- 记录每个 task family 的实际 group exposure；
- 每个 optimizer step 记录抽到的 task_id、weakness_id、group_id；
- 不允许因为任务困难就只保留成功 rollout；
- 不启用动态难度筛选，避免与 Credit 方法同时变化。

### 2.4 Probe/训练覆盖验收

每个 P0 weakness 必须有以下统计：
- Probe 中出现多少个有效对照组；
- 正式训练中抽到多少组；
- 这些组中实际出现多少次目标错误动作；
- 这些错误动作获得了什么局部 Advantage；
- 目标正确动作是否获得了正向或至少非负的局部 Advantage。

不能只报告“任务进入了 Train Split”。**进入 Split 不等于模型真的见到了错误，也不等于产生了有效梯度。**

---

## 3. Phase A：TaskSpec-derived DFA / Verifiable Process Labels

### 3.1 DFA 状态定义

从现有 TaskSpec 的有序 `steps` 编译有限状态机，不手写每个任务的专用状态机。

对具有 L 个预期工具动作的任务，定义：
- `q_0 ... q_L`：已按顺序验证通过 0 到 L 个预期工具步骤；
- `q_accept`：全部必要动作已完成，且终止语义正确；
- `q_error`：第一次偏离任务契约；
- `q_truncated`：生成预算或轮数耗尽，尚不能判定为错误；
- `q_environment_invalid`：Setup / Snapshot / 环境基础设施失败。

每个决策转移必须检查：
1. Tool Name；
2. 生效参数（使用 `registry.effective_arguments()`，与 Benchmark 的默认值语义一致）；
3. 论文引用实际解析身份（使用 Observation 中解析出的 paper_id；不能只比传入 ref 字符串）；
4. 工具是否真实执行成功；
5. 用户约束和工具副作用；
6. FINISH 是否符合完整步骤和 terminal semantics。

正确动作转移到下一状态；第一次错误工具、错误参数、错误论文、额外调用或提前 FINISH 转移到 q_error。q_error 对当前严格序列任务是吸收态，不要给后续动作伪造正确标签。

blocked task 必须沿用 `expected_terminal_mode` / `expected_terminal_reason` 的现有语义：正确行为可能是零工具调用并说明具体阻塞原因，不能强制要求至少调用一个工具。

### 3.2 语义一致性

不要复制一套与 Benchmark 不一致的参数比较逻辑。优先复用或提取共享 helper：
- `benchmark.metrics.argument_match_score`
- `benchmark.metrics.resolved_paper_id`
- `benchmark.metrics._same_paper`（必要时提取成公开 helper）
- `tools.tool_registry.registry.effective_arguments`
- blocked terminal semantics 的现有校验。

必须覆盖：
- aspect 的大小写和 cs. 前缀；
- default 参数省略与显式默认值；
- bool 与数值类型；
- save_to_file / output_path 的实际副作用；
- ref 的序号、ID、标题片段和 null 最近论文语义；
- 论文版本号及实际解析到的 paper_id；
- PENDING 与 READY 的区别；
- 多篇论文逐篇核验；
- FINISH、额外调用和提前结束；
- blocked 任务的具体原因。

### 3.3 DFA 输出

每条轨迹必须得到逐步结构化记录：

- `automaton_state_before`
- `action_key`
- `automaton_state_after`
- `transition_label`（correct / first_error / post_error / accept / truncated / environment_invalid / neutral）
- `first_error_index`
- `first_error_kind`
- `verified_prefix_length`
- `strict_success`

不能只保存一个 `phi` 数值。必须能追溯到每一步的判定依据。

---

## 4. Phase A：重新设计 Credit Assignment

保留旧 Prefix-tree 代码和测试，新增独立的 v2 实现。建议文件：
- `AgenticArxiv/rl/task_automaton.py`
- `AgenticArxiv/rl/anchor_state.py`
- `AgenticArxiv/rl/credit_v2.py`

最终文件名可以根据现有架构调整，但不得把所有逻辑堆进一个函数。

### 4.1 E2：DFA Local Credit + First-error Protection

这一分支不使用 GiGPO Anchor-state Grouping。

DFA Local Credit：
- 对同一 task group 中处于相同 DFA contract state 的决策进行分组；
- 正确完成当前预期动作的决策标记为 correct；
- 错误工具、错误参数、错误 ref、提前 FINISH 标记为 wrong；
- 环境无效、截断、无法判定的决策不参与比较；
- 仅当同一 contract state 下至少存在两个可比较决策，且标签/回报存在差异时，才产生局部信号；
- 组内 Advantage 必须中心化/归一化，避免给所有样本无差别的常数正奖励；
- 如果组内所有轨迹在该状态都选择同一正确动作，不伪造相对优势。

First-error Protection：
- 对失败轨迹，第一次错误之前经过验证的正确前缀不得继承负向 episode Advantage；
- 第一个错误动作获得明确的负向局部信号；
- 第一个错误之后的 suffix 默认不再按“专家正确/错误”强行打标签；除非有独立的状态级证据，否则 mask 或置零；
- 对成功轨迹保留正向 episode Advantage；
- 对环境无效轨迹不施加策略惩罚；
- 对截断轨迹保留已完成前缀的可验证信息，但不能把未生成的 FINISH 伪造为成功。

实现后必须输出一个可读的单条轨迹解释，例如：
- step 0：correct，prefix protected；
- step 1：correct，prefix protected；
- step 2：first_error，negative credit；
- step 3：post_error，masked。

### 4.2 E3：GiGPO-style Anchor-state Credit，不启用 First-error Protection

参考 GiGPO 的 episode-level + step-level 双层 Advantage。

Anchor State 必须表示“模型当前实际面对的决策状态”，不能简单把相同 Action Prefix 或相同 DFA 进度当作完整状态。

至少包含：
- task identity；
- 当前完整模型可见上下文的确定性 fingerprint（包括此前 Assistant 输出和 Tool Observation）；
- 当前 MockArxivEnv / MemoryStore 中影响下一步工具决策的状态；
- 当前 TaskSpec automaton state；
- 必须排除无语义的随机 session_id、临时路径、时间戳、统计计数器等字段。

实现一个确定性 canonical serializer + hash：
- 同一模型可见上下文和同一语义环境状态必须得到相同 fingerprint；
- 不同论文列表、不同 PDF cache、不同 translation status 必须得到不同 fingerprint；
- 不要直接对 Python pickle 或对象内存地址求 hash；
- 在训练和离线重放中使用相同的 canonicalization 规则。

重要区分：
- **Exact Anchor State**：用于 GiGPO 局部比较，要求模型可见上下文和决策相关环境状态相同。
- **DFA Contract State**：用于专家步骤标签和任务进度诊断；不能在没有状态等价证据时冒充 Exact Anchor State。

GiGPO-style Advantage：
- Episode Advantage：来自同一 prompt 的 G=4 条完整轨迹的 Strict Outcome Reward 组内归一化。
- Anchor Advantage：对跨轨迹重复出现的 Exact Anchor State，收集从该状态采样的动作及其后续 return，在同一 Anchor State 组内计算相对 Advantage。
- 同一 Anchor State 下必须至少有两个有效样本；若没有重复状态、无回报方差或只有同一动作且同一结果，则不产生伪造局部信号。
- 不额外增加 rollout；只使用当前 GRPO 已采样的轨迹。
- 对不同状态下的动作不得直接比较。
- 计算局部 return 时，终点必须使用 Strict Outcome Reward；过程信息只来自本任务明确启用的过程信号，不得重新引入旧五维加权总 Reward。
- 记录 anchor group size、不同 action 数、return variance、局部 Advantage、coverage 和 no-signal 原因。

E3 不使用 First-error Protection，目的是单独测量 GiGPO 的效果；DFA 仍可用于诊断，但不能将 First-error mask 混入 E3。

### 4.3 E4：GiGPO + First-error Protection

组合 E2 与 E3：
- episode-level Strict Outcome Advantage；
- GiGPO Exact Anchor-state Advantage；
- DFA First-error Protection；
- 所有信号必须分别记录，最终合并过程可审计。

不要在 E4 中把两种局部 Advantage 简单相加后不做尺度检查。先分别归一化，再用固定权重组合，并报告各通道的均值、方差、正负比例和对最终 token Advantage 的贡献。

### 4.4 Token Advantage 对齐

沿用现有 Native token span / tool mask 基础设施：
- Assistant Thought / Tool Call / Final Answer 的真实生成 token 可以参与 loss；
- Tool Observation、Setup、Padding 不参与 loss；
- 每个 Assistant turn 必须准确映射到对应的 DFA decision / Anchor State；
- 不允许把下一轮的 credit 对齐到上一轮 token；
- 终止 FINISH 的 token span 必须单独保留；
- 若 rollout 没有对应 turn span，不得猜测映射。

建议最终形式：

\[
A_{i,t}^{token}
=
(1-\lambda) A_i^{episode}
+
\lambda A_{i,t}^{local}
\]

`lambda=0.5` 作为本轮固定初始值，与现有 0.5 step-credit 比例保持可比；E1 令 `lambda=0`。E4 的 local channel 可先将归一化后的 DFA 和 Anchor Advantage 等权合并，再进入上述公式。若代码实现发现该组合导致尺度异常，必须在 CPU 离线测试阶段修复，不要在 GPU 正式训练时临时改权重。

---

## 5. Reward v2：Strict Outcome 为唯一终局优化目标

### 5.1 Strict Outcome Reward

E1–E4 统一使用：

\[
R_{outcome}(\tau)=
\begin{cases}
1,& \text{is\_strict\_success}(\tau)=True\\
0,& \text{otherwise}
\end{cases}
\]

其中 Strict Success 继续调用当前 `benchmark.metrics.is_strict_success()`，不得修改定义。

环境异常必须单独 mask：
- Snapshot coverage miss；
- Setup failure；
- 与模型无关的 Mock 环境执行异常；
- 轨迹记录或 tool_call_id 对齐基础设施错误。

模型自己生成的非法 JSON、Schema 错误、错误工具、错误参数、错误引用、提前 FINISH 等仍属于策略错误，不能伪装成 environment_invalid。

### 5.2 旧 Reward 的处理

旧五维 Reward（Format / Tool / Argument / Process / Outcome）：
- 保留在旧实现和历史复现实验中；
- 本轮可继续计算并记录 component breakdown，作为诊断；
- 不再作为 E1–E4 的优化总 Reward；
- 不得把 component 平均分当成 Strict Success；
- 不得把 Reward 上升解释为 Benchmark 成功率上升。

### 5.3 为什么 E1 需要 Strict-only

E1 是严格 Outcome-only Vanilla GRPO：
- 只用 0/1 Strict Outcome Reward；
- 不用 DFA Local Credit；
- 不用 GiGPO；
- 不用 First-error Protection；
- 用来测量 Strict Outcome 稀疏奖励在当前任务上的可训练性。

如果四条轨迹全部失败或全部成功，组内 Advantage 为零是预期现象，必须报告 zero-variance group fraction，不能通过临时加入旧 Reward 来“救活”E1。

### 5.4 Reward / Advantage Calibration（IRC 思路）

参考 Iterative Reward Calibration 的核心教训：Dense Reward 可能与最终 Advantage 方向错位。

本轮先做离线校准，不动态调参：
- 对每个 action class 统计 Strict Success / Failure；
- 统计同一 DFA state 下 correct vs wrong 的配对结果；
- 统计每类动作的正向、负向、零向 Advantage 比例；
- 检查经过 group normalization 后，正确动作是否仍得到较高 Advantage；
- 检查错误动作是否被错误鼓励；
- 检查已验证正确的前缀是否被负向 Advantage 污染。

如果校准结果显示某类 process credit 与 Strict Success 方向冲突：
- 该类别的局部 credit 应置零或标记为 unresolved；
- 不得为了覆盖率强行赋正负值；
- 记录为什么该类信号不可信。

---

## 6. 新的 Credit 诊断指标：不要继续只报告旧版 40%

必须实现并输出：

1. **DFA First-error Localization Accuracy**：定位到的第一个错误位置是否与独立 TaskSpec / Benchmark 判定一致。
2. **Verified-prefix Preservation Rate**：失败轨迹中，已验证正确前缀没有被负向 process Advantage 污染的比例。
3. **First-error Suppression Rate**：首个错误动作获得负向 local Advantage 的比例（仅统计有有效对照信号的组）。
4. **Correct-action Promotion Rate**：正确动作在存在有效对照时获得正向 local Advantage 的比例。
5. **False-positive Credit Rate**：错误动作却获得正向 local Advantage 的比例。
6. **False-negative Credit Rate**：正确动作却获得负向 local Advantage 的比例。
7. **Anchor-state Coverage**：参与有效 Anchor Group 的决策数 / 可参与决策总数。
8. **Anchor Group Size Distribution**：重复 Anchor State 的 group size 分布。
9. **Anchor Pairwise Ranking Accuracy**：同一状态下，正确动作的局部 Advantage 是否高于错误动作；没有正确/错误动作对照的状态不计入。
10. **Advantage Misalignment Rate**：局部 Advantage 方向与可验证 Strict Outcome / TaskSpec 动作标签冲突的比例。
11. **No-signal Reason Breakdown**：identical action / zero variance / singleton state / truncation / environment invalid 等。
12. **Per-weakness Exposure**：每类目标弱点实际出现次数、产生有效 credit 次数和被更新 token 数。

所有指标必须提供 numerator / denominator，不能只给百分比。无有效样本时输出 N/A，不能输出 0% 造成误解。

---

## 7. 消融矩阵：固定数据、固定起点、固定超参数

### 7.1 E0–E4 定义

| ID | Outcome Reward | Local Credit | First-error Protection | 用途 |
|---|---|---|---|---|
| E0 | 旧五维 Reward | 旧 Prefix-tree | 无 | 已有方法基线，优先复用 2026-11-24 25-step Pilot |
| E1 | Strict 0/1 | Vanilla episode GRPO | 无 | 测量 Strict-only 稀疏奖励 |
| E2 | Strict 0/1 | DFA Local Credit | 有 | 测量规则步骤监督 + 正确前缀保护 |
| E3 | Strict 0/1 | GiGPO Exact Anchor-state | 无 | 测量同状态动作比较 |
| E4 | Strict 0/1 | DFA + GiGPO | 有 | 完整方法 |

注意：
- E2 和 E3 都可以使用 DFA 做标签/诊断；E3 的实际局部 Advantage 必须来自 GiGPO Anchor State，而不是 DFA First-error mask。
- E4 同时启用 DFA Local Credit、GiGPO Anchor Credit 和 First-error Protection。
- E0 使用已存在的 25-step Pilot；如果资产或 Split 不同，重跑 E0。
- 如果 E0–E4 使用的任务集合不同，必须统一任务集合并重跑 E0。
- 不允许 E0 用 100 steps，而 E1–E4 用 25 steps 后直接比较成功率。

### 7.2 固定配置

沿用现有 Qwen3.5-2B Native GRPO 配置，除消融变量外全部固定：

- Native SFT Adapter 起点；
- Base Model / Tokenizer；
- Native System Prompt；
- Tool Schema；
- Qwen Chat Template；
- Mock Snapshot；
- Train Split；
- Dev Split；
- seed=42；
- temperature=1.0；
- num_generations=4；
- learning_rate=5e-6；
- KL beta=0.04；
- scale_rewards="group"（只作用于各实验定义的 Outcome Advantage）；
- bf16；
- per_device_train_batch_size=1；
- gradient_accumulation_steps=4；
- max_completion_length=1792（先核验最长专家轨迹）；
- max_new_tokens_per_turn=704；
- max_turns=5（若 TaskSpec 最长任务需要更多轮，先通过 CPU 审计后统一调整所有实验）；
- 25 optimizer steps；
- dynamic_sampling=false；
- 单 GPU、单进程、实验顺序执行，不并行加载多个模型。

若实际代码的参数名不同，适配现有配置，不要静默改变数值。

### 7.3 评测协议

每个实验训练后：
- 在同一个 Dev Split 上运行同一组 24 次试验（8 tasks × 3 trials）；
- 使用相同的 seed、Prompt、Tools、Snapshot、解码参数和 Strict Success 实现；
- 保存逐试次结果并做 paired comparison；
- 报告 Strict Success、错误类型、多步骤完整执行、参数、论文引用、PENDING、工具错误、重复调用、截断；
- 不用训练 Reward 选模型；
- 不允许根据 Dev 结果临时修改评测任务或阈值。

SFT 当前 Dev 基线为 20/24；已有 E0 Pilot 为 18/24。新方法必须至少报告是否超过这两个基线。

IID/OOD 不用于训练选择。只有某个候选在 Dev 达到或超过 Native SFT 的 20/24，且没有明显的状态/约束退化，才对该候选运行 IID/OOD 留出评测。

### 7.4 25-step Pilot 的解释边界

25 steps 是机制验证，不足以证明最终收敛。
- 不能把 1–2 次成功差异夸大为稳定提升；
- 同时报告每个 task family 的 paired case；
- 对 24 次 Dev 结果给出置信区间或精确计数；
- 如果 E1–E4 都低于 SFT，不继续扩大训练，先解释信号覆盖和 Advantage alignment；
- 如果 E2/E3/E4 某一项明显更好，再为该项安排 100-step 后续正式实验。

---

## 8. Phase A：Codex 必须先完成的 CPU 工作

Codex 应自主完成以下所有工作，不需要用户启动 GPU：

### A1. 代码审计
- 读取本任务书引用的实验报告；
- 审计 `reward.py`、`grpo_reward.py`、`step_credit.py`、`step_credit_trainer.py`、`native_grpo_rollout.py`、`env.py`、`store_memory.py`、`side_effects.py`、`task_spec.py`、`metrics.py`；
- 明确现有 TRL 1.17.1 的 Advantage / mask / token span 对齐方式；
- 不破坏旧 Prefix-tree 的可复现性。

### A2. Weakness Bank 和数据审计
- 从已有报告构建 Weakness Bank；
- 检查 Train v2 的 18 个任务是否覆盖 P0 弱点；
- 做 Train / Dev / IID / OOD 泄漏审计；
- 固定 E0–E4 使用的共同 Train Manifest；
- 输出训练采样计划和目标弱点 exposure 计划。

### A3. DFA / Credit v2 实现
- TaskSpec-derived DFA；
- 精确的模型可见上下文与环境状态 fingerprint；
- GiGPO Anchor-state grouping；
- DFA Local Credit；
- First-error Protection；
- per-token Advantage 对齐；
- 按样本处理 truncated / environment_invalid，不能让单个无效样本无理由关闭整个四样本组的有效 Credit；
- 记录可解释的逐步 Credit 轨迹。

### A4. Reward v2 实现
- E1–E4 统一 Strict Outcome 0/1；
- 旧五维 Reward 只保留为诊断和历史兼容；
- 过程信号独立进入局部 Advantage 通道；
- 环境异常 mask 与策略错误惩罚严格区分；
- 实现 IRC 风格离线对齐审计。

### A5. 单元测试与离线验证
至少覆盖：
- DFA 正确工具、错误工具、错误参数、错误 ref、额外调用、提前 FINISH、成功 FINISH；
- blocked task、PENDING/READY、双论文缓存、全 CS 搜索范围；
- 相同 Anchor State 得到相同 fingerprint；
- 不同 paper/cache/translation state 得到不同 fingerprint；
- ephemeral session_id / timestamp 不污染 fingerprint；
- Anchor Group 单例、零方差、同动作、不同动作；
- local Advantage 正负方向；
- First-error 前缀保护、first-error 惩罚、post-error mask；
- environment_invalid 和 truncation；
- Assistant token span、Tool Observation mask、Padding mask；
- GRPO batch 顺序与 G=4 group 对齐；
- 旧 Prefix-tree 现有测试继续通过。

先跑：
- 相关 pytest；
- 完整 AgenticArxiv 测试套件（若环境允许）；
- py_compile / import smoke；
- CPU dry-run；
- reward/credit offline audit。

### A6. GPU 运行脚本和报告
新增一个统一入口，例如：
- `scripts/run_grpo_credit_v2_ablation.sh`
- `scripts/run_grpo_credit_v2.py`
- `scripts/audit_grpo_credit_v2.py`

具体文件名可按仓库惯例调整。

要求：
- 支持 `--dry-run`；
- 支持单独运行 E1/E2/E3/E4；
- 支持从检查点恢复；
- 每个实验使用独立 output directory；
- 自动核验 Base / Adapter / Snapshot / Split SHA；
- 自动保存 config、seed、runtime、GPU 型号、峰值显存、训练日志、rollouts、Credit audit、Adapter SHA、Dev paired results；
- 任何资产身份不匹配必须 fail fast；
- 不允许把旧结果覆盖。

### A7. CPU 阶段完成条件

生成：

`reports/grpo_credit_v2/gpu_readiness.md`

必须包含：
- 完成的代码改动；
- 相关测试结果；
- 固定 Train / Dev Manifest SHA；
- Base / Adapter / Snapshot SHA；
- 预计显存；
- GPU 命令；
- 实验顺序；
- 每个实验的预计时间；
- 预计总 GPU 时间；
- 当前已知风险；
- 是否需要重跑 E0。

**只有该文件完整且 CPU 测试通过后，才通知用户启动 GPU。此时停止，不要提前运行 GPU 训练。**

---

## 9. Phase B：用户启动 RTX 4090 后的 GPU 工作

### 9.1 硬件和运行方式

目标 GPU：单张 NVIDIA RTX 4090，24GB VRAM。

沿用：
- BF16；
- micro batch 1；
- gradient accumulation 4；
- G=4；
- gradient checkpointing；
- 单进程；
- 每次只加载一个实验模型；
- 每个实验结束后释放模型、清理 CUDA cache，再加载下一实验。

之前的 25-step Native GRPO 在 RTX 4090 上：
- 峰值已分配显存约 6.01 GiB；
- 100-step 实验约 6.11 GiB；
- 因此 24GB 对当前 2B LoRA 训练配置足够。
- 新的状态快照 / Anchor State 主要增加 CPU 侧数据结构和日志开销；必须实测并记录新峰值，不得承诺一定与旧峰值相同。

### 9.2 GPU 执行顺序

1. 检查 `nvidia-smi`、CUDA、Torch、BF16、磁盘空间；
2. 重新核验所有 SHA；
3. 运行每个配置的 4-step Smoke；
4. Smoke 通过后，按 E1 → E2 → E3 → E4 顺序执行 25-step Pilot；
5. 每个实验完成后立即做 24 次 Dev 评测并保存；
6. 不并行运行多个实验；
7. E0 复用已验证的 25-step Pilot；若资产 / Split 不一致则在 E1 前重跑 E0；
8. 生成总消融报告和 paired case audit；
9. 根据 Dev 预先设定的选择规则决定是否对单个最佳候选跑 IID/OOD；
10. 完成后通知用户关闭 GPU。

如果某一配置 Smoke 失败、Advantage 非有限、Credit 与 token span 不一致、出现大量 environment_invalid 或出现明显的错误动作正向 Credit，立即停止该配置，不要继续消耗 GPU。

### 9.3 GPU 时间预算

根据本项目已有 RTX 4090 实测：
- 25-step 训练约 25.2 分钟；
- 4-step Smoke 约 5.5 分钟；
- 24 次 Dev 评测约 8–10 分钟（按旧流程估算，须记录新实测）。

E1–E4 四个新配置的 GPU 预算：
- 4 × 25-step train：约 100–120 分钟；
- 4 × Smoke：约 22–30 分钟；
- 4 × Dev evaluation：约 32–40 分钟；
- 模型加载、保存、审计和清理：约 20–40 分钟。

**预计 GPU 连续占用约 3–4 小时。** 这是规划估计，不是保证。若必须重跑 E0，再增加约 40–50 分钟。若发现必须重新做大规模 Frozen Probe，再额外预留约 2 小时；除非固定 Split 不可用，否则不要重跑。

CPU 阶段开发时间不计入 GPU 时间，Codex 应尽可能并行完成。

---

## 10. 最终报告与交付物

至少提交：

- `reports/grpo_credit_v2/weakness_bank.json`
- `reports/grpo_credit_v2/gpu_readiness.md`
- `reports/grpo_credit_v2/ablation_report.md`
- `reports/grpo_credit_v2/ablation_results.json`
- `reports/grpo_credit_v2/paired_dev.json`
- `reports/grpo_credit_v2/credit_alignment_audit.json`
- `reports/grpo_credit_v2/anchor_state_audit.json`
- `reports/grpo_credit_v2/reward_calibration_audit.json`
- 各实验独立的 config、training metrics、rollout trace、checkpoint/Adapter SHA 和评测记录。

报告必须回答：

1. Weakness Bank 中的 P0 问题是否真正进入训练？
2. 正式训练实际采样到了多少目标错误动作？
3. E1 Strict-only 的零方差比例是多少？
4. DFA 是否准确定位了第一次错误？
5. 正确前缀是否避免了负向污染？
6. GiGPO Anchor State 有多少有效重复状态？覆盖率是多少？
7. 同一状态下正确动作的 Advantage 是否高于错误动作？
8. E2 / E3 / E4 哪一项改善了 Strict Success？
9. 是否存在 Reward / Advantage Misalignment？
10. 与 Native SFT 20/24、E0 18/24 相比，是否有可靠改善？
11. 哪些失败是策略错误，哪些是环境或截断问题？
12. 下一轮值得增加 100 steps 的唯一候选是什么？为什么？

不允许只报告平均 Reward、Loss 或训练成功。主结果必须是同协议 Strict Success 和逐试次配对变化。

---

## 11. 论文依据与实现边界

本轮实现参考以下工作。应阅读论文方法部分和必要的官方代码；只迁移与 ArXivFetch 环境兼容的部分，不照搬其他 benchmark 的任务假设。

1. **GiGPO — Group-in-Group Policy Optimization for LLM Agent Training, NeurIPS 2025**  
   https://proceedings.neurips.cc/paper_files/paper/2025/hash/420c9f777c0b4f78d515e53cf74d58b2-Abstract-Conference.html  
   核心参考：episode-level + anchor-state grouping 的 step-level relative advantage；同一环境状态下的动作比较；不依赖额外 critic 或额外 rollout。官方代码：https://github.com/langfengQ/verl-agent

2. **Multi-Turn Reinforcement Learning for Tool-Calling Agents with Iterative Reward Calibration, 2026**  
   https://arxiv.org/abs/2604.02869  
   核心参考：Dense Reward 可能与最终 Advantage 方向错位；先通过 rollout 数据分析 reward discriminativeness 和 advantage alignment，再校准过程信号。

3. **Save the Good Prefix: Precise Error Penalization via Process-Supervised RL to Enhance LLM Reasoning, Findings of ACL 2026**  
   https://aclanthology.org/2026.findings-acl.1767/  
   核心参考：定位 first error，保护已经验证正确的 prefix，把负向信号集中到错误位置，而不是惩罚整条轨迹。

4. **GLARE: Scalable Neuro-Symbolic Reward Shaping for LLM Agents via Group-Level Automata, ICML 2026**  
   https://proceedings.mlr.press/v306/yan26k.html  
   核心参考：将任务进度转换为确定性自动机状态，使用可验证状态转移产生过程信号。我们的第一版直接从 TaskSpec 编译 DFA，不需要额外 LLM 抽取 LTL。

5. **ProVer: Targeting Pivotal Decisions for Credit Assignment in Agentic Reinforcement Learning, 2026**  
   https://arxiv.org/abs/2609.36178  
   作为后续 P1 方向：Judge 只提议关键片段，最终信用通过当前策略在片段前后恢复状态并续跑的 Strict Success 差异验证。本轮不实现全量 ProVer，以控制额外 Rollout 成本。

---

## 12. Codex 给用户的阶段性反馈格式

CPU 阶段完成后，只需要一次明确反馈：

- 分支和最新 commit；
- 实现了哪些模块；
- 哪些测试通过 / 未通过；
- Train / Dev / SFT / Snapshot SHA；
- E0 是否可复用；
- GPU readiness report 路径；
- RTX 4090 预计连续运行时间；
- 用户需要启动 GPU 后执行的唯一入口命令。

不要要求用户逐个选择 E1–E4，也不要要求用户重新解释本任务书。

GPU 阶段结束后：
- 给出 E0–E4 Strict Success 对比；
- 给出 Credit Alignment 指标；
- 给出最典型的成功 / 失败配对轨迹；
- 给出是否值得扩大到 100 steps 的明确结论；
- 告知用户可以关闭 GPU。

**最终成功标准不是训练 Loss 下降，也不是平均 Reward 上升，而是：在同一评测协议下，正确动作获得可验证的正向局部信号、错误动作获得负向局部信号，且 Strict Success 相比 Native SFT 基线出现可复现的改善。**
