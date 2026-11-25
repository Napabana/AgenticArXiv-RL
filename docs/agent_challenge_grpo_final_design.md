# AgenticArXiv-RL：固定 Harness 下的复杂任务 GRPO 设计

**文档类型：** 最终一轮 Agent GRPO 实验设计与 Codex 实施任务书\
**研究主线：** SFT v2 提供基础工具调用能力；GRPO
优化复杂、长程、多状态依赖任务中的策略执行。\
**实现仓库：** `Napabana/AgenticArXiv-RL`\
**实现基线分支：** `exp/native-sft-v2`\
**基线提交（编写本文时）：** `77c4536fbce4e3346f10e00da9a1d6269a44a250`\
**建议实现分支：** `exp/agent-challenge-grpo-final`\
**策略模型：** `Qwen/Qwen3.5-2B-Base` + Native SFT v2 Adapter\
**训练硬件：** 单张 RTX 4090，24GB\
**建议独立设计仓库名：** `AgenticArXiv-Challenge-Benchmark`

------------------------------------------------------------------------

# 0. 研究方向：SFT 学基础能力，GRPO 学复杂任务执行

## 0.1 核心研究问题

本轮研究问题调整为：

> 在固定 Harness、System Prompt、Native Tool Schema 和 Mock Snapshot
> 的条件下，已经通过 SFT v2 获得基础工具调用能力的小参数 Agent，能否通过
> GRPO 进一步提高复杂、长程、多状态依赖任务的 Strict Success？

本轮不再把"模型能否学会单个工具调用、参数格式或基础规则"作为主要目标。这些属于
Native SFT v2 已经覆盖的能力。

新的能力分工：

  -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  阶段                    主要职责                                                                                                   不作为本轮重点的内容
  ----------------------- ---------------------------------------------------------------------------------------------------------- ------------------------------------------------
  Native SFT v2           基础工具调用、参数映射、引用解析、用户约束、基本缓存与异步状态处理、基础终止行为                           不再通过机械增加单步示范追求更高训练集准确率

  GRPO                    多目标任务分解、跨工具依赖、长程状态跟踪、条件计划执行、多论文身份保持、复杂任务完整性、必要时的错误恢复   不把单步工具格式或简单参数映射作为主要训练目标
  -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

GRPO
仍然可能更新工具选择和参数生成，但训练任务必须要求模型在复杂任务中组合这些基础能力，而不是重复训练
SFT v2 已经掌握的孤立动作。

## 0.2 为什么必须换成复杂 Benchmark

此前在旧任务集上的 GRPO 没有产生稳定收益：

-   100-step Native GRPO：同协议 Strict Success 从 72/93 降至 65/93。
-   扩充任务后的 25-step GRPO：Dev 从 SFT 的 20/24 降至 18/24。
-   后一轮虽然多步骤完整执行从 5/6 提高到 6/6，但状态与约束任务从 5/6
    降到 3/6，PENDING 任务也出现回退。

旧报告： - `reports/native_grpo_20261123/native_grpo_report.md` -
`reports/native_grpo_followup_20261124/experiment.md`

需要明确实验起点：上述 100-step 和 25-step 旧 GRPO 结果使用的是此前 Native SFT v1 策略（旧报告记录的 SFT Adapter SHA 为 `789af8cf28960892a1fde034d32f9b8ddf230e8450552be70d15024c964ed6c8`），不是本轮 Native SFT v2 Adapter。它们说明旧任务集上的 GRPO 曾出现回退，但**不能据此断言 GRPO 在 SFT v2 起点上也无效**。本轮必须从 Native SFT v2 重新开始，使用新的复杂任务集验证其增量收益。

这些旧结果同时说明：只优化旧任务中的局部行为，可能减少重复调用，却不一定改善复杂任务的整体完成能力。本轮应将主要训练预算投入多步骤、多目标、状态分支和跨论文工作流。

## 0.3 本轮明确不做什么

1.  不修改或删减生产 `NATIVE_TOOL_SYSTEM_PROMPT`。
2.  不新增 Protocol-only Prompt，不做 Prompt Ablation。
3.  不改变 `build_visible_setup_context()` 的文本或行为。
4.  不修改 Native Tool Schema、工具描述和 Chat Template。
5.  不通过增加工具调用次数、降低 Strict 标准或改变 Reward
    定义制造训练信号。
6.  不新增外部论文，不联网补充 Snapshot，不伪造论文正文、工具
    Observation 或 READY 状态。
7.  不重新训练 SFT，不做 DPO、OPD 或教师模型蒸馏。
8.  不因 Probe 信号不足而强行启动 GPU 训练。

------------------------------------------------------------------------

# 1. 固定实验条件：Prompt、Harness、工具和模型

## 1.1 固定生产 Prompt

所有阶段必须使用 Native SFT v2 使用的生产提示词：

-   `AgenticArxiv/agents/prompt_templates.py::NATIVE_TOOL_SYSTEM_PROMPT`
-   `AgenticArxiv/agents/prompt_templates.py::build_visible_setup_context()`

Native SFT v2 记录的 System Prompt SHA256：

``` text
31674f2183d642a9035ecf8360c138135c5ef4645904017eadf4ec7bdf4abf05
```

要求： - Codex 必须在实际工作树中重新计算 SHA256。 - 如果与 Native SFT
v2 Manifest 不一致，停止并报告；不能自动接受新 Prompt。 - Setup Context
也必须保持原样，不能把"事实描述"改写成另一套提示词。 - 不新增 few-shot
示例，不删除策略规则，不修改措辞。 -
不通过隐藏状态或工具描述来人为提高任务难度。

**本轮研究的是：在真实生产 Prompt 已经提供基本规则的前提下，GRPO
能否提高复杂任务执行能力。** 结论只适用于这一固定 Harness
条件，不声称模型在没有 Prompt 的情况下独立内化了所有规则。

## 1.2 固定模型身份

所有 Probe、E1--E4 训练和评测都使用同一个 Native SFT v2 起点：

``` text
Base Model:
Qwen/Qwen3.5-2B-Base

Native SFT v2 Adapter SHA256:
f0c33a4d32a175bcaf1e83c55231a0108a84859cb748c318db49988a52f7911f

SFT v2 Data SHA256:
8763ff0ef7c19bf748d3d036d3730e8f30c44709161df614302a7a5df5bb3dfd

SFT v2 Manifest SHA256:
b8aa518f3f8aba708029bb1525b43d63372840a5bb785bf0fdde1439faf10751
```

以上 SHA 是当前 GitHub 训练 Manifest 中记录的身份。Codex
必须在实际运行环境再次核验文件，不能只复制本文中的字符串。

每个 E1--E4 模式必须独立从同一个 SFT v2 Adapter 初始化： - E2 不从 E1
继续训练； - E3 不从 E2 继续训练； - E4 不从其他模式继续训练； - 不覆盖
SFT v2 原 Adapter； - 每个模式输出独立 Adapter SHA。

## 1.3 固定 Native Tool Calling 协议

继续使用现有 10 个工具及其真实 Schema：

1.  `get_recently_submitted_cs_papers`
2.  `search_arxiv_papers`
3.  `download_arxiv_pdf`
4.  `get_paper_cache_status`
5.  `get_paper_content`
6.  `summarize_paper`
7.  `extract_paper_figures`
8.  `analyze_figure`
9.  `translate_arxiv_pdf`
10. `get_translated_content`

必须固定： - Native Tool Calling，不切换为 ReAct 文本协议； - Tool
Schema SHA256； - Tokenizer 和 Chat Template SHA256； -
当前推理模板设置； - 工具执行路径和 Mock 环境版本； - Prompt SHA256。

所有评测模型和所有 E1--E4 模式使用同一版本。

------------------------------------------------------------------------

# 2. Mock Snapshot：严格限制在现有 50 篇论文

## 2.1 固定 Snapshot

当前 Native SFT v2 / Native GRPO Manifest 记录的 Snapshot SHA256：

``` text
d97035db326ca11935e882fd2aca7fe932e9e76cf1bc8a21078be7bec1ae8794
```

Snapshot 文件没有直接作为普通 Git 跟踪文件提供给
Codex，因此必须从实际运行环境读取用户本地 Snapshot 并核验 SHA。

用户确认当前 Mock Snapshot 只有约 50 篇论文。Codex 必须实际统计唯一
`paper_id`： - 如果确认为 50 篇，继续； - 如果实际数量不是
50，停止并报告实际数量和差异； - 不得重复记录凑数； - 不得新增真实 arXiv
数据； - 不得联网补齐缺失字段。

## 2.2 Snapshot Inventory

先生成：

``` text
data/agent_challenge_final/snapshot_inventory.json
reports/agent_challenge_final/snapshot_audit.md
```

Inventory 至少记录：

-   Snapshot 路径、SHA256、唯一 paper_id 数量和完整 ID；
-   每篇论文的 title、abstract、categories、published、updated；
-   `get_recently_submitted_cs_papers` 的可用 aspect、days、max_results
    和结果顺序；
-   `search_arxiv_papers` 的全部精确 query keys 和对应结果；
-   `get_paper_content` 的 paper_id + section 覆盖；
-   `summarize_paper` 的 paper_id + normalized style + max_words bucket
    覆盖；
-   `extract_paper_figures` 的 paper_id 覆盖；
-   `analyze_figure` 的 paper_id + figure_no + question 覆盖；
-   `get_translated_content` 的 paper_id + page 覆盖；
-   哪些操作由 Snapshot 回放，哪些由本地 Mock
    实现，哪些是异步副作用模拟。

## 2.3 工具可执行性边界

必须尊重当前 `MockArxivEnv` 的真实实现：

-   `get_recently_submitted_cs_papers`：只能使用 Snapshot
    中实际存在的论文池和已验证的参数。
-   `search_arxiv_papers`：只能使用 Snapshot 已记录且 Gold Replay
    通过的精确 query；不能任意发明关键词。
-   `download_arxiv_pdf`：当前是离线下载桩，可能创建占位 PDF 并更新
    MemoryStore；占位 PDF 不等于真实论文正文。
-   `get_paper_cache_status`：从真实 Mock MemoryStore 状态读取。
-   `translate_arxiv_pdf`：当前 LocalSideEffectManager 默认返回
    PENDING；不得假设稍后自动 READY。
-   `get_paper_content`：必须命中实际 paper_id + section 记录。
-   `summarize_paper`：必须命中实际 paper_id + style + max_words bucket
    记录。
-   `extract_paper_figures`：必须命中实际 paper_id 记录。
-   `analyze_figure`：必须命中实际 paper_id + figure_no + question
    记录。
-   `get_translated_content`：必须命中实际 paper_id + page 记录。

额外限制： - 不使用 Snapshot 未覆盖的 section、figure、page、summary
style 或 question。 - 不把 Snapshot miss 归因为模型错误。 -
不把环境执行异常计为 Strict Failure。 - 不伪造 READY。只有 Snapshot
中真实存在翻译记录时才能建立 READY 状态。 - `days`
必须使用已记录参数，或逐篇验证论文时间确实满足用户请求。 - `max_results`
不能超过相应论文池长度。

## 2.4 Snapshot Views 与论文重叠

优先尝试按 paper_id 划分 Train / Dev / Sealed Test： - 先根据 Inventory
统计各工具覆盖； - 如果能在不丢失关键工具能力的前提下实现
paper-disjoint，优先使用； - 如果 50 篇论文不足以支持工具覆盖，允许退回
task-signature-disjoint； - 退回时必须报告 Train/Dev/Test 的 paper_id
overlap； - 不得将存在 paper_id overlap 的结果描述为 paper-level
generalization。

------------------------------------------------------------------------

# 3. Agent Challenge Benchmark v1：目标任务集

## 3.1 目标规模与切分

目标构建 44 个真正不同的任务实例：

  Split               目标数量 用途
  ----------------- ---------- -----------------------------------
  Train Candidate           24 Frozen Probe 后筛选 GRPO 训练任务
  Dev                        8 训练方案比较和诊断
  Sealed Test               12 最终独立评测
  合计                      44 ---

44 是目标，不是必须凑满的 KPI。如果 Snapshot
覆盖不足，宁可减少任务数量，也不能伪造论文、工具 Observation
或任务能力。

每个 Dev / Sealed Test 任务重复 3 次。Sealed Test 不参与
Probe、任务筛选、Reward 调整、超参数选择或 Checkpoint 选择。

## 3.2 任务复杂度标准

本轮复杂度不按"工具调用越多越好"定义，而按真实决策依赖定义。

一个复杂任务至少满足以下两项： - 有两个或以上相互关联的子目标； -
需要跨两个或以上工具类型； - 有状态分支，不同初始状态要求不同动作； -
需要在多篇论文之间保持身份和引用； - 前一步 Observation
会改变后续动作； - 有必须完成的顺序依赖； - 存在一个可验证的终止条件。

建议： - L1 控制任务：2--3 个模型工具调用，约 10%； - L2 组合任务：4--5
个模型工具调用，约 45%； - L3 长程任务：5--7 个模型工具调用，约 45%。

至少 60% 的 Train Candidate 应包含 4 个或以上模型工具调用。至少 50%
应包含两个或以上子目标。

不允许为了增加长度重复查询同一状态、重复下载同一论文或增加与用户目标无关的工具调用。

## 3.3 六类任务家族

  家族                              目标数量 主要能力
  ------------------------------- ---------- ------------------------------------------
  A. 多步骤长程工作流                     10 多子目标、跨工具依赖、完整执行
  B. 状态条件与缓存分支                   10 cache hit/miss、force、部分缓存状态
  C. 多论文身份与引用保持                  8 多论文、引用变化、paper identity
  D. 内容 / 摘要 / 图表组合                6 Snapshot 支持的内容读取和图表工作流
  E. 多约束组合                            6 搜索范围、保存限制、参数约束和多目标组合
  F. PENDING / READY / Recovery            4 异步状态、可恢复错误或明确阻塞
  合计                                    44 ---

若 Snapshot 不支持某类任务，例如没有 READY
翻译记录，则该类任务不得伪造。应将名额转给 Snapshot
支持的其他复杂任务，并记录原因。

## 3.4 家族 A：多步骤长程工作流（10 个）

重点：多个子目标必须全部完成，不能只完成第一个目标就 FINISH。

示例模板 1：

> 从当前论文列表中处理第 2 篇和第 4 篇论文。分别确认两篇 PDF
> 的缓存状态；只下载缺失的论文；读取第 2 篇的指定章节和第 4
> 篇的摘要；最后分别汇报两篇论文的处理结果。

Gold 轨迹必须由实际 Snapshot
和初始缓存状态确定，不能在任务文本中直接泄漏 expected tools。

示例模板 2：

> 检索指定类别的论文，下载目标论文，确认缓存后读取 Snapshot
> 支持的章节，再提取另一篇论文的图表并分析，最后分别汇报来源。

要求： - 至少 4 个必要模型工具调用； - 多个子目标有真实依赖； -
任何一步的 Observation 都可能影响下一步； - FINISH
前必须完成所有必要子目标； - 不允许通过无关工具调用获得局部 Reward。

## 3.5 家族 B：状态条件与缓存分支（10 个）

这是本轮重点之一。

使用**相同或高度相似的用户任务 + 不同真实 Setup 状态**，形成配对任务。

示例：

用户任务：

> 确保当前会话中指定的两篇论文都已经下载，并分别读取指定内容。如果文件已经存在，就保留现有文件；如果不存在，再下载。

不同初始状态：

-   B1：两篇均未下载；
-   B2：两篇均已下载；
-   B3：第一篇已下载，第二篇未下载；
-   B4：第一篇未下载，第二篇已下载。

正确策略由真实缓存状态决定。Setup 必须通过当前 Native Agent 和 Mock
工具执行，不得直接写 MemoryStore 假造状态。

其他任务： - 已缓存 + 用户明确要求强制重新下载：使用正确
`force=true`； - 已缓存 + 用户未要求强制下载：不得无意义重复下载； -
用户只要求查询缓存：不能自动下载； - 多篇论文混合缓存状态：逐篇处理； -
检索列表变化后，必须重新解析引用，不能把旧 ref 当作稳定 paper_id。

每种状态分支使用独立 TaskSpec，保证 Gold 计划唯一、可验证。

## 3.6 家族 C：多论文身份与引用保持（8 个）

重点不是单次引用，而是在多轮交互和多次检索后仍然正确操作目标论文。

任务可覆盖： - 一次检索建立候选列表； - 对两个不同 ref 执行下载； -
再次检索改变候选列表； - 继续操作之前指定的论文； - 使用完整 arXiv ID
或标题片段解析目标（仅当工具 Schema 支持且 Gold Replay 证明有效）； -
对论文 A 读取内容，对论文 B 分析图表； - 最终回答中区分两篇论文。

必须记录每个步骤解析出的真实 paper_id。不能只检查 ref 数字相同。

不得仅把旧任务中的 ref=1 改成 ref=2 就当作新任务。

## 3.7 家族 D：内容 / 摘要 / 图表组合（6 个）

仅使用 Snapshot 中真实存在的记录。

可构造： - 下载论文 A → 读取指定 section； - 下载论文 B → 提取图表； -
对论文 B 的指定图表执行 analyze； - 对论文 A 的指定内容执行
summarize； - 最后按论文身份分别汇报。

如果 Snapshot 不支持目标 section、figure、style 或
question，任务必须排除。

严格指标只验证工具序列、有效参数、paper identity
和终止语义。不能把没有确定性验证器的自然语言摘要质量混入 Strict
Success。

可以另行记录内容输出诊断，但不能使用临时 LLM Judge 改变主 Reward。

## 3.8 家族 E：多约束组合（6 个）

单独的 aspect、days、save_to_file、force、keep_dual 等基础参数映射属于
SFT 已覆盖能力。本轮只在它们与长程任务组合时使用。

例如： - 指定检索范围和时间窗； - 处理两篇论文； - 不允许保存到文件； -
对其中一篇强制重新下载； - 另一篇保留已有缓存； -
最终只读取指定内容并汇报。

每个约束都必须影响真实动作或参数。不能把多个约束写在任务文本中但不让它们影响轨迹。

## 3.9 家族 F：PENDING / READY / Recovery（4 个）

PENDING / READY： - PENDING 状态下不能读取尚未就绪的译文； -
如果用户还有不依赖译文的独立子目标，应继续完成； - READY 只能来自
Snapshot 中真实翻译记录； - `translate_arxiv_pdf` 默认 PENDING 的 Mock
语义必须保持不变； - 不得假设等待后自动 READY。

Recovery： - 只有当前 Mock
环境能够产生真实、确定性、可恢复的工具错误时才纳入； - 错误 Observation
必须由工具执行真实返回； - 不得把 Snapshot
miss、环境异常或人为伪造文本当作可恢复工具错误； -
如果需要新增故障注入，必须在 Mock 层显式定义 fault
type、触发条件、恢复条件和可复现 Seed，并单独审计； -
如果无法实现真实、可恢复的错误轨迹，本轮不强行加入 Recovery
任务，将名额转给其他状态分支任务。

------------------------------------------------------------------------

# 4. TaskSpec、Gold Replay 与数据防泄漏

## 4.1 复用现有 TaskSpec

继续使用现有 `TaskSpec` / `Step`： - `steps` 是唯一标准工具动作序列； -
`expected_tools` / `expected_tool_args` 从 steps 派生； - `setup`
用于建立初始状态，不计入模型动作； - `terminal_mode` / `terminal_reason`
使用现有语义； - `max_iterations` 必须覆盖 Gold 工具调用数 + FINISH。

不同初始状态或不同正确动作序列拆成独立 TaskSpec。

## 4.2 Task Catalog 元数据

每个任务记录： - task_id、family、difficulty、split； -
用户可见任务文本； - TaskSpec steps / setup； - target paper_ids； -
snapshot keys； - terminal semantics； - max_iterations； - parent Train
task IDs； - novelty signature； - prompt SHA256。

以上标准答案和元数据只供构造、Replay、审计和评分使用，绝不能放入模型可见
Prompt。

## 4.3 Gold Replay

每个任务必须： 1. 新建独立 MemoryStore、MockArxivEnv、PDF root 和
Session。 2. 使用当前 Native Agent 执行真实 Setup。 3. 验证 Setup
后缓存、候选列表、最近操作论文和异步状态。 4. 按 Gold Actions
执行完整轨迹。 5. 保存每个 Tool Call、tool_call_id、effective args、原始
Observation 和 paper_id。 6. 验证工具命中 Snapshot 或合法 Mock 行为。 7.
验证终止状态和最终回答。 8. 调用当前
`benchmark.metrics.is_strict_success()`。 9. Gold Strict Success 必须为
1。 10. 重放至少一次，结果确定性一致。

任何 Snapshot miss、Setup
failure、环境异常或无法复现的任务均排除，不能计入 Probe。

## 4.4 防止 SFT / Benchmark 泄漏

与以下文件做泄漏审计： -
`data/sft/native_sft_v2/native_sft_v2_task_catalog.jsonl` -
`data/splits/v4_86.json` - 原 Native Dev / IID / OOD 任务； -
`data/splits/native_grpo_qwen35_candidates_v2.json` -
`data/splits/native_grpo_qwen35_train_v1.json` -
`data/splits/native_grpo_qwen35_train_v2.json` -
`reports/native_grpo_followup_20261124` 中的 24 个新变体。

至少比较： - task_id； - 规范化用户任务文本； - target paper_id 集合； -
Setup state signature； - effective action sequence； - terminal
semantics； - 去掉措辞后的行为模板签名； - 任务依赖图签名。

新任务可以复用基础工具，但不能只是替换论文序号或重写句子。

如果 paper_id 在 Train / Dev / Test 之间重叠，必须明确报告。只有真正
paper-disjoint 时才可声称论文级泛化。

------------------------------------------------------------------------

# 5. Frozen Probe：用 SFT v2 验证复杂任务的学习空间

## 5.1 Probe 配置

-   Policy：Native SFT v2 Adapter；
-   Prompt：原生产 Prompt，SHA 必须与 SFT v2 相同；
-   Tool Schema / Chat Template：与 SFT v2 相同；
-   Snapshot：固定 SHA；
-   Train Candidate：目标 24 个；
-   每个任务 8 个独立 Prompt Groups；
-   每组 G=4；
-   最多 24 × 8 × 4 = 768 条 Rollout；
-   每条 Rollout 使用独立 Session 和 MemoryStore；
-   不更新模型参数；
-   保存全部原始轨迹、Seed、Strict、Reward、Tool Events、Token
    数和终止状态。

## 5.2 三种有效信号分别统计

**A. Strict-informative group**

同一 Prompt 的 G=4 轨迹中同时出现 Strict Success=1 和 Strict Failure=0。

示例： - `[1, 1, 0, 0]`：有效； - `[0, 0, 0, 0]`：Strict 无组内信号； -
`[1, 1, 1, 1]`：ceiling group； -
含环境异常或不可判定轨迹：不能计为有效组。

**B. DFA-informative group**

在相同 DFA Contract State
下出现可验证的正确动作与首错动作，且局部优势不为零。

**C. GiGPO Anchor-informative group**

在完全相同的真实决策状态下重复出现不同动作，并且后续 Strict Return
存在差异。

Anchor 必须包含影响下一步决策的真实状态： - 当前用户可见历史； -
当前工具列表； - 当前候选论文； - 缓存状态； - 最近操作论文； - PENDING
/ READY 状态。

不能仅凭相同 task_id、相同 action prefix 或相同 TaskSpec
进度认为状态相同。不得包含随机
session_id、时间戳、临时路径或无语义计数器。

## 5.3 Probe Go / No-Go

单任务候选入选： - Gold Replay 通过； - 8 组中至少 2 组
Strict-informative，或存在至少 2 个独立、可复核的 DFA 对照； -
环境异常为 0； - 没有大比例截断； - 无 SFT / Held-out 行为签名泄漏。

整个任务池 Go 条件： 1. 至少 12 个任务入选； 2. 至少覆盖 4 个任务家族；
3. 任一任务家族不超过入选任务的 40%； 4. Strict-informative group
fraction 至少 30%； 5. 至少 8 个入选任务含 4 个或以上模型工具调用； 6.
至少 8 个入选任务包含真实状态分支或跨工具依赖； 7. DFA Local Credit
覆盖至少 10% 的可判定决策； 8. GiGPO Anchor-informative decision
coverage 至少 10%； 9. 环境无效组比例为 0； 10. Train / Dev / Sealed
Test 泄漏审计通过。

若未达到： - 不修改 Strict Reward； - 不人为指定失败轨迹； -
不降低任务成功标准； - 可在 Snapshot 支持范围内补充一次候选任务并重新
Probe； - 仍不达标则 No-Go，停止本轮正式训练。

注意：Strict-informative、DFA-informative 和 Anchor-informative
必须分别报告。不能把三种信号合并成一个数字后宣称 E1--E4
都有相同学习空间。

------------------------------------------------------------------------

# 6. E1--E4：复用现有 Credit v2，不改算法定义

本轮复用： - `AgenticArxiv/rl/credit_v2.py` -
`AgenticArxiv/rl/credit_v2_trainer.py` -
`AgenticArxiv/rl/task_automaton.py` -
`AgenticArxiv/rl/anchor_state.py` - 现有 Credit v2 单元测试。

四种模式定义：

  模式   Episode Advantage   Local Credit               First-error Protection
  ------ ------------------- -------------------------- ------------------------
  E1     Strict 0/1          无，lambda=0               无
  E2     Strict 0/1          DFA Contract-state         有
  E3     Strict 0/1          GiGPO Exact Anchor-state   无
  E4     Strict 0/1          DFA + GiGPO                有

具体实现以当前 `credit_v2.py::credits()` 为准： - E1：只使用组内 Strict
Episode Advantage； - E2：DFA Local Credit，lambda=0.5，启用 First-error
Protection； - E3：Anchor Local Credit，lambda=0.5，不启用 First-error
Protection； - E4：DFA 与 Anchor 的组合 Local Credit，lambda=0.5，启用
First-error Protection。

主优化 Reward 继续使用：

``` python
reward = 1.0 if is_strict_success(task, trajectory) else 0.0
```

约束： - Strict Success 定义不变； - 旧五维 Reward 只作诊断，不参与本轮
E1--E4 主优化； - 环境异常、Snapshot miss、Setup failure 必须 mask； -
模型自身的错误工具、错误参数、错误引用、提前 FINISH 仍然是策略错误； -
raw DFA、raw Anchor、raw combined、effective Credit 分开记录； -
First-error Protection 的方向保证不能冒充原始估计器准确率。

## 6.1 训练前必须审计的 Credit 细节

-   截断轨迹不能把已验证正确前缀整体错误排除；
-   `q_truncated` 不等同于 `q_error`；
-   环境异常与策略错误分开；
-   Anchor fingerprint 只包含真实决策状态；
-   Tool Observation 不参与 Policy Loss；
-   Assistant Tool Call / Assistant Content / Final Answer Token Span
    对齐准确；
-   E1--E4 使用完全相同的 G=4 组顺序；
-   所有原有 Credit v2 单元测试通过；
-   新复杂任务至少有 3 条完整 Gold 轨迹通过 Token Alignment 审计。

## 6.2 Loss Type 与 TRL 版本

旧 E1--E4 使用 TRL 1.17.1，`GRPOConfig` 没有显式传入
`loss_type`。该版本默认 Loss Type 为 `dapo`。

本轮为避免静默变化： - 固定 TRL 版本为旧实验版本； -
读取旧运行环境和日志核验 Loss Type； - 若确认旧运行时为
`dapo`，本轮显式设置 `loss_type="dapo"`； - 不启用 DAPO Dynamic
Sampling、Clip-Higher 或其他新组件； - 报告准确称为"GRPO 组内优势 + DAPO
Loss Normalization + E1--E4 Credit Ablation"，不能称为完整 DAPO。

如果旧实际运行版本与以上不同，停止并报告，不能静默切换。

------------------------------------------------------------------------

# 7. GRPO 训练计划

## 7.1 统一训练起点

所有 E1--E4 独立从同一 Native SFT v2 Adapter 加载。

每个模式： - 独立加载 Policy； - 独立保存 Adapter； -
不从其他模式继续训练； - 不覆盖 SFT v2； -
记录模型、Tokenizer、Prompt、Tool Schema、Snapshot、Train Split 和配置
SHA。

## 7.2 固定训练预算

沿用当前 RTX 4090 已经跑通的 Native Credit v2 配置：

  参数                                          值
  ------------------------------- ----------------
  `num_generations`                              4
  `learning_rate`                           `5e-6`
  `beta`                                    `0.04`
  `seed`                                        42
  `temperature`                                1.0
  `scale_rewards`                          `group`
  `dynamic_sampling`                         false
  `per_device_train_batch_size`                  1
  `gradient_accumulation_steps`                  4
  `bf16`                                      true
  `gradient_checkpointing`                    true
  Optimizer Steps / Mode                        25
  Modes                             E1、E2、E3、E4

每个模式 25 个 optimizer steps，每步一个 G=4 Prompt Group，共 100 条训练
Rollout。四个模式共 400 条正式训练 Rollout。

不因为任务更复杂而同时大幅改变学习率、KL、Batch、Temperature 和 Credit
权重。

## 7.3 固定 Group Schedule

Probe 结束后生成一个 25-group 固定训练计划：

  任务类型                        目标比例
  ----------------------------- ----------
  多步骤长程工作流                     40%
  状态条件与缓存分支                   24%
  多论文身份与引用                     16%
  内容 / 摘要 / 图表                   12%
  多约束 / PENDING / Recovery           8%

要求： - 四条 Rollout 必须来自同一个 Prompt 和 TaskSpec； -
每条轨迹从相同初始 Setup 状态开始； - 四种模式使用完全相同的
task_id、group_id 和 group 顺序； - 单个任务最多重复 3 次； - 不启用
Dynamic Sampling； - 不因为单条轨迹失败而丢弃同组其他轨迹； -
不把环境异常当成模型失败； - 训练计划中至少一半 Group 来自 4
个或以上模型工具调用的任务。

## 7.4 长轨迹 Token Budget

复杂任务可能超过旧的 `max_completion_length=1792`，不能静默截断。

Gold Replay 后统计： - 最大完整 Gold completion tokens； - P95 Gold
completion tokens； - 最大单轮 Assistant generation tokens； -
每个任务的工具调用数； - 每个任务需要的 max_iterations； - Tool
Observation 对 Token Budget 的贡献。

设置原则： - `max_turns` 至少覆盖 Gold tool calls + FINISH； - 每轮
`max_new_tokens` 覆盖 Gold 单轮 Assistant 长度并留余量； -
`max_completion_length` 根据 Gold 完整轨迹长度确定，并留 20% 余量； -
优先保留旧的 704 tokens/turn； - 若必须超过旧 1792 completion
budget，先做 GPU Smoke； - 如果 Snapshot Observation
过长，只允许按确定性规则裁剪非关键文本字段，必须保留
paper_id、status、引用和影响决策的关键字段； - 原始 Observation
必须完整保存用于审计； - 若任务无法在 24GB 4090
稳定运行，则简化或排除，不把截断作为主要学习信号。

------------------------------------------------------------------------

# 8. 评测设计

## 8.1 评测模型

同一 Challenge Dev / Sealed Test 评测： 1. Frozen Native SFT v2； 2.
E1； 3. E2； 4. E3； 5. E4。

所有模型使用相同生产 Prompt、Setup Context、Tool Schema、Snapshot、Chat
Template、解码参数和 Trial Seeds。

## 8.2 主指标

-   Strict Success；
-   fail-to-pass；
-   pass-to-fail；
-   net gain；
-   任务家族 Strict Success；
-   L2 / L3 任务 Strict Success。

## 8.3 次级指标

-   多子目标完整率；
-   多步骤工具序列正确率；
-   缓存分支决策准确率；
-   force 参数正确率；
-   paper identity 保持准确率；
-   跨论文引用解析准确率；
-   内容 / 摘要 / 图表工作流完整率；
-   PENDING / READY 状态处理；
-   Recovery 成功率（仅在有效故障任务存在时）；
-   必要调用遗漏率；
-   额外重复调用率；
-   工具执行错误率；
-   提前 FINISH；
-   max-turn / token truncation；
-   Snapshot coverage miss；
-   环境无效 Rollout。

**不把调用次数下降单独视为收益。** 必须同时检查必要步骤是否完整执行。

## 8.4 统计与结果解释

-   Dev：8 tasks × 3 trials = 24 episodes；
-   Sealed Test：12 tasks × 3 trials = 36 episodes；
-   Sealed Test 只能在训练和 Dev 配置冻结后运行一次；
-   配对结果按 task_id 进行 cluster bootstrap；
-   不能把同一道任务的 3 次重复视为 3 个完全独立任务；
-   必须保存逐任务、逐试次的原始轨迹。

## 8.5 生产 Benchmark 回归

新 Challenge 的主评测与旧 Native Benchmark 使用相同生产
Prompt，但任务集不同。

在 Dev 选出候选后，只对 Dev 最佳候选额外运行原 Native Benchmark 的 31
tasks × 3 trials，与 SFT v2 的 83/93 比较。

必须检查 GRPO 是否损害 SFT v2 已经掌握的基础能力，特别是： - 全 CS
范围错误； - force=true 回退； - 第二篇缓存检查遗漏； - PENDING / READY
退化； - 不支持能力误调用。

Challenge 提升但生产 Benchmark 明显退化，不能直接宣称 Agent
整体能力提升。

------------------------------------------------------------------------

# 9. GPU 预算与执行顺序

## 9.1 RTX 4090 执行约束

-   单 GPU、单进程；
-   E1--E4 串行运行；
-   不并行加载多个 Policy；
-   每个模式结束后释放模型和 CUDA Cache；
-   记录峰值显存、耗时、Rollout 吞吐和 Token 长度；
-   先完成 4-step Smoke，再启动 25-step Pilot；
-   不因赶时间跳过 Gold Replay、Prompt SHA 或 Token Alignment 检查。

## 9.2 预计预算

按目标 24 Train Candidate × 8 groups × G=4： - Frozen Probe：最多 768 条
Rollout； - E1--E4：每模式 100 条正式训练 Rollout； - 四模式合计 400
条； - 每个策略 Dev + Sealed Test：60 episodes； - 最佳候选额外运行旧
Native Benchmark：93 episodes。

旧 33 候选 Probe 实测约 2
小时，但新任务更长，不能直接线性推算。建议预留： - Snapshot
Inventory、Gold Replay、CPU 审计：1--2 小时； - 新 Frozen Probe：2--4
小时； - E1--E4 Smoke + 25-step 训练：约 2--3 小时； - Dev / Test
和生产回归：1--2 小时。

GPU 阶段总预算先按 6--9 小时准备，最终以 CPU 阶段的 Token 统计和 GPU
Smoke 实测更新。

## 9.3 必须按阶段执行

**Phase 0：CPU 资产审计** - 核验 SFT v2 Adapter、Prompt、Tool
Schema、Snapshot SHA； - 生成 50-paper Inventory； -
审计现有任务、Credit v2 代码和测试。

**Phase 1：CPU 任务构造** - 生成 44 个候选任务； - Gold Replay； -
Snapshot Coverage Audit； - Train / Dev / Sealed Test 切分； - SFT / 旧
Benchmark 泄漏审计； - 生成 `gpu_readiness.md`。

**Phase 2：GPU Frozen Probe** - 使用 SFT v2 和原生产 Prompt； - 完成 24
× 8 × 4 Probe； - 计算 Strict / DFA / Anchor 三类信号； - Go / No-Go。

**Phase 3：E1--E4** - 各 4-step Smoke； - 各 25-step Pilot； - 同一
Group Schedule； - 串行运行。

**Phase 4：评测** - Dev； - 冻结配置； - Sealed Test； - 最佳候选生产
Benchmark 回归； - 最终报告。

CPU 阶段完成后，Codex 必须生成 `gpu_readiness.md` 并暂停，等待用户启动
GPU。不得自动启动 Probe 或训练。

------------------------------------------------------------------------

# 10. Codex 实施目录与任务

从 `exp/native-sft-v2` 最新提交创建实现分支：

``` text
exp/agent-challenge-grpo-final
```

建议新增：

``` text
AgenticArxiv/benchmark/agent_challenge_final/
  __init__.py
  task_builder.py
  snapshot_inventory.py
  challenge_audit.py

scripts/agent_challenge_final/
  build_tasks.py
  audit_snapshot.py
  audit_tasks.py
  run_probe.py
  run_ablation.py
  evaluate.py

data/agent_challenge_final/
  snapshot_inventory.json
  task_catalog.jsonl
  train_candidates.jsonl
  train_selected.jsonl
  dev.jsonl
  sealed_test.jsonl
  manifest.json

reports/agent_challenge_final/
  snapshot_audit.md
  task_audit.md
  gold_replay.json
  leakage_audit.json
  probe_report.md
  gpu_readiness.md
  E1/
  E2/
  E3/
  E4/
  ablation_report.md
  final_report.md

outputs/agent_challenge_final/
  E1/
  E2/
  E3/
  E4/
```

文件名可以按现有仓库约定调整，但必须保留同样的审计能力。

## Phase 1：CPU 阶段交付

Codex 必须： 1. 创建实现分支； 2. 核验 SFT v2 Adapter / Base / Prompt /
Tool Schema / Snapshot SHA； 3. 生成 50-paper Inventory； 4. 不修改生产
Prompt 和 Setup Context； 5. 构建 44 个复杂候选任务； 6. 完成 Gold
Replay； 7. 完成 Snapshot Coverage Audit； 8. 完成 Train / Dev / Sealed
Test 切分； 9. 完成 SFT / Held-out / 旧 GRPO 行为签名泄漏审计； 10.
复用并审计 E1--E4 Credit v2； 11. 完成 CPU 单元测试； 12. 生成 GPU
Readiness。

CPU 阶段不得加载模型或启动 GPU。

## Phase 2：Frozen Probe

用户启动 GPU 后： - 使用 Native SFT v2 Adapter； - 使用原生产 Prompt； -
24 候选 × 8 groups × G=4； - 保存全部原始 Rollout； - 分别统计 Strict /
DFA / Anchor 信号； - 按 Go 条件选 Train Split； - 未达到 Go
条件则停止，不运行正式 E1--E4。

## Phase 3：E1--E4

-   同一 SFT v2 起点；
-   同一 Train Split；
-   同一 Prompt；
-   同一 Snapshot；
-   同一固定 Group Schedule；
-   同一 TRL / Loss Type；
-   同一训练超参数；
-   E1--E4 只改变 Credit Assignment；
-   每个模式独立 Adapter 和日志；
-   串行运行。

## Phase 4：评测与报告

必须交付： - Snapshot Inventory； - Prompt / Tool Schema / Chat Template
SHA； - Challenge Task Catalog； - Gold Replay Audit； - Train / Dev /
Sealed Test Manifest； - Leakage Audit； - Frozen Probe 原始轨迹； -
三类信号统计； - E1--E4 训练配置和 Adapter SHA； - E1--E4 Credit
Alignment Audit； - Dev / Sealed Test 配对结果； - Native Benchmark
回归； - 最终结论。

------------------------------------------------------------------------

# 11. 最终成功标准

最终报告只能得出以下四种结论之一：

1.  **GRPO 有额外收益：** Challenge Sealed Test 相比同 Prompt 的 SFT v2
    有净提升，且生产 Benchmark 没有明显退化。
2.  **特定 Credit 方法有效：** 例如 E2 优于 E1，E3
    无效；结论严格限定在当前 Snapshot 和任务分布。
3.  **没有额外收益：** SFT v2 已经能够完成这些任务，或 GRPO
    没有进一步改善。
4.  **实验不可判定：** Snapshot 覆盖、Probe 信号、Token Budget、Credit
    对齐或训练链路不满足要求。

不能只根据 Reward 均值、Loss 下降、Credit Accuracy、工具调用次数减少或
Dev 单次成功率宣称 GRPO 有效。

最终主结论必须基于 Sealed Test 的 Strict Success、逐任务配对结果和生产
Benchmark 回归。

------------------------------------------------------------------------

# 12. 给 Codex 的明确执行指令

请完整阅读本文，在 `Napabana/AgenticArXiv-RL` 的
`exp/agent-challenge-grpo-final` 分支实现。

**最重要的约束：保留 Native SFT v2 的完整生产 Prompt，不做
Protocol-only，不改变 Harness。**

先完成 CPU 阶段，生成 `gpu_readiness.md` 后暂停。

优先核验： - SFT v2 Adapter SHA； - 生产 System Prompt SHA； -
`build_visible_setup_context()` 是否未变； - 10 个 Native Tool
Schema； - Snapshot 实际唯一论文数量和 SHA； - 每个任务的 Gold
Replay； - TaskSpec 和 effective args； - Train / Dev / Sealed Test
泄漏； - E1--E4 Credit v2 代码与单元测试； - 长轨迹 Token Budget。

禁止： - 修改生产 Prompt； - 隐藏真实状态或工具 Schema； - 重新联网构建
Snapshot； - 伪造论文、Observation、READY 或错误历史； - 把 Snapshot
miss 作为策略失败； - 通过同义改写、替换 ref 数字凑任务数量； - 为提高
Probe 信号修改 Strict Reward； - 未经用户确认启动 GPU。

**本轮的核心目标不是让 GRPO 在简单任务上获得高分，而是检验它能否在 SFT
v2 已经掌握基础能力之后，进一步改善真实复杂 Agent 工作流的完成率。**