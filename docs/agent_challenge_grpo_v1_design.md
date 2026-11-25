# AgenticArXiv-RL：固定生产 Prompt 的复杂任务 Benchmark 与最终 GRPO 设计

> 本文是本轮最终 Agent GRPO 实验的设计基线和 Codex 实施任务书。它取代此前 Protocol-only Prompt 版本的 Challenge 设计。**本轮不改变生产 Prompt，不改变 Native Tool Calling 协议，不增加或删除 Prompt 中的行为规则。**

- 仓库：`Napabana/AgenticArXiv-RL`
- 基线分支：`exp/native-sft-v2`
- 基线提交：`77c4536fbce4e3346f10e00da9a1d6269a44a250`
- 本设计分支：`exp/agent-challenge-grpo-v1`
- Base Model：`Qwen/Qwen3.5-2B-Base`
- Policy 起点：Native SFT v2 Adapter
- GPU：单张 RTX 4090 24GB

---

## 0. 研究定位：GRPO 不重复 SFT 已经掌握的基础能力

### 0.1 当前证据

Native SFT v2 的配对留出评测结果：

| Split | SFT v1 | SFT v2 |
|---|---:|---:|
| Dev | 20/24 | 18/24 |
| IID | 44/57 | 53/57 |
| OOD | 8/12 | 12/12 |
| 合计 | 72/93 | **83/93** |

同题同试次有 14 次修复、3 次回退，净增加 11 次。SFT v2 的多步骤任务达到 39/39，指定缓存检查达到 27/27，PENDING 提交后的正确结束为 18/18，Blocked 零调用达到 6/6。

仍然存在的明确问题：
- `opt_force_dl` 在 Dev 从 3/3 回退到 0/3：模型在候选列表存在时错误地声称 ref 越界，没有执行 `download_arxiv_pdf(ref=1, force=true)`。
- Blocked 的能力边界解释仍不稳定。
- SFT v2 的留出轨迹中没有实际工具执行错误，Recovery 成功率无法估计。
- 旧 GRPO E1–E4 的总体 Strict Success 与 SFT v1 持平（72/93），旧训练任务池和 SFT v1 能力分布已经不适合直接作为本轮最终实验的主要任务集。

### 0.2 本轮要验证的核心假设

> 在固定生产 Prompt 和工具协议的条件下，Native SFT v2 已掌握基础工具调用、参数理解和常见状态处理；GRPO 应重点优化模型在复杂任务中的组合决策、长程执行、跨论文状态保持和错误恢复，而不是重复训练单工具调用或基础参数规则。

SFT 与 GRPO 的分工：

| 能力 | Native SFT v2 | 本轮 GRPO |
|---|---|---|
| 基础工具选择 | 主要建立 | 不作为主要目标 |
| 基础参数与引用解析 | 主要建立 | 在多约束、多论文场景中检验 |
| 用户约束遵循 | 主要建立 | 检验多约束同时存在时能否持续遵守 |
| 单论文缓存判断 | 已有较强覆盖 | 检验多论文、混合缓存状态下的组合决策 |
| 常规 2–4 步任务 | 已有较强覆盖 | 不作为主要训练目标 |
| 5–7 步依赖链 | 部分覆盖 | 重点优化 |
| 多论文交替处理 | 覆盖有限 | 重点优化论文身份保持和引用稳定性 |
| 多目标组合 | 覆盖有限 | 重点优化子目标完成和依赖管理 |
| 工具执行错误恢复 | 证据不足 | 在可复现的受控错误条件下验证 |
| 长轨迹提前结束 | 已有改善 | 重点降低复杂任务中的遗漏和提前结束 |

Prompt 中已经包含“跟踪任务进度”“遵守约束”“根据错误修正调用”等通用要求。这些规则是生产 Harness 的组成部分，本轮保留。GRPO 的目标不是重新教模型这些文字规则，而是让模型在具体复杂轨迹中更稳定地执行它们。

### 0.3 不再采用 Protocol-only Prompt

此前提出的 Protocol-only Prompt 方案在本轮废弃，原因：
1. Native SFT v2 的训练样本和评测都使用完整生产 System Prompt。
2. 改 Prompt 会改变模型输入分布，无法直接与 SFT v2 的生产评测基线比较。
3. 删除策略提示会同时改变 Harness 条件和模型训练条件，难以归因 GRPO 的增量效果。
4. 我们真正要验证的是“相同 Harness 条件下，GRPO 能否改善复杂任务执行”，而不是“模型脱离 Harness 后能否独立复述规则”。

**必须保持原始 `NATIVE_TOOL_SYSTEM_PROMPT`、`build_visible_setup_context()`、Tool Schema、Chat Template 不变。** 不新增 Challenge Prompt，不 monkeypatch Prompt，不删减 Prompt 中的行为规则，不改变每轮一个工具调用的 Native Tool Calling 协议。

---

## 1. 固定实验资产与身份校验

Codex 必须从实际仓库文件核验，不得只相信本设计文档中的路径或 SHA。

### 1.1 Native SFT v2

| 资产 | SHA256 / 数量 |
|---|---|
| Base Model | `Qwen/Qwen3.5-2B-Base` |
| SFT v2 Adapter | `f0c33a4d32a175bcaf1e83c55231a0108a84859cb748c318db49988a52f7911f` |
| SFT v2 Adapter Config | `dabf01d58fe513575d19d81dee6ff3bc527e1f376a9925d05ec6e5635a0f6c0a` |
| SFT v2 Train Data | `8763ff0ef7c19bf748d3d036d3730e8f30c44709161df614302a7a5df5bb3dfd` |
| SFT v2 Train Manifest | `b8aa518f3f8aba708029bb1525b43d63372840a5bb785bf0fdde1439faf10751` |
| 唯一决策样本 | 590 |
| 专家轨迹 | 223 |
| SFT Optimizer Steps | 83 |

### 1.2 Prompt / Tool / Snapshot / Chat Template

从 `data/sft/native_sft_v2/native_sft_v2_train.jsonl.manifest.json` 读取并再次核验：

```text
system_prompt_sha256 = 31674f2183d642a9035ecf8360c138135c5ef4645904017eadf4ec7bdf4abf05
tools_sha256         = 01ed14f7f8a041ca9752475c1bf1513996c375c5a32cadf2514089fb04401ae0
chat_template_sha256 = 273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80
snapshot_sha256      = d97035db326ca11935e882fd2aca7fe932e9e76cf1bc8a21078be7bec1ae8794
split_sha256         = 0a176b0f67b8b61d4f84e27a40071ab81a5be3d2040c2a58aa695432f5a6eacc
```

若任一 SHA 与本地实际资产不符，停止并报告。不得静默更新哈希或换用其他模型、Snapshot、Prompt、Tool Schema。

### 1.3 固定 Prompt 的具体要求

1. 直接使用生产 `NATIVE_TOOL_SYSTEM_PROMPT`。
2. SFT v2 数据中的 system message 与 GRPO rollout 生成的 system message 必须逐字节一致。
3. 继续使用 `NativeToolCallingAgent.setup_context()` 和现有 `build_visible_setup_context()`。
4. 不新增 `CHALLENGE_PROTOCOL_SYSTEM_PROMPT`。
5. 不新增或替换 Setup Context 文案。
6. 不更改 Tool Schema、工具描述、默认参数和参数归一化。
7. 不使用 `get_routed_argument_prompt()` 固定模型工具选择。
8. 不切换到 ReAct 文本协议，不要求模型额外输出 JSON 字符串。
9. Prompt SHA、Tools SHA、Chat Template SHA 必须写入每一份任务集 Manifest、Probe、训练和评测报告。

Prompt 中已有的策略规则不是本轮要删除的内容。新任务要考察模型能否在复杂状态和长轨迹中正确应用这些规则。

---

## 2. Mock Snapshot 边界：只用现有 50 篇论文

### 2.1 Snapshot 不得扩充

当前 Snapshot 约有 50 篇论文，固定 SHA：

```text
d97035db326ca11935e882fd2aca7fe932e9e76cf1bc8a21078be7bec1ae8794
```

Snapshot 文件不一定由 Git 跟踪，Codex 必须在当前运行环境读取用户实际提供的文件，核验 SHA 并统计唯一 `paper_id`。

- 如果唯一论文数为 50：继续。
- 如果不是 50：停止并报告差异。
- 不得复制记录凑数。
- 不得联网补充 arXiv 论文。
- 不得伪造 title、abstract、paper_id、PDF 正文、图表、翻译或工具 Observation。

### 2.2 必须先做 Snapshot Inventory

生成：
- `data/agent_challenge_v1/snapshot_inventory.json`
- `reports/agent_challenge_v1/snapshot_audit.md`

至少记录：
- Snapshot 路径、SHA、唯一 paper_id 数量和 ID 列表；
- 每篇论文的真实 title、categories、published、updated；
- `get_recently_submitted_cs_papers` 每个 aspect 的真实记录池、days、结果数量和顺序；
- `search_arxiv_papers` 的精确 query keys 和真实结果列表；
- `get_paper_content` 的 paper_id + section 覆盖；
- `summarize_paper` 的 paper_id + normalized style + bucketed max_words 覆盖；
- `extract_paper_figures` 的 paper_id 覆盖；
- `analyze_figure` 的 paper_id + figure_no + question 覆盖；
- `get_translated_content` 的 paper_id + page 覆盖；
- 每个工具究竟由 Snapshot 回放、本地 MemoryStore 还是异步副作用模拟。

### 2.3 必须遵守当前 Mock 工具的真实语义

当前工具集合为：
1. `get_recently_submitted_cs_papers`
2. `search_arxiv_papers`
3. `download_arxiv_pdf`
4. `get_paper_cache_status`
5. `get_paper_content`
6. `summarize_paper`
7. `extract_paper_figures`
8. `analyze_figure`
9. `translate_arxiv_pdf`
10. `get_translated_content`

重要约束：
- `get_recently_submitted_cs_papers` 使用 Snapshot 中真实存在的 aspect / days 记录。
- `search_arxiv_papers` 只能使用 Snapshot 中存在且 Gold Replay 通过的精确 query key；不能任意编造 keyword query。
- `max_results` 不得超过对应结果池长度。
- `download_arxiv_pdf` 是离线下载桩，会创建本地占位 PDF 并更新 MemoryStore；不代表存在真实 PDF 正文。
- `get_paper_cache_status` 读取本地 MemoryStore。
- `translate_arxiv_pdf` 默认由 LocalSideEffectManager 返回 PENDING；不能假定它会自动 READY。
- READY 只能来自 Snapshot 中确实存在的翻译记录。
- 内容、摘要、图表、翻译页操作必须命中真实 Snapshot key。
- Snapshot miss、Setup failure、环境异常不是策略错误，不得作为负 Reward。

### 2.4 Snapshot Coverage Gate

每个候选任务必须先通过 Gold Replay。若某个动作缺少 Snapshot key、论文 ID 无法解析、Setup 无法建立所需状态、工具返回 Snapshot miss，直接排除任务并记录原因。

不要为了凑够任务数量降低 Coverage 标准。

---

## 3. 新 Benchmark 的设计原则

### 3.1 不再做基础工具题的重复扩增

以下内容不应占据本轮 Train 主体：
- 单独调用一次搜索；
- 单独下载一篇论文；
- 单独查一次缓存；
- 仅测试 `keep_dual` 一个参数；
- 仅测试一个简单 ref；
- 只有 2–3 个模型工具动作的常规任务；
- 只更换用户措辞、论文序号或 aspect 的同义任务。

这些能力已经有 SFT v2 监督和旧 Benchmark 评测证据。可以保留少量作为回归控制，但不能成为主要 GRPO 训练目标。

### 3.2 新任务的复杂度来自决策结构，而不是文字长度

每个主要 Challenge 任务至少满足以下三项中的两项：
- 需要处理 2 篇或更多不同论文；
- 包含 5 次或更多模型工具调用；
- 有至少 2 个跨工具依赖关系；
- 有缓存、翻译或 Session 状态差异；
- 需要在多次搜索或工具调用后继续保持论文身份；
- 同时遵守至少 2 个用户约束；
- 存在一个真实可恢复的工具错误；
- 需要完成多个明确子目标后才能结束。

目标难度：
- L1：4 次模型工具调用，约 20%；
- L2：5–6 次模型工具调用，约 50%；
- L3：6–7 次模型工具调用，约 30%。

原则上不超过 7 次模型工具调用 + FINISH。不能为了增加长度而插入无意义查询、重复调用或没有用户价值的工具动作。

### 3.3 继续使用现有线性 TaskSpec / DFA

当前 `TaskSpec.steps`、`task_automaton.transitions()` 和 E1–E4 Credit v2 使用有序工具步骤。为了控制最后一轮实验的实现风险：

- 本轮每个任务必须定义一条明确、可审计的 canonical action sequence。
- 用户请求应明确表达步骤顺序，或前一步结果确实是后一步的必要输入。
- 不要在同一个 TaskSpec 中隐藏多条不同的合法顺序。
- 不要为了引入并行计划而在本轮重写整个 Strict Scorer 或 DFA。
- 如果某任务存在多条同样合法的动作顺序，先改写任务使其具有明确顺序；无法做到则排除。

本轮目标是复杂的有序依赖链、跨论文状态保持和长程执行，不是实现一个新的通用计划图算法。

---

## 4. Agent Challenge Benchmark v1：任务集构造

### 4.1 目标规模和 Split

目标 44 个不同任务实例：

| Split | 数量 | 用途 |
|---|---:|---|
| Train Candidate | 24 | SFT v2 Frozen Probe 后筛选 |
| Dev | 8 | 选择 E1–E4 候选、诊断 |
| Sealed Test | 12 | 最终独立评测 |

44 是目标规模，不是必须完成的 KPI。如果 Snapshot 实际覆盖不足，可以少于 44，但必须说明原因；不能伪造数据。

Dev 和 Sealed Test 每题重复 3 次：
- Dev：最多 24 episodes；
- Sealed Test：最多 36 episodes。

### 4.2 六类任务家族

| 家族 | 目标数量 | 主要训练目标 |
|---|---:|---|
| A. 长程依赖工作流 | 10 | 多步骤完整性、依赖顺序、避免提前结束 |
| B. 多论文交错处理 | 8 | 论文身份保持、跨搜索列表引用、交错操作 |
| C. 状态条件组合决策 | 8 | 混合缓存状态、force、不同论文的差异化处理 |
| D. 内容 / 摘要 / 图表链 | 8 | 跨工具工作流、长上下文状态保持 |
| E. 多目标研究任务 | 6 | 多篇论文的多项子目标、结果汇总 |
| F. 受控错误恢复 | 4 | 观察真实工具错误后修正并完成任务 |
| 合计 | 44 | — |

如果 F 类受控错误无法通过真实、可复现的 Mock 机制实现，不能伪造错误 Observation。此时允许减少 F 类任务，并把剩余名额补到 A–E 中，但必须报告 Recovery 本轮不可测的原因。

### 4.3 A 类：长程依赖工作流（10）

目标：从用户请求中识别多个有依赖关系的操作，完整执行 5–7 次模型工具调用。

任务模板示例（仅示意，具体论文和参数必须从 Snapshot Inventory 选择）：

> 检索最近 X 天 cs.CV 方向的论文，下载候选列表中的第 2 篇，确认缓存状态；随后读取该论文的指定章节并生成指定长度的摘要，最后汇报下载、缓存和内容处理结果。

Canonical plan 示例：
1. 检索论文；
2. 下载目标论文；
3. 检查该论文缓存；
4. 读取 Snapshot 确实支持的 section；
5. 调用 Snapshot 确实支持的 summarize key；
6. FINISH。

扩展变体：
- 搜索 → 下载 A → 检查 A → 读取 A → 下载 B → 检查 B → FINISH；
- 搜索 → 下载 → 缓存检查 → 读取指定章节 → 摘要 → 图表提取 → 图表分析；
- 两篇论文分别完成不同子目标后再汇报。

要求：
- 每个工具调用的 paper_id、ref 和 Snapshot key 可审计；
- 不允许用工具数量凑复杂度；
- 如果某论文没有对应 section 或 summary 记录，不得使用它。

### 4.4 B 类：多论文交错处理（8）

目标：训练模型在多次操作后保持正确的论文身份，不把 ref、title 和 paper_id 混淆。

任务设计：
- 同一 Session 中处理至少两篇不同论文；
- 在第一次搜索列表建立后执行部分操作；
- 再执行第二次搜索或其他操作，导致当前候选列表发生变化；
- 后续明确要求回到第一篇论文继续读取或处理；
- 使用完整 arXiv ID、标题片段或 ref，要求模型根据当前真实 Session 解析；
- 对每篇论文分别检查缓存或内容。

任务必须通过真实 Native Agent 验证引用语义。若 ref 在当前实现中指向当前列表序号，而非历史列表，任务描述和 Gold plan 必须严格匹配这一行为。

不要只把旧任务的 ref=1 改成 ref=2。新任务必须在多论文状态、列表变化、跨步骤身份保持或工具链组合上有实质差异。

### 4.5 C 类：状态条件组合决策（8）

这是本轮重点之一，同时覆盖 SFT v2 已暴露的 `opt_force_dl` 回退。

使用真实 Setup 创建不同初始状态：
- 论文 A 已缓存、论文 B 未缓存；
- 两篇论文均已缓存，但用户只要求其中一篇强制重新下载；
- 一篇已缓存，另一篇未缓存，用户要求分别处理；
- 用户明确禁止重新下载某篇论文，但要求下载另一篇；
- 用户指定 force=true 的目标必须通过有效 ref / paper_id 解析。

任务模板示例：

> 请处理当前会话中的两篇论文。第一篇如果已缓存就保留现有文件；第二篇无论是否已缓存都重新下载。最后分别确认两篇论文的缓存状态。

Gold plan 必须根据真实 Setup 状态确定：
- 对第一篇使用正常缓存策略；
- 对第二篇使用 `force=true`；
- 对两篇分别检查缓存；
- 最后结束并汇报。

要求：
- 同一用户任务可以构造不同真实初始状态的配对 TaskSpec；
- Setup 只负责建立真实状态，不能直接把答案写入模型可见上下文；
- 不允许把 `force=true` 写成所有下载都默认使用的规则；
- 每个任务必须核验最终 MemoryStore 状态。

### 4.6 D 类：内容 / 摘要 / 图表工作流（8）

只使用 Snapshot 中确实存在的内容记录。

可用组合：
- 下载 → 读取指定 section → 摘要；
- 下载 → 提取图表 → 分析指定图表；
- 两篇论文分别读取指定 section → 分别摘要 → 汇总；
- 一篇论文读取方法章节，另一篇读取实验章节，再做结构化比较；
- 多个图表任务与缓存状态交错。

严格区分两种评价：
1. **Primary Strict Success：** 现有 Scorer 能可靠验证的工具序列、参数、论文引用、正常终止和执行错误。
2. **Content Quality Diagnostic：** 摘要是否准确、跨论文比较是否有事实依据。除非能建立确定性验证器，否则只保存结果并作为人工审阅材料，不得混入 Strict Reward，也不得临时使用 LLM-as-Judge 改变主指标。

### 4.7 E 类：多目标研究任务（6）

目标：一个用户请求中包含多个不同的论文处理目标，模型需要保持任务清单和论文身份。

示例：
> 在当前候选列表中处理两篇指定论文。对第一篇读取方法章节并提取图表；对第二篇读取实验章节并生成摘要。最后分别汇报每篇论文完成了哪些操作。

要求：
- 用户目标明确，工具能力足够；
- 论文和 Snapshot keys 均已核验；
- 多个子目标按用户明确指定的顺序执行；
- 不能把“摘要文本内容正确”假装成现有 Strict Scorer 已经验证的事实。

### 4.8 F 类：受控错误恢复（4）

SFT v2 的 93 次留出评测中没有出现实际工具执行错误，Recovery 能力目前缺少评测证据。本轮可以尝试构造少量确定性故障注入任务，但必须满足以下条件：

- 故障必须由显式、版本化的 Mock Fault Plan 产生；
- 故障发生在一个原本合法、Snapshot 覆盖完整的工具调用上；
- 例如对有效下载操作注入一次确定性的 transient I/O failure，第二次相同合法调用可以成功；
- 错误 Observation 必须由当前 Mock 执行路径实际生成并保存，不得手写一段看似工具返回的文本；
- 故障注入不能修改论文 Snapshot；
- 故障事件必须标记为 `expected_injected_tool_error`，与 `snapshot_coverage_error`、`environment_execution_error` 分开；
- 第一次错误不能改变缓存或下载状态；
- 重试后必须真实更新 MemoryStore；
- Gold Replay 必须重复执行并获得完全一致的故障和恢复结果。

如果现有 NativeRollout 将该错误直接标记为 `environment_invalid`，Codex 必须先增加独立分类和单元测试，证明该故障是策略可观察、可恢复的任务事件，而不是基础设施错误。

**如果无法实现可靠的故障注入和严格评分，本轮排除 F 类训练任务。禁止用 Snapshot miss 或任意异常冒充 Recovery。**

### 4.9 PENDING / READY / Blocked 的定位

SFT v2 已经在现有评测中表现出：
- PENDING 提交后正确结束 18/18；
- Blocked 零调用 6/6；
- Blocked 解释仍有不足。

因此：
- PENDING / READY 只保留少量 Dev/Test 回归控制；
- 不把大量简单 PENDING 任务放入 GRPO Train；
- READY 只能来自 Snapshot 的真实翻译记录；
- Blocked 任务不作为本轮主要训练目标，因为 Prompt 已经给出明确的能力边界规则，且当前任务目标应转向复杂可执行工作流；
- 如果 Sealed Test 有 Blocked 题，必须报告零调用与解释准确率，但不因此扩充训练任务。

---

## 5. TaskSpec、Gold Replay 和数据防泄漏

### 5.1 复用现有 TaskSpec / Step

每个任务继续使用当前 `TaskSpec` / `Step`：
- `steps` 是唯一标准工具动作序列；
- `expected_tools` 和 `expected_tool_args` 由 steps 派生；
- `setup` 建立初始环境状态，不计入模型动作；
- `expected_paper_ids` 按现有任务格式逐步记录；
- `max_iterations` 覆盖 Gold tool calls + FINISH；
- blocked 仅用于明确不可执行的控制任务。

本轮不重写 Strict Scorer，不引入多路径 DAG，不改现有线性 DFA。

### 5.2 每个任务必须记录的元数据

至少记录：
- task_id；
- family；
- difficulty；
- split；
- 用户任务文本；
- steps；
- setup；
- target paper_ids；
- expected_paper_ids；
- Snapshot keys；
- terminal_mode / terminal_reason；
- max_iterations；
- parent task IDs；
- action signature；
- setup-state signature；
- workflow signature；
- 任务与 SFT v2 的新颖性说明。

以上 TaskSpec 标准答案和元数据绝不能出现在模型可见 Prompt 中。

### 5.3 Gold Replay

每个任务必须：
1. 创建独立 MemoryStore、MockArxivEnv、PDF root 和 Session；
2. 使用真实 Native Agent 执行 Setup；
3. 验证 Setup 后的初始状态；
4. 逐步执行全部 Gold Actions；
5. 保存真实 Tool Call、tool_call_id、参数、Observation、paper_id；
6. 验证所有 Snapshot keys；
7. 验证最终 FINISH / blocked semantics；
8. 调用现有 `is_strict_success()`；
9. Gold Strict Success 必须为 1；
10. 至少重复 Replay 一次，结果完全确定。

环境异常、Snapshot miss、Setup failure 一律排除，不进入 Probe / GRPO。

### 5.4 任务新颖性与泄漏审计

与以下历史资产比较：
- `data/sft/native_sft_v2/native_sft_v2_task_catalog.jsonl`
- `data/splits/v4_86.json`
- 旧 Dev / IID / OOD；
- `data/splits/native_grpo_qwen35_candidates_v2.json`
- `data/splits/native_grpo_qwen35_train_v2.json`
- 旧 E1–E4 训练 Group Schedule。

至少比较：
- task_id；
- 规范化用户请求；
- target paper_id 集合；
- setup-state signature；
- effective action sequence；
- terminal semantics；
- 工具链长度；
- workflow signature；
- 多论文身份切换模式。

不能只更换措辞、ref 序号或论文 ID 就称为新任务。

### 5.5 50 篇 Snapshot 下的泛化边界

优先尝试按 paper_id 生成 Snapshot views，但不能为了 paper-level disjoint 导致某个 split 缺少工具记录。

若无法得到覆盖完整的 30/10/10 paper-level views：
- 使用 task-signature-disjoint 的 Train / Dev / Sealed Test；
- 允许同一论文出现在不同 split；
- 必须输出 paper_id overlap 矩阵；
- 明确本轮验证的是“新工作流组合泛化”，不是“未见论文泛化”。

---

## 6. Frozen Probe：只在 Native SFT v2 + 完整生产 Prompt 下运行

### 6.1 Probe 配置

- Policy：Native SFT v2 Adapter；
- Prompt：原始完整 `NATIVE_TOOL_SYSTEM_PROMPT`；
- Setup Context：原始 `build_visible_setup_context()`；
- Tool Schema：与 SFT v2 一致；
- Snapshot：固定 50-paper Snapshot；
- 每个 Train Candidate 8 个独立 Prompt Groups；
- 每组 G=4；
- Temperature=1.0；
- 每个 Rollout 独立 Session / MemoryStore；
- 不更新模型参数；
- 保存全部原始轨迹、Seed、Token 长度、Strict、Credit 和环境状态。

24 个候选任务的最大 Probe 量：
[
24 	imes 8 	imes 4 = 768 	ext{Rollouts}
]

### 6.2 必须区分三种信号

**A. Strict-informative Group**

四条轨迹均可判定，且同组 Strict Outcome 同时包含成功和失败。

- `[1,1,0,0]`：有效；
- `[0,0,0,0]`：零 Strict 信号；
- `[1,1,1,1]`：ceiling，无 Strict 信号；
- 含环境异常、Snapshot miss 或不可判定轨迹：不计入有效 Strict Group。

**B. DFA Process-informative Decision**

相同 DFA Contract State 下存在可比较的正确动作和首错动作，并且局部 Advantage 非零。

**C. GiGPO Exact-anchor-informative Decision**

同一个精确 Anchor State 在多个 Rollout 中重复出现，并满足：
- 环境状态一致；
- 模型可见上下文一致；
- 动作不同；
- 后续 Strict Return 有差异；
- Anchor fingerprint 不包含 session_id、时间戳、临时路径和随机计数器。

### 6.3 Go / No-Go

单任务入选条件：
- 8 组中至少 2 组 Strict-informative；
- Gold Replay 通过；
- 环境异常为 0；
- 没有大比例截断；
- 不存在训练数据和 Held-out 的行为签名泄漏。

整体 Go 条件：
1. 至少 12 个任务入选；
2. 至少覆盖 4 个任务家族；
3. 任一任务家族不超过 Train Pool 的 40%；
4. Strict-informative Group Fraction 至少 25%；
5. DFA Local Credit 有效覆盖至少 10% 的可判定决策；
6. Anchor-informative decision coverage 至少 10%；
7. 至少 8 个入选任务包含 5 次或更多模型工具调用；
8. 环境异常率为 0；
9. Gold completion 无静默截断；
10. Train / Dev / Sealed Test 泄漏审计通过。

未达到 Go：
- 不修改 Reward 凑信号；
- 不降低 Strict Success 标准；
- 允许根据 Snapshot 真实覆盖再补充一次候选任务；
- 仍不达标则停止本轮 GRPO，输出 No-Go 报告，不启动 E1–E4。

---

## 7. E1–E4：复用 Credit v2，不再重做算法

### 7.1 实验定义

| 实验 | 终局 Reward | Local Credit | First-error Protection |
|---|---|---|---|
| E1 | Strict 0/1 | 无，Episode Advantage | 无 |
| E2 | Strict 0/1 | DFA Contract-state | 有 |
| E3 | Strict 0/1 | GiGPO Exact Anchor | 无 |
| E4 | Strict 0/1 | DFA + GiGPO | 有 |

具体要求：
- E1：只用同一 G=4 Prompt Group 的 Strict Outcome Advantage。
- E2：用 DFA 状态比较正确/错误动作；首错前缀保护，首错负向，首错后无独立证据则 Mask。
- E3：用 Exact Anchor State 比较同一真实状态下不同动作的后续 Strict Return。
- E4：结合 DFA、Anchor 和 First-error Protection。

所有模式必须分别记录：
- raw_dfa_adv；
- raw_anchor_adv；
- raw_combined_local；
- effective_local；
- first-error localization；
- verified-prefix preservation；
- correct-action promotion；
- first-error suppression；
- anchor coverage / ranking；
- advantage misalignment；
- environment invalid；
- truncation。

不能把保护规则保证的方向一致性冒充为原始 Credit Estimator 的 Accuracy。

### 7.2 Strict Reward 不变

统一主 Reward：

```python
reward = 1.0 if is_strict_success(task, trajectory) else 0.0
```

- 继续使用现有 `benchmark.metrics.is_strict_success()`；
- 不修改工具序列、参数、引用、终止和执行错误的 Strict 定义；
- 不把旧五维 Reward 或格式分数混入主 Reward；
- Snapshot miss、Setup failure、环境异常必须 mask；
- 模型自身的工具错误、参数错误、引用错误、提前 FINISH 仍然是策略失败。

受控 Fault Injection 任务只有在单独通过 Strict 语义扩展审计后才能加入 Train。不得悄悄把预期注入错误计为普通模型工具失败，也不得修改普通任务的 Strict 口径。

### 7.3 GPU 前审计旧 Credit v2

审计：
- `AgenticArxiv/rl/task_automaton.py`
- `AgenticArxiv/rl/anchor_state.py`
- `AgenticArxiv/rl/credit_v2.py`
- `AgenticArxiv/rl/step_credit_trainer.py`

重点检查：
- 正确前缀不会因为后续截断而整体丢弃；
- `q_truncated` 不等于 `q_error`；
- 环境异常与策略错误分离；
- Anchor fingerprint 具有真实环境状态；
- Tool Observation 不参与 Policy Loss；
- Assistant Tool Call / Content / Final Answer Token Span 与决策对齐；
- raw / effective Credit 分开；
- 所有旧 Credit v2 测试通过。

不在正式 GPU 训练过程中临时修改 Credit 规则或权重。

---

## 8. 最后一轮 GRPO 训练计划

### 8.1 统一训练起点

E1–E4 全部独立从 Native SFT v2 Adapter 初始化：

```text
Base Model: Qwen/Qwen3.5-2B-Base
SFT v2 Adapter SHA256:
f0c33a4d32a175bcaf1e83c55231a0108a84859cb748c318db49988a52f7911f
Snapshot SHA256:
d97035db326ca11935e882fd2aca7fe932e9e76cf1bc8a21078be7bec1ae8794
Prompt SHA256:
31674f2183d642a9035ecf8360c138135c5ef4645904017eadf4ec7bdf4abf05
Tools SHA256:
01ed14f7f8a041ca9752475c1bf1513996c375c5a32cadf2514089fb04401ae0
Chat Template SHA256:
273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80
```

每个模式：
- 独立加载 SFT v2 Adapter；
- 独立保存 GRPO Adapter；
- 不得 E2 从 E1 继续；
- 不得 E3 从 E2 继续；
- 不得覆盖 SFT v2 Adapter；
- 使用相同 Train Split 和固定 Group Schedule。

### 8.2 固定训练预算

沿用旧 Native GRPO 已在 RTX 4090 上运行通过的配置：

| 参数 | 值 |
|---|---:|
| `num_generations` | 4 |
| `learning_rate` | `5e-6` |
| `beta` | 0.04 |
| `seed` | 42 |
| `temperature` | 1.0 |
| `scale_rewards` | `group` |
| `dynamic_sampling` | false |
| `step_credit_weight` | 0.5 |
| `bf16` | true |
| `per_device_train_batch_size` | 1 |
| `gradient_accumulation_steps` | 4 |
| `gradient_checkpointing` | true |
| Optimizer Steps / Mode | 25 |
| Modes | E1、E2、E3、E4 |

四个模式各 25 steps，共 100 optimizer steps、约 400 条正式训练 Rollout（每步一个 G=4 Prompt Group）。

### 8.3 Loss Type 必须显式固定

旧 Native GRPO 使用 TRL 1.17.1，之前的 `GRPOConfig` 没有显式传入 `loss_type`，该版本默认 `loss_type="dapo"`。

为保持与旧 E1–E4 的 Policy Loss 归一化方式一致，本轮必须：
- 核验实际安装版本为 TRL 1.17.1；
- 显式设置 `loss_type="dapo"`；
- 不启用 DAPO Dynamic Sampling；
- 不启用 Clip-Higher；
- 不引入其他新 DAPO 组件。

报告准确称为“GRPO 组内优势 + DAPO token-level loss normalization”，不能称为完整 DAPO。

如果实际环境的 TRL 版本或旧实验 Loss Type 不一致，先停止并报告，不能静默切换。

### 8.4 固定 Group Schedule

Probe 后生成一个固定的 25-group 训练计划：
- 约 40% 长程依赖链 / 多目标任务；
- 约 25% 多论文交错 / 身份保持；
- 约 20% 状态条件 / 缓存 / force；
- 约 10% 内容 / 摘要 / 图表链；
- 约 5% 受控 Recovery（仅当真实可复现且有 Probe 信号）。

要求：
- 每个 group 对应一个 Prompt，G=4；
- 每个 group 的四条轨迹从相同初始 Snapshot / Setup 状态开始；
- 每个 task_id、group_id、seed 固定并记录；
- 单个任务最多重复 3 次；
- 不启用动态采样；
- 不因为某条轨迹失败而丢弃同组其他轨迹；
- E1–E4 使用完全相同的 25-group 顺序和 seed schedule。

如果最终可入选任务少于 12，或某个类别没有足够任务，不得通过重复一个任务填满整个 Schedule。

### 8.5 长轨迹 Token Budget

新任务比旧 18-task Train Split 更长，不能无条件沿用旧 `max_completion_length=1792`。

Gold Replay 后统计：
- 每条 Gold 完整轨迹 tokens；
- 最大值和 P95；
- 每轮 Assistant generation tokens；
- 每个任务的工具调用次数；
- 每个任务的 `max_iterations`。

设置：
- `max_iterations` 至少覆盖 Gold 工具调用数 + FINISH；
- `max_new_tokens_per_turn` 覆盖 Gold 单轮最大 Assistant tokens 并留余量；
- `max_completion_length` 至少为 Gold 完整轨迹最大 Token 数的 1.2 倍；
- 不允许静默截断 Gold；
- 先用 4-step Smoke 验证 24GB 显存和轨迹长度；
- 超出可稳定运行预算的任务应简化或排除，不能让截断成为主要训练信号。

E1–E4 必须使用完全相同的 Token Budget。

---

## 9. 评测协议

### 9.1 主要比较

所有模型使用相同完整生产 Prompt、Tool Schema、Snapshot、Chat Template、解码参数和 Trial Seeds：

- Frozen Native SFT v2；
- E1；
- E2；
- E3；
- E4。

### 9.2 Dev / Sealed Test

- Dev：8 tasks × 3 trials = 24 episodes；
- Sealed Test：12 tasks × 3 trials = 36 episodes；
- Sealed Test 不参与 Probe、Reward 调整、超参数选择或 Checkpoint 选择；
- E1–E4 训练配置冻结后，一次性运行 Sealed Test；
- 保存每个 task/trial 的原始轨迹和配对变化。

Primary Metrics：
- Strict Success；
- Fail-to-pass；
- Pass-to-fail；
- Net gain。

Secondary Metrics：
- 工具序列准确率；
- effective argument accuracy；
- paper reference resolution；
- 多步骤完整率；
- 多论文身份保持；
- cache / force decision；
- content / figure pipeline 完整率；
- recovery success（仅适用于真实故障注入任务）；
- extra calls / repeats；
- tool errors；
- Snapshot coverage misses；
- truncation / max-turn stop；
- completion tokens 和每任务工具调用数。

置信区间按 task_id 聚类 Bootstrap，不得把同一道任务的三次重复当成完全独立任务。

### 9.3 生产 Benchmark 回归

新 Challenge 与原有生产 Benchmark 使用相同完整 Prompt，但任务集不同。

Dev 选出候选配置后，只对 Dev 最佳候选额外运行原有 31-task / 93-trial Native Benchmark，与 SFT v2 的 83/93 比较，确认 GRPO 没有破坏已掌握的生产能力。

不能把 Challenge 的 Strict Success 和旧生产 Benchmark 的 83/93 合并。

### 9.4 结果解释边界

- 25 steps、单训练 Seed 是 Pilot，不代表收敛。
- 44 个任务的 Challenge 规模仍有限，结论限定于当前 Snapshot 和工具集合。
- 只有 Challenge Sealed Test 的配对 Strict Success 提升，且生产 Benchmark 无明显退化，才认为 GRPO 有额外收益。
- 训练 Reward、局部 Credit Accuracy、Dev 单次成功率不能代替最终结果。
- 如果 Probe 信号不足，实验不可判定，不能据此宣称 GRPO 无效。
- 如果 E2/E4 优于 E1 而 E3 无效，应如实说明 DFA/First-error 在本任务集上更有用，不强行强调 GiGPO。

---

## 10. 执行顺序与 GPU 预算

### 10.1 CPU 阶段

Codex 先完成：
1. 创建工作分支并记录基线 commit；
2. 核验 SFT v2 Adapter / Base / Prompt / Tool / Chat Template / Snapshot SHA；
3. 生成 Snapshot Inventory；
4. 读取并保留原生产 Prompt，不改写；
5. 根据真实 Snapshot 生成任务候选；
6. 任务 Gold Replay；
7. Strict Scorer 与 TaskSpec 审计；
8. Train / Dev / Sealed Test 泄漏审计；
9. E1–E4 Credit v2 单元测试；
10. 生成 GPU Readiness。

CPU 阶段不加载 SFT 模型，不运行 Frozen Probe，不启动训练。

### 10.2 GPU 阶段

用户确认启动后：
1. SFT v2 + 完整 Prompt Frozen Probe；
2. 检查 Go / No-Go；
3. E1–E4 各 4-step Smoke；
4. E1–E4 各 25-step Pilot，串行；
5. Dev 配对评测；
6. 冻结配置后 Sealed Test；
7. Dev 最佳候选运行生产 Benchmark 回归；
8. 生成最终报告，停止，不自动追加 100 steps。

### 10.3 预算

以 24 Train Candidate × 8 groups × G=4 估算：
- Probe 最多 768 条 Rollout；
- 每个 E 模式 25 groups × 4 = 100 条训练 Rollout；
- 四个模式合计约 400 条正式训练 Rollout；
- Dev/Test 每个模型最多 60 episodes。

新任务轨迹更长，不能直接套用旧 Probe 的时间估算。Codex 必须在 CPU Readiness 中根据 Gold token 长度、max_turns 和首批 Smoke 预估实际 GPU 时间。

RTX 4090 要求：
- 单 GPU、单进程；
- E1–E4 串行；
- 不并行加载多个 Policy；
- 每个模式结束后释放模型和 CUDA Cache；
- 记录峰值显存、训练时间和 Rollout 吞吐；
- 不得为了节省时间跳过 Probe、Gold Replay 或 Sealed Test。

CPU 阶段完成后生成 `reports/agent_challenge_v1/gpu_readiness.md`，列出任务数、Snapshot 覆盖、Gold Replay 通过率、Prompt SHA、Split SHA、Token Budget、Probe 预计耗时和 GPU 命令。**生成 Readiness 后暂停，等待用户明确启动 GPU。**

---

## 11. Codex 交付目录

建议目录：

```text
AgenticArxiv/benchmark/challenge_v1/
  __init__.py
  task_builder.py
  snapshot_inventory.py
  task_audit.py

scripts/agent_challenge_v1/
  build_tasks.py
  audit_snapshot.py
  audit_tasks.py
  run_probe.py
  run_ablation.py
  evaluate.py

data/agent_challenge_v1/
  snapshot_inventory.json
  task_catalog.jsonl
  train_candidates.jsonl
  train_selected.jsonl
  dev.jsonl
  sealed_test.jsonl
  manifest.json

reports/agent_challenge_v1/
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

outputs/agent_challenge_v1/
  E1/
  E2/
  E3/
  E4/
```

文件名可以按仓库规范调整，但必须保留全部审计信息和原始轨迹。

---

## 12. 最终成功标准

本轮只允许给出以下四类结论之一：

1. **GRPO 有额外收益：** Challenge Sealed Test 相比同 Prompt 下的 SFT v2 有净 Strict Success 提升，且原生产 Benchmark 没有明显退化。
2. **只有特定 Credit 方法有效：** 例如 E2/E4 优于 E1，而 E3 无效；结论限定于当前任务集和 Snapshot。
3. **没有额外收益：** SFT v2 已经掌握这些复杂任务，或 GRPO 没有进一步改善。
4. **实验不可判定：** Probe 信号不足、Snapshot 覆盖不足、任务泄漏、环境错误或训练链路审计失败。

不允许只根据训练 Reward、Loss、Credit Accuracy 或 Dev 单次成功率宣布 GRPO 有效。

---

## 13. 给 Codex 的明确执行指令

请在 `exp/agent-challenge-grpo-v1` 分支严格执行本文。

**本设计取代此前 Protocol-only Prompt 方案。**

第一阶段只完成 CPU 工作，生成 GPU Readiness 后暂停。

重点核验：
- SFT v2 Adapter SHA；
- 完整生产 Prompt SHA 必须等于 SFT v2 训练 Manifest；
- Tool Schema / Chat Template SHA；
- Snapshot SHA 和唯一论文数量；
- 所有任务的 Snapshot keys 和 Gold Replay；
- 新任务是否真正具有多论文、长程依赖或多状态组合，而非旧任务换词；
- Strict Scorer 与 E1–E4 是否可直接复用；
- 所有模式是否使用同一固定训练计划。

禁止：
- 修改或删减生产 System Prompt；
- 新增 Protocol-only Prompt；
- 修改 `build_visible_setup_context()`；
- 改变 Native Tool Calling 协议；
- 修改旧 Benchmark Strict Success 定义；
- 联网扩充 Snapshot；
- 伪造论文、Observation、PENDING/READY 或工具错误；
- 将 Snapshot miss 当成模型策略错误；
- 通过改变 Reward 凑 Probe 信号；
- 未经用户确认启动 GPU。

**本轮研究目标不是让 GRPO 重复 SFT v2 已经学会的基础工具调用，而是在相同生产 Harness 条件下，检验它能否进一步改善复杂、长程、多论文和状态交错任务的执行成功率。**
