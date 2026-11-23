# Qwen3.5-2B-Base Native GRPO：目标、Rollout、Reward 与实施计划

> 本文是 Native SFT 之后的 GRPO 实施规格。目标不是重写 GRPO，而是把现有 ReAct GRPO 生成链路迁移到 Qwen3.5 Native Tool Calling，并在此基础上复用已有的可验证 Reward 和 Prefix-tree Step Credit。
>
> 工作分支：\`feat/qwen35-native-grpo\`  
> 基线分支：\`feat/qwen35-base-native-sft\`  
> 基线提交：Native SFT 最终结果所在分支的当前 HEAD  
> 策略模型：Qwen3.5-2B-Base + 2026-11-21 Native SFT Adapter  
> 环境：固定 Snapshot + MockArxivEnv Replay  
> 本轮先做：Native Rollout、Reward 审计、Frozen-policy Probe、Step Credit 验证。审计通过后再启动正式训练。

---

## 0. 本轮要回答的研究问题

本轮 GRPO 不以“继续提高 Tool Call 格式解析率”为主要目标。Native SFT 已经把格式解析成功率从 95.5% 提升到 99.2%，Schema 合规率为 100%。继续把主要训练预算放在格式上，收益有限。

本轮核心假设：

> 在 Native SFT 已经学会基础工具选择、参数表达和常见工具链的基础上，GRPO 能否利用多轮环境反馈和组内相对奖励，进一步改善模型在多步骤、条件依赖和训练中未直接出现的工具组合上的决策能力？

需要验证的能力目标：

1. **状态条件决策**：根据真实 Observation 判断下一步是否需要调用工具、调用哪个工具、是否已经满足用户要求。
2. **多步骤完整执行**：完成搜索、选择论文、下载、缓存检查、翻译或读取等有依赖关系的步骤，不提前 FINISH，也不漏掉必需步骤。
3. **用户约束与参数决策**：正确处理“只下载 / 下载后查缓存 / 只查缓存”、是否保存文件、force、keep_dual、论文引用和范围限制。
4. **多对象任务完成**：对用户明确要求的多个论文逐一执行操作，不只处理第一个对象。
5. **阻塞与终止**：在缺少上下文、无效引用或工具能力不支持时不调用无关工具，并说明具体原因；正确区分 PENDING 和 READY。
6. **组合泛化**：把已掌握的工具能力组合到训练中未直接出现的新任务中。

不把“增加工具调用次数”或“让任务更长”本身作为目标。任务必须包含真实的状态依赖或条件分支，额外步骤必须由用户目标要求。

---

## 1. SFT 结果给出的 GRPO 方向

当前最终 Native SFT 结果：

| 指标 | Base | Native SFT |
|---|---:|---:|
| Strict Success | 46/93（49.5%） | 69/93（74.2%） |
| Dev | 17/24（70.8%） | 17/24（70.8%） |
| IID | 20/57（35.1%） | 45/57（78.9%） |
| OOD | 9/12（75.0%） | 7/12（58.3%） |
| 工具序列完全匹配 | 54.8% | 86.0% |
| 生效参数平均得分 | 62.5% | 84.1% |
| 论文引用全对 | 62.1% | 84.8% |
| 多步骤完成 | 59.0% | 82.1% |
| 超出要求的重复调用试次 | 12/93 | 3/93 |
| PENDING 提交后正常结束 | 8/14 | 21/21 |
| 阻塞任务零调用且原因正确 | 0/6 | 0/6 |

当前 SFT 剩余问题：

- OOD 的 \`multi_cv5_dl1_cache_tr\` 三次都在下载后跳过缓存检查，直接提交翻译。
- OOD 的 \`multi_cl5_dl2_cache\` 有一次最终回答重复并达到生成长度上限。
- Dev 的“全 CS 检索且禁止写文件”仍然可能把范围选成 CL。SFT 三次都正确处理了不保存，但范围错误。
- 阻塞任务零调用且正确说明原因仍为 0/6。
- SFT 后仍有参数错误、论文引用错误和提前结束。

这些测试任务本身属于留出集，**不得直接加入 GRPO 训练**。只能从 Train 父任务构造语义相近、参数或 Session 状态不同的新任务。

### 1.1 对“之前 GRPO 提升不明显”的判断

2026-11-18 的历史报告中，旧 SFT → GRPO 的总体 Strict Success 是 69/93 → 72/93，增加 3.2 个百分点；新增成功集中在一个 IID 任务。该实验使用的是旧版 2,496 行 ReAct SFT 数据、旧 SFT Adapter 和当时的 ReAct GRPO 路径，不是本次 505 条 Native SFT。

因此：

- 历史结果说明之前的 GRPO 增量有限，但不能直接预测本次 Native SFT → Native GRPO 的结果。
- “任务太短”是待验证假设，不是已经证实的原因。
- 还需要检查训练任务与当前策略的匹配程度、组内奖励方差、奖励是否正确区分行为、Rollout 是否完整，以及 Prefix-tree 信用分配是否真正启用。

---

## 2. 当前代码状态与必须修复的协议差异

### 2.1 已有、可复用的部分

- \`agents/native_tool_agent.py\`：Native Tool Calling Agent。
- \`tools/tool_registry.py\`：工具注册和模型可见 Schema。
- \`rl/env.py\`、\`rl/multiturn_env.py\`：MockArxivEnv 和多轮工具环境。
- \`rl/reward.py\`：五个主要奖励分量及安全门控。
- \`rl/step_credit.py\`：规范化动作前缀树、Reward Credit、Progress Credit。
- \`rl/step_credit_trainer.py\`：将 Step Credit 对齐到 TRL token-level advantages。
- \`rl/grpo_reward.py\`：已有 Native messages 到内部 history 的转换函数，以及 ReAct 多轮 Rollout。

### 2.2 当前 GRPO 仍是 ReAct 路径

\`train_grpo.py\` 目前调用：

~~~python
rows = build_prompt_dataset(tasks)
~~~

而 \`build_prompt_dataset()\` 使用 \`get_react_prompt()\`。多轮 Rollout 使用 \`make_multiturn_rollout_func()\`，内部通过 \`parse_react_action()\` 解析 \`Thought:\` / \`Action:\`。

因此，当前代码还不是完整的 Native GRPO。已有 \`messages_to_trajectory()\` 只解决了 Native 消息到内部 Reward history 的一部分转换，并未替代正式 Rollout。

**本轮必须实现独立的 Native GRPO Prompt + Rollout 路径，不能只把 Native SFT Adapter 传给旧 ReAct GRPO。**

### 2.3 Native 转换中已发现的缺陷

现有 \`messages_to_trajectory()\` 有两处需要修正：

1. 当 \`function.arguments\` JSON 解析失败时，当前实现把参数改成空字典 \`{}\`。这会掩盖格式错误，必须改为显式 parse failure，且不得执行该工具。
2. 当前通过 FIFO 顺序把 Tool Message 配给 pending tool call。Native 协议应使用 \`tool_call_id\` 精确配对；未知、重复或缺失 ID 必须作为协议错误处理。

此外，Native Rollout 必须强制每个 Assistant turn 最多一个 tool call，与当前 Native Agent 约定一致。

---

## 3. Native GRPO Rollout 设计

### 3.1 模型输入

模型看到的协议必须与 Native Benchmark 一致：

- 当前 Native System Prompt。
- 用户任务。
- 当前会话状态和历史消息。
- 通过运行时 \`tools\` Schema 注入的 10 个工具。
- Qwen3.5 官方 Chat Template。
- 关闭 thinking。
- 不使用旧 ReAct 的 \`Thought:\` / \`Action:\` / \`Observation:\` 文本协议。

不要在 GRPO 中另写一份简化 System Prompt。复用 Native Agent 当前的 Prompt、Tool Schema 和模板渲染方式，并在 Manifest 中记录对应 SHA256。

### 3.2 每条 Rollout 的执行流程

同一个 Prompt 由当前策略采样 G 条独立轨迹（首轮固定 G=4）。每条轨迹都必须创建独立 Mock 环境和 Session。

~~~text
Task Prompt + Native Tools Schema
              |
              v
      Qwen Native Generation
              |
       +------+------+
       |             |
       v             v
  Native Tool Call  Final Answer
       |             |
       v             v
  解析与 Schema 校验   FINISH
       |
       v
  MockArxivEnv 执行
       |
       v
  真实 Tool Observation
       |
       v
  role=tool + tool_call_id
       |
       v
  按 Qwen Chat Template 回填上下文
       |
       v
  下一轮 Native Generation
~~~

要求：

1. \`TaskSpec.setup\` 在模型生成前执行，用于初始化会话状态，不进入模型动作序列、不计入 Reward 和 Policy Loss。
2. 每轮只生成一个 Assistant Message。
3. 如果生成 Native Tool Call，解析工具名和参数，校验 Schema，再执行 MockArxivEnv。
4. 将工具结果作为 \`role=tool\` 消息回填，使用对应的 \`tool_call_id\`。
5. 下一轮重新基于完整消息历史生成。
6. 如果生成普通 Assistant Final Answer，记为 FINISH 并结束。
7. 不允许模型生成或伪造 Tool Observation。
8. 工具解析失败、Schema 错误、工具执行错误必须保留在轨迹中，不能静默修复为成功。
9. PENDING 是工具的真实 Observation，不代表 READY。是否继续执行取决于用户要求和后续步骤依赖。
10. 达到 max_turns 或 completion token 上限时，标记为 truncated，不得自动补成成功 FINISH。

### 3.3 Token 与环境消息边界

TRL 的 GRPO Loss 只能作用于策略生成的 Assistant Token，不能把工具 Observation 当成模型生成内容训练。

Rollout 必须保存：

- \`prompt_ids\`：完整输入。
- \`completion_ids\`：本轮所有 Assistant 输出和环境回填内容。
- \`tool_mask\` / 等价环境 Mask：标记工具 Observation。
- \`turn_token_spans\`：每个 Assistant turn 在 completion 中的 \`[start, end)\` 区间。
- \`history\`：每轮 Native Tool Call、真实 Observation、Final Answer。
- \`clipped\`、\`reached_max_turns\`、\`termination_reason\`。
- \`setup_receipts\`：Setup 执行记录，供审计但不进入模型动作序列。

Step Credit 的 token spans 必须只覆盖 Assistant 生成部分。Tool Observation 和 padding 不参与 Policy Loss。

Content 和 Tool Call 同属 Assistant Message，第一版沿用相同 turn-level advantage。暂不单独设计自然语言 Content Reward；必须监控训练前后 Content 是否变得冗长、与实际状态不符或退化。

### 3.4 Qwen / TRL 兼容性

- \`requirements-qwen35.txt\` 固定 \`trl==1.17.1\`。
- \`StepCreditGRPOTrainer\` 也明确要求 TRL 1.17.1。
- 本次 SFT Manifest 记录的 TRL 是 0.29.1；不要直接假设 SFT 虚拟环境也能运行当前 Step Credit。
- Native GRPO 使用独立的、锁定依赖版本的环境。
- 必须在真实安装环境中测试 Qwen Chat Template、单轮生成停止边界、Tool Message 回填、TRL 自定义 Rollout 接口和 tool mask。

---

## 4. Reward 设计：先审计现有五维奖励，不先改权重

### 4.1 第一轮保留现有权重

现有 RewardCalculator 的默认权重：

| 分量 | 权重 | 含义 |
|---|---:|---|
| Format | 1 | 输出是否形成有效动作 / 终止 |
| Tool | 3 | 工具调用序列是否符合任务要求 |
| Argument | 2 | 参数是否符合要求，包含有效参数和论文指代 |
| Process | 1 | 有效步骤、解析失败、工具执行失败和额外调用 |
| Outcome | 3 | 是否完成任务、是否正确终止、是否正确解释阻塞 |

总奖励采用加权归一化，分量范围为 [-1, 1]。

另外：

- \`result_quality\`：检查工具 Observation 是否有实际、有效、可用的结果。
- \`efficiency\`：检查超出任务要求的调用和错误开销。

当前这两个分量记录在诊断信息中，不进入五维加权平均；它们参与安全门控或奖励上限。不要把它们误写成已经拥有独立权重的第六、第七维。

第一轮固定权重为 \`format:1, tool:3, argument:2, process:1, outcome:3\`。先不做权重搜索，也不同时修改课程学习、KL、采样数量和奖励定义。

第一轮沿用已有 Qwen3.5 GRPO Launcher 的 \`reward_curriculum_steps=0\`，全程使用完整权重。若之后要验证课程学习，再单独做消融。

### 4.2 Native 轨迹转成统一 Reward History

内部统一格式：

~~~json
{
  "history": [
    {
      "thought": "简短的 Assistant Content",
      "action": "{\"name\":\"download_arxiv_pdf\",\"args\":{\"ref\":2}}",
      "observation": "真实工具返回"
    },
    {
      "thought": "PDF 已就绪，用户还要求查缓存。",
      "action": "{\"name\":\"get_paper_cache_status\",\"args\":{\"ref\":2}}",
      "observation": "真实缓存状态"
    },
    {
      "thought": "PDF 已就绪，译文未就绪。",
      "action": "FINISH",
      "observation": "任务完成"
    }
  ]
}
~~~

注意：

- Native 的工具名、参数和 Tool Observation 必须来自真实交互。
- 最终 Assistant Answer 转为 FINISH，文本保存在 \`thought\` 字段，供终止语义评估使用。
- 无效 JSON / Schema 错误不得变成空参数的正常工具调用。
- Tool Message 按 \`tool_call_id\` 配对。
- 每条轨迹必须保存原始 Native Messages，便于复核转换前后的信息没有丢失。

### 4.3 各分量必须通过 Native 轨迹测试

| 分量 | Native GRPO 必须验证的行为 |
|---|---|
| Format | 完整 Tool Call、参数对象、工具 Schema、结束标记；非法调用不能被当成有效格式 |
| Tool | 工具名和调用顺序；不支持任务要求零调用 |
| Argument | 调用参数经过 \`registry.effective_arguments()\` 后再与标准答案比较；论文引用按真实 Observation 解析结果判断 |
| Process | 重复调用、漏步骤、工具错误、解析错误、提前结束、超出 max_turns |
| Outcome | 完整完成、正确 FINISH、blocked 任务明确拒绝、PENDING 不得误报 READY |
| Result Quality | 工具结果真实可用，不把 Snapshot miss、fallback 或空图表答案当成成功 |
| Efficiency | 不执行用户未要求的额外工具调用 |

特别检查：

1. 用户要求下载后查缓存时，不能因为下载返回 PDF READY 就直接 FINISH。
2. 用户只要求下载时，不得为了拿额外 Reward 擅自查缓存。
3. 用户要求“不保存”时，工具默认值不能覆盖用户约束。
4. 默认参数省略与显式传默认值必须按实际生效参数判断。
5. 论文引用必须依据 Observation 中实际解析到的 paper_id 判断。
6. blocked 任务正确拒绝与普通 FINISH 必须区分。
7. PENDING 不能被当成 READY。若任务要求依赖译文内容的下一步，必须说明未就绪；若有用户明确要求且不依赖译文的独立后续步骤，可以继续。

### 4.4 一个当前 Reward 的边界：Final Answer 内容

当前 Reward 主要评估工具序列、参数、工具执行结果、终止和 blocked 语义。它没有通用地判断所有 Final Answer 是否完整交付了摘要或论文正文。

因此，第一版 GRPO 的主要优化目标是**工具决策和多轮执行策略**，不是自由文本答案质量。

- 不要为了“奖励自然语言写得好”直接引入不稳定的 LLM-as-Judge。
- 对 PENDING / READY、阻塞拒绝、任务是否完成等状态，优先增加确定性状态一致性检查。
- 对摘要和正文等长文本，暂时保留 SFT 已学习的表达能力，通过 KL 约束和最终 Benchmark 监控是否退化。
- 如果后续确实要优化答案内容，再单独定义可验证的答案交付指标。

---

## 5. Prefix-tree Step Credit：复用现有实现，但必须 Native 化审计

### 5.1 原理和现有实现

现有 \`step_credit.py\` 已实现两种信号：

**模块 A：Reward Credit**

- 对同一任务的 G 条轨迹按规范化动作前缀构建树。
- 节点价值是经过该前缀的轨迹最终 Reward 均值。
- 每步分配相邻前缀节点的价值差。
- 叶子使用该轨迹自身的 Reward。
- 各步信用之和与整条轨迹的 GRPO Advantage 守恒。

**模块 B：Progress Credit**

- 当组内最终 Reward 零方差、但规则进度有差异时启用。
- 按顺序匹配任务的标准工具、有效参数、论文引用和 FINISH。
- 只在第一次偏离前累计进度，后续步骤标记为 off-prefix。
- 为原本没有最终奖励差异的组提供规则进度信号。

模式：

| 模式 | 行为 |
|---|---|
| \`off\` | 只使用标准 GRPO 轨迹级 Advantage |
| \`reward\` | 只使用基于最终 Reward 的前缀树信用 |
| \`progress\` | 优先使用 Reward Credit；最终 Reward 零方差时使用 Progress Credit |

### 5.2 Native 化必须修正的内容

1. \`canonical_action_key()\` 使用工具的有效参数，而不是仅使用模型原始传入的参数字典。
2. \`label_trajectory()\` 使用与 Benchmark 一致的参数默认值、类型和语义比较，不能把省略默认值误判为错误。
3. ref 的动作键以实际解析到的论文 ID 规范化，保持已有能力。
4. 每轮 Native Assistant Tool Call 对应一个 history step 和一个 token span。
5. Final Answer 对应 FINISH step 和一个 Assistant span。
6. Tool Observation 不生成 step span，不参与 Policy Loss。
7. 如果任一轨迹被截断或达到 max_turns，按当前设计禁用该组 Step Credit，避免伪造终止叶子污染前缀价值。
8. 维持 Step Credit 与 TRL 原始 Advantage 的尺度对齐和 Reward Credit 守恒检查。
9. 记录 reward / progress / none / truncated group fraction、singleton step fraction、credit coverage、credit accuracy 和 max conservation error。

### 5.3 首轮训练的对照方式

不要一开始同时打开所有训练技巧。

先用完全相同的 SFT Adapter、任务、Snapshot、随机种子、训练参数和 Rollout 预算，至少保留：

- Native scalar GRPO：\`step_credit=off\`。
- Native Prefix-tree GRPO：\`step_credit=progress\`。

先用少量更新步做对照 Smoke / Pilot。若 Step Credit 的信号覆盖、守恒误差和轨迹审计均通过，再决定是否跑正式训练。

\`step_credit=reward\` 可以作为额外诊断模式，不必第一晚就进行完整训练。

第一轮不同时启用 dynamic sampling。先用 Frozen-policy Probe 筛选具有有效组内奖励差异的任务，避免动态补采机制与 Step Credit 同时改变训练样本分布。

---

## 6. GRPO 任务集：围绕“决策点”设计，不是简单拉长任务

### 6.1 不复用旧的最终训练切分

当前 \`data/splits/qwen35_grpo_train.json\` 的来源模型是：

\`outputs/qwen35_sft_full_e2/final\`

它来自旧版 SFT Adapter 的 Frozen-policy Probe，不是 2026-11-21 的 Native SFT Adapter。

因此：

- 不得直接把它当作本轮最终 GRPO split。
- 可以检查其中的任务定义，作为候选池参考。
- 必须用当前 Native SFT Adapter 重新运行 Frozen-policy Probe，并重新生成 model-specific split。
- Manifest 的 \`source.model\` 必须指向本次 Native SFT Adapter，\`policy_frozen=true\`、\`learning_rate=0\`。

### 6.2 候选任务来源

- 使用 \`v4_86\` 的 Train 父任务及其允许的参数化派生任务。
- 可从 \`EXPANDED_SPECS\` 中选择训练任务。
- 不允许使用 Dev、IID Test、OOD Test 的任务 ID、请求原文、精确参数组合或标准答案。
- 新增组合任务必须记录 Train 父任务来源、Snapshot 依赖和步骤契约。
- 所有 Snapshot 键在加入候选池时先检查，缺键任务不能进入正式 Probe。

### 6.3 优先任务类型

建议首轮候选任务包含：

| 类型 | 例子 | 主要决策 |
|---|---|---|
| 多步骤论文处理 | 搜索 → 下载指定论文 → 查缓存 → 翻译 | 根据每轮 Observation 继续 |
| 条件下载 | 若 PDF 未就绪才下载，之后查缓存 | 根据 Setup / Observation 选择是否调用 |
| 条件翻译 | 用户要求翻译才提交；不要求时结束 | 避免额外操作 |
| 多论文操作 | 对指定的第 1、2 篇分别下载或查缓存 | 完成全部子目标 |
| 论文定位 | 搜索后按标题片段 / ID 选论文 | 正确引用 |
| 异步状态 | 翻译返回 PENDING 后判断是否有独立后续任务 | 不把 PENDING 当 READY |
| 阻塞拒绝 | 无 Session、无效引用、不支持能力 | 零调用并说明原因 |

任务长度优先选择 3–5 次真实决策的任务（工具调用加最终终止），但保留少量 1–2 步任务作为参数和终止控制组。不要为凑长度添加用户没有要求的操作。

### 6.4 条件任务如何构造

条件决策应该由同一类自然语言目标和不同的 Session / Setup 状态构成配对任务。

例如：

- Session 中 PDF 已经 READY：正确行为是直接查缓存或结束，不重复下载。
- Session 中 PDF 尚未 READY：正确行为是下载，再按用户要求查缓存。
- 用户明确要求翻译：提交翻译。
- 用户没有要求翻译：不得擅自增加翻译。

这类配对任务能测试模型是否根据状态作出决策，而不是记住固定工具链。

### 6.5 Frozen-policy Probe

建议首轮：

- 每个候选任务采样 4 generations / group。
- 每个任务先运行多个独立 groups，统计最终 Reward 均值、标准差和有效组比例。
- 同时记录五维 Reward 分量、错误类型、轨迹长度、正确前缀长度、首次错误位置。
- 选择存在足够 informative groups 的任务。
- 全成功任务可保留为 ceiling control，不强行纳入训练。
- 所有生成都失败且没有进度差异的任务暂不纳入正式训练，先检查任务难度或奖励设计。
- 若最终 Reward 零方差但严格前缀进度存在差异，标记为 Step Credit progress 候选。

任务选择规则必须在看 Dev / IID / OOD 结果前确定。

---

## 7. 实验执行顺序

### Phase 1：Native GRPO 实现

Codex 负责：

1. 新增 Native Prompt Dataset Builder。
2. 新增 Native Multi-turn Rollout。
3. 复用 Native Agent 的 Prompt、Tool Schema 和 Qwen Chat Template。
4. 每条轨迹独立 Mock 环境。
5. 正确回填 Tool Message 和 tool_call_id。
6. 保存 Assistant spans、tool_mask、原始 messages 和统一 history。
7. 修复 Native messages_to_trajectory 的错误 JSON 和 tool_call_id 配对。
8. 把 Native Rollout 接入 \`train_grpo.py\`，旧 ReAct 路径保留，不覆盖。
9. 修复并测试 Reward 对 Native history 的适配。
10. 修复 Prefix-tree 对有效参数和 Native turn spans 的适配。

验收：相同任务的 Native Benchmark 与 Native GRPO Rollout 在相同输入和环境下，工具调用、Observation、终止语义一致。

### Phase 2：单元测试和 CPU / 小模型协议测试

至少测试：

- 一个工具调用后 FINISH。
- 两个连续工具调用。
- 三个及以上工具调用。
- 多对象工具调用。
- 普通 Final Answer。
- PENDING 正常终止。
- PENDING 后允许的独立后续操作。
- 缺少 Session 的零调用拒绝。
- 非法 ref 的零调用拒绝。
- Unsupported capability 的零调用拒绝。
- malformed tool arguments。
- Schema-invalid arguments。
- tool_call_id 错配、重复和缺失。
- 工具执行失败。
- Snapshot coverage miss。
- 达到 max_turns。
- 达到 completion token 上限。
- 同一任务四条 Rollout 的环境状态互相隔离。
- Assistant spans 与 TRL tool_mask 完全一致。
- Reward history 与原始 Native messages 可相互核验。

Step Credit 测试必须检查：

- 同一有效动作的 canonical key 一致。
- 默认参数省略 / 显式填写等价。
- 不同有效参数不能错误合并。
- 同一 ref 的等价引用能按真实 paper_id 规范化。
- Reward Credit 守恒。
- Progress Credit 只在最终 Reward 平坦且进度存在差异时启用。
- 截断组不使用伪造的 Step Credit。
- 单进程和 G=4 约束明确生效。

### Phase 3：Frozen-policy Probe 与 Reward 审计

使用当前 Native SFT Adapter，\`learning_rate=0\`。

输出：

- 每个任务的 4-generation 组内 Reward。
- 五个 Reward 分量及 safety gate 触发原因。
- Reward std / zero-variance group fraction。
- Prefix-tree reward / progress / none 的组别分布。
- 每条轨迹的完整 Native Messages。
- 每条轨迹的统一 Reward History。
- 每步标准动作匹配结果和 first error。
- 被排除任务和原因。

**用户需要审核一批代表性轨迹和 Reward 排名后，才能启动正式训练。**

### Phase 4：Smoke 与 Pilot

先做 GPU Smoke：

- 加载当前 Qwen3.5-2B-Base。
- 加载 2026-11-21 Native SFT Adapter。
- 验证 LoRA 参数可训练、Reference Adapter 正确冻结。
- 每个 Prompt 4 条 generations。
- 至少覆盖 2 个多步骤任务和 1 个 blocked 任务。
- 完整运行 Native Tool Calling、Mock Observation 回填和 Reward。
- 验证 GRPO loss、KL、梯度、LoRA 指纹变化。
- 验证训练日志、checkpoint 和 rollout traces。

Smoke 通过后再跑短 Pilot。先比较 \`step_credit=off\` 和 \`step_credit=progress\`，不同时启用 dynamic sampling。

### Phase 5：正式 GRPO

只有在用户审核任务和 Reward 后启动。

第一轮尽量固定：

- 同一 Base 和 Native SFT Adapter。
- 同一 Prompt / Tools Schema / Snapshot。
- \`num_generations=4\`。
- 同一 GRPO Train split。
- 同一随机种子。
- 同一 Rollout turn / token budget。
- 同一 KL 系数、学习率和 optimizer。
- 先不改五维 Reward 权重。

训练中保存：

- 标量 Reward 和五维分量。
- reward std、zero variance fraction。
- Prefix-tree reward / progress / none fraction。
- Step Credit coverage / accuracy / conservation error。
- tool error、parse error、false finish、blocked refusal。
- rollout traces。
- KL、policy loss、grad norm、learning rate。
- 每个 checkpoint 的 Adapter 和训练状态。

Dev 可作为 Canary / Checkpoint 选择依据。IID / OOD 不参与任务筛选、Reward 调参和训练中模型选择。

### Phase 6：最终评测

对选定的最终 Adapter 与 SFT Adapter 使用完全相同的 Native Benchmark：

- Dev / IID / OOD。
- 同一 Snapshot。
- 同一 Prompt / Tools Schema。
- 同一解码配置。
- 每任务相同试次数和种子。

报告：

- Strict Success。
- Tool Sequence Accuracy。
- Effective Argument Accuracy。
- Reference Accuracy。
- Multi-step Completion。
- Conditional Decision Accuracy。
- Cache-after-download Compliance。
- Refusal / Blocked Zero-call Accuracy。
- PENDING / READY Consistency。
- Repeated Call Rate。
- Tool Execution Failure。
- Final Answer Truncation / Repetition。
- Token / Tool Call / Wall-clock Cost。

同时保留 SFT → GRPO 的逐任务配对结果。不能只报告平均 Reward 上升。

---

## 8. 哪些由用户负责，哪些由 Codex 负责

### 8.1 用户负责的设计与审核

用户不需要自己改底层 Rollout 代码，但必须负责决定“什么行为才是正确行为”。

1. 确认 GRPO 的主要目标是状态条件决策、多步骤完整执行和组合泛化，而不是继续追求格式解析率。
2. 审核任务类型和代表性自然语言请求，确认没有把不必要的操作写进标准动作。
3. 决定哪些子目标有顺序依赖、哪些可以独立执行。
4. 审核 PENDING 后哪些后续操作可以执行，哪些必须等待。
5. 审核 blocked 任务的正确拒绝条件。
6. 审核 Reward 排名：正确轨迹必须比漏步骤、错参数、错论文、提前 FINISH 和无关调用的轨迹得分更高。
7. 审核 Frozen-policy Probe 后的任务池和组内方差报告。
8. 批准正式训练的 Reward 模式、任务池和资源预算。

### 8.2 Codex 可以直接实施的工程工作

1. Native GRPO Prompt Builder。
2. Native Multi-turn Rollout。
3. Qwen Chat Template / 工具 Schema 注入。
4. Tool Message 回填和 tool_call_id 配对。
5. Assistant Token spans / tool_mask。
6. Native history adapter。
7. Reward / metrics / Step Credit 适配。
8. effective arguments 规范化。
9. 测试、Frozen-policy Probe、审计脚本。
10. 训练入口、日志、checkpoint、Benchmark 对照。

Codex 不得擅自决定：

- 把哪个 Dev / IID / OOD 任务加入训练。
- 把不必要的工具调用当成正确步骤。
- 修改 Reward 权重后直接启动正式训练。
- 因为某条任务没有奖励方差就伪造 Observation 或修改标准答案。
- 让模型对 Snapshot miss 承担本不属于模型的错误惩罚。

---

## 9. 本轮开始前必须处理的两个配置问题

### 9.1 当前 SFT Adapter 与旧 GRPO Split

\`scripts/run_qwen35_grpo.sh\` 的默认 Adapter 路径仍是旧版：

\`outputs/qwen35_sft_full_e2/final\`

必须显式指向本次 Native SFT 的最终 Adapter，并在 Manifest 中校验：

- Base model SHA256。
- Adapter SHA256。
- Native SFT data Manifest SHA256。
- Prompt / Tools Schema SHA256。
- Snapshot SHA256。
- 当前 GRPO split 的 source.model 与实际 Adapter 一致。

旧 \`qwen35_grpo_train.json\` 不得直接作为最终 split。

### 9.2 GRPO 依赖环境

- \`requirements-qwen35.txt\` 锁定 TRL 1.17.1。
- \`StepCreditGRPOTrainer\` 只接受 TRL 1.17.1。
- Native SFT 训练 Manifest 使用 TRL 0.29.1。

因此应单独建立或检查 Native GRPO 虚拟环境，按 \`requirements-qwen35.txt\` 安装和核对依赖。不要在未检查版本的情况下复用 SFT venv。

---

## 10. 今晚的最小执行计划

按下面顺序推进，确保今晚至少能启动 Native Rollout 和 Reward Probe，而不是直接冒险跑完整训练。

1. Codex 从本分支读取本文，完成代码审计并确认 Native GRPO 的实际缺口。
2. 实现 Native Prompt / Rollout / Tool Observation 回填，先不训练。
3. 修复 Reward history 转换、Schema 错误和 Step Credit 的 effective arguments。
4. 跑 CPU / 小模型回归测试。
5. 在服务器上完成 TRL 1.17.1 环境检查。
6. 用当前 Native SFT Adapter 做 4-generation Frozen-policy Probe。
7. 用户审核代表性轨迹、Reward 分量、组内方差和任务池。
8. 审核通过后做 2–5 optimizer steps 的 GRPO Smoke，再决定是否跑短 Pilot。

今晚不要求一次性跑完正式 GRPO。**Native Rollout 正确、Reward 排名可信、组内有训练信号，是启动正式训练的三个前置条件。**

---

## 11. Codex 交付物

### 代码

- Native GRPO prompt builder。
- Native multi-turn rollout。
- Native messages → reward history adapter。
- Native Reward / metrics tests。
- Native Prefix-tree Step Credit tests。
- Frozen-policy Probe / split builder。
- GRPO launch config 支持当前 Native SFT Adapter。
- 保留原 ReAct GRPO 路径，不覆盖旧实现。

### 报告

- \`reports/native_grpo_design_audit.md\`
- \`reports/native_grpo_rollout_audit.md\`
- \`reports/native_grpo_reward_audit.md\`
- \`reports/native_grpo_frozen_probe.md\`
- \`reports/native_grpo_smoke.md\`
- \`data/splits/qwen35_native_grpo_train.json\` 及其 Manifest

### 正式训练前的停止条件

以下任一项不通过，必须停止，不得开始正式 GRPO：

- Native 工具调用没有被正确解析。
- Tool Message 没有按 tool_call_id 回填。
- Observation 或 Setup 混入模型生成目标。
- Tool Observation 参与 Policy Loss。
- Reward 对有效默认参数或论文引用评分不一致。
- Snapshot miss 被当成模型工具失败。
- Prefix-tree 信用不守恒。
- 组内 Reward / Progress 全部无差异。
- Train / Dev / IID / OOD 存在任务或标准答案泄漏。
- 当前 Adapter、Base、Snapshot、Prompt、Tools Schema 的来源无法核验。

**本轮总原则：先把 Native Rollout 和 Reward 做正确，再讨论权重；先证明当前 SFT 策略能在任务组内产生可区分的轨迹，再训练；先让模型学会根据状态完成正确任务，再追求更高的 Reward 数字。**
