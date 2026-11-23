# Qwen3.5-2B Native GRPO 执行方案（Prefix-tree Step Credit）

> **本版为执行规格，设计决策已锁定。** Codex 按本文自主实现、测试、构造任务、审计 Reward、运行 Probe 和训练。用户只负责在 Codex 通知时启动 GPU 实例，以及实验完成后关闭实例；除非遇到本文未覆盖的实质性设计歧义，否则不要反复要求用户确认。
>
> 工作分支：`feat/qwen35-native-grpo`  
> 基线分支：`feat/qwen35-base-native-sft`  
> 基线提交：`d9ead3c3763800bff6da867ba16fed270810d31c`  
> 策略起点：2026-11-21 完成的 Qwen3.5-2B-Base Native SFT Adapter  
> 训练方法：Native Tool Calling GRPO + Prefix-tree Step Credit（progress 模式）  
> 本轮原则：不重写旧任务体系，不重新做普通 GRPO 对照，不调 Reward 权重；先通过 CPU/静态测试，再请求一次 GPU，完成 Probe、训练和评测。

---

## 1. 研究目标：GRPO 要让 SFT 模型进一步学会什么

本轮不再把 Tool Call 格式正确率作为主要优化目标。Native SFT 已经把格式解析成功率从 95.5% 提升到 99.2%，Schema 合规率为 100%。GRPO 应该集中解决 SFT 后仍存在的策略问题。

### P0：根据状态做多步骤决策

- 读懂每轮真实 Tool Observation。
- 根据当前状态判断下一步，不提前 FINISH。
- 完成用户明确要求的全部步骤。
- 不把固定工具序列当作无条件模板。

### P0：条件执行和用户约束

- 区分“只下载”“下载后查缓存”“只查缓存”“下载后翻译”等目标。
- 根据缓存状态决定是否需要下载。
- 用户没有要求的操作不能擅自增加。
- 用户明确要求的操作不能因工具默认值或模型习惯而省略。
- 正确处理论文引用、参数默认值、文件副作用和操作顺序。

### P1：组合泛化和任务边界

- 对多个论文对象逐一完成操作。
- 将已经学会的搜索、下载、缓存、翻译、读取能力组合到新任务。
- 缺少会话上下文、引用无效、论文不存在或工具能力不支持时，不调用无关工具，并说明具体阻塞原因。
- 正确区分 PENDING 和 READY。

**不把“调用更多工具”或“任务更长”本身作为目标。** 每一个标准步骤都必须由用户请求和环境状态决定。

---

## 2. 现有任务来源：保留任务体系，但不能直接复用旧 Split

已核对仓库中的任务血缘：

| 文件 / 版本 | 任务数量 | 说明 |
|---|---:|---|
| `data/splits/v3_81.json` Train | 51 | 旧的 Train 候选来源 |
| `data/splits/v4_86.json` Train | 55 | 当前 Native SFT 使用的 Train 父任务来源 |
| `qwen35_probe_candidates_d970_quality.json` Train | 47 | 从旧 Train 池筛出的 Frozen Probe 候选 |
| 旧 `qwen35_grpo_train.json` | 39 | 旧 SFT 模型 Probe 后选出的 GRPO 任务 |

关键事实：

- 旧 Probe 的 47 个候选任务 ID 全部包含在本次 Native SFT 的 55 个 `train_parent_ids` 中。
- 旧 GRPO Split 的 39 个任务 ID 也全部包含在本次 SFT 的 55 个父任务 ID 中。
- 旧 `qwen35_grpo_train.json` 的来源模型是 `outputs/qwen35_sft_full_e2/final`，不是本次 Native SFT Adapter。

因此：

1. **保留** `tasks_expanded.py`、`TaskSpec`、`family()`、`v4_86` Train 的任务定义和参数化机制。
2. **不直接使用**旧的 39-task Split，也不把旧 Probe 的选择结果当成本次结论。
3. 旧的 47 个候选任务只作为任务模板和候选来源；本次必须基于最新 Native SFT Adapter 重新 Probe。
4. 由于这些任务大多已经作为 SFT 父任务出现，不能简单把原任务原样重复训练。优先改变参数、论文引用、初始 Session 状态和工具组合，构造同一 Train 任务家族下的新决策情境。
5. Dev、IID Test、OOD Test 的任务 ID、自然语言原文、精确参数组合和标准答案全部保持隔离。尤其不能把 SFT 报告中失败的 OOD 任务原样加入 GRPO Train。

---

## 3. 任务改造方向：从 SFT 剩余错误出发

SFT 的主要证据：

- Strict Success：46/93 → 69/93；Dev 持平，IID 明显提升，OOD 从 9/12 回退到 7/12。
- 工具序列完全匹配：54.8% → 86.0%。
- 生效参数平均得分：62.5% → 84.1%。
- 多步骤完成：59.0% → 82.1%。
- 超出要求的重复调用：12/93 → 3/93。
- PENDING 提交后正常结束：21/21。
- 阻塞任务零调用且正确解释原因：0/6。
- SFT 仍有 9 次参数不合格、3 次论文引用不合格、7 次未完成必需步骤便结束。
- 训练集中 103 条下载监督中，只有 3 条下一步紧接缓存查询。
- 最终 OOD 的 `multi_cv5_dl1_cache_tr` 三次都跳过缓存检查，直接翻译。

据此构造以下任务家族。

### A. 下载 / 缓存 / 翻译的条件任务（最高优先级）

以现有 `state_dl_then_cache`、`state_cache_before_dl`、`constraint_download_only`、`constraint_cache_only`、`chain_cv5_cache_dl_tr_cache` 等 TaskSpec 为模板，构造新的 Train 变体。

必须覆盖以下互相区分的要求：

1. 只下载：下载成功后 FINISH，不查缓存、不翻译。
2. 下载后查缓存：下载后必须调用缓存工具，再 FINISH。
3. 只查缓存：不下载、不翻译。
4. 先查缓存、缺失才下载：根据真实初始状态决定是否下载。
5. 用户要求翻译：只有在满足工具前置条件时提交翻译。
6. 用户未要求翻译：不得擅自提交翻译。

对条件任务使用不同的 TaskSpec / Setup 表示不同的初始环境状态和标准步骤。不要在 Reward 中硬编码一个任务同时存在两套不明确的标准答案。

### B. 搜索范围和工具副作用（高优先级）

SFT 后仍有“全 CS 范围”与参数错误的残留问题。构造 Train 变体覆盖：

- 全 CS 范围 `aspect="*"`，不要误选成某个子领域。
- 用户要求不保存文件时，正确设置 `save_to_file=false`。
- 用户要求保存时，保留对应默认或显式行为。
- 搜索时间范围和篇数要与用户请求一致。

不要复制 Dev 的 `constraint_search_no_file` 原文和精确参数；使用不同的 Train 参数组合，并让任务描述与 TaskSpec 标准参数由同一份参数生成。

### C. 多对象和多步骤组合（高优先级）

使用 Train 父任务派生 2–4 个工具调用的任务，例如：

- 搜索后分别操作两篇不同论文。
- 下载指定论文后查缓存，再按要求翻译。
- 搜索、下载、读取论文内容或摘要。
- 同一任务中需要保留不同论文引用，不能把 ref 混淆。

只在用户明确提出时才加入对应操作。不要为了拉长轨迹添加额外步骤。

### D. 引用、参数和状态指代（中优先级）

使用新的 Train 参数组合覆盖：

- 序号引用、完整 arXiv ID、标题子串、最近操作论文（null）。
- 保留版本号。
- 多对象引用不串篇。
- 缺少会话状态时不猜测最近论文。

### E. 阻塞和拒绝（中优先级）

保留现有 `infeasible_no_session`、`infeasible_unknown_id`、`infeasible_zero_index` 等任务定义作为模板，构造不同表达和不同合法 Train 条件的变体，覆盖：

- 缺少会话上下文。
- 无效或越界引用。
- 快照中不存在的论文 ID。
- 工具不支持的操作。

正确行为是零工具调用，并在最终回答中说明具体原因。不能只输出泛化的“任务完成”或无理由拒绝。

### 不作为本轮主目标的能力

- PENDING 提交后正常结束已达到 21/21，保留少量控制任务即可，不要过度采样。
- 重复调用已经明显减少，作为 Process / Efficiency 监控，不作为主要任务来源。
- 摘要、论文正文和图表分析的自然语言质量目前没有可靠的通用 Reward。不要声称本轮 GRPO 能优化这些内容质量；可以保留少量工具选择控制任务，但不把自由文本质量作为主要训练目标。
- 不要把模型已经稳定完成的简单单步检索任务大量放入 Train。

### 任务规模

- 首轮 Frozen Probe 候选池控制在 **16–20 个任务**。
- 至少覆盖 A、B、C、E 四类；D 可穿插在 A/C 中。
- 其中优先使用旧 Train TaskSpec 作为模板，再新增约 6–10 个有明确来源的新变体。
- 最终 GRPO Train Split 由 Probe 结果决定，不要求 16–20 个候选全部入选。
- 每个候选任务先跑 4 个独立 group，每组 4 generations（共 16 条轨迹 / task）。若某任务结果临界且值得保留，可追加至 8 groups；不要默认对所有任务跑 8 groups。

所有新增任务必须：
- 记录 Train 父任务 ID 和派生来源；
- 自然语言描述、Setup、标准工具序列和参数来自同一份 TaskSpec；
- Setup 只初始化 Mock 环境，不算模型动作、不计入 Reward；
- Snapshot 中涉及的论文、PDF、缓存和译文必须确实存在；
- 与 Dev / IID / OOD 的任务 ID、请求文本、Setup 和标准步骤签名做泄漏审计。

---

## 4. Frozen-policy Probe：使用最新 SFT Adapter 重新筛选

必须使用 2026-11-21 Native SFT Adapter，而不是旧的 `outputs/qwen35_sft_full_e2/final`。

已知产物身份（以实际本地文件和 manifest 再核验一次）：

- Adapter SHA256：`789af8cf28960892a1fde034d32f9b8ddf230e8450552be70d15024c964ed6c8`
- Native SFT Train Manifest SHA256：`34196a2639203ca290a43c1012daa1f82f4621a4f8beb2203c597713d013bf70`
- Snapshot SHA256：`d97035db326ca11935e882fd2aca7fe932e9e76cf1bc8a21078be7bec1ae8794`

Probe 设置：

- policy frozen，learning rate = 0。
- `num_generations=4`。
- 首轮每个任务 4 groups；相同 Prompt 的四条轨迹组成一组。
- 固定 seed，并保存每条 Native Rollout。
- 使用正式训练同一份 System Prompt、Tools Schema、Qwen Chat Template 和 Snapshot。
- 记录最终 Reward、五个分量、Safety Gate、Reward std、严格进度 `phi`、Prefix-tree 模式、首次错误位置和轨迹长度。

### 任务筛选规则

因为本轮使用 `step_credit=progress`，不能只按最终 Reward 方差筛选：

- **Reward informative**：组内最终 Reward 有差异，Prefix-tree 模块 A 能提供 Reward Credit。
- **Progress informative**：最终 Reward 零方差，但严格前缀进度 `phi` 有差异，模块 B 能提供 Progress Credit。
- **Ceiling control**：所有采样都完成同一正确轨迹，Reward 和 Progress 都无差异；保留作诊断，不放入正式训练。
- **No-signal**：Reward 和 Progress 都无差异，或任务本身不具备可区分的有效行为；排除。
- **Environment-invalid**：Snapshot 缺键、Setup 失败、工具实现异常；修复环境或排除，不能记为模型失败。

正式候选任务至少应有 2/4 个 group 能提供 Reward 或 Progress 信号。若 4 groups 结果临界，可追加至 8 groups 再判断。

输出一个 model-specific split，Manifest 必须记录当前 Adapter SHA、Base Model、Prompt / Tools Schema SHA、Snapshot SHA、seed、groups_per_task、Reward 配置和选择理由。不能沿用旧 split 的 source metadata。

---

## 5. Reward 和 Prefix-tree Step Credit：本轮直接启用，不做普通 GRPO 对照

### 5.1 固定五维 Reward，不调整权重

首轮保持现有 RewardCalculator 权重：

| Reward | 权重 |
|---|---:|
| Format | 1 |
| Tool | 3 |
| Argument | 2 |
| Process | 1 |
| Outcome | 3 |

沿用 `reward_curriculum_steps=0`，从训练开始使用完整权重。

`result_quality` 和 `efficiency` 当前是诊断分量，并参与安全门控 / 奖励上限，不是五维加权平均中的额外两个独立权重。保持现有定义，先修复 Native 适配问题，不进行主观调权。

### 5.2 本轮直接使用 Prefix-tree Step Credit

正式配置固定：

- `step_credit=progress`
- `step_credit_weight=0.5`
- `step_credit_beta=1.0`
- `num_generations=4`
- 单进程训练（Step Credit v1 的 Collector 是进程内状态）
- `dynamic_sampling=false`

`progress` 模式的准确含义：

1. 若组内最终 Reward 有方差，优先使用前缀树模块 A：根据相同动作前缀后的最终 Reward 差异分配步骤信用。
2. 若最终 Reward 零方差、但严格任务进度有方差，使用模块 B：按标准工具、有效参数、论文引用和 FINISH 的正确顺序进度分配信用。
3. 两者都没有差异时，不伪造训练信号，保留标准 GRPO 的零优势并将该组记为 no-signal。

当前 `step_credit_weight=0.5` 表示 Assistant token 的 Advantage 为 50% 轨迹级 Advantage + 50% 步骤级 Advantage。它不是新的 Reward 分量权重。

本轮不额外跑 scalar GRPO 对照，以节省时间。报告中必须明确：本轮能评估 Native SFT → Native GRPO + Prefix-tree 的总体增益，但不能单独声称 Prefix-tree 相比普通 GRPO 的因果增益。后续有资源再补小规模消融。

### 5.3 Native Reward 必须正确处理的语义

统一把 Native Messages 转换成现有 Reward 可理解的内部 history，但保留原始消息以便审计：

- Assistant Tool Call → 一条 action step。
- 对应的 Tool Message → 该 action 的真实 Observation。
- Assistant Final Answer → FINISH step，最终回答文本保存在终止语义字段。
- Setup receipts 只用于审计环境状态，不加入模型动作 history。
- JSON 参数解析失败必须标记 parse failure，不能静默改成空字典。
- Tool Message 必须按 `tool_call_id` 精确配对，不能仅按 FIFO 顺序猜测。
- 每个 Assistant turn 最多一个 Tool Call。
- Tool Observation 不得伪造成模型生成内容。

Reward 必须通过下列对照：

1. 下载后查缓存：漏掉缓存步骤的轨迹必须低于完整轨迹。
2. 只下载：擅自查缓存或翻译必须被扣分。
3. 只查缓存：擅自下载必须被扣分。
4. 默认参数省略和显式传默认值，按工具的有效参数语义判等。
5. `save_to_file=false` 与工具默认保存行为冲突时，按用户任务要求评分。
6. ref 按实际 Observation 解析出的 paper_id 判定，不能只看传入数字。
7. PENDING 不能当 READY；异步提交成功后不能声称译文已经可读。
8. blocked 任务必须正确说明具体阻塞原因；无工具调用但泛泛说“完成”不能拿满分。
9. 工具执行错误、Schema 错误、解析失败不能被其他奖励分量补偿成成功。
10. Snapshot miss / Setup failure 是环境问题，不能惩罚策略模型。

---

## 6. Codex 必须实现的 Native GRPO 工程工作

### 6.1 保留旧 ReAct 实现，新增 Native 路径

当前 `train_grpo.py` 仍使用 `build_prompt_dataset()`、`get_react_prompt()` 和 `parse_react_action()`。已有 `messages_to_trajectory()` 只解决部分转换，并没有完成 Native Rollout。

请新增 Native GRPO 路径，保留旧 ReAct 路径，不要直接覆盖旧逻辑。

### 6.2 Native Rollout 具体要求

每个 prompt 采样四条独立轨迹，每条轨迹有独立 MockArxivEnv / Session：

1. 使用 Native System Prompt、10 个 Tool Schema 和 Qwen3.5 官方 Chat Template。
2. 每轮生成一个 Assistant Message。
3. 若包含 Tool Call，解析并校验工具名和参数，再执行 MockArxivEnv。
4. 将真实工具结果以 `role=tool` 消息回填，按 `tool_call_id` 关联。
5. 下一轮模型读取完整历史继续生成。
6. 若产生普通 Final Answer，则记为 FINISH。
7. 模型不能生成 Tool Observation；在生成边界结束当前 Assistant turn。
8. 不合法 Tool Call 不执行，保留错误轨迹。
9. 达到 max_turns / token budget 时标记 truncated，不得自动补成成功 FINISH。
10. 记录原始 Native Messages、内部 history、tool_call_id、tool mask、Assistant token spans、终止原因、clipped 状态和 setup receipts。

### 6.3 Token Loss Mask 与 Step Credit 对齐

- Policy Loss 只覆盖模型实际生成的 Assistant token。
- Tool Observation、Setup 和 Padding 不参与 Policy Loss。
- 每个 Assistant turn 的 `turn_token_spans=[start,end)` 必须准确。
- Tool Observation 对应的 `tool_mask` 必须与 Assistant spans 完全互补且逐 token 一致。
- Assistant Content 和 Tool Call 同属当前 Assistant Message，第一版使用同一个 turn-level advantage。
- Final Answer 也作为一个 Assistant turn，映射为 FINISH step。
- 若任一轨迹被截断或达到 max_turns，按现有实现将整个组标记为 truncated，禁用该组 Step Credit，不能制造伪造 FINISH 叶子。

### 6.4 修复 Prefix-tree 的参数一致性

现有 `step_credit.py` 的 `canonical_action_key()` 和 `_tool_step_matches()` 主要基于模型原始参数字典；必须与 Benchmark / Reward 的有效参数判定统一：

- 使用 `registry.effective_arguments(tool_name, args)` 规范化默认值、类型和合法参数。
- 默认值省略与显式传默认值必须得到相同 canonical key。
- 不同生效参数不能错误合并。
- ref 在可从 Observation 解析出 paper_id 时，按实际论文身份规范化。
- `label_trajectory()` 必须复用与 Benchmark 一致的参数和引用比较语义。
- 保留严格有序前缀匹配：第一次偏离之后的动作不再获得正确前缀进度。
- blocked 任务的 FINISH 必须通过具体拒绝语义检查。

### 6.5 运行约束

- `requirements-qwen35.txt` 锁定 `trl==1.17.1`；`StepCreditGRPOTrainer` 也要求该版本。
- Native SFT 训练记录使用 TRL 0.29.1。必须使用独立 GRPO 环境，不要直接复用 SFT 环境。
- Step Credit v1 要求单进程，且当前 Qwen3.5 配置要求 G=4。
- 不启用 Dynamic Sampling，避免同时改变采样分布和 Step Credit 分组。
- `max_turns` 至少覆盖任务最长标准工具步骤数 + 最终回答；由任务清单自动计算。
- `max_completion_length` 通过实际 Qwen tokenizer 预检，必须保证最长标准 Native 轨迹可以生成，不得截断标准答案。

---

## 7. 自动执行流程：用户只负责 GPU 开关

Codex 按顺序自主执行，不要在每个阶段都停下来要求用户批准。

### Phase A：无需 GPU 的工作

1. 切换本分支，读取本文。
2. 审计当前 Native Agent、Tool Schema、Mock 环境、Reward 和 Step Credit。
3. 实现 Native Prompt / Rollout / history conversion / token spans / tool mask。
4. 修复参数 JSON 错误、tool_call_id 配对和有效参数 canonicalization。
5. 构造 16–20 个候选任务及必要的 Train 派生变体。
6. 完成任务血缘、Snapshot coverage、Holdout leakage 和 TaskSpec 标准答案审计。
7. 完成 CPU 单元测试、静态检查、py_compile、依赖版本检查和 dry-run。
8. 生成 GPU 启动前报告，列出准确的模型路径、Adapter SHA、Snapshot SHA、环境、预期显存、Probe 规模和后续训练配置。

**Phase A 通过后，Codex 才向用户请求启动 GPU。** 请求必须明确告诉用户需要哪种 GPU、预计连续运行多久、需要提供什么访问方式。不要提前占用 GPU。

### Phase B：GPU 启动后一次性完成

GPU 启动后，Codex 自主完成：

1. 检查 GPU、CUDA、Torch、TRL 1.17.1 和模型文件。
2. 使用最新 Native SFT Adapter 运行 Frozen-policy Probe。
3. 计算 Reward / Progress informative groups，生成新的 model-specific Train Split。
4. 审计代表性轨迹、Reward 排名、Step Credit 模式和 no-signal 比例。
5. 若出现 Reward 逻辑错误、Snapshot 问题或大部分任务没有任何信号，先停止训练并修复 / 报告，不得伪造标签或强行开跑。
6. 若 Probe 通过，运行 2–5 optimizer steps 的 Native GRPO Smoke，检查 loss、KL、grad norm、LoRA 参数指纹、rollout traces 和 Step Credit 守恒。
7. Smoke 通过后，直接运行第一轮 **100 optimizer steps** 的 Native GRPO + Prefix-tree Step Credit。
8. 每 25 steps 保存 checkpoint 和训练状态。
9. 完成训练后在 Dev 上评估候选 checkpoint，选择 Dev 表现较好的 checkpoint；不要用 IID/OOD 选择模型。
10. 对选定 checkpoint 运行同协议 Dev / IID / OOD Benchmark，与 Native SFT Adapter 做逐任务配对比较。
11. 保存所有日志、轨迹、Manifest、Adapter SHA 和结果报告。
12. 向用户报告 GPU 工作已完成，并明确通知可以关闭实例。

若 GPU 启动后发现资源不足，先尝试降低 micro-batch / 调整 gradient accumulation 或减少 Probe groups，不得擅自改变 G=4、Step Credit 算法或 Reward 权重。若仍无法运行，给出具体资源需求后停止。

### 第一轮 GRPO 固定参数

除非 GPU 预检证明必须调整显存相关项，其他配置保持：

- optimizer steps：100
- learning rate：`5e-6`
- KL beta：`0.04`
- generations per prompt：4
- per-device batch：4
- gradient accumulation：1
- Reward curriculum steps：0
- Reward weights：1 / 3 / 2 / 1 / 3
- Step Credit：`progress`
- Step Credit weight：0.5
- Step Credit beta：1.0
- Dynamic Sampling：关闭
- seed：42
- 单进程
- checkpoint：每 25 steps
- max_turns 和 max_completion_length：根据候选任务和 Qwen tokenizer 自动预检

100 steps 是第一轮受控训练，不代表必须训练更多 Epoch。若 Dev 没有改善或出现明显退化，保留最佳 checkpoint 和 SFT Adapter，不自动续训。

---

## 8. 通过标准和停止条件

以下任一项不通过，禁止启动正式 100-step 训练：

- Native Tool Call 解析不完整或参数错误被静默修复。
- Tool Message 没有按 tool_call_id 配对。
- Setup 或 Tool Observation 混入模型生成目标 / Policy Loss。
- Assistant spans 与 TRL tool_mask 不一致。
- 有效默认参数的 Reward 与 Step Credit 判定不一致。
- PENDING 被判定为 READY。
- blocked 任务可以通过泛化“任务完成”拿到正确奖励。
- Snapshot / Setup 失败被错误归因于模型。
- Step Credit Reward Credit 守恒检查失败。
- 训练组全都没有 Reward 或 Progress 差异。
- Train / Dev / IID / OOD 发生任务或标准答案泄漏。
- Adapter、Base、Prompt、Tools Schema、Snapshot 或依赖版本无法核验。

训练完成后至少报告：

- Native SFT vs Native GRPO + Prefix-tree 的 Strict Success。
- Dev / IID / OOD 分项结果。
- 多步骤完整执行率。
- 条件决策准确率。
- 缓存检查步骤完成率。
- 参数与论文引用准确率。
- Blocked 拒绝正确率。
- PENDING / READY 状态一致性。
- 重复调用率、工具错误率、截断率。
- Reward 五个分量、组内方差。
- Reward / Progress / None / Truncated Step Credit 组比例。
- Step Credit coverage、accuracy、conservation error。
- KL、policy loss、grad norm、learning rate。
- 训练成本、GPU 时长、checkpoint 和 Adapter SHA。

报告结论必须区分：
- GRPO 相比 SFT 是否改善；
- 哪些能力改善、哪些没有改善；
- 本轮使用 Prefix-tree，但没有 scalar GRPO 对照，因此不能把全部增益单独归因于 Prefix-tree。

---

## 9. Codex 执行边界

### Codex 自主负责

- 任务方向落地、Train 变体生成和任务池筛选。
- Native Rollout / Mock 环境接入。
- Reward / Step Credit 代码修改和单元测试。
- Frozen-policy Probe、Split、审计。
- GPU Smoke、100-step 训练、Checkpoint 选择和 Benchmark。
- 更新文档、报告、Manifest，并 push 到当前分支。

### 用户只负责

1. Codex 在 Phase A 全部通过后，按它给出的配置启动 GPU 实例。
2. 提供 Codex 执行所需的服务器访问方式。
3. Codex 完成训练和评测并通知后，关闭 GPU 实例。

不要把任务选择、Reward 权重、Step Credit 模式或训练配置再交给用户逐项确认；这些设计已经在本文确定。

如果遇到本文未覆盖、会改变任务正确性或奖励语义的实质性歧义，Codex 必须暂停该部分并提出一个具体问题。不要为了继续运行而自行猜测。

---

## 10. Codex 开始执行的指令

请切换到 `feat/qwen35-native-grpo`，完整阅读本文，并按 Phase A → Phase B 执行。

这次直接使用 Native Tool Calling + `step_credit=progress`（Reward Prefix-tree 优先，最终 Reward 平坦时使用严格 Progress fallback）。不要先做普通 scalar GRPO 对照，不要改五维 Reward 权重，不要复用旧的 39-task Split，也不要把 SFT 已经解决的任务原样重复塞入训练。

先完成所有无需 GPU 的实现、测试、任务构造和数据审计。全部通过后再向用户请求一次 GPU 启动。GPU 启动后自主完成 Probe、Step Credit 审计、Smoke、100-step 训练、Dev checkpoint 选择和最终 Benchmark。最后 push 全部代码和报告，并通知用户关闭 GPU。

**用户的目标是尽快得到一轮可信的 Native SFT → Native GRPO + Prefix-tree 实验结果；不要把本轮扩展成无止境的框架重构或 Reward 权重搜索。**
