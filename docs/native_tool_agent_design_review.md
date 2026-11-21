# Native Tool Calling：Agent 设计问题审计与修复规格

> 用途：交给 Codex 修复 Native Tool Calling Agent 的设计问题。本文先区分 Harness / 工具契约 / 评分器问题与模型策略问题，再规定修复顺序和验收测试。  
> 基线分支：feat/qwen35-2b-base-tool-calling  
> 实施分支：feat/qwen35-base-native-sft  
> 证据：用户提供的 native_sft_badcase_audit.md、2026-11-21 Base Benchmark 报告，以及当前分支源码。

## 0. 结论

当前 Native Tool Calling 的失败不能全部归因于 Qwen3.5-2B-Base。

Base 的 60 条失败轨迹中，确实有大量模型决策错误；但同时存在工具参数契约表达不充分、默认值语义不一致、评分器把语义等价调用判错、终止规则与任务定义冲突、setup 可见状态不够准确等设计问题。

本轮先修复 Agent 设计，再生成 SFT 数据。不要把设计问题做成模型训练样本，否则模型会被要求学习一套不一致的工具协议。

本轮的原则：

1. System Prompt 只规定通用行为，不逐个解释工具参数。
2. 每个工具的名称、用途、参数意义、合法取值和默认行为写在 Tool Schema / Description 中。
3. SFT 目标采用简短的 Assistant Content + Native Tool Call；Content 是用户可见的状态说明，不是隐藏 CoT。
4. 评分以工具序列和参数的实际语义为准，不把“省略一个有默认值的参数”误判为参数错误。
5. 先修复并重跑 Base 基线，再生成数据和训练，保证前后对比公平。

---

## 1. System Prompt：删除逐参数指令

当前 NATIVE_TOOL_SYSTEM_PROMPT 中以下内容不应继续放在系统提示词：

- “检索的篇数上限填入 max_results”
- “最近多少天填入 days”
- “用户未要求文件路径时省略 output_path”
- “明确给出论文序号、ID 或标题时如何设置 ref”
- 把 translate_arxiv_pdf 的异步细节写成系统提示词中的单工具规则

这些内容属于各工具的参数契约，应放到相应工具的 Schema 和 Description。当前提示词一边让模型“把明确参数约束传给工具”，一边单独规定某些字段应该传或不传，容易让模型把参数理解成需要猜测的特殊规则。

尤其是 output_path：不应该让 System Prompt 教模型“没有路径就省略”。应该在 Schema 中明确其类型和默认值；SFT 的标准调用可以使用类型正确的默认值。不要把空字符串当成所有未指定参数的通用占位符。

### 建议替换为

~~~text
你是一个 arXiv 计算机科学论文研究助手。

根据用户目标和当前会话状态，使用提供的工具完成任务。每轮最多调用一个工具。优先利用当前会话和历史工具结果，不要重复已经成功完成的操作；用户要求多个子任务时，逐项完成后再结束。

需要调用工具时，可以用一句简短、自然的语言说明当前已知状态和下一步动作。说明必须依据用户请求、已有会话状态和真实工具 Observation，不得编造已经发生的操作；不要输出冗长的内部推理或隐藏思考标签。

工具名称、参数含义、参数类型、合法取值和默认行为以对应 Tool Schema 与 Description 为准。只调用提供的工具，不编造工具名、参数名或参数值。

工具返回错误时，不要原样重复相同工具和相同参数；根据错误信息判断能否修正。无法修正时，说明具体阻塞原因并结束。

只有真实工具结果支持时，才能声称操作成功。缺少必要上下文、引用无效或请求超出工具能力时，不调用无关工具，应明确说明原因。

如果工具返回异步任务已受理或 PENDING，遵循该工具的异步契约：不要重复提交，也不要无意义轮询；如果用户明确要求的后续独立步骤仍然可执行，可以完成该步骤，然后如实汇报状态。
~~~

这段 Prompt 只保留通用策略，不列举 aspect、days、max_results、output_path、ref、service 等具体字段。

---

## 2. Tool Schema：建立单一、清楚、一致的参数契约

NativeToolCallingAgent.build_messages 当前把 registry.list_tools() 直接作为 tools 传给模型。因此，模型实际能看到什么参数语义，主要由 Tool Schema 决定，而不是由 ReAct 时代的文本工具手册决定。

当前多个 Schema 的 description 过于简略，只写“计算机科学子领域代码”“翻译服务”“图表问题”等，没有解释模型最容易混淆的行为。

### 2.1 分类搜索与关键词搜索必须明确区分

涉及文件：

- AgenticArxiv/tools/arxiv_tool.py
- AgenticArxiv/tools/tool_registry.py
- AgenticArxiv/agents/native_tool_agent.py

get_recently_submitted_cs_papers 是按 CS 子领域和提交时间检索。aspect 的含义应明确写成：

- “*”表示所有计算机科学子领域。
- AI、CV、RO、CL、LG、CR 等是子领域短代码。
- 用户说 cs.CV 时，语义对应 CV，而不是把字符串 cs.CV 原样拼成参数。
- days 是最近多少天；max_results 是返回数量上限。
- save_to_file 是是否保存文件；output_path 是保存路径。
- output_path 的默认值应明确为 null / None，运行时使用工具自己的默认路径。
- save_to_file 的默认值必须与 Python 函数实际默认值一致。

search_arxiv_papers 是按关键词、标题或作者搜索。它需要 query；不要把 cs.CV、cs.RO 这样的类别标识当作关键词查询。其 days 表示可选的时间过滤，没有时间限制时使用明确定义的默认值。

建议在工具描述中直接写清楚“按子领域筛选用哪个工具，按关键词/标题/作者搜索用哪个工具”，不要依赖模型从函数名猜。

**类别别名处理建议：** 对形如 cs.CV、cs.RO 的标准 arXiv 类别写法，在进入枚举校验前增加确定性的归一化，将 cs. 前缀去掉并转成规范短码。保留原始参数和规范化参数两份日志。SFT 目标统一使用规范短码。不要把任意未知字符串都自动猜成某个类别。

### 2.2 所有论文工具的 ref 语义必须统一

涉及：

- download_arxiv_pdf
- get_paper_cache_status
- translate_arxiv_pdf
- get_paper_content
- summarize_paper
- extract_paper_figures
- analyze_figure
- get_translated_content

当前函数默认值并不一致：

| 工具 | 当前函数签名中的 ref 默认值 |
|---|---|
| download_arxiv_pdf | 1 |
| get_paper_cache_status | None |
| translate_arxiv_pdf | None |
| get_paper_content | 1 |
| summarize_paper | 1 |
| extract_paper_figures | 1 |
| analyze_figure | 1 |
| get_translated_content | 1 |

但系统提示词又说，指代最近操作的论文时可以省略 ref 或传 null。对部分工具省略 ref 实际会指向第 1 篇，而不是最近操作的论文。这是明确的工具契约不一致。

统一约定：

- ref 为整数：当前会话候选列表中的 1-based 序号。
- ref 为字符串：完整 arXiv ID（包括版本后缀）或标题片段。
- ref 为 null：最近一次操作的论文。
- 不要让“省略 ref”在不同工具中产生不同语义。
- 所有论文操作工具使用一致的 ref 默认语义；建议函数默认值统一为 None，并在模型可见 Schema 中声明 default=null。
- 如果用户明确说第 2 篇，传 2；明确给出 ID，原样传完整字符串；指代最近操作的论文，传 null。
- 如果 ref=null 时没有 last_active_paper_id，工具应返回明确的缺少上下文错误，不应擅自退回第 1 篇。

注意：不能只改 Schema，不改 Python 函数默认值和 BaseAgent 的特殊执行路径。三者必须一致。

### 2.3 默认值必须真实、可执行、类型正确

用户希望模型在没有额外约束时使用工具默认值，而不是被 Prompt 指示“某字段不要传”。建议：

1. 每个正常用户可见参数都在 Schema 中写明 description 和 default。
2. Schema default、Python 函数默认值、BaseAgent 特殊分发逻辑使用同一套默认值。
3. 对“无值”有意义的参数，使用类型正确的 null，而不是随便填空字符串。
4. 对用户明确指定的值，必须覆盖默认值。
5. SFT 的规范目标中，正常用户可见参数尽量显式写出有效值或默认值；不要把 session_id 等框架字段混入目标。
6. 评分器要接受“显式传默认值”和“省略后由工具使用同一默认值”这两种语义等价形式。

需要重点对齐的默认项：

| 工具 / 参数 | 当前代码中可见的默认语义 | 要求 |
|---|---|---|
| get_recently_submitted_cs_papers.aspect | * | 描述 * = 所有 CS 类别 |
| get_recently_submitted_cs_papers.days | 7 | Schema 与函数一致 |
| get_recently_submitted_cs_papers.max_results | 50 | Schema 与函数一致 |
| get_recently_submitted_cs_papers.output_path | None | Schema 支持 null，默认 null |
| get_recently_submitted_cs_papers.save_to_file | True | 用户明确不保存时必须传 false |
| search_arxiv_papers.max_results | 10 | Schema 与函数一致 |
| search_arxiv_papers.days | None | 无时间过滤时使用 null |
| download_arxiv_pdf.force | False | Schema 与函数一致 |
| translate_arxiv_pdf.force | False | Schema 与 BaseAgent 一致 |
| translate_arxiv_pdf.service / threads | 当前依赖 settings 默认值 | Schema default 必须从实际 settings 对齐，不能硬编码后与运行时配置分离 |
| translate_arxiv_pdf.keep_dual | False | 用户要求双语时显式 true |
| get_paper_content.section | None | null 表示默认读取标题与摘要 |
| summarize_paper.style / max_words | tldr / 120 | Schema、normalize_style、bucket_max_words 一致 |
| analyze_figure.figure_no / question | 1 / describe | Schema 与工具函数一致 |
| get_translated_content.page | 1 | Schema 与函数一致 |

不要把所有参数都填成空字符串。对于 output_path，使用 null 表示未指定；对于布尔值使用 true/false；对于数值使用真实默认数值；对于 ref 使用明确的整数、ID、标题片段或 null。

### 2.4 区分模型参数和 Harness 参数

BaseAgent._execute_with_side_effects 已经会覆盖并注入 session_id，模型不应该负责生成它。

建议把 session_id 标记为内部参数，不放进模型可见的 tools Schema 和 SFT Completion，但保留在内部执行契约中供 Harness 注入。

translate_arxiv_pdf 的 paper_id、pdf_url、input_pdf_path 等直接文件路径/内部覆盖参数也应检查是否需要暴露给策略模型。若当前任务集不需要模型直接控制这些内部参数，应从模型可见 Schema 隐藏，由 Harness 或专用接口处理。

---

## 3. 评分器：按“实际生效的参数”评分，而不是按原始 JSON 字段是否出现评分

涉及：

- AgenticArxiv/benchmark/metrics.py
- AgenticArxiv/rl/reward.py
- AgenticArxiv/benchmark/protocol_audit.py
- 对应测试文件

当前 argument_match_score 主要按 TaskSpec expected_args 中的键逐个比对原始参数。它没有把工具默认值展开后再比较。因此，参数没有写出来但函数默认值正好正确时，也会被扣分。

### 3.1 必须修正的例子：constraint_search_no_file

专家动作要求 aspect="*"，而 get_recently_submitted_cs_papers 的 Python 函数默认 aspect="*"。

Base 试次 0/1/2 没有显式传 aspect，但工具实际按默认值检索了所有 CS 类别，结果数量和语义均正确。模型额外传 output_path=""，而 save_to_file=false，且空路径本身会回退到工具默认路径。

这不是模型把 aspect 传错了，也不是模型漏掉了一个实际必需参数。它是评分器没有做默认值归一化造成的假阴性。

修复后，这类调用应视为参数语义正确；不能作为 SFT badcase。

### 3.2 参数评分规则

对每一步工具调用：

1. 先检查工具名和工具序列。工具序列必须与专家步骤完全一致，额外调用、重复调用、缺少步骤都应失败。
2. 从工具 Schema / 注册函数中取得默认值，构造 actual effective args。
3. 用专家动作和相同默认规则构造 expected effective args。
4. 对实际影响行为的参数进行比较。
5. 忽略 Harness 注入的 session_id。
6. 允许语义等价的规范化形式，例如类别 cs.CV → CV；数字 ref 的字符串/整数表示在确实解析到同一论文时等价。
7. 对 ref 优先比较工具实际解析到的 paper_id，而不是只比较原始 ref 字符串。
8. 如果用户明确指定了带版本号的 arXiv ID，例如 v1，版本号必须保留；显式版本请求不能被无条件降级为“忽略版本”的比较。
9. 不应惩罚与默认行为等价的省略字段；但如果省略导致 Python/运行时默认值与用户意图不同，则仍然是参数错误。
10. 对非默认、且会改变行为的额外参数必须评分。例如专家要求 keep_dual=true，模型漏传导致默认 false，应判错；专家要求普通下载，模型擅自 force=true，也应判错。
11. 未知参数仍然是 Schema 错误，不能因为“参数多传了”就一概忽略。

metrics.py 与 GRPO 使用的 RewardCalculator 必须共用同一参数语义，避免 Benchmark 认为正确、GRPO 却给负奖励，或反过来。

### 3.3 修复自然语言误报

Badcase 审计发现 search_CV_1d_1 的工具调用和结果都正确，但最终 Assistant 回答中出现“ref 参数无法解析”，旧版 _count_parse_failures 对最终文本做关键词匹配，把它误判成协议解析失败。

协议解析失败应来自结构化 parser / runtime event / parse_failed 标志，不能仅凭最终回答里出现“无法解析”四个字判断。

同样检查 RewardCalculator._process_score 和 observation_reports，避免把用户可见的普通解释当成协议解析错误。

**修复后必须重算旧 Base 原始轨迹，报告旧分数与新评分口径分数；不要直接把 33/93 当作纯模型能力指标。** 审计中提到的 35/93 只是移除两条明确文本误报后的局部诊断复算，不是最终修正分数。

---

## 4. Setup / Session / Observation：模型必须看到真实且一致的状态

涉及：

- AgenticArXiv/benchmark/runner.py
- AgenticArXiv/agents/prompt_templates.py::build_visible_setup_context
- AgenticArXiv/agents/base_agent.py
- AgenticArXiv/agents/native_tool_agent.py
- AgenticArXiv/agents/side_effects.py
- MockArxivEnv

### 4.1 Setup 执行结果和模型可见状态要一致

BenchmarkRunner._apply_setup 会真实执行 setup，但 build_visible_setup_context 主要根据 TaskSpec 声明的动作推导文本，而不是根据实际执行回执构造状态。

建议让 setup 执行返回结构化的结果记录，并用实际状态生成模型可见上下文。不能因为 TaskSpec 声明“下载过”就无条件告诉模型“已下载成功”；必须以工具回执或环境状态为准。

### 4.2 两处代码口径不一致

- BaseAgent.PAPER_SEARCH_ACTIONS 同时包含 get_recently_submitted_cs_papers 和 search_arxiv_papers。
- BenchmarkRunner._apply_setup 对搜索结果写入论文列表时，目前只显式处理 get_recently_submitted_cs_papers。

请统一两种搜索工具的 Session 写入行为，确保 setup 与 Agent 正常执行的副作用一致。

build_visible_setup_context 目前显式处理了搜索、下载、翻译、缓存、正文、摘要和抽图，但没有完整覆盖 get_translated_content 和 analyze_figure 对 last_active 的更新。补齐所有会改变会话状态的工具。

### 4.3 翻译 PENDING 与“已经翻译完成”必须区分

BenchmarkRunner._apply_setup 对 translate_arxiv_pdf 使用 enqueue_translate，返回的是异步提交状态。build_visible_setup_context 当前却写成“此前已翻译”，容易让模型误以为译文已经 READY。

上下文应明确区分：

- 翻译任务已提交 / PENDING；
- 翻译已完成 / READY；
- 翻译失败 / FAILED。

get_translated_content 只有在 READY 时才可读取。不能只根据“执行过 translate_arxiv_pdf”就推断译文已完成。

### 4.4 PENDING 规则与现有 Train TaskSpec 有冲突

当前 NATIVE_TOOL_SYSTEM_PROMPT 的规则倾向于：translate_arxiv_pdf 返回 PENDING 后立即结束。

但当前 Train TaskSpec 中的 chain_cv5_cache_dl_tr_cache 明确包含：

1. 搜索；
2. 查缓存；
3. 下载；
4. 提交翻译；
5. 最后再查一次缓存。

所以“PENDING 后无条件 FINISH”与这个合法任务的后续明确步骤冲突。

统一规则应为：

- PENDING 表示翻译请求已受理，不表示翻译完成。
- 不得重复提交相同翻译任务。
- 不得为了等待 READY 而反复轮询。
- 如果用户明确要求的剩余步骤是独立且可立即执行的，例如提交后按要求查一次缓存，可以执行该步骤一次，再结束并如实报告 translated_ready=false。
- 如果剩余步骤依赖译文 READY，例如立即读取第 2 页，而当前仍为 PENDING，则停止并说明尚未完成，不要调用 get_translated_content。
- 翻译是最后一个目标步骤时，直接生成简短最终回答。

### 4.5 迭代预算

审计所有 TaskSpec 的工具步骤数和 max_iterations，保证预算至少能容纳全部工具调用以及最终回答。不要让“最后一个工具调用后没有机会生成 Final Answer”成为框架造成的失败。

### 4.6 Snapshot Miss 单独归因

Badcase 中有两次额外 analyze_figure 调用触发 MockArxivEnv replay snapshot missing。它们是模型做了专家轨迹之外的额外调用，同时也暴露了 Mock 快照对非法/额外动作的覆盖边界。

要求：

- 不得用固定回退论文伪装成正确工具结果。
- Snapshot 未覆盖必须返回可识别、稳定的环境错误。
- 记录 snapshot_coverage_error，与普通工具参数错误分开统计。
- 正确专家轨迹所需的工具调用必须有完整快照；数据生成时遇到专家动作缺键应中止该条轨迹，不得静默跳过。

---

## 5. Badcase 归因：哪些是设计问题，哪些是模型问题

### 设计 / Harness / 评分器问题

| 现象 | 归因 | 修复 |
|---|---|---|
| aspect 缺省但函数默认就是 *，仍被扣参数分 | 评分器问题 | 比较 effective args |
| output_path="" 被误当成需要专门学习的行为 | 工具契约 / 评分问题 | 定义 null 默认值，忽略无行为差异的空路径 |
| 最终回答出现“无法解析”就被算 parser failure | 评分器误报 | 只依据结构化协议事件 |
| ref 在不同工具中省略后语义不同 | 工具契约不一致 | 统一 ref 默认值与 null 语义 |
| service / threads Schema default 与动态 settings 可能分离 | Schema / Runtime 一致性 | 从真实 settings 对齐 |
| session_id 出现在模型 Schema，但 Harness 会覆盖 | 模型接口暴露了内部参数 | 模型侧隐藏，Harness 注入 |
| PENDING 一律终止与 chain_cv5_cache_dl_tr_cache 冲突 | Prompt / TaskSpec 冲突 | 采用“禁止重提交/轮询，但允许明确剩余独立步骤”的规则 |
| setup 根据声明推导成功状态 | 状态构造问题 | 使用实际 setup 回执 |
| setup 对搜索工具和 last_active 的覆盖不完整 | 状态副作用问题 | 统一状态写入与可见上下文 |
| 两次额外图表调用触发 snapshot miss | Mock 覆盖边界 | 单独分类，不伪造成功 Observation |

### 设计修复后仍应归因于模型策略的问题

| Base 行为 | 应学习的行为 |
|---|---|
| 空会话先查 ref=1 的缓存 | 没有候选列表时，先执行用户真正要求的搜索；没有足够上下文时拒绝 |
| 已有候选列表仍重新搜索 | 直接利用当前 Session 和原有 ref |
| 把 cs.CV / cs.RO 当关键词搜索 | 分类搜索与关键词搜索选对工具 |
| 生成未注册工具名 | 只从可用工具列表中选择 |
| 用 get_paper_content 代替 download_arxiv_pdf | 区分下载 PDF、读取正文、查询缓存 |
| 已有工具错误后重复同样调用 | 根据 Observation 修正，不能原样循环 |
| 只完成多个子目标中的一部分就 FINISH | 所有用户要求的子目标都完成后再结束 |
| pdf_ready=false 仍声称下载成功 | Final Answer 必须依据真实 Observation |
| 缺少上下文仍调用无关工具 | 说明缺失上下文并拒绝 |
| 第 20 篇超出当前 5 篇列表仍重复查/下载 | 根据现有列表边界拒绝 |
| PENDING 后重复输出状态、轮询或重新提交 | 简短报告状态，遵守异步契约 |
| 用户给出完整 arXiv ID 却丢掉版本号并改用关键词搜索 | 原样保留完整 ID，直接调用目标工具 |
| 用户要求 keep_dual=true 却使用默认 false | 保留用户明确参数约束 |

注意：类别短码问题属于混合项。当前 Schema 只列出短码，描述又没有充分解释 cs.CV 与 CV 的映射；需要先补齐 Schema 和确定性归一化。修复后若模型仍选择错误工具或生成错误类别，才归因于模型。

---

## 6. 修复顺序与验收

### P0：开始 Native SFT 前必须完成

1. 重写 NATIVE_TOOL_SYSTEM_PROMPT，删除逐参数指令。
2. 补齐 10 个工具的 Schema / Description，统一 ref 和默认值。
3. 隐藏 session_id 等 Harness 管理参数。
4. 修复 effective-argument scoring；Benchmark 与 GRPO 共用。
5. 修复自然语言“无法解析”误报。
6. 修复 setup 状态和 PENDING 契约冲突。
7. 对 Qwen 原生 XML 参数解码保留 arXiv ID 的字符串类型回归测试。

### 必须增加的测试

- get_recently_submitted_cs_papers 未显式传 aspect，与显式 aspect="*" 的实际行为和评分一致。
- output_path=null 与 output_path="" 在工具语义等价时不被判为不同；save_to_file=false 与 true 必须区分。
- keep_dual=true 不得被默认 false 替代。
- service / threads 的 Schema default 与运行时 settings 一致。
- cs.CV 规范化为 CV；未知类别仍拒绝。
- ref=null 与显式 ref=1 只有在实际解析到同一篇论文时才算语义等价。
- 完整 arXiv ID（含 v1）在解析、执行和评分中保留版本。
- session_id 不出现在模型生成目标中，执行时由 Harness 注入。
- PENDING 后明确有 cache follow-up 的任务可以执行一次 cache；没有后续步骤的任务直接结束；PENDING 时读取译文必须被阻止。
- setup 失败时，模型可见上下文不能声称 setup 成功。
- Final Answer 中出现“无法解析”文字，但结构化协议正常时，不得增加 parse_failures。

### P1：修复后再生成数据

- 重跑纯 Base 的 Native Tool Calling Benchmark，得到新 Prompt / Schema / Scorer 版本下的公平基线。
- 保存每轮原始 response、工具参数、Observation、终止状态。
- 使用第二份文档中的训练任务设计生成 Native SFT 数据。
- 审计通过后训练 Qwen3.5-2B-Base LoRA。
- 使用相同 v4_86 Dev / IID / OOD 评测集、当前 Snapshot、相同重复次数和解码设置评估。

**不要在修复前开始 SFT。** 否则模型会学到不一致的默认值、错误的终止规则和评分器偏差。
