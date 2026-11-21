# Qwen3.5-2B-Base Native SFT：Badcase 到训练任务的数据设计

> 用途：交给 Codex 直接构造 Native SFT 数据并完成训练前验证。本文基于 2026-11-21 的 Base Badcase 审计和当前仓库 TaskSpec / 数据 Manifest。  
> 实施分支：feat/qwen35-base-native-sft  
> 目标模型：Qwen/Qwen3.5-2B-Base  
> 训练目标：Assistant Content + Native Tool Call + Final Answer

---

## 0. 本次 SFT 要解决什么

本次不是训练模型记住某条固定工具调用，也不是把旧 ReAct 文本改成另一种 JSON 格式。

我们要让模型学会：

1. 用户提出需求时，选对工具。
2. 从自然语言中抽取参数，并使用工具定义的默认值。
3. 识别当前 Session 已有的论文列表、下载状态和最近操作对象。
4. 根据真实 Observation 决定下一步，而不是重做已经成功的步骤。
5. 完成用户提出的全部子目标后再结束。
6. 正确处理 PENDING、缺少上下文、非法引用和工具能力边界。
7. 工具调用前生成简短、准确、与上下文衔接的 Assistant Content。

最终训练目标不是单独的 Tool Call，也不是长篇 Thought，而是：

- 需要工具时：Assistant Content + assistant.tool_calls。
- 不需要工具时：普通 Assistant Content，作为 Final Answer / refusal。
- Tool Observation：只作为下一轮上下文，不作为模型预测目标。

## 1. 数据版本与文件来源：不要再把 V3、V4、V7 混在一起

### 1.1 当前任务切分

本分支中有以下文件：

| 文件 | 任务数 | Train | Dev | IID Test | OOD Test | 说明 |
|---|---:|---:|---:|---:|---:|---|
| data/splits/v3_81.json | 81 | 51 | 8 | 18 | 4 | 旧版 81-task split |
| data/splits/v4_86.json | 86 | 55 | 8 | 19 | 4 | 当前 86-task split |
| data/splits/v7_86.json | 86 | 55 | 8 | 19 | 4 | 当前仓库中四个 split 的任务 ID 与 v4_86 完全相同 |

Badcase 审计文件把当次结果的历史追溯标签写为 v7_86；这不代表新 SFT 要改用 V7。

**本次新 SFT 以 data/splits/v4_86.json 为唯一规范切分文件。**

只允许使用其中 split.train 的 55 个父任务。Dev / IID / OOD 的原始任务文本、参数组合和专家标准动作不能直接进入训练。

### 1.2 旧 SFT 数据的真实来源

当前仓库 Manifest 给出的来源关系：

| 文件 | 数量 | 来源 / 说明 |
|---|---:|---|
| data/sft/sft_v3_t5_parametric_seed.jsonl | 249 行 | 86 个派生任务、32 个 Train 父任务的专家动作样本 |
| data/sft/sft_v3_t5_parametric_linguistic.jsonl | 2988 行 | 249 行扩成 6 种任务包装 × 2 种 Thought 表达 |
| data/sft/sft_v4_multiturn.jsonl | 2508 行 | 将旧语言扩增样本重新组织成多轮 ReAct 决策样本 |
| data/sft/sft_v4_multiturn_6144.jsonl | 2496 行 | V4 产物按 6144 token 上限过滤，排除 12 行 |

这些旧数据的 Manifest 指向：

- split：data/splits/v3_81.json
- 旧 Snapshot SHA256：dcde74d7f86da03d544b6a7e27fee3e07df078be85f1b1c39d8f47e81c34aa3f

本次 Base Badcase 审计使用的 Snapshot SHA256：

- d97035db326ca11935e882fd2aca7fe932e9e76cf1bc8a21078be7bec1ae8794

所以旧 V4 的 2496 行不能直接作为本次 Native SFT 的最终训练数据：它们是 ReAct 目标格式，而且历史 Observation 来自旧 Snapshot。

### 1.3 旧数据怎么复用

复用的是任务定义、参数变体、setup 和专家 Steps，不是旧的 Thought/Action 文本，也不是旧 Observation。

执行规则：

1. 读取 v4_86.train 的 55 个 TaskSpec。
2. 检查 V3 的 86 个派生任务和 249 个专家样本；只保留 parent_task_id 属于当前 v4_86.train 的任务。
3. 将旧参数化任务作为基础任务池，补上 V4 新增的 4 个 Train 任务：
   - trread_ai5_default
   - trread_ai5_null_page3
   - trread_cl5_default
   - trread_cv5_page2
4. 使用本次 Base Benchmark 对应的 Snapshot 重新执行专家动作，生成新的 Observation。
5. 对没有旧派生任务覆盖的 Train 父任务，直接从 TaskSpec 生成基础 Native 轨迹。
6. 不再做 6 种任务包装 × 2 种 Thought 的机械扩增。

注意：当前 GitHub 分支上存在旧数据 Manifest，但 sft_v3_t5_parametric_seed.jsonl 本体未必已提交。Codex 必须检查实际文件；如果本地/服务器没有该文件，就通过现有生成脚本和 TaskSpec 重新生成，并核对 Manifest / SHA256。不得根据 Manifest 猜测文件内容。

当前 Snapshot 必须通过 SHA256 核验。若 d97035... 对应文件不在仓库中，先从本次 Base Benchmark 产物中定位，不得静默退回旧 dcde74... Snapshot。

---

## 2. Badcase 审计结论与训练优先级

用户提供的 Badcase 审计覆盖 60 条失败轨迹、23 个任务。关键计数：

| 现象 | 审计证据 | 训练优先级 |
|---|---:|---|
| 工具序列与专家不一致 | 54/60 失败轨迹 | 最高 |
| 至少重复一次相同工具和参数 | 16 条轨迹 | 最高 |
| 不可执行任务（邮件发送、越界下载） | 6/6 试次未做到零调用并解释原因 | 最高 |
| Schema 拒绝 | 17 轮 | 高，但先修 Schema / 参数契约 |
| 真正的协议长度失败 | 2 轮，均为 PENDING 后重复生成 | 高 |
| 工具执行错误 | 28 条试次、68 次调用 | 高，先区分模型错误和 Snapshot 覆盖错误 |
| 评分器误报 | search_CV_1d_1 的 2 次真实成功被文本正则误判 | 不作为模型 Badcase |

### 2.1 不能作为模型 Badcase 的案例

**constraint_search_no_file：**

模型没有显式传 aspect，但 get_recently_submitted_cs_papers 的实际默认值就是 aspect="*"。工具返回了正确的 5 篇结果，save_to_file=false 也正确。额外的 output_path="" 在当前工具语义下没有改变结果。

这是旧评分器没有按 effective args 处理默认值的问题，不应通过 SFT 强迫模型记住“必须显式输出 aspect=*”。修复评分器后，该类调用应按实际默认行为判定。

**search_CV_1d_1：**

工具调用、参数、Observation 和任务结果都正确。最终回答中的“ref 参数无法解析”触发旧评分器的文本正则，是评分误报。不要为这两条轨迹生成针对性训练样本。

**arXiv ID 类型解码：**

历史审计发现无版本 ID 被 XML 解码器转成浮点数。当前分支已有按 Schema 保留 numeric-looking string 的解码逻辑和测试。Codex 应保留并补齐回归测试，不要把已修复的 Harness Bug 当成模型能力问题。模型确实丢掉 v1 后缀、改用关键词搜索的行为仍然是需要训练的模型错误。

**Snapshot missing：**

两次图表分析缺键发生在模型额外调用专家轨迹之外的工具之后。它同时说明模型多做了无关操作、Snapshot 对额外调用的覆盖有限。不要用编造的 Observation 填充，也不要把它作为专家正确轨迹。单独标记 snapshot_coverage_error。

---

## 3. 训练任务按行为设计，不按旧样本行数设计

所有新任务都存入独立 Catalog。建议字段：

~~~json
{
  "task_id": "native_sft_...",
  "data_source": "core",
  "parent_task_id": "v4_86_train_task_id",
  "source_badcase_id": "badcase_task_id_or_pattern",
  "error_pattern": "wrong_tool_selection",
  "task": "给模型看的用户请求",
  "setup": [],
  "steps": [],
  "terminal_mode": "completed",
  "terminal_reason": null,
  "decision_notes": []
}
~~~

data_source 使用：

- core：v4_86 Train 的基础专家任务。
- error_targeted：由 Base Badcase 错误类型派生、但不复制测试原题的新训练任务。
- sft_only：为补充工具能力边界而新建、不加入 Benchmark 切分的训练任务。
- recovery：上下文中包含一个真实工具错误 Observation，监督模型如何修正下一步。

### 3.1 类别检索与关键词检索

**Badcase 来源：**

- multi_cv3_dl1
- search_AI_1d_3
- search_CR_14d_10
- search_RO_30d_25
- summary_lg5_tldr120
- multi_ro3_dl_two
- analyze_ai5_ref4_desc

**复用 Train 父任务：**

- search_AI_30d_25
- search_CL_30d_10
- search_CL_7d_5
- search_CR_7d_5
- search_CV_14d_3
- search_LG_14d_1
- search_LG_3d_10
- search_RO_3d_8
- search_kw_llm
- constraint_search_only

**需要覆盖的任务实例：**

1. 用户要求“最近 7 天所有计算机科学方向论文，最多 5 篇”：调用分类检索工具，规范参数 aspect="*"、days=7、max_results=5。
2. 用户要求机器人学、计算机视觉、自然语言处理、机器学习、密码学等 CS 子领域：分别使用 RO、CV、CL、LG、CR 短码。
3. 用户要求关键词“agentic reinforcement learning”：使用 search_arxiv_papers，传真实关键词 query；不能把 cs.AI、cs.CV 等类别代码当关键词。
4. 用户只要求搜索结果：搜索成功后直接回答，不额外查缓存、读取正文或下载。

训练重点：工具选择和参数映射。不要把“cs.CV 与 CV 的表面写法差异”训练成语义错误；规范化后的类别含义相同。

### 3.2 下载、读取正文、缓存查询必须区分

**Badcase 来源：**

- state_dl_then_cache
- multi_cv3_dl1
- figure_ai5_ref4
- trread_ro5_page2

**复用 Train 父任务：**

- constraint_download_only
- constraint_cache_only
- state_cache_before_dl
- constraint_two_downloads_only
- chain_ai5_read_then_summary
- figure_cv5_ref1
- analyze_cv5_ref1_desc
- trread_ai5_default
- trread_ai5_null_page3
- trread_cl5_default
- trread_cv5_page2

**需要覆盖的行为：**

- “下载 PDF” → download_arxiv_pdf。
- “读取摘要/方法章节” → get_paper_content。
- “查询是否已下载/翻译” → get_paper_cache_status。
- “提取 PDF 图表” → extract_paper_figures。
- “描述第 1 张图” → 先确保图表已抽取，再调用 analyze_figure。
- “读取中文译文第 2 页” → 只在译文 READY 时调用 get_translated_content；不能把读取译文误认为重新提交翻译。

Content 必须说明当前工具的真实作用。例如，不能在 get_paper_content 返回正文后声称 PDF 已下载；只有 download_arxiv_pdf 的真实 Observation 才能证明下载状态。

### 3.3 Session 状态与 ref 指代

**Badcase 来源：**

- constraint_translate_only
- ref_ctrl_id_download
- ref_stress_learning_state
- multi_ai3_tr1
- opt_service
- trread_ro5_page2

**复用 Train 父任务：**

- state_ref_last_active
- state_ref_last_of_10
- state_translate_active
- state_cache_before_dl
- ref_ctrl_null_translate
- ref_ctrl_word_handover
- ref_ctrl_word_marionette
- ref_stress_state_across
- ref_stress_uncertainty_aware
- ref_stress_image_restoration

**需要构造的对照场景：**

1. 空会话 + “把刚才那篇论文翻译一下”：没有可解析的论文对象，应零调用并请求补充信息。
2. 只有搜索列表 + “翻译第 2 篇”：使用 ref=2；不要因为没有 last_active 就重新搜索。
3. 搜索并下载过 + “翻译刚才下载的那篇”：使用 ref=null，直接提交翻译。
4. 用户明确给出完整 arXiv ID：将完整 ID（含 v1）作为字符串传给目标工具，不要先关键词搜索，不得删除版本后缀。
5. 用户给出标题片段：直接把标题片段作为 ref；不需要为了获得序号而重搜。
6. 用户指定第 10 篇：只有当前候选列表确实包含该项时才可调用；使用原列表引用，不要覆盖 Session。

至少包含“同一句指代请求在空 Session 与已有活跃论文 Session 下产生不同决策”的成对样本。这是状态理解任务，不是简单措辞扩增。

### 3.4 多子目标完整执行

**Badcase 来源：**

- state_cache_two
- multi_ro3_dl_two
- analyze_ai5_ref4_desc
- chain_cl5_dl3_tr3
- multi_cv5_dl1_cache_tr

**复用 Train 父任务：**

- constraint_two_downloads_only
- chain_ro5_dl_three
- multi_lg10_dl3
- chain_ai5_dl2_tr2
- chain_ai5_read_then_summary
- chain_cv5_cache_dl_tr_cache
- chain_lg10_dl_tr_last

需要覆盖：

- 用户要求下载第 1、2 篇：两篇都下载后才结束。
- 用户要求前三篇依次下载：不能漏掉第 2、3 篇，也不能重复第 1 篇。
- 用户要求先读方法章节，再总结：读正文和总结是两个独立步骤。
- 用户要求检索、下载、抽图、分析：按工具依赖顺序执行全部步骤。
- 每一步 Content 说明已经完成什么、用户还要求什么；不能在第一项完成后就说整个任务完成。

### 3.5 可选参数和默认参数

**复用 Train 父任务：**

- opt_keep_dual
- opt_threads
- opt_force_tr
- chain_lg10_dl_tr_last
- summary_ai5_tldr60
- summary_cl5_bullet250
- summary_cv5_structured120
- summary_ro5_structured250
- figure_cv5_ref1
- analyze_cv5_ref1_desc

需要覆盖：

- 用户明确要求保留双语：keep_dual=true。
- 用户明确指定 Google：service="google"。
- 用户指定 8 个线程：threads=8。
- 用户要求强制重新翻译：force=true。
- 用户指定摘要风格和字数：正确传 style、max_words。
- 用户没有指定可选值：按 Schema / Tool Description 的默认值生成，不要自己编造空字符串。

SFT 的规范 Tool Call 建议显式包含正常用户可见参数的有效值或默认值，例如 force=false、keep_dual=false、style="tldr"、page=1。框架管理字段 session_id 不进入模型目标。评分器仍需接受省略字段与实际默认值等价的旧调用。

### 3.6 完成、PENDING 和拒绝

**复用 Train 父任务：**

- infeasible_no_session
- infeasible_unknown_id
- infeasible_zero_index
- state_ref_last_active
- chain_lg10_dl_tr_last
- chain_cv5_cache_dl_tr_cache
- trread_ai5_default
- trread_ai5_null_page3
- trread_cl5_default
- trread_cv5_page2

需要覆盖四种终止行为：

**A. 普通完成**

所有要求的工具步骤均已成功，生成最终回答。不要提前结束。

**B. 翻译 PENDING 且翻译是最后一步**

报告翻译任务已受理和当前 PENDING 状态，立即结束；不重复提交、不轮询、不读取尚未 READY 的译文。

**C. 翻译 PENDING，但用户明确要求还有一个独立步骤**

chain_cv5_cache_dl_tr_cache 是 Train 中的真实例子：翻译提交后，用户还要求最后查一次缓存。允许执行一次 get_paper_cache_status，然后结束；应如实报告 translated_ready=false。不要把“PENDING 后绝不允许任何工具调用”写成无条件规则。

**D. Blocked / Refusal**

- 没有 Session，用户指代“刚才那篇”：零调用，说明缺少上下文。
- ref=0 或超过现有候选列表：零调用，说明引用越界。
- 明确 ID 不在当前 Snapshot：不编造论文，不反复尝试。
- 工具集不支持的操作：零调用，说明能力边界。

infeasible_no_session、infeasible_unknown_id、infeasible_zero_index 已在 Train 中。infeasible_unsupported_action 是 Dev 任务，不能把它的原始请求和专家答案直接复制进训练。可以创建新的 SFT-only 任务，例如“把已下载论文 PDF 上传到不支持的外部云盘”或“把论文附件发送给邮箱”，标记 data_source=sft_only，明确 no-tool / blocked。不要把这些新任务加入 v4_86 Benchmark split。

### 3.7 错误后的恢复样本

Base 会在关键词检索未命中、缓存查找缺少论文列表等错误后重复同一个调用。

可以额外构造少量 recovery 样本：

1. Prompt 历史包含一次错误的 search_arxiv_papers(query="cs.CV") 调用及真实“关键词未命中离线快照”Tool Observation。
2. 当前 User Task 仍然要求按 cs.CV 类别检索。
3. Completion 是简短 Content，说明前次关键词检索不适合类别筛选，然后调用 get_recently_submitted_cs_papers(aspect="CV", ...)，不能重复原查询。

另一个恢复样本：

1. Prompt 历史包含一次在空 Session 中 get_paper_cache_status(ref=1) 的真实错误 Observation。
2. 如果用户原任务是检索论文，则下一步调用分类检索工具。
3. 如果用户只说“把刚才那篇翻译一下”，且没有 Session 上下文，则应拒绝，不应继续尝试其他工具。

Recovery 样本必须单独标记 data_source=recovery。错误动作只能作为历史上下文，不能作为当前 Completion 的正确目标。不得伪造工具 Observation。

---

## 4. Assistant Content 怎么生成

这是 Native SFT 的重要监督目标。不要把它删掉，也不要重新生成旧 ReAct 的长篇 Thought。

### 4.1 Content 的定义

Content 是普通 Assistant 自然语言，用户可以看到。它的作用是说明当前已知状态和下一步行动，使多轮消息衔接清楚。

它不是隐藏 CoT，也不是要求模型暴露完整内部推理。

每次工具调用前通常一两句话即可，包含：

- 当前已经确认的状态；
- 用户接下来要求什么；
- 下一步准备执行什么。

Final Answer 则根据实际工具结果说明完成状态、PENDING 或阻塞原因。

### 4.2 Content 生成规则

创建类似以下接口：

~~~python
build_assistant_content(
    task,
    current_state,
    previous_observations,
    next_action,
    decision_note=None,
)
~~~

要求：

1. 优先采用人工审核的 decision_note。
2. 普通场景使用基于 TaskSpec、Session 状态、上一轮 Observation 和下一步专家动作的确定性模板。
3. 不默认调用另一个 LLM 自由生成 Content，避免生成错误状态或虚构理由。
4. Content 必须被实际状态支持；不能在下载前说“下载成功”，不能在翻译 PENDING 时说“翻译完成”。
5. 不要每轮都重复“好的，我来帮您处理”。
6. 不要复述整个用户请求或列出与当前决策无关的步骤。
7. 不要生成长篇分析、伪造的内部推理或 <think> 标签。
8. 对多子目标任务，Content 要指出已完成的子目标和剩余目标。
9. 对恢复样本，Content 应说明上一工具错误后为什么改变下一步，但不能把错误说成成功。

### 4.3 Content 示例

| 决策场景 | 合格 Content 示例 |
|---|---|
| 空会话，准备分类检索 | 当前会话没有候选论文，我先按用户指定的类别和时间范围检索。 |
| 已有候选列表，下载第 2 篇 | 已获得候选列表，用户指定第 2 篇，接下来直接下载 ref=2。 |
| 下载后继续读取摘要 | PDF 下载结果已确认 READY，接下来读取用户要求的摘要。 |
| 下载后继续翻译 | PDF 已就绪，用户还要求翻译这篇论文，我现在提交翻译任务。 |
| 多篇任务 | 第 1 篇已处理完成，用户还要求第 2 篇，我继续处理第 2 篇。 |
| PENDING 且为最后一步 | 翻译任务已受理，目前为 PENDING；本次提交已完成。 |
| PENDING 后还有明确缓存检查 | 翻译请求已受理；用户还要求核对缓存，我现在只查询一次缓存状态。 |
| 译文 READY 后读取第 2 页 | 译文已经 READY，接下来读取用户指定的第 2 页。 |
| 缺少上下文 | 当前会话没有可解析的论文对象，无法确定“刚才那篇”指什么，因此不调用工具。 |
| 超出能力范围 | 当前工具集不支持发送邮件，我无法代为发送，因此不调用无关工具。 |
| 工具错误后恢复 | 刚才的关键词查询未命中；用户要求的是 cs.CV 分类检索，我改用分类搜索工具，不重复相同查询。 |

不要把这些示例原样复制到所有任务。它们是 Content 质量标准，具体文本要由每条任务的真实状态和专家动作决定。

---

## 5. 专家轨迹和 Native SFT 格式

### 5.1 完整轨迹

创建完整 Native 交互记录：

- User Task
- Assistant Content + assistant.tool_calls
- Tool Observation
- Assistant Content + 下一次 tool_calls
- ...
- Final Answer / Refusal

工具执行必须复用 Native Agent 的真实执行路径，包括：

- Tool Registry 的 Schema；
- BaseAgent 的 session_id 注入；
- NativeToolCallingAgent 的消息历史结构；
- BaseAgent._execute_with_side_effects 的副作用处理；
- MockArxivEnv 的 Snapshot Replay；
- 翻译异步提交和 PENDING 行为。

不要自己重新写一套工具执行逻辑。否则训练 Observation 会和推理 Observation 不一致。

每条轨迹使用独立 Session。setup 先执行并保存真实结果，再生成模型可见的 setup context。setup 不作为当前策略的监督动作。

输出：

- data/sft/native_sft_trajectories.jsonl
- data/sft/native_sft_content_preview.jsonl

tool_call_id 必须唯一，并且 Assistant tool_calls 与后续 role=tool 的 tool_call_id 一一对应。为便于复现，训练数据可使用确定性 ID；不要在每次重建时随机改变完整轨迹内容。

### 5.2 决策样本

完整轨迹按每一个 Assistant 决策时刻拆分：

| 样本 | Prompt | Completion |
|---|---|---|
| 首次工具调用 | System + User + Tools | Assistant Content + Tool Call |
| 第二次工具调用 | System + User + 历史 Assistant Tool Call + Tool Result + Tools | Assistant Content + Tool Call |
| 最终回答 | System + User + 完整历史 + Tools | Final Assistant Content |

建议训练记录：

~~~json
{
  "prompt": [],
  "completion": [],
  "tools": [],
  "task_id": "native_sft_...",
  "trajectory_id": "...",
  "trajectory_step": 0,
  "data_source": "error_targeted",
  "error_pattern": "unnecessary_search"
}
~~~

要求：

- 当前 Assistant Content 参与 Loss。
- 当前 Assistant Tool Call 的工具名、参数和调用结构参与 Loss。
- Final Answer / refusal 的 Assistant Content 参与 Loss。
- System、User、Tools Schema、历史 Assistant 消息、历史 Tool Observation 不参与当前样本的 Loss。
- Tool Observation 只作为下一轮上下文，不作为模型预测目标。
- 不生成 Thought 字段，不生成旧 ReAct Action 文本。
- 使用当前 Qwen3.5 Chat Template 渲染结构化消息，不手工拼接 Qwen XML。
- 核对当前 TRL 版本对 tool_calls 的真实数据接口；不要假设 function.arguments 一定应是字符串或字典，必须以实际 Chat Template 和 TRL 版本做渲染测试。
- 必须验证 Assistant Content 和 Tool Call 的 token labels 都是有效监督标签，Prompt / 历史 / Tool Result 都被 mask。

输出：

- data/sft/native_sft_train.jsonl
- data/sft/native_sft_train.jsonl.manifest.json

---

## 6. 轨迹生成和数据审计要求

创建：

- scripts/build_native_sft_data.py
- scripts/audit_native_sft_data.py
- reports/native_sft_data_audit.md

### 6.1 任务来源审计

- core / error_targeted 的 parent_task_id 必须属于 data/splits/v4_86.json 的 split.train。
- 不允许把 Dev / IID Test / OOD Test 的原始任务、参数组合和专家答案复制进训练集。
- SFT-only 任务必须有单独标记，不能偷偷混入 Benchmark split。
- 每条 targeted task 必须有 source_badcase_id 和 error_pattern。
- 不能只根据失败任务 ID 生成相同任务；必须生成新的自然语言任务实例或参数实例，且仍然属于训练能力范围。

### 6.2 专家执行审计

- setup 真实执行成功，且模型可见上下文与实际状态一致。
- 每一步专家 Action 符合 Registry Schema。
- 工具顺序、参数、ref 解析结果与任务定义一致。
- 每一步 Observation 来自当前 Snapshot。
- 完整执行全部专家步骤。
- PENDING 与后续独立步骤按第 3.6 节规则处理。
- blocked 任务必须零工具调用，并生成具体拒绝理由。
- 失败、Snapshot 缺键、工具回执缺失的轨迹不得静默进入训练。

### 6.3 Content 审计

- Content 只陈述已经发生或当前可确认的状态。
- 每个工具调用前的 Content 与下一步专家动作一致。
- 不得出现“已经下载成功”但 pdf_ready=false。
- 不得出现“翻译完成”但 status=PENDING。
- 不得在已有候选列表时声称“当前没有论文”。
- 不得在空 Session 时声称“刚才下载的论文”可解析。
- 不得把用户未要求的额外操作写进 Content。
- 不得出现长篇重复话术或 <think> 标签。
- 最终回答必须与实际 Observation 一致。

### 6.4 长度与重复审计

- 使用当前 Qwen3.5-2B-Base Tokenizer 和真实 Chat Template 计算长度。
- 6144 作为当前数据样本长度上限，不要把它误写成模型架构的输入长度上限。
- 不得直接截断 Completion 或截断 Tool Call。
- 超长时优先压缩历史 Observation 的展示长度，或排除并记录；不能破坏监督目标。
- 统计完全重复样本、同一 TaskSpec 的参数变体、相同轨迹的多份语言包装。
- 不再进行 6×2 的机械语言 / Thought 扩增。

Manifest 至少记录：

- v4_86 split 文件和 SHA256；
- 55 个 Train TaskSpec ID；
- 当前 Snapshot SHA256；
- Tool Schema SHA256；
- 代码 commit；
- core / error_targeted / sft_only / recovery 数量；
- 完整轨迹数、决策样本数；
- 各工具调用次数和各终止类型数量；
- Content / Tool Call / Final Answer 数量；
- 被排除轨迹及原因；
- Token 长度分布；
- 输出文件 SHA256。

---

## 7. 训练与评测顺序

不要只生成数据然后直接用旧 Base 成绩比较。Agent Prompt、Schema、默认值、评分器都修复后，旧基线已经不是同一个实验条件。

按以下顺序执行：

1. 完成 native_tool_agent_design_review.md 中的 P0 修复和测试。
2. 用修复后的 Agent / Schema / Scorer 重跑纯 Qwen3.5-2B-Base，保存新基线和完整逐轮轨迹。
3. 以 v4_86 Train 55 个任务、当前 Snapshot 和本文件的 targeted task plan 构造 Native SFT 数据。
4. 完成完整轨迹、Content、Tool Call、Schema、Loss Mask、数据泄漏和 Token 审计。
5. 使用 Qwen/Qwen3.5-2B-Base 训练 LoRA。参考已有 SFT 启动脚本和配置，但必须确认 base_model 指向 Base，不能误加载官方后训练 Qwen3.5-2B。
6. 先跑短 Smoke Test，检查 Loss、LoRA 更新、Content 和 Tool Call 标签、Adapter 加载、Native 推理。
7. Smoke Test 通过后完成正式 SFT。
8. 使用相同的 Native Tool Calling Agent、相同 Snapshot、相同 v4_86 Dev / IID / OOD、相同重复次数和解码配置评测。
9. 分别报告 Strict Success、工具序列准确率、effective-argument 准确率、Schema 合规、工具执行错误、重复调用、PENDING 终止、拒绝成功率。
10. 保存完整轨迹、模型 Adapter、训练日志、数据 Manifest 和评测报告。

本阶段先不启动 GRPO。SFT 的增量结果确认后，再决定是否进入 GRPO。

---

## 8. Codex 最终交付

1. Agent / Schema / Scorer 修复及测试。
2. 修复后的纯 Base 新基线和完整 Badcase 轨迹。
3. data/sft/native_sft_task_catalog.jsonl
4. data/sft/native_sft_trajectories.jsonl
5. data/sft/native_sft_content_preview.jsonl
6. data/sft/native_sft_train.jsonl
7. data/sft/native_sft_train.jsonl.manifest.json
8. scripts/build_native_sft_data.py
9. scripts/audit_native_sft_data.py
10. reports/native_sft_data_audit.md
11. Native SFT 训练入口和启动脚本
12. Smoke Test 结果
13. 正式 SFT Adapter
14. SFT 后 Benchmark 报告和逐轮轨迹

所有旧 ReAct 数据、旧 SFT Adapter、GRPO 代码、原始 Benchmark 和现有 Snapshot 都必须保留，不覆盖。

**执行原则：先修设计，再重跑 Base，再生成 Native 数据，再 SFT。不要为了凑数据量扩增措辞；每条新增数据都必须覆盖一个明确的工具选择、参数、状态、完整执行、终止或拒绝行为。**
