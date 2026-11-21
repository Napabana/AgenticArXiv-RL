# Qwen3.5-2B-Base Native SFT：最终提示词、Badcase 数据与训练执行方案

> 本文是本轮 Native SFT 的唯一执行规格，整合 Base/SFT 配对错误复盘、用户约束补齐方案及此前 Native SFT 数据设计。Codex 应按本文执行；如与旧规划冲突，以本文为准。
>
> 实施分支：feat/qwen35-base-native-sft  
> 基线分支：feat/qwen35-2b-base-tool-calling  
> 目标模型：Qwen/Qwen3.5-2B-Base  
> 数据格式：Qwen 原生 Tool Calling + 简短 Assistant Content + Tool Result + Final Answer  
> 本轮目标：先修正 Agent 设计和评分口径，再构造针对性 Native SFT 数据，完成 SFT 和同协议评测。本轮不启动 GRPO。

---

## 0. 本轮实验要回答的问题

本轮不是继续扩增旧 ReAct 文本，也不是教模型记住固定工具调用序列。

要验证的是：在 Qwen3.5-2B-Base 上，使用高质量专家交互轨迹进行 Native SFT，能否改善以下行为：

1. 正确理解用户目标、禁止条件、范围限制和步骤顺序。
2. 在工具之间做出正确选择，生成符合工具契约的参数。
3. 根据真实 Tool Observation 维护论文列表、论文引用和 Session 状态。
4. 完整执行用户要求的多个子目标，不重复已成功步骤。
5. 区分下载、读取正文、缓存查询、翻译、译文读取、图表抽取和图表分析。
6. 正确结束任务，包括普通完成、异步 PENDING、缺少上下文、无效引用和工具能力不支持。
7. 在每次工具调用前生成简短、准确、与前序状态衔接的 Assistant Content。

本轮 SFT 的监督目标是：

- 需要工具时：Assistant Content + 结构化 Native Tool Call。
- 不需要工具时：普通 Assistant Content，作为 Final Answer / refusal。
- Tool Result：作为下一轮上下文，不作为模型预测目标。
- 不生成旧 ReAct 的 Thought / Action / Observation 文本，也不生成冗长隐藏 CoT。

---

## 1. 已有实验结果与本轮数据依据

### 1.1 上一轮 Base / SFT 配对评测

依据用户提供的《Native SFT 与 Base 的配对错误复盘》：

| Split | Base | 上一轮 SFT | 修复试次 | 回退试次 |
|---|---:|---:|---:|---:|
| Dev | 15/24 | 15/24 | 6 | 6 |
| IID Test | 33/57 | 37/57 | 15 | 11 |
| OOD Test | 9/12 | 12/12 | 3 | 0 |
| Total | 57/93 | 64/93 | 24 | 17 |

Dev 总分没有变化，但 6 次改善、6 次回退；IID 和 OOD 有改善，但仍有明确的参数、引用和终止错误。

上一轮 SFT 的 29 个失败试次按首个不达标层级互斥归因：

| 首个不达标层级 | 试次数 | 主要问题 |
|---|---:|---|
| 工具序列 | 11 | 不支持能力仍调用工具、越界仍下载、只查第一篇缓存、下载后漏查缓存 |
| 生效参数或论文引用 | 12 | 全 CS 与禁止保存、活跃论文双语翻译、显式摘要章节、标题片段定位 |
| 最终回答终止 | 6 | 工具序列和参数正确，但最终回答重复论文 ID / PDF READY 信息，触及生成上限 |

其他诊断计数可能与上述互斥归因重叠，不应相加：

- 工具序列不匹配：Base 29，SFT 11。
- 序列匹配但生效参数不合格：Base 7，SFT 12。
- 序列匹配但论文指代不合格：Base 0，SFT 3。
- 工具步骤未完成便结束：Base 2，SFT 5。
- 存在协议解析失败：Base 4，SFT 6。
- 存在工具执行失败：Base 17，SFT 3。
- 存在超出任务要求的重复调用：Base 16，SFT 1。
- 阻塞任务拒绝语义不合格：Base 5，SFT 6。
- 存在 PENDING 调用拦截：Base 4，SFT 0。
- 存在 Snapshot 覆盖错误：Base 11，SFT 0。

### 1.2 已明确的数据覆盖缺口

上一轮 SFT 数据统计：

- 83 条分类检索监督全部使用 save_to_file=true。
- 全 CS 的 aspect="*" 只出现 1 次。
- 95 个“下载后继续决策”的监督样本中，35 个直接生成最终回答，0 个直接调用缓存查询工具。
- 25 条翻译调用中，7 条 keep_dual=true，但没有一条同时覆盖 ref=null 与 keep_dual=true。

这解释了为什么模型会把工具默认保存行为当成固定行为、下载后直接结束，以及在“最近操作论文 + 保留双语”组合上出错。数据需要补充对照，而不是简单增加同义句。

### 1.3 必须保留的历史结果

- 冻结旧 Base / SFT 结果，不覆盖历史报告。
- 本轮修复 Prompt、Schema 或评分器后，旧成绩不再是同一实验条件下的直接基线。
- 新 Base 和新 SFT 必须使用相同 Prompt、Schema、Snapshot、解码配置、任务列表和随机种子配对评测。
- 不把旧 ReAct 38.7%、旧 Native 77.4% 与本轮修复后的分数混在同一张因果对照表中。

---

## 2. Prompt 最终设计：三个逻辑部分

工具注入已经实现。本轮不要重写 Tool Injection 机制，也不要把工具参数逐项写进固定 System Prompt。

Prompt 逻辑分为三部分：

1. Agent 身份：固定的简短身份和交互方式。
2. 工具注入：运行时通过 tools Schema 注入，不属于固定 System Prompt 的参数说明文本。
3. 用户约束与任务完成规则：固定的通用行为规则，专门约束用户意图、任务进度和阻塞处理。

### 2.1 第一部分：Agent 身份

建议文本：

~~~text
你是一个 arXiv 计算机科学论文研究助手。根据用户目标和当前会话状态，使用运行时提供的工具完成任务。每轮最多调用一个工具。

调用工具前可以用一句简短、自然的语言说明当前已知状态和下一步动作。说明必须依据用户请求和真实工具结果，不输出冗长的内部推理或隐藏思考标签。
~~~

不要在身份部分重复具体工具的参数规则。

### 2.2 第二部分：工具注入

保留当前 Native Tool Calling 的运行时工具注入机制：

- Tool Registry 产生模型可见的 tools Schema。
- Qwen Chat Template 接收 tools。
- 模型生成原生 tool_call。
- Agent 执行工具，并以 Tool Message 回填真实结果。

工具定义、参数含义、类型、合法取值和默认行为，以对应 Schema / Description 为准。

本轮只修正本文件第 3 节明确指出的工具描述问题；不要把所有参数说明复制进 System Prompt。

### 2.3 第三部分：用户约束与任务完成规则

将以下三条作为独立章节追加到 System Prompt：

~~~text
### 用户约束与任务完成规则

1. 遵守用户约束：用户明确提出的禁止条件、范围限制、参数要求和步骤顺序必须遵守。工具默认行为与用户限制冲突时，选择能够满足限制的合法参数；只执行与用户目标相关的操作。

2. 跟踪任务进度：根据用户请求、历史交互和真实工具 Observation，判断已完成、待完成和受阻的步骤。按用户要求及步骤依赖继续执行；只有全部必需步骤完成或存在明确阻塞时，才能给出最终回答。区分 PENDING 和 READY，不重复提交或无意义轮询。

3. 处理阻塞与部分完成：缺少必要上下文、引用无效或工具能力不支持时，说明具体原因，不猜测或编造。若其他独立子目标仍可执行且不违反用户约束，应继续完成，并分别汇报完成情况和阻塞原因。
~~~

要求：

- 不再加入 aspect、days、max_results、ref、output_path、save_to_file 等具体参数映射。
- 不把“所有可选参数必须显式传递”写进 Prompt。
- 不把 PENDING 解释成无条件禁止后续所有工具调用；明确的独立后续步骤仍可执行。
- 不重复旧 Prompt 中相同含义的规则，合并后保持简洁。
- 每轮最多一个工具调用的约束保留。
- 工具注入本身已经实现，不得为了本轮 Prompt 修改而重写 Native Tool Calling Runtime。

---

## 3. Tool Schema / Description 的必要修正

System Prompt 不负责逐参数教模型。工具 Schema 必须把模型需要的参数契约讲清楚。

### 3.1 分类检索：save_to_file

工具：get_recently_submitted_cs_papers。

模型可见的参数说明必须明确：

- aspect="*" 表示所有 CS 子领域。
- aspect="CV"、"CL" 等表示相应 CS 子领域。
- days 是最近多少天。
- max_results 是最多返回多少篇。
- save_to_file=true 表示将检索列表写入文件。
- save_to_file=false 表示只返回检索结果，不写文件。
- save_to_file 的默认值是 true。
- output_path 是保存路径；默认 null / None 时使用工具约定的默认路径。
- 当 save_to_file=false 时，output_path 不影响行为，训练目标统一为 null；不要生成空字符串作为路径。

用户明确说“不要保存 / 不写文件”时，模型必须覆盖默认值，使用 save_to_file=false。用户未提出禁止保存时，工具默认行为可以生效，不应仅因没有显式传 save_to_file 就判参数错误。

离线 Replay 不一定产生真实文件副作用。除 Replay 外，使用 stub arXiv Client 和临时目录验证：

- save_to_file=false 不创建文件。
- save_to_file=true 使用工具约定路径创建文件。
- 两种情况下返回的论文结果符合工具契约。

### 3.2 缓存查询工具

工具：get_paper_cache_status。

Description 必须明确：

- 它只读取当前论文的下载 / 翻译缓存状态。
- 它不会启动 PDF 下载。
- 它不会启动论文翻译。
- 用户说“只查缓存”时，只调用该工具。
- 用户说“下载后查缓存”时，先下载，再查缓存。
- 用户只说“下载”时，不擅自增加缓存查询。

### 3.3 其他工具说明

检查并确保工具描述能够区分：

- download_arxiv_pdf：下载 PDF。
- get_paper_content：读取论文正文 / 指定章节。
- get_paper_cache_status：只读缓存状态。
- translate_arxiv_pdf：提交异步翻译任务。
- get_translated_content：读取已经 READY 的译文。
- extract_paper_figures：从论文 PDF 提取图表。
- analyze_figure：分析已抽取的图表。
- summarize_paper：生成摘要 / 总结。

不要让工具描述暗示一个工具会自动完成另一个工具的职责。

---

## 4. 评分规则：比较实际生效参数，不惩罚默认值等价写法

用户关心的是工具序列是否正确、参数是否真正改变了工具行为，而不是 JSON 是否逐字段长得和专家答案一模一样。

### 4.1 工具序列

- 工具选择和顺序必须符合任务要求。
- 多余的工具调用、重复操作、漏掉必需步骤都应判为序列错误。
- “只下载”“下载后查缓存”“只查缓存”是三个不同任务，不能互相替代。

### 4.2 参数

按工具 Schema / 函数默认值构造 effective arguments 后比较：

- 可选字段省略且实际默认值符合用户要求：参数正确。
- 用户明确禁止某个默认副作用时，模型必须覆盖默认值。例如默认 save_to_file=true，但用户要求不保存，必须使用 false。
- 用户明确要求 keep_dual=true 时，不能用默认 false。
- 用户没有指定 output_path，且工具默认路径符合任务要求，不因省略而扣分。
- save_to_file=false 时，output_path 不影响行为；null 与省略应视为语义等价。
- 未知参数、错误类型、非法枚举值仍属于参数 / Schema 错误。
- 不能因为模型少传一个可选默认字段就判错；也不能因为模型多传了一个会改变行为的参数而放过。

### 4.3 摘要章节的语义边界

当前严格评测要求 section="abstract"，但模型 section=null 的默认结果包含标题和摘要。

为避免任务语言歧义：

- “读取标题和摘要”对应默认 section=null。
- “只读取摘要章节，不要其他章节”对应 section="abstract"。
- 训练数据使用明确措辞，不制造模糊监督。
- 评分器对语义范围单独测试，不要把包含额外标题的默认输出无条件判成与“只读摘要”完全等价。

### 4.4 Final Answer

- 任务步骤和参数都正确，但最终回答重复论文 ID / PDF READY 信息并触及生成上限，属于终止 / 输出设计问题。
- 训练目标使用简短完整的最终回答，每个结果只汇报一次。
- 不要在 Final Answer 中重复整个历史 Observation。
- 检查 max_new_tokens、EOS / turn-ending token 和停止条件，避免把运行时长度设置问题全部归因于模型。
- 评分器必须分别记录工具执行完成和最终回答完成，不能只看文本里是否出现某个关键词。

---

## 5. SFT 数据划分与旧数据复用

### 5.1 唯一规范切分

本轮以 data/splits/v4_86.json 为规范文件：

- Train：55 个父任务。
- Dev：8 个任务。
- IID Test：19 个任务。
- OOD Test：4 个任务。

当前仓库的 v7_86.json 与 v4_86.json 的任务 ID 划分一致，但本轮固定使用 v4_86.json，避免混淆版本。

### 5.2 旧 V3 / V4 数据怎么处理

| 文件 | 历史内容 | 本轮处理 |
|---|---|---|
| data/sft/sft_v3_t5_parametric_seed.jsonl | 249 行专家决策，86 个派生任务，32 个 Train 父任务 | 只复用任务定义、参数实例、setup 和专家 Steps |
| data/sft/sft_v3_t5_parametric_linguistic.jsonl | 2988 行，6 种任务包装 × 2 种 Thought 表达 | 不直接训练，不做 12 倍语言扩增 |
| data/sft/sft_v4_multiturn.jsonl | 2508 行旧 ReAct 多轮重建数据 | 不直接训练 |
| data/sft/sft_v4_multiturn_6144.jsonl | 2496 行旧 ReAct 长度过滤数据 | 不直接训练 |

旧 V3 Manifest 指向 v3_81 split 和旧 Snapshot；本轮不能直接复用旧 Observation。

处理原则：

1. 检查 249 行 Seed JSONL 是否实际存在。若不存在，根据现有 TaskSpec 和旧脚本重建，不得仅凭 Manifest 假设文件存在。
2. 只保留 parent_task_id 属于 v4_86.train 的任务。
3. 复用其专家任务定义和合法参数变体。
4. 使用本轮当前 Snapshot 重新执行专家动作，重新获取 Observation。
5. 补充 v4_86 Train 中尚未覆盖的任务，特别是译文读取任务：
   - trread_ai5_default
   - trread_ai5_null_page3
   - trread_cl5_default
   - trread_cv5_page2
6. 不直接使用旧 ReAct Thought / Action 文本作为训练目标。

### 5.3 测试集隔离

- Dev / IID Test / OOD Test 的原始任务文本、精确参数组合、setup 和标准答案不得直接复制到 SFT。
- 可以分析它们暴露的错误机制，但必须从 Train 父任务构造新的任务实例、参数组合或 Session 状态。
- SFT-only 能力边界任务单独标记，不加入 Benchmark split。
- 每个新增任务记录 parent_train_task_id、source_badcase_id 和 error_pattern。
- 生成数据前对 task text、setup、steps 和归一化参数做去重 / 泄漏检查。
- Dev 中的以下评测继续保留，不进入训练：
  - 全 CS、近 7 天、最多 5 篇、明确不保存。
  - AI、近 7 天、最多 5 篇 setup 后下载第 1 篇并查缓存。

---

## 6. Badcase 驱动的任务构造

训练任务不是复制模型错误输出，而是从错误中提取行为缺口，再生成新的正确专家任务。每个训练任务必须有可验证的专家行为和真实 Observation。

### 6.1 首批约 21 条对照轨迹

#### A. 保存开关：3 对

目的：让模型把检索范围和文件副作用当成两个独立条件。

1. Train: constraint_search_only
   - 全 CS、近 3 天、最多 4 篇。
   - 对照一：明确不要写文件，专家调用 aspect="*"、save_to_file=false、output_path=null。
   - 对照二：明确保存检索结果，专家调用 aspect="*"、save_to_file=true。

2. Train: search_CV_14d_3
   - CV、近 14 天、最多 4 篇。
   - 分别构造不保存 / 保存。

3. Train: search_CL_7d_5
   - CL、近 5 天、最多 3 篇。
   - 分别构造不保存 / 保存。

共 6 条候选轨迹。

#### B. 下载与缓存：3 组、每组 3 个分支

目的：明确区分只下载、下载后查缓存、只查缓存。

每组使用相同论文对象和候选状态，只改变用户要求：

- 只下载：download_arxiv_pdf。
- 下载后查缓存：download_arxiv_pdf → get_paper_cache_status。
- 只查缓存：setup 先建立同一论文对象，当前任务只调用 get_paper_cache_status。

候选：

1. Train: constraint_download_only / constraint_cache_only / state_cache_before_dl
   - CL 近 14 天、5 篇候选、第 2 篇。
   - 候选论文 ID：2611.18527v1。

2. Train: constraint_download_only / constraint_cache_only / state_cache_before_dl
   - CV 近 10 天、4 篇候选、第 3 篇。
   - 候选论文 ID：2608.14539v1。

3. Train: constraint_download_only / constraint_cache_only / state_cache_before_dl
   - LG 近 3 天、5 篇候选、第 4 篇。
   - 候选论文 ID：2611.18559v1。

共 9 条候选轨迹。所有 ID、候选列表位置和缓存状态必须在当前 Snapshot 中重新核验。

#### C. 允许子目标与不支持操作：1 对

使用同一已有论文对象：

- “查一下这篇论文的下载和翻译缓存”：只调用 get_paper_cache_status。
- “只把原 PDF 移动到另一个目录”：当前工具不支持文件移动，应零调用并解释能力边界。

不允许为了完成不支持的移动操作，擅自下载、读取或翻译论文。

#### D. 相同指代、不同 Session 状态：1 对

使用相同请求：“读出先前处理过的那篇论文摘要。”

- 空 Session：没有论文列表和 last_active，零调用并说明缺少论文对象。
- 有 Session：setup 已检索 AI 候选并处理第 2 篇，模型看到真实状态；直接调用 get_paper_content(ref=null)，返回摘要。

候选 AI 第 2 篇 ID：2608.14530v1。执行前核验 Snapshot。

#### E. 能力范围区分：1 对

使用同一已下载论文：

- “列出 PDF 中的图表”：若当前 Snapshot 有图表记录，按需 extract_paper_figures。
- “只给 PDF 加水印”：当前工具没有 PDF 编辑能力，应零调用并说明原因。

图表候选可使用 CV 第 1 篇 2608.14546v1；必须核验图表记录。

以上首批约 21 条轨迹。数字是候选规模，不是强制凑数指标。若 Snapshot 缺键或任务不满足前提，替换候选并记录原因，不得编造 Observation。

### 6.2 根据上一轮 SFT 的 29 个失败试次补充

首批 21 条之后，还要覆盖以下行为。使用 v4_86 Train 中的父任务和其他未用于评测的论文 / 参数组合构造新实例。

#### A. 工具完成后的最终回答：3 类

针对缓存、翻译、摘要：

- 查缓存后：简洁汇报下载和翻译状态。
- 翻译 PENDING：只汇报任务已受理和 PENDING，不声称翻译完成。
- 摘要成功：交付实际摘要正文，不重复论文 ID 和 PDF READY 历史信息。

目标 Completion 要短、完整，并包含正确的 Assistant 结束标记。检查生成预算，避免重复输出导致截断。

#### B. 活跃论文 + 双语选项：1 对

使用已下载且成为 last_active 的论文，对照：

- 用户说“翻译刚下载的论文，并保留中英文”：ref=null、keep_dual=true。
- 用户说“翻译刚下载的论文，只保留中文”：ref=null、keep_dual=false。

不得重新搜索、重新下载或把 ref=null 改成 ref=1。用户明确的 keep_dual 值必须覆盖默认值。

#### C. 默认章节与显式摘要章节：1 对

使用有完整 Snapshot 的其他 Train 论文：

- “读取标题和摘要”：按工具默认章节语义处理，通常 section=null。
- “只读取摘要章节，不要其他章节”：section="abstract"。

任务文本必须明确范围，不能用模糊的“读摘要”制造不确定监督。

#### D. 多对象缓存查询

构造 2 篇 / 3 篇论文的缓存检查任务：

- 用户要求检查所有指定论文时，每个对象都必须检查。
- 不允许只检查第一篇就结束。
- 最终回答分别报告每篇论文状态。
- 使用不同 Train 论文和不同列表位置，不复制测试任务的原始组合。

#### E. 标题片段定位

构造一个候选列表，其中第 1 篇是干扰项，目标论文位于第 3 篇，且标题包含唯一片段（例如 Learning State）。

用户要求下载标题包含该片段的论文时，应使用标题片段 ref 或经过工具契约允许的精确 ID；不能默认下载 ref=1，也不能无必要地重新搜索。

#### F. 合法边界与越界拒绝

在相同候选列表下构造相邻对照：

- 合法最后一个序号：执行对应操作。
- 超出列表的下一个序号：零调用，说明当前列表不包含该引用。

不得在越界后擅自扩大检索范围或改用其他论文。

### 6.3 任务 Catalog 字段

创建 data/sft/native_sft_task_catalog.jsonl，每条至少包含：

~~~json
{
  "task_id": "native_sft_...",
  "data_source": "core",
  "parent_train_task_id": "constraint_download_only",
  "source_badcase_id": "state_dl_then_cache",
  "error_pattern": "missing_required_cache_check",
  "task": "用户可见的自然语言任务",
  "setup": [],
  "steps": [],
  "terminal_mode": "completed",
  "terminal_reason": null,
  "decision_notes": []
}
~~~

data_source 取值：

- core：当前 v4_86 Train 的基础任务。
- error_targeted：从 Badcase 错误机制派生的新训练实例。
- sft_only：不加入 Benchmark split 的能力边界 / 额外训练任务。
- recovery：历史中包含真实工具错误 Observation，监督模型如何修正下一步。

每条 error_targeted / recovery 必须记录 source_badcase_id 和 error_pattern。

---

## 7. Assistant Content：简短状态说明，不是长篇 CoT

### 7.1 训练目标

保留自然语言 Content。不要只训练 Tool Call。

每轮工具调用前的 Content 通常一两句话，说明：

- 当前已经确认的状态；
- 用户还要求什么；
- 为什么下一步应执行这个工具。

Content 是用户可见的普通 Assistant 文本，不是隐藏推理，不要求输出完整内部分析。

Final Answer 必须依据真实 Observation，简短汇报完成情况和阻塞。

### 7.2 生成方式

实现类似接口：

~~~python
build_assistant_content(
    task,
    current_state,
    previous_observations,
    next_action,
    decision_note=None,
)
~~~

规则：

1. 优先使用人工审核的 decision_note。
2. 普通任务用确定性模板，根据 Task、Session 状态、真实 Observation 和下一步专家动作生成。
3. 不默认调用另一个 LLM 自由生成长篇推理。
4. 不得在工具执行前声称操作成功。
5. 不得在 PENDING 时声称翻译完成。
6. 不得在空 Session 时声称已有论文列表或活跃论文。
7. 不要每轮机械重复“好的，我来帮你处理”。
8. 不重复整个用户请求或完整历史 Observation。
9. 对多子目标任务说明已完成和剩余的相关子目标。
10. 工具失败后的 Content 应说明失败事实和下一步修正，不得把失败说成成功。

### 7.3 Content 示例

| 场景 | 推荐 Content |
|---|---|
| 首次检索 | 当前会话没有候选论文，我先按用户指定的分类和时间范围检索。 |
| 已检索，下载第 2 篇 | 已获得 5 篇论文，用户指定第 2 篇，接下来直接下载该论文。 |
| 下载后查缓存 | PDF 已下载成功，用户还要求核对缓存状态，我现在查询缓存。 |
| 只下载 | PDF 已下载成功，用户没有要求额外操作，接下来汇报下载结果。 |
| 下载后翻译 | PDF 已就绪，用户还要求翻译这篇论文，我现在提交翻译任务。 |
| PENDING | 翻译任务已受理，目前为 PENDING；本次提交已完成。 |
| 空 Session 指代 | 当前会话没有可解析的论文对象，无法确定“先前处理过的那篇”指什么，因此不调用工具。 |
| 不支持操作 | 当前工具不支持移动或编辑 PDF，因此无法执行该操作，不调用无关工具。 |
| 多对象缓存 | 第 1 篇状态已确认，接下来检查用户指定的第 2 篇。 |

Content 必须来自当前任务和真实状态。不要把示例原样复制到所有样本。

---

## 8. Native 专家轨迹与 SFT 样本格式

### 8.1 完整轨迹

创建 data/sft/native_sft_trajectories.jsonl。

每条轨迹保存完整交互：

- System / User。
- Assistant Content + assistant.tool_calls。
- Tool Message。
- 下一轮 Assistant Content + tool_calls。
- 最终 Assistant Content。

要求：

1. 每条轨迹使用独立 Session。
2. setup 真实执行，模型可见状态根据实际执行回执生成。
3. 专家 Steps 由审核过的 TaskSpec 提供。
4. 工具 Schema 直接使用当前 Registry 注入的定义。
5. 使用当前 Base Benchmark 对应 Snapshot，核对 SHA256。
6. 通过与 Native Agent 一致的工具执行 / side-effect 路径取得 Observation。
7. tool_call_id 唯一，且与后续 Tool Message 正确配对。
8. 每轮最多一个工具调用。
9. PENDING 处理遵循本文件规则；允许明确要求且独立可执行的后续步骤，但禁止重复提交和无意义轮询。
10. blocked 任务应生成零工具调用的最终拒绝回答。
11. 专家步骤失败、Snapshot 缺键、工具返回缺失、状态不一致时，整条轨迹标记失败并排除；不得静默截断后继续。

输出：

- data/sft/native_sft_trajectories.jsonl
- data/sft/native_sft_content_preview.jsonl

### 8.2 决策样本

创建 data/sft/native_sft_train.jsonl。

每个 Assistant 决策时刻生成一条样本：

~~~json
{
  "prompt": [
    "system message",
    "user message",
    "previous assistant tool calls",
    "previous tool messages"
  ],
  "completion": [
    "current assistant message"
  ],
  "tools": [],
  "task_id": "native_sft_...",
  "trajectory_id": "...",
  "trajectory_step": 0,
  "data_source": "error_targeted",
  "error_pattern": "missing_required_cache_check"
}
~~~

训练目标：

- 当前 Assistant Content + Tool Call：两部分都参与 Loss。
- 当前 Final Answer / refusal Content：参与 Loss。
- System、User、Tools Schema：不参与 Loss。
- 历史 Assistant 消息：不参与当前样本 Loss。
- 历史 Tool Observation：不参与 Loss，只作为上下文。
- 不生成 Thought 字段。
- 不把 Tool Result 当成模型生成目标。

使用 Qwen3.5-2B-Base 对应 Chat Template 渲染结构化消息。不得手工拼接 Qwen 的 XML 工具标记。

Codex 必须核对实际安装的 Transformers / TRL 版本及数据接口，确认：

- function.arguments 的数据类型符合当前 Template / Trainer 要求。
- assistant content 与 tool_calls 同时存在时可以正确渲染。
- Tool Message 与 tool_call_id 配对正确。
- 当前 Assistant Content 和 Tool Call 都有有效 labels。
- Prompt、历史消息和 Tool Result 均被正确 mask。
- Final Answer 的标签和结束 Token 正确。
- 不意外生成 <think> 内容。

如果 TRL 内置工具数据格式不能保证正确的 Loss Mask，使用自定义 Token Labels。不要假设 completion_only_loss 自动处理所有工具消息和多轮历史。

---

## 9. 数据审计与产物

实现：

- scripts/build_native_sft_data.py
- scripts/audit_native_sft_data.py

输出：

- data/sft/native_sft_task_catalog.jsonl
- data/sft/native_sft_trajectories.jsonl
- data/sft/native_sft_content_preview.jsonl
- data/sft/native_sft_train.jsonl
- data/sft/native_sft_train.jsonl.manifest.json
- reports/native_sft_data_audit.md

审计必须覆盖：

### 9.1 数据来源和泄漏

- core / error_targeted 的 parent_task_id 属于 v4_86.train。
- sft_only 有明确标记，不加入 Benchmark split。
- 不包含 Dev / IID / OOD 原始任务、精确参数组合或答案。
- source_badcase_id 和 error_pattern 可追溯。
- 任务文本、setup、steps、参数组合进行去重和泄漏检查。

### 9.2 轨迹正确性

- setup 成功并与模型可见状态一致。
- 每一步专家工具名、参数和顺序正确。
- 每条 Observation 来自当前 Snapshot / 当前环境的真实回执。
- 论文 ID、版本号、ref 解析结果正确。
- 下载 / 翻译 / 缓存状态与真实执行一致。
- 完整完成所有专家步骤。
- PENDING 不被误写为 READY。
- blocked 任务没有调用无关工具。
- 工具错误和 Snapshot miss 单独记录。

### 9.3 Content 正确性

- Content 依据用户请求、Session 和真实 Observation。
- 不在操作前声称成功。
- 不重复已完成的动作。
- 不漏掉用户明确的禁止条件、范围限制和步骤顺序。
- 不在最终回答重复论文 ID / READY 信息导致冗长或截断。
- 多子目标分别汇报。
- PENDING、READY、FAILED 区分准确。
- 不生成隐藏 CoT 或冗长模板话术。

### 9.4 长度和重复

- 使用当前 Qwen3.5-2B-Base Tokenizer 和实际 Chat Template 统计完整样本长度。
- 6144 是本轮训练样本长度预算，不是模型架构的上下文上限。
- 不得截断 Completion、Tool Call、结束 Token。
- 超长时先压缩历史 Observation 的展示内容，但必须保留支撑后续决策的论文标题、ID、状态和引用顺序；压缩后重新检查语义。
- 无法安全压缩时排除并记录。
- 不再做 6 种任务包装 × 2 种 Thought 的机械扩增。

### 9.5 Manifest

至少记录：

- split 文件路径和 SHA256。
- Train TaskSpec ID 列表。
- Snapshot SHA256。
- Tool Schema SHA256。
- Prompt 版本 / SHA256。
- 代码 commit。
- core / error_targeted / sft_only / recovery 数量。
- 任务数量、完整轨迹数量、决策样本数量。
- 每个工具调用数量。
- Tool Call / Final Answer / refusal 数量。
- 各类错误模式覆盖数量。
- 被排除轨迹和原因。
- Token 长度分布。
- 输出文件 SHA256。

审计不通过不得启动正式训练。

---

## 10. 实验执行顺序

### Step 1：修正 Prompt 和工具描述

- 按第 2 节组织 Prompt 三部分。
- Tool Injection 保持现状。
- 按第 3 节补齐 save_to_file 和 cache 工具 Description。
- 完成默认参数和用户明确约束的测试。

### Step 2：Prompt-only A/B

在不训练模型的情况下，使用同一个 Qwen3.5-2B-Base 比较：

- A：现有 Prompt。
- B：新三部分 Prompt。

固定模型权重、Tool Schema、Snapshot、任务、种子、解码参数和最大迭代次数。只改变用户约束与任务完成规则部分。

重点测试：

- 全 CS + 不保存。
- 下载后查缓存。
- 只下载 / 只查缓存。
- 空 Session / 已有 Session。
- 用户禁止额外操作。
- 多子目标完成。
- PENDING 后独立后续步骤。
- 不支持操作的零调用拒绝。

Dev 的全 CS 不保存任务和 AI setup 下载后查缓存任务保留为评测，不加入 SFT。每族至少保留一组未参与训练的诊断对照。

输出 Prompt A/B 报告和完整逐轮轨迹。若新 Prompt 引入回退，先修正规则，不要直接进入 SFT。

### Step 3：构造任务 Catalog

- 读取 v4_86 的 55 个 Train 父任务。
- 检查旧 V3 249 行 Seed 是否存在。
- 复用旧任务定义、参数变体和专家 Steps。
- 生成第 6 节所列约 21 条首批对照轨迹及上一轮 SFT 失败补齐任务。
- 逐个验证 Snapshot 中的论文、章节、图表和状态。
- 输出任务 Catalog 和候选统计。

### Step 4：专家执行与 Native 数据生成

- 生成完整 Native 轨迹。
- 生成 Assistant Content Preview。
- 拆分决策样本。
- 运行完整数据审计。
- 检查 Chat Template / Loss Mask。
- 输出 Manifest。

### Step 5：人工审核门槛

将以下内容汇总给用户：

- 新增任务列表和来源。
- 每类任务的专家工具序列和参数。
- 每条轨迹的 Content Preview。
- PENDING / refusal / partial completion 示例。
- 被排除任务及原因。
- 数据类别和 Token 统计。
- 审计结论。

只有当数据审计全部通过，且 Content / 专家动作没有明显语义错误时，才继续正式 SFT。遇到无法从工具定义、TaskSpec 或真实 Observation 确定的语义，不得自行猜测，列出具体问题。

### Step 6：Native SFT

- Base Model 必须是 Qwen/Qwen3.5-2B-Base。
- 不加载官方 Qwen3.5-2B 后训练权重或旧 ReAct Adapter。
- 使用 BF16 LoRA，参考当前已验证训练配置。
- 先 Smoke Test，检查 Loss、LoRA 更新、Content 标签、Tool Call 标签、Final Answer 结束 Token 和 Adapter 加载。
- Smoke Test 通过后进行正式训练。
- 保留训练日志、配置、Adapter、数据 Manifest 和 checkpoint。
- 不同时大幅改变模型、LoRA、学习率、Batch、Max Length 等多个因素。

### Step 7：同协议评测

使用最终选定 Prompt、同一 Tool Schema、Snapshot、Benchmark、解码配置和试次种子，配对评测：

- 修复后的 Base。
- Base + Native SFT。

至少报告 Dev / IID / OOD 和总体结果。除 Strict Success 外，必须报告：

- 工具序列准确率。
- effective-argument 准确率。
- 论文引用准确率。
- 多子目标完整率。
- save_to_file 约束遵守率。
- 下载后缓存查询遵守率。
- 重复调用率。
- PENDING 正确终止率。
- refusal / blocked 零调用成功率。
- Final Answer 截断和重复率。
- 工具执行错误及 Snapshot miss。

保留逐任务、逐试次完整轨迹。不要只报告总体百分比。

### Step 8：GRPO 决策

本文件不要求立即训练 GRPO。

先分析 Native SFT 的剩余错误。如果仍存在可通过奖励优化的策略错误，且 Rollout / Reward / 组内方差可靠，再单独制定 Native GRPO 计划。

---

## 11. Codex 最终交付清单

### Prompt / Agent / Scorer

1. 更新后的 NATIVE_TOOL_SYSTEM_PROMPT。
2. Tool Schema / Description 修正。
3. save_to_file、缓存查询、用户限制、步骤顺序、PENDING、拒绝和 Final Answer 的回归测试。
4. 参数 effective-value 评分测试。
5. Prompt-only A/B 报告。

### SFT 数据

6. data/sft/native_sft_task_catalog.jsonl
7. data/sft/native_sft_trajectories.jsonl
8. data/sft/native_sft_content_preview.jsonl
9. data/sft/native_sft_train.jsonl
10. data/sft/native_sft_train.jsonl.manifest.json
11. scripts/build_native_sft_data.py
12. scripts/audit_native_sft_data.py
13. reports/native_sft_data_audit.md

### 训练和评测

14. Native SFT 训练入口 / 启动脚本。
15. Smoke Test 报告。
16. 正式 SFT Adapter 和训练日志。
17. Base vs SFT 同协议 Benchmark 报告。
18. 完整逐任务 / 逐试次轨迹。

所有旧 ReAct 数据、旧 SFT Adapter、GRPO 代码、Benchmark 历史结果和 Snapshot 均保留，不覆盖。

**最终执行原则：先修 Prompt / Schema / Scorer，再做 Prompt-only 对照；随后从 Train 任务和 Badcase 行为构造新的 Native 专家轨迹，训练 Assistant Content + Tool Call + Final Answer。不要再做无意义的语言扩增，不要让模型学习错误的工具契约，也不要把评分器问题转化成训练目标。**
