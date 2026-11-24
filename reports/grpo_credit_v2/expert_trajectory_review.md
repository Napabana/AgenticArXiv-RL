# 失败任务的专家轨迹审阅

原提示下 E4 在 31 道题、93 个试次中严格成功 72 次。21 次失败集中在 8 道题，其中 6 道为 0/3。另列 SFT 或 E0–E3 曾失败而 E4 通过的 4 道题，以及 Train 中的 3 道受阻题。每一道都在固定 Snapshot 上用脚本专家重放了任务，核对了语义目标论文 ID。全程无需模型生成。任务定义给出应调用的工具和参数，文中的专家答复是本次脚本回放的具体措辞；正常完成任务的评分器不会要求逐字复述。

## E4 的 21 次失败

| 任务 | 切分 | SFT 成功 | E4 成功 | E4 失败试次（1 起算） |
|---|---|---:|---:|---|
| 约稿邮件 | dev | 0/3 | 0/3 | 1、2、3 |
| CL 第 3 篇，查缓存后翻译 | iid_test | 0/3 | 0/3 | 1、2、3 |
| CV 第 1 篇，查缓存后翻译 | ood_test | 0/3 | 0/3 | 1、2、3 |
| 只翻译并保留双语 | iid_test | 0/3 | 0/3 | 1、2、3 |
| 按标题片段下载 | iid_test | 0/3 | 0/3 | 1、2、3 |
| 5 篇中请求第 20 篇 | iid_test | 0/3 | 0/3 | 1、2、3 |
| 下载、抽图并描述图 1 | iid_test | 3/3 | 1/3 | 1、2 |
| 强制重新下载 | dev | 3/3 | 2/3 | 3 |

三次全部失败的六题中，CL 和 CV 两条查缓存后翻译任务、双语翻译任务，三次的工具、实参、结束动作完全相同。越界下载的三次首错相同，但后续结束方式不同；按标题下载的三次都选错，选中的论文有两种；邮件题有误翻译和错误追问两种路径。以上不要求自然语言最终答复逐字相同。

## 五道受阻题的提示词复核

每格为 3 次试验中的严格成功数。原提示在两种模型上均测完五题。加入立即 blocked 指令后，SFT 测完五题，E4 仅邮件题有完整结果。三道 Train 题只用于诊断，未计入 93 次正式评测。

| 请求 | 切分 | 原提示 SFT | 原提示 E4 | 加入指令 SFT | 加入指令 E4 |
|---|---|---:|---:|---:|---:|
| 给第 1 篇论文作者发约稿邮件 | Dev | 0/3 | 0/3 | 0/3 | 0/3 |
| 从 5 篇中下载第 20 篇 | IID | 0/3 | 0/3 | 0/3 | 未完成 |
| 引用第 0 篇 | Train | 0/3 | 0/3 | 0/3 | 未完成 |
| 指定未知论文 ID | Train | 0/3 | 0/3 | 0/3 | 未完成 |
| 空会话中提到之前的论文 | Train | 3/3 | 3/3 | 0/3 | 未完成 |

邮件题上，两种模型零工具调用各从 1/3 升至 3/3，但新增指令后的六次都追问论文编号。空会话题上，原提示下两个模型都能直接请求澄清，新增指令后 SFT 三次都先调用了工具。第 0 篇题虽然没有成功执行工具，但都尝试传入非法引用并被 Schema 拒绝，不能算作主动停下。逐次回答、实际执行的工具数和终止状态见[受阻题结果](blocked_task_review.json)。本次取回的补测归档 SHA-256 为 `6f96a40574dc2c538fa644db7bbeeb48f5ba713c042034ba8a943fbc36ab92d1`。

### 供审阅的提示词草案

以下草案按当前专家要求处理两步请求。草案未实测，本次审阅未修改默认提示词。

> 调用前先核对两个问题：用户最终要求的动作，哪项现有工具能实际完成？这个动作需要的对象和参数，能否从当前会话及已有工具结果确定？
>
> 如果没有工具能完成必需动作，立即结束本轮，零工具调用。直接对用户说清楚做不了的具体动作和工具能力限制。即使某个前置步骤可行，也不要用它代替用户要求的结果。
>
> 如果工具能完成动作，再核对引用和参数。优先使用已有结果。需要澄清时，指出已有证据和具体缺口，只询问能解除该阻塞的信息。列表里没有出现某个对象，只能说明当前列表未列出它。
>
> 遇到阻塞，用自然语言给出最终回答后停止（blocked），不要声称成功。能够执行时，按用户要求继续。

## 邮件任务设计上的取舍

用户的请求包含先查找联系方式、再发送邮件两个动作。当前标准把它整体标为 unsupported_capability，预期第一次 Assistant 输出就是解释原因并 FINISH，期望工具序列为空。Setup 中已有 5 篇论文，第 1 篇已确定；原始轨迹没有显示模型要求补编号后真的能继续查邮箱。工具表有论文检索和论文内容读取，未提供查通讯邮箱或发邮件的专用接口。读取标题或摘要也不保证取得经过核验的邮箱。

如产品希望助手先完成独立可行的前半部分，应在任务中说明能取得什么中间结果、如何验证邮箱及如何判部分完成。现有 Strict 会把任何工具调用判为序列错误。SFT 里有一个真实混合示范：先查 PDF 缓存，再说明不能加水印。把所有带能力缺口的请求都一律零调用，会与这种示范发生冲突。

## E4 的真实训练组

25 步各采 G=4，98 条有效、2 条环境异常被屏蔽。14 组在有效轨迹中有成有败（两组各有一条无效）；8 组 4/4 成功，3 组 0/4。3 个全失败组的不同动作序列数分别为 2、3、2。版本号下载→查缓存→翻译组出现了局部对照，仍有有效 Advantage；无效序号、搜索→下载→摘要两组最终策略 Advantage 全为零。10/25 组的有效决策全部为零 Advantage；这没有计入 KL 项。仅 1/25 组的所有有效 Assistant token 序列完全一致。文字多样仍可能对应同一个错误。

例如训练组 0 的四条版本号下载轨迹全部失败，其中三条第一步参数错误，第一步 effective_local 约 −0.25；另一条第一步正确获约 +0.75，随后选错工具。该组可以区别第一步优劣，但没有提供完整正确解的正例。训练组 12 的无效序号任务则四条全败，尽管采到三种动作序列，策略 Advantage 仍全部为零。该核验以 E4 保存的逐决策日志为准。

模型没有从 25 步训练取得总体严格成功净收益。训练集的 25 组不是在上述 8 道留出题上采得，所以不能用单题三次评测直接断言该题训练时四条轨迹也相同。

## 严格成功的终止判据

取最后 FINISH 的回答文本，按 TaskSpec 原因码调用词表或专门规则，再结合工具、参数、引用、结束及错误标志计算 Strict。以下是直接调用当前评分函数的独立示例，均构造零工具调用、正常 FINISH 的邮件题轨迹。

| 回答 | 原因判定 | Strict |
|---|---|---:|
| 当前工具无法发送约稿邮件，无法代为联系作者。 | explained_block | 1 |
| 当前会话没有指定论文引用。请提供论文编号。 | ambiguous | 0 |
| 我手头的接口仅供论文检索，寄信请使用您自己的账户。 | ambiguous | 0 |
| 因为缺少论文编号，我无法发送邮件。 | explained_block | 1 |

末行说明仅共现“无法”和“邮件”等词也能通过，即使把障碍归错；委婉同义表述也可能漏判。真实六条追问编号的回答另有实质错误，列表中已存在目标论文。

## E4 出现失败的 8 道题

### 约稿邮件 · `infeasible_unsupported_action`

用户原话：**把第1篇论文的作者邮箱找出来发一封约稿邮件**

- 请求前的 Setup：`get_recently_submitted_cs_papers(aspect="AI", days=7, max_results=5)`。返回 5 篇，1=2608.14539v1、2=2608.14530v1、3=2608.14528v1、4=2611.18553v1、5=2611.18543v1。
- 专家第一轮零工具调用。阻塞原因码为 `unsupported_capability`。
- 专家随后 FINISH，回答原文如下。

> 当前工具无法发送约稿邮件，无法代为联系作者。

严格成功次数：SFT 0/3，E0 0/3，E1 0/3，E2 0/3，E3 0/3，E4 0/3。

三次失败。前两次提交翻译并准确报告 PENDING；第三次零调用，却追问已有的论文编号。标准答案要求解释无法发邮件。

### CL 第 3 篇，查缓存后翻译 · `chain_cl5_dl3_tr3`

用户原话：**搜索最近7天自然语言处理(cs.CL)论文5篇，下载第3篇，确认缓存后翻译它**

- 请求前无 Setup。
- 专家步骤 1：`get_recently_submitted_cs_papers(aspect="CL", days=7, max_results=5)`。
- 专家步骤 2：`download_arxiv_pdf(ref=3)`，应解析为 `2611.18514v1`。
- 专家步骤 3：`get_paper_cache_status(ref=3)`，应解析为 `2611.18514v1`。
- 专家步骤 4：`translate_arxiv_pdf(ref=3)`，应解析为 `2611.18514v1`。
- 专家随后 FINISH，回答原文如下。

> 2611.18514v1 的 PDF 已就绪，译文未就绪。
> 2611.18514v1 的翻译任务已受理（PENDING，任务编号 mocktask0001），译文尚未就绪。

严格成功次数：SFT 0/3，E4 0/3。

三次均正确检索、下载第 3 篇，随后直接提交翻译。三次均漏掉用户点名的缓存确认。

### CV 第 1 篇，查缓存后翻译 · `multi_cv5_dl1_cache_tr`

用户原话：**检索计算机视觉(cs.CV)最近7天5篇论文，下载第1篇，确认缓存后翻译**

- 请求前无 Setup。
- 专家步骤 1：`get_recently_submitted_cs_papers(aspect="CV", days=7, max_results=5)`。
- 专家步骤 2：`download_arxiv_pdf(ref=1)`，应解析为 `2608.14546v1`。
- 专家步骤 3：`get_paper_cache_status(ref=1)`，应解析为 `2608.14546v1`。
- 专家步骤 4：`translate_arxiv_pdf(ref=1)`，应解析为 `2608.14546v1`。
- 专家随后 FINISH，回答原文如下。

> 2608.14546v1 的 PDF 已就绪，译文未就绪。
> 2608.14546v1 的翻译任务已受理（PENDING，任务编号 mocktask0001），译文尚未就绪。

严格成功次数：SFT 0/3，E4 0/3。

三次均正确下载第 1 篇，接着提交翻译，均漏掉确认缓存。

### 只翻译并保留双语 · `constraint_translate_only`

用户原话：**只把刚下载的那篇论文翻译成双语版本，不要重新搜索，也不要检查缓存状态**

- 请求前的 Setup：`get_recently_submitted_cs_papers(aspect="AI", days=7, max_results=5)`。返回 5 篇，1=2608.14539v1、2=2608.14530v1、3=2608.14528v1、4=2611.18553v1、5=2611.18543v1。
- 请求前的 Setup：`download_arxiv_pdf(ref=1)`。目标 2608.14539v1，状态 READY。
- 专家步骤 1：`translate_arxiv_pdf(ref=null, keep_dual=true)`，应解析为 `2608.14539v1`。
- 专家随后 FINISH，回答原文如下。

> 2608.14539v1 的翻译任务已受理（PENDING，任务编号 mocktask0001），译文尚未就绪。

严格成功次数：SFT 0/3，E4 0/3。

三次都选对了论文和翻译工具，唯一未通过的检查是参数。用户要求双语版本，模型三次均使用 keep_dual=false。显式论文 ID 与当前活动论文等价。

### 按标题片段下载 · `ref_stress_learning_state`

用户原话：**下载标题里包含 "Learning State" 的那篇论文**

- 请求前的 Setup：`get_recently_submitted_cs_papers(aspect="AI", days=7, max_results=5)`。返回 5 篇，1=2608.14539v1、2=2608.14530v1、3=2608.14528v1、4=2611.18553v1、5=2611.18543v1。
- 专家步骤 1：`download_arxiv_pdf(ref="Learning State")`，应解析为 `2608.14528v1`。
- 专家随后 FINISH，回答原文如下。

> 2608.14528v1 的 PDF 已就绪。

严格成功次数：SFT 0/3，E4 0/3。

三次都调用了下载工具，第一次选第 2 篇，后两次选第 1 篇。标题命中的目标为第 3 篇 2608.14528v1。

### 5 篇中请求第 20 篇 · `infeasible_index_out_of_range`

用户原话：**把第20篇论文下载下来**

- 请求前的 Setup：`get_recently_submitted_cs_papers(aspect="AI", days=7, max_results=5)`。返回 5 篇，1=2608.14539v1、2=2608.14530v1、3=2608.14528v1、4=2611.18553v1、5=2611.18543v1。
- 专家第一轮零工具调用。阻塞原因码为 `invalid_reference`。
- 专家随后 FINISH，回答原文如下。

> 指定论文序号超出当前候选列表的范围。请提供有效的论文序号。

严格成功次数：SFT 0/3，E4 0/3。

三次的第一错都是直接尝试下载 ref=20。第一次和第三次随后因生成长度结束而解析失败；第二次反复尝试直到轮数上限。

### 下载、抽图并描述图 1 · `analyze_ai5_ref4_desc`

用户原话：**检索最近7天人工智能(cs.AI)论文5篇，下载第4篇，抽出图表并描述第1张图**

- 请求前无 Setup。
- 专家步骤 1：`get_recently_submitted_cs_papers(aspect="AI", days=7, max_results=5)`。
- 专家步骤 2：`download_arxiv_pdf(ref=4)`，应解析为 `2611.18553v1`。
- 专家步骤 3：`extract_paper_figures(ref=4)`，应解析为 `2611.18553v1`。
- 专家步骤 4：`analyze_figure(ref=4, figure_no=1, question="describe")`，应解析为 `2611.18553v1`。
- 专家随后 FINISH，回答原文如下。

> 2611.18553v1 的 PDF 已提取到 8 张图表。
> 已分析指定图表（2611.18553v1）。
> Figure 1. The WeightBench evaluation pipeline.。

严格成功次数：SFT 3/3，E4 1/3。

失败的两次均走完四个工具且实参、引用正确。最后一轮因生成长度而解析失败，没有正常 FINISH。第三次严格成功。

### 强制重新下载 · `opt_force_dl`

用户原话：**强制重新下载第1篇论文**

- 请求前的 Setup：`get_recently_submitted_cs_papers(aspect="AI", days=7, max_results=5)`。返回 5 篇，1=2608.14539v1、2=2608.14530v1、3=2608.14528v1、4=2611.18553v1、5=2611.18543v1。
- 专家步骤 1：`download_arxiv_pdf(ref=1, force=true)`，应解析为 `2608.14539v1`。
- 专家随后 FINISH，回答原文如下。

> 2608.14539v1 的 PDF 已就绪。

严格成功次数：SFT 3/3，E0 3/3，E1 2/3，E2 3/3，E3 3/3，E4 2/3。

前两次通过。第三次声称列表为空并直接结束；Setup 实际已经检索到五篇。

## 其他配置失败而 E4 三次通过的 4 道题

### 全 CS 检索且不保存 · `constraint_search_no_file`

用户原话：**检索最近7天所有计算机科学方向的新论文，最多5篇；只返回结果，不要保存到文件，也不要下载论文**

- 请求前无 Setup。
- 专家步骤 1：`get_recently_submitted_cs_papers(aspect="*", days=7, max_results=5, save_to_file=false)`。
- 专家随后 FINISH，回答原文如下。

> 共检索到 5 篇论文。
> 1. Pairing AIS with RF: Analysis of a Markovian Approach for Maritime Data Fusion（2611.18563v1）
> 2. FETTUCCINE: Fast and efficient brain-to-text decoding on mobile devices（2611.18561v1）
> 3. Training Reasoning Models in Simulation to Natively Use Real Robots（2611.18562v1）
> 4. BAL: Bidirectional Autoregression in Latent Space for Learning Human Movement Representations（2611.18559v1）
> 5. WeightBench: Evaluating Language Models as Weight-Space Programmers（2611.18553v1）
> 按要求仅返回检索结果。

严格成功次数：SFT 3/3，E0 0/3，E1 3/3，E2 3/3，E3 3/3，E4 3/3。

实际失败路径：

- E0，3 次：`get_recently_submitted_cs_papers,FINISH`。失败检查：args。

### 下载后确认缓存 · `state_dl_then_cache`

用户原话：**下载第1篇，然后确认它是否已经缓存**

- 请求前的 Setup：`get_recently_submitted_cs_papers(aspect="AI", days=7, max_results=5)`。返回 5 篇，1=2608.14539v1、2=2608.14530v1、3=2608.14528v1、4=2611.18553v1、5=2611.18543v1。
- 专家步骤 1：`download_arxiv_pdf(ref=1)`，应解析为 `2608.14539v1`。
- 专家步骤 2：`get_paper_cache_status(ref=1)`，应解析为 `2608.14539v1`。
- 专家随后 FINISH，回答原文如下。

> 2608.14539v1 的 PDF 已就绪，译文未就绪。

严格成功次数：SFT 2/3，E0 3/3，E1 3/3，E2 3/3，E3 0/3，E4 3/3。

实际失败路径：

- SFT，1 次：`download_arxiv_pdf,FINISH`。失败检查：tools/args/ref。
- E3，3 次：`download_arxiv_pdf,FINISH`。失败检查：tools/args/ref。

### 读取现有译文第 2 页 · `trread_ro5_page2`

用户原话：**读取第1篇论文中文译文的第2页**

- 请求前的 Setup：`get_recently_submitted_cs_papers(aspect="RO", days=7, max_results=5)`。返回 5 篇，1=2611.18562v1、2=2611.18546v1、3=2611.18505v1、4=2611.18501v1、5=2611.18496v1。
- 请求前的 Setup：`download_arxiv_pdf(ref=1)`。目标 2611.18562v1，状态 READY。
- 请求前的 Setup：`translate_arxiv_pdf(ref=1)`。目标 2611.18562v1，状态 READY。
- 专家步骤 1：`get_translated_content(ref=1, page=2)`，应解析为 `2611.18562v1`。
- 专家随后 FINISH，回答原文如下。

> 已读取 2611.18562v1 中文译文第 2 页。
> Training Reasoning Models in Simulation to Natively Use Real Robots
>
> 可以很好地推广到训练任务之外：它们可以 在思维链中推理的能力对于泛化尤其重要。
> 迁移到未知的机器人、现实世界以及新的指
> 令。在模拟环境中对 LIBERO 和 RoboDojo 进 然而，到目前为止，这种性能仅在最强的专有模型中
> 行训练后，该模型在真实的双臂机器人上在 获得。开源权重模型在相同的基准上成功率较低，并
> 我们的试验中成功率达到 60%，而基础模型 且在一些简单任务上也会失败。一些工作构建了一个
> 则完全未成功。在 LIBERO 上，训练将成功 框架，添加技能和组件，例如分割模型 SAM 3（Cario
> 率从 1.6% 提升到 74.4%，接近 GPT-6 Astra n 等，2026），并将一些动作转换为一小组选项（Yan
> 和 GPT-6.1 Sol 在最大推理努力下的表现。与 g 等，2026c；Lu 等，2026；Zhang 等，2026c；Chu
> 在示例上微调的视觉-语言-动作模型 π0.5 相比 等，2026；Huang 等，2026；Si 等，2026；RoboDojo
> ，我们的模型对任务本身的变化更为鲁棒。 团队，2026）。这样的框架允许开源权重模型完成一
> 在 LIBERO-Pro 上，当为熟悉场景分配新任务 些任务（Yang 等，2026c），但它可能会限制其动作
> 时，π0.5 的成功率仅为 0.75%，而我们的模型 空间，从而使其他任务无法实现。随着模型能力的提
> 成功率达到 57.4%。最后，该模型保留了其通 升，框架中的一些组件也可能被纳入模型本身。
> 用能力，在九个推理、编码和视觉基准上没
> 有可测量的性能下降。这些结果突显了机器
> 计算机使用代理，例如 OpenAI 的代理，是通过在计算 人使用强化学习作为基础模型后训练的一个
> 机任务上进行强化学习（RL）训练的（OpenAI, 2025a; 有前景的组成部分。
> b；Wang 等，2025），而这两种设置之间的高度相似
> 性表明，机器人使用也可以以相同的方式学习。因此
> ，我们使用机器人使用强化学习（RURL）训练一个开
> 放权重推理模型，完全在仿真中进行，并且不依赖示
> 1. 引言 范。为了支持这种大规模的长时程、多轮 RURL，我
> 们构建了 PEAR（Agentic Robotics 后训练引擎），这
> 机器人使用代理（RUA）最近已成为通用机器人控制 是一个专门为训练 RUA 设计的基础设施（第 3.2 节）
> 的一个有前景的方向（Isola，2026；Yang 等，2026c； 。在每一轮中，模型观察相机图像和机器人的状态，
> Zhang 等，2026c；Lu 等，2026）。类似于从屏幕截图 对任务进行推理，并输出下一个末端执行器位置（图 1
> 操作计算机的计算机使用代理（CUA）（OpenAI，20 ）。逆向运动学将该位置转换为关节目标，机器人控
> 25a；Anthropic，2024），RUA 使用一个视觉-语言模 制器执行运动，然后模型接收下一个观察结果。之前
> 型（VLM）来观察机器人的摄像头图像、推理任务并 的轮次会保留在上下文中，每个回合都会获得一个二
> 输出下一个末端执行器的姿态。因为 VLM 直接操作机 元奖励，指示任务是否完成
> 器人，
> （节选）。

严格成功次数：SFT 2/3，E4 3/3。

实际失败路径：

- SFT，1 次：`get_translated_content,ERROR`。失败检查：FINISH/parse。

### CL 第 2 篇，下载后查缓存 · `multi_cl5_dl2_cache`

用户原话：**检索最近7天自然语言处理(cs.CL)论文5篇，下载第2篇，再查它的缓存状态**

- 请求前无 Setup。
- 专家步骤 1：`get_recently_submitted_cs_papers(aspect="CL", days=7, max_results=5)`。
- 专家步骤 2：`download_arxiv_pdf(ref=2)`，应解析为 `2611.18527v1`。
- 专家步骤 3：`get_paper_cache_status(ref=2)`，应解析为 `2611.18527v1`。
- 专家随后 FINISH，回答原文如下。

> 2611.18527v1 的 PDF 已就绪，译文未就绪。

严格成功次数：SFT 2/3，E4 3/3。

实际失败路径：

- SFT，1 次：`get_recently_submitted_cs_papers,download_arxiv_pdf,get_paper_cache_status,ERROR`。失败检查：FINISH/parse。

## Train 中额外的 3 道受阻题

### 下载第 0 篇 · `infeasible_zero_index`

用户原话：**下载第0篇论文**

- 请求前的 Setup：`get_recently_submitted_cs_papers(aspect="AI", days=7, max_results=5)`。返回 5 篇，1=2608.14539v1、2=2608.14530v1、3=2608.14528v1、4=2611.18553v1、5=2611.18543v1。
- 专家第一轮零工具调用。阻塞原因码为 `invalid_reference`。
- 专家随后 FINISH，回答原文如下。

> 指定论文序号超出当前候选列表的范围。请提供有效的论文序号。

### 下载快照之外的显式 ID · `infeasible_unknown_id`

用户原话：**下载 2999.99999v1 这篇论文**

- 请求前的 Setup：`get_recently_submitted_cs_papers(aspect="AI", days=7, max_results=5)`。返回 5 篇，1=2608.14539v1、2=2608.14530v1、3=2608.14528v1、4=2611.18553v1、5=2611.18543v1。
- 专家第一轮零工具调用。阻塞原因码为 `paper_not_found`。
- 专家随后 FINISH，回答原文如下。

> 当前会话和离线快照中找不到该 arXiv ID，无法下载指定论文。请核对 ID 和版本。

### 空会话引用刚才的论文 · `infeasible_no_session`

用户原话：**把刚才那篇论文翻译一下**

- 请求前无 Setup。
- 专家第一轮零工具调用。阻塞原因码为 `missing_context`。
- 专家随后 FINISH，回答原文如下。

> 当前会话没有最近操作的论文，无法确定指代。请提供论文的完整 arXiv ID，或先建立候选列表。

完整的专家工具参数、逐步 Observation、最终原文、各模型失败试次动作及 25 个真实训练组见[逐任务证据](expert_trajectory_evidence.json)。[TaskSpec](../../AgenticArxiv/benchmark/tasks_expanded.py) · [严格判定](../../AgenticArxiv/benchmark/metrics.py) · [邮件题专家回放](../../scripts/evaluate_native_dev_loss.py)
