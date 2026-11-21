# Native SFT 与 Base 的配对错误复盘

以同一任务、同一试次配对。分母为实际完整跑完的试次。

| 切分 | Base 严格成功 | SFT 严格成功 | 修复 | 回退 |
| --- | ---: | ---: | ---: | ---: |
| dev | 15/24 | 15/24 | 6 | 6 |
| iid_test | 33/57 | 37/57 | 15 | 11 |
| ood_test | 9/12 | 12/12 | 3 | 0 |

## 错误类型

以下按出现该错误的试次数计数。同一试次可以同时具备多个错误。

| 类型 | Base | SFT |
| --- | ---: | ---: |
| 工具序列不匹配 | 29 | 11 |
| 序列匹配但生效参数不合格 | 7 | 12 |
| 序列匹配但论文指代不合格 | 0 | 3 |
| 工具步骤未完成便结束 | 2 | 5 |
| 存在协议解析失败 | 4 | 6 |
| 存在 Schema 拒绝 | 1 | 0 |
| 存在工具执行失败 | 17 | 3 |
| 存在超出任务要求的重复调用 | 16 | 1 |
| 阻塞任务的拒绝语义不合格 | 5 | 6 |
| 存在 PENDING 调用拦截 | 4 | 0 |
| 存在 Snapshot 覆盖错误 | 11 | 0 |

## 严格成功发生变化的任务

| 切分 | 任务 | 改善试次 | 回退试次 |
| --- | --- | ---: | ---: |
| dev | `constraint_search_no_file` | 0 | 3 |
| dev | `ref_ctrl_id_download` | 3 | 0 |
| dev | `search_AI_1d_3` | 3 | 0 |
| dev | `state_dl_then_cache` | 0 | 3 |
| iid_test | `chain_cl5_dl3_tr3` | 0 | 3 |
| iid_test | `constraint_translate_only` | 0 | 1 |
| iid_test | `infeasible_index_out_of_range` | 0 | 1 |
| iid_test | `multi_ai3_tr1` | 3 | 0 |
| iid_test | `qa_cr5_abstract` | 0 | 2 |
| iid_test | `ref_ctrl_id_cache` | 3 | 0 |
| iid_test | `search_CV_1d_1` | 3 | 0 |
| iid_test | `search_RO_30d_25` | 3 | 0 |
| iid_test | `state_cache_two` | 0 | 2 |
| iid_test | `state_ref_cache_active` | 1 | 0 |
| iid_test | `summary_lg5_tldr120` | 0 | 2 |
| iid_test | `trread_ro5_page2` | 2 | 0 |
| ood_test | `multi_ai5_dl1_tr` | 3 | 0 |

## 与训练数据覆盖对应的观察

- 训练中的 83 条分类检索监督只有 0 条使用 `save_to_file=false`。全 CS 短码 `*` 出现 1 次。
- 直接承接下载的监督决策有 95 条，其中 0 条调用缓存工具，35 条生成最终回答。
- 在后续提示约束对照中，可把用户的“不要”“只”“禁止”写为通用的行动约束，并在训练中用快照支持的新实例验证如何覆盖有副作用的默认值。
- 逐试次的调用参数、论文 ID 和协议事件见同名 JSON。


## 29 个 SFT 失败的互斥归因

| 首个不达标层级 | 试次 | 证据 |
| --- | ---: | --- |
| 工具序列 | 11 | 不支持能力 3、越界仍下载 3、只查第一篇缓存 3、下载后漏查缓存 2 |
| 生效参数或论文引用 | 12 | 全 CS 与禁止保存 3、活跃论文双语翻译 3、显式摘要章节 3、标题片段定位 3 |
| 最终回答终止 | 6 | 查缓存 1、提交翻译 3、生成摘要 2。工具序列与参数均对，最终文本因重复论文 ID 或 PDF READY 状态触到生成上限 |

显式摘要题的目标论文选择正确。当前严格评测要求 section=abstract，SFT 生成默认的 section=null。核对 Snapshot 可见，默认版本含标题及摘要，显式 abstract 版本含摘要。后续设计可单独厘清用户只要求摘要时应接受哪些输出范围。

诊断 JSON 中保留了每个试次的实际 Tool Call、参数、解析论文 ID、协议事件和终止类型。对应的候选 SFT 修改见 `docs/native_sft_constraint_followup_plan.md`。
