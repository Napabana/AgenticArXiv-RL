# Challenge Frozen Probe

**NO_GO**。冻结 Native SFT v2，在 24 个训练候选上各采 8 组，每组 G=4，共 192 组、768 条轨迹。

| 准入条件 | 实测 | 通过 |
|---|---|---|
| 入选任务 ≥12 | 5 | 否 |
| 入选家族 ≥4 | 4 | 是 |
| 最大家族占比 ≤40% | 2/5 (40.0%) | 是 |
| Strict 混合组 ≥30% | 15/192 (7.8%) | 否 |
| 四次以上调用的入选任务 ≥8 | 3 | 否 |
| 含状态分支或跨工具依赖的入选任务 ≥8 | 5 | 否 |
| DFA 非零覆盖 ≥10% | 482/3124 (15.4%) | 是 |
| Anchor 非零覆盖 ≥10% | 62/3124 (2.0%) | 否 |
| 环境无效组为 0 | 16/192 (8.3%) | 否 |
| 泄漏审计通过 | 0 个完整行为签名冲突，操作目标 ID 跨 split 交集为空 | 是 |

## 信号与运行

| 信号 | 全部采样组 | 环境有效组 |
|---|---|---|
| Strict 混合 | 15/192 (7.8%) | 15/176 (8.5%) |
| 同 DFA 状态的正确动作与首错 | 104/192 (54.2%) | 104/176 (59.1%) |
| 同真实 Anchor 的不同动作与不同回报 | 15/192 (7.8%) | 15/176 (8.5%) |

## 排除环境无效组后的敏感性分析

保留全部原始记录，仅从分析分母排除 16 个环境无效组。剩余 176 组中 Strict 混合为 15/176 (8.5%)，与原定 30% 门槛对照。DFA 非零决策覆盖为 482/3124 (15.4%)，Anchor 非零决策覆盖为 62/3124 (2.0%)。这两项覆盖率原本就只在有效组计算。

下表保留原定的组内对照次数与截断比例，只放宽整题环境异常排除条件。各题有效组数会不同，表中记录实际曝光量。原设计的 Go 决议单列保留。

| 题目 | 保留组数 | Strict 混合组 | DFA 对照组 | 截断轨迹 | 按有效组重算可入选 |
|---|---:|---:|---:|---:|---|
| challenge_train_01_A | 8 | 2 | 7 | 15 | 否 |
| challenge_train_02_A | 8 | 0 | 0 | 5 | 否 |
| challenge_train_03_A | 8 | 1 | 2 | 10 | 否 |
| challenge_train_04_A | 7 | 4 | 5 | 22 | 否 |
| challenge_train_05_A | 8 | 0 | 1 | 2 | 否 |
| challenge_train_06_A | 6 | 0 | 6 | 13 | 否 |
| challenge_train_07_B | 7 | 0 | 7 | 10 | 否 |
| challenge_train_08_B | 8 | 0 | 7 | 10 | 否 |
| challenge_train_09_B | 8 | 0 | 7 | 3 | 是 |
| challenge_train_10_B | 5 | 1 | 5 | 5 | 是 |
| challenge_train_11_B | 7 | 0 | 4 | 3 | 是 |
| challenge_train_12_B | 7 | 2 | 7 | 2 | 是 |
| challenge_train_13_C | 7 | 0 | 6 | 10 | 否 |
| challenge_train_14_C | 8 | 0 | 3 | 20 | 否 |
| challenge_train_15_C | 7 | 0 | 1 | 11 | 否 |
| challenge_train_16_C | 5 | 0 | 4 | 8 | 否 |
| challenge_train_17_D | 7 | 2 | 7 | 9 | 否 |
| challenge_train_18_D | 8 | 0 | 1 | 8 | 否 |
| challenge_train_19_D | 8 | 1 | 2 | 7 | 是 |
| challenge_train_20_E | 8 | 0 | 7 | 5 | 是 |
| challenge_train_21_E | 8 | 0 | 7 | 2 | 是 |
| challenge_train_22_E | 8 | 0 | 0 | 0 | 否 |
| challenge_train_23_F | 8 | 1 | 7 | 5 | 是 |
| challenge_train_24_F | 7 | 1 | 1 | 2 | 否 |

环境有效组中，161 组为四条全部失败，0 组为四条全部成功。在全部失败组中仍有 89 组出现同状态的正确前缀与首错。组内终局回报相同，Strict 和依赖回报差异的 Anchor 在这些组没有对照。

环境可评分轨迹的 Strict 成功为 24/749 (3.2%)。环境无效 19 条；触及轮数或 Token 上限 201 条。原始 DFA/Anchor 非零决策的分母为环境有效组的 3124 个可判定决策。投影 E4 后理论上带非零优势且掩码开启的 Assistant Token 为 38057；冻结 Probe 实际参数更新次数为 0。

首组开始到末组落盘约 3.33 小时，合计生成 285193 个 Assistant Token；吞吐 23.8 Token/秒。单采样进程最高 CUDA allocated 为 7.35 GiB。原始数据分组保存，每组均核对任务、Seed、SHA 和信用诊断重算结果。

## 逐题

| 任务 | 组 | Strict 混合 | DFA 对照 | Anchor 对照 | 环境无效组 | 截断轨迹 | Strict 成功 | 入选 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| challenge_train_01_A | 8 | 2 | 7 | 2 | 0 | 15 | 4 | 否 |
| challenge_train_02_A | 8 | 0 | 0 | 0 | 0 | 5 | 0 | 否 |
| challenge_train_03_A | 8 | 1 | 2 | 1 | 0 | 10 | 1 | 否 |
| challenge_train_04_A | 8 | 4 | 5 | 4 | 1 | 25 | 5 | 否 |
| challenge_train_05_A | 8 | 0 | 1 | 0 | 0 | 2 | 0 | 否 |
| challenge_train_06_A | 8 | 0 | 6 | 0 | 2 | 15 | 0 | 否 |
| challenge_train_07_B | 8 | 0 | 7 | 0 | 1 | 12 | 0 | 否 |
| challenge_train_08_B | 8 | 0 | 7 | 0 | 0 | 10 | 0 | 否 |
| challenge_train_09_B | 8 | 0 | 7 | 0 | 0 | 3 | 0 | 是 |
| challenge_train_10_B | 8 | 1 | 5 | 1 | 3 | 7 | 7 | 否 |
| challenge_train_11_B | 8 | 0 | 4 | 0 | 1 | 4 | 0 | 否 |
| challenge_train_12_B | 8 | 2 | 7 | 2 | 1 | 2 | 2 | 否 |
| challenge_train_13_C | 8 | 0 | 6 | 0 | 1 | 10 | 0 | 否 |
| challenge_train_14_C | 8 | 0 | 3 | 0 | 0 | 20 | 0 | 否 |
| challenge_train_15_C | 8 | 0 | 1 | 0 | 1 | 11 | 0 | 否 |
| challenge_train_16_C | 8 | 0 | 4 | 0 | 3 | 11 | 0 | 否 |
| challenge_train_17_D | 8 | 2 | 7 | 2 | 1 | 10 | 2 | 否 |
| challenge_train_18_D | 8 | 0 | 1 | 0 | 0 | 8 | 0 | 否 |
| challenge_train_19_D | 8 | 1 | 2 | 1 | 0 | 7 | 1 | 是 |
| challenge_train_20_E | 8 | 0 | 7 | 0 | 0 | 5 | 0 | 是 |
| challenge_train_21_E | 8 | 0 | 7 | 0 | 0 | 2 | 0 | 是 |
| challenge_train_22_E | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 否 |
| challenge_train_23_F | 8 | 1 | 7 | 1 | 0 | 5 | 1 | 是 |
| challenge_train_24_F | 8 | 1 | 1 | 1 | 1 | 2 | 1 | 否 |

## 每题追加采样实际观察到什么

使用相同的候选题和预定 Seed，仅截取各题前 k 个组。环境无效轨迹从严格成功分母中剔除。

| 每题采样 | 至少一次严格成功的题 | 成功轨迹 / 可评分轨迹 | 出现 DFA 对照的题 | 出现 Anchor 对照的题 |
|---|---:|---:|---:|---:|
| 1 组 / 4 条 | 1 | 2/92 (2.2%) | 11 | 0 |
| 2 组 / 8 条 | 2 | 5/185 (2.7%) | 16 | 1 |
| 4 组 / 16 条 | 5 | 12/371 (3.2%) | 19 | 4 |
| 8 组 / 32 条 | 9 | 24/749 (3.2%) | 22 | 9 |

## 首错与引用

只计可评分轨迹中的第一个可验证错误，同一条轨迹至多计一次。

- wrong_ref：399 条。
- wrong_tool：121 条。
- tool_error：95 条。
- wrong_args：53 条。
- schema_error：29 条。
- early_finish：15 条。
- translation_not_ready：6 条。
- pending_guard：3 条。

| 工具 | 指定序号 | 实际引用 | 应解析到论文 | 实际解析到论文 | 次数 |
|---|---:|---|---|---|---:|
| download_arxiv_pdf | 4 | 2 | 2611.18553v1 | 2608.14530v1 | 152 |
| download_arxiv_pdf | 2 | 1 | 2608.14530v1 | 2608.14539v1 | 71 |
| get_paper_cache_status | 4 | 2 | 2611.18553v1 | 2608.14530v1 | 35 |
| get_paper_content | 4 | 2 | 2611.18553v1 | 2608.14530v1 | 32 |
| get_paper_cache_status | 2 | 1 | 2608.14530v1 | 2608.14539v1 | 23 |
| extract_paper_figures | 4 | 2 | 2611.18553v1 | 2608.14530v1 | 18 |
| get_paper_content | 2 | 1 | 2608.14530v1 | 2608.14539v1 | 15 |
| get_paper_cache_status | 4 | 3 | 2611.18553v1 | 2608.14528v1 | 8 |
| get_paper_cache_status | 2 | 2611.18553v1 | 2608.14530v1 | 2611.18553v1 | 7 |
| translate_arxiv_pdf | 2 | 2611.18553v1 | 2608.14530v1 | 2611.18553v1 | 6 |
| translate_arxiv_pdf | 2 | 1 | 2608.14530v1 | 2608.14539v1 | 6 |
| download_arxiv_pdf | 2 | 2611.18553v1 | 2608.14530v1 | 2611.18553v1 | 5 |

## 环境异常

次数按轨迹统计。同组另外三条轨迹保留原始判定，含环境异常的组不计入三类有效信号。

- 8 条。工具执行失败: '[MockArxivEnv] replay 模式下快照缺失: tool=get_paper_content key={"paper_id": "2611.18553v1", "section": "method"}
- 5 条。工具执行失败: '[MockArxivEnv] replay 模式下快照缺失: tool=get_paper_content key={"paper_id": "2608.14539v1", "section": "method"}
- 1 条。工具执行失败: '[MockArxivEnv] replay 模式下快照缺失: tool=get_paper_content key={"paper_id": "2611.18553v1", "section": "result"}
- 1 条。工具执行失败: '[MockArxivEnv] replay 模式下快照缺失: tool=get_paper_content key={"paper_id": "2608.14528v1", "section": "method"}
- 1 条。工具执行失败: '[MockArxivEnv] replay 模式下快照缺失: tool=get_paper_content key={"paper_id": "2611.18561v1", "section": null}
- 1 条。工具执行失败: '[MockArxivEnv] replay 模式下快照缺失: tool=get_paper_content key={"paper_id": "2611.18563v1", "section": "result"}
- 1 条。工具执行失败: 关键词未命中离线快照；返回固定回退子集，不代表查询匹配结果。请检查 query 与 days 是否完整匹配任务要求。
- 1 条。工具执行失败: '[MockArxivEnv] replay 模式下快照缺失: tool=get_paper_content key={"paper_id": "2608.14539v1", "section": "result"}
