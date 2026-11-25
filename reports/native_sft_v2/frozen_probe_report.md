# Native SFT v2 Frozen Probe

33 个 Train 候选各采 8 组，每组 4 条独立采样。共 264 组、1056 条轨迹。264 组的四条轨迹均可判定，0 组至少一条无法判定。

有效组中 81 组同时有 Strict 成功和失败（30.7%），183 组 Strict 方差为零（69.3%）。五维 Reward 有非零组内方差的组数为 93。

27/33 道题至少出现一次有效对照；17/33 道题达到 2/8 入围条件。

## 与 v1 的共同采样范围

历史 v1 后四组按信号自适应加采，比较只采用每题预先共有的前四组。两版相同题目和组 seed、双方均有效的组共 132/132。v1 和 v2 的 Strict 混合组分别为 52 和 34；组内成功轨迹分别为 352/528 和 413/528。v1 的 528 条归档轨迹经当前评分器复算，与原判定逐条一致。

## 各任务信号

| Train 候选 | 有效组 | Strict 混合组 | 全成功组 | 全失败组 | Strict 成功轨迹 |
|---|---:|---:|---:|---:|---:|
| constraint_cache_only | 8/8 | 2 | 6 | 0 | 30/32 |
| chain_ro5_dl_three | 8/8 | 6 | 2 | 0 | 21/32 |
| chain_lg10_dl_tr_last | 8/8 | 8 | 0 | 0 | 21/32 |
| trread_ai5_default | 8/8 | 4 | 4 | 0 | 28/32 |
| infeasible_no_session | 8/8 | 7 | 0 | 1 | 11/32 |
| native_grpo_two_targets | 8/8 | 0 | 8 | 0 | 32/32 |
| native_grpo_versioned_download_cache_translate | 8/8 | 5 | 3 | 0 | 26/32 |
| native_grpo_invalid_index | 8/8 | 2 | 0 | 6 | 2/32 |
| native_grpo_search_download_abstract | 8/8 | 6 | 2 | 0 | 21/32 |
| native_grpo_v2_scope_all_5_0 | 8/8 | 1 | 7 | 0 | 31/32 |
| native_grpo_v2_scope_all_5_1 | 8/8 | 1 | 7 | 0 | 31/32 |
| native_grpo_v2_scope_CL_5_0 | 8/8 | 5 | 3 | 0 | 25/32 |
| native_grpo_v2_scope_CL_5_1 | 8/8 | 1 | 7 | 0 | 31/32 |
| native_grpo_v2_scope_all_12_0 | 8/8 | 0 | 8 | 0 | 32/32 |
| native_grpo_v2_scope_all_12_1 | 8/8 | 2 | 6 | 0 | 29/32 |
| native_grpo_v2_scope_CV_12_0 | 8/8 | 1 | 7 | 0 | 31/32 |
| native_grpo_v2_scope_CV_12_1 | 8/8 | 3 | 5 | 0 | 29/32 |
| native_grpo_v2_scope_all_19_0 | 8/8 | 3 | 5 | 0 | 29/32 |
| native_grpo_v2_scope_all_19_1 | 8/8 | 3 | 5 | 0 | 27/32 |
| native_grpo_v2_scope_AI_19_0 | 8/8 | 0 | 8 | 0 | 32/32 |
| native_grpo_v2_scope_AI_19_1 | 8/8 | 0 | 8 | 0 | 32/32 |
| native_grpo_v2_pair_CV_3 | 8/8 | 1 | 7 | 0 | 31/32 |
| native_grpo_v2_pair_CV_1 | 8/8 | 0 | 8 | 0 | 32/32 |
| native_grpo_v2_conditional_CV_0 | 8/8 | 1 | 0 | 7 | 1/32 |
| native_grpo_v2_conditional_CV_1 | 8/8 | 1 | 7 | 0 | 31/32 |
| native_grpo_v2_pair_AI_2 | 8/8 | 2 | 6 | 0 | 29/32 |
| native_grpo_v2_pair_AI_4 | 8/8 | 4 | 4 | 0 | 27/32 |
| native_grpo_v2_conditional_AI_0 | 8/8 | 0 | 0 | 8 | 0/32 |
| native_grpo_v2_conditional_AI_1 | 8/8 | 7 | 1 | 0 | 18/32 |
| native_grpo_v2_pair_RO_4 | 8/8 | 1 | 7 | 0 | 31/32 |
| native_grpo_v2_pair_RO_2 | 8/8 | 1 | 7 | 0 | 31/32 |
| native_grpo_v2_conditional_RO_0 | 8/8 | 1 | 0 | 7 | 1/32 |
| native_grpo_v2_conditional_RO_1 | 8/8 | 2 | 6 | 0 | 29/32 |

首错是每条轨迹第一次偏离可验证的专家决策。没有可判定首错的轨迹不计入下表。

| 首错类型 | 轨迹数 |
|---|---:|
| wrong_tool | 90 |
| bad_terminal | 45 |
| wrong_args | 44 |
| wrong_ref | 25 |
| early_finish | 17 |
| schema_error | 9 |
| extra_call | 7 |
| tool_error | 2 |
| pending_guard | 1 |
| pending_claimed_ready | 1 |

Probe 原始组文件 SHA256 为 `79f4d00e5e3eeb149b67e3ada3bc6b8511d8736aeb5fe52ed6f88400595336b4`。完整结果保存在本地归档，入围列表和总量见 [Probe 汇总](frozen_probe.json)。
