# PENDING 专项冻结策略实测

2026-11-24 在单张 RTX 4090 上，用冻结的 Native SFT Adapter 对六个候选各采 3 组，每组 4 条。模型自然生成，温度 1，seed 42–59。18/18 组满足轨迹完整性和组内初始状态一致性检查，严格成功 2/72。采样用时 702 秒。原始轨迹 SHA-256 为 `66e7d747e18148508c506ef12433866c17eed7c1af298874d3f5860a6dc859d5`。

| 候选 | strict 成功 | 有效信用对照组 | exact-anchor 正确/错误决策对 |
|---|---:|---:|---:|
| pending_v1_a_cv | 0/12 | 0/3 | 0 |
| pending_v1_a_ai | 0/12 | 0/3 | 0 |
| pending_v1_b_ro | 1/12 | 1/3 | 3 |
| pending_v1_b_cl | 1/12 | 1/3 | 3 |
| pending_v1_c_cr | 0/12 | 0/3 | 0 |
| pending_v1_c_lg | 0/12 | 0/3 | 0 |

两道 B 类任务均在初始翻译已提交的状态下采到了一次成功。两组中分别有三对共享 exact anchor 的正确/错误决策，多对比较共享同一条成功轨迹。每题要求至少两组有自然对照，六题均未入选。保留 Train v2 及历史 E0。PENDING 专项在 Weakness Bank 中维持 `no_signal`，本次冻结采样的优化器更新 token 为 0。

70 条失败轨迹的首错分布为 PENDING guard 22、工具错误 17、参数错误 14、提前结束 7、译文未就绪 6、引用错误 4。结束方式为自然结束 67、用尽轮数 4、解析失败 1。PENDING 行为按决策计数，提前读取 36 次、正确查一次缓存 2 次、正确终止 2 次，重复提交、重复轮询、错误声称就绪各 0 次。该行为计数可能包含首错之后的调用。

逐组 strict、DFA、Anchor、E4 及保护前后诊断见 [selection.json](selection.json)。[运行环境](runtime.json) 和 [专项覆盖记录](pending_weakness_bank.json) 随报告保存。完整原始轨迹在本次本地实验目录 `outputs/server56517/probe/groups.jsonl`。
