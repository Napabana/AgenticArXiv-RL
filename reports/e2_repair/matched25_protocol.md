# 相同步数的模型比较

在本轮 DAPO 第一次更新前确定 checkpoint 和评测对象。已完成的 GRPO 训练、SFT 和 GRPO-25 的 Dev 输出保留。DAPO 从 SFT v2 独立训练 72 步，在第 25、48、72 次优化器更新后保存完整 checkpoint。第 24 步照常在线采样和优化。

| 评测数据 | 预登记模型 | 每个模型的试次 |
|---|---|---:|
| E2-Repair Dev | SFT v2、GRPO-25、DAPO-25、DAPO-48、DAPO-72 | 8 × 3 |
| E2-Repair 新最终留出 | 同上 | 12 × 3 |
| 生产回归 | DAPO-25、DAPO-48、DAPO-72，对照已验证的 SFT v2 | 93 |

同一试次使用原始评分器、任务、预算、提示词和固定种子。复用已验证的 SFT v2、GRPO-25 Dev 原始记录。新留出集按既定顺序使用从 85000 开始的种子。先记录模型来源和 checkpoint 步数，再冻结五份权重 SHA 和五份完整 Dev 摘要 SHA，然后每个模型各完成一遍留出评测。三方比较的主指标为共同可评分试次上的 Strict、fail-to-pass、pass-to-fail，另外报告分模型覆盖、截断和局部行为。Dev 已经观察过，不能视为独立终测。

本次调整只涉及 checkpoint 保存和模型评测名单。基础训练 freeze 为 `9463c4d16daa4787571341b8f15b851b935c51cec6746747f94dedcd1e2bc0d2`。
