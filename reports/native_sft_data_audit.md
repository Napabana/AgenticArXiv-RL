# Native SFT 数据与 Loss Mask 审计

- v4_86 Train 父任务 55 个。完整轨迹 154 条。
- 审计通过的决策样本 429 条。工具调用 281 条，最终回答 148 条。
- 当前 Assistant 的内容和工具调用已逐样本通过 Qwen 模板渲染和 TRL labels 核对（TRL 0.29.1）。
- System、User、工具定义、历史 Assistant 和 Tool Observation 均位于 loss mask 覆盖的前缀。
- Snapshot SHA256 `d97035db326ca11935e882fd2aca7fe932e9e76cf1bc8a21078be7bec1ae8794`。
- Train JSONL SHA256 `0d13b8acd3c23f0f939ddd1007d8b7b0f0865902e93c6b9469d4a9c2c42abe5b`。
- Token 长度 P50 3297、P95 4425、最大 5337。上限 6144。
- 为 6 个任务重新绑定了 Snapshot 已覆盖的论文序号，保持其检索范围、工具链和阅读或处理目标。
- 排除轨迹 0 条，排除决策 6 条。完整原因见 Manifest。
- Dev、IID Test 和 OOD Test 的任务 ID、请求原文、专家参数与 setup 组合均经过泄漏核对。


## 历史输入定位

旧 V3 seed 本体在本地工作树和 GPU 实例均未找到。本地预期路径为 `<repo-root>/outputs/qwen35_2b_base_tool_calling_worktree/data/sft/sft_v3_t5_parametric_seed.jsonl`。原 Manifest 给出的预期 SHA256 为 `1d4d2831225e68ecf773ac9314fb6be14e085ab104dbe48266db68f80882011c`。

从原生成器与 TaskSpec 重建的 86 个任务定义逐字段通过 Manifest 校验，定义 SHA256 为 `6d5b26fe521a34837b98e4cd986c54e9a9bb64710f5b0a63e0f9c1b266c2fa32`。当前专家 Observation 来自固定的 d970 Snapshot。对齐前 6 处原引用及修正后的快照键列在本次 Manifest 的 `snapshot_alignments`。
