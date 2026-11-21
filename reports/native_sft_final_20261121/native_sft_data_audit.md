# Native SFT 数据与 Loss Mask 审计

- v4_86 Train 父任务 55 个。完整轨迹 188 条。
- 审计通过的决策样本 505 条。工具调用 317 条，最终回答 188 条。
- 当前 Assistant 的内容、工具调用和 EOS 已逐样本通过 Qwen 模板和 TRL labels 核对（TRL 0.29.1）。
- System、User、工具定义、历史 Assistant 和 Tool Observation 均位于 loss mask 覆盖的前缀。
- Snapshot SHA256 `d97035db326ca11935e882fd2aca7fe932e9e76cf1bc8a21078be7bec1ae8794`。
- Train JSONL SHA256 `42002233e1fe03cd4e4fbb31b44f9565f588ec1d0de5e42995f540aa87c58e68`。
- Token 长度 P50 3456、P95 4474、最大 6138。上限 6144。
- 为 6 个任务重新绑定了 Snapshot 已覆盖的论文序号，保持其检索范围、工具链和阅读或处理目标。
- 排除轨迹 0 条，排除决策 0 条。完整原因见 Manifest。
- 压缩长 Observation 的决策 5 条。完整轨迹、调用参数和训练目标均保留。
- Dev、IID Test 和 OOD Test 的任务 ID、请求原文、专家参数与 setup 组合均经过泄漏核对。

| 针对性训练行为 | 完整 SFT 轨迹 |
| --- | ---: |
| active_or_missing | 2 |
| active_with_dual | 2 |
| capability_boundary | 4 |
| concise_finish | 3 |
| download_cache_order | 9 |
| index_boundary | 2 |
| multi_object_cache | 2 |
| partial_completion | 1 |
| search_save_constraint | 6 |
| section_default | 2 |
| title_resolution | 1 |