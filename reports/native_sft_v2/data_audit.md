# Native SFT v2 数据审计

**CPU 审计通过。** 188 条 v1 轨迹均通过当前 Native Agent 和固定 Mock 环境的专家回放。38 条新增候选中，35 条通过严格执行及防泄漏检查。最终共 223 条轨迹、590 个唯一监督决策，其中 505 个源自冻结 v1、85 个来自新增轨迹。数据文件独立保存；原 v1 Manifest、训练数据、Adapter 与归档评测均通过指纹校验。

| 采样类别 | 轨迹 | 唯一决策 | 正式一轮计划曝光 |
|---|---:|---:|---:|
| Core / Parametric | 131 | 396 | 396（60%） |
| Error-targeted | 60 | 149 | 198（30%） |
| Recovery / PENDING / Blocked | 32 | 45 | 66（10%） |
| 合计 | 223 | 590 | 660 |

每个独立样本至少出现一次；只有采样索引重复。累积 8 个微批时预计更新 83 步。训练时从实际进入 loss 的 batch 读取样本索引，与预期多重集合逐一比对。以上为计划曝光；实际值由正式训练写入 `training.json`。

## 覆盖与排除

新增任务包括同一请求在缓存已命中和未命中时采取不同动作、全 CS 与单领域搜索时保存或禁止保存、分别处理两篇论文并核对缓存、读取章节后总结、翻译服务与双语保留参数、PENDING 后结束或继续独立缓存检查、READY 后读指定页、越界引用、超出能力但仍含可执行子目标，以及在真实工具错误后修正行为。保留的 Recovery 从 v1 的 2 条增加到 5 条。各错误机制的逐项数量见 [分布文件](data_distribution.json)，包含任务和有效参数的原始目录见 [任务目录](../../data/sft/native_sft_v2/native_sft_v2_task_catalog.jsonl)。

| 排除的候选 | 原因 |
|---|---|
| sft_v2_pending_read | 虽然初始状态不同，有效工具调用及参数序列与留出任务 ref_ctrl_id_cache 重合。 |
| sft_v2_recover_active | 恢复后的有效工具调用及参数序列与留出任务 state_ref_cache_active 重合。 |
| sft_v2_recover_scope | Snapshot 缺少关键词查询记录，无法把快照缺项验证为真实空结果。 |

全部新增候选均有 Train 父任务。对 31 道留出任务检查 ID、规范化请求和带初始状态的任务签名；新增数据另查去掉初始状态后的完整有效调用序列及参考终止文本。通过这些检查的样本没有检出重合。现有 SFT-only 任务沿用 v1 的来源记录。Strict replay 使用独立会话、生产 Native 协议和离线工具；核验参数、引用对象、执行观测与终止状态。恢复样本中的失败动作仅进入历史。

## 监督和长度

590 条唯一 token 序列全部经 TRL 0.29.1 collator 逐 token 核验。367 个目标为带 Assistant Content 的 Native Tool Call，223 个为终止回答。提示和历史工具输出的 label 均为 -100；当前 completion 的 label 逐 token 对齐。长度中位数 3456，P95 为 4491，最长 6138，限制 6144。新增决策无需压缩。

## 复现与证据

- Base 权重 SHA256 为 `928acbf11878c32185bbd863514d191769285065ab9ea14fbfe431303f5fdf2d`，v1 Adapter 为 `789af8cf28960892a1fde034d32f9b8ddf230e8450552be70d15024c964ed6c8`。
- v2 训练 JSONL SHA256 为 `8763ff0ef7c19bf748d3d036d3730e8f30c44709161df614302a7a5df5bb3dfd`，数据 Manifest SHA256 为 `b8aa518f3f8aba708029bb1525b43d63372840a5bb785bf0fdde1439faf10751`。
- Split SHA256 为 `0a176b0f67b8b61d4f84e27a40071ab81a5be3d2040c2a58aa695432f5a6eacc`，Snapshot 为 `d97035db326ca11935e882fd2aca7fe932e9e76cf1bc8a21078be7bec1ae8794`。工具、模板、提示词及回放输出的完整 SHA 见 [训练 Manifest](../../data/sft/native_sft_v2/native_sft_v2_train.jsonl.manifest.json)。
- CPU 回归测试命令为 `pytest -q AgenticArxiv/tests --ignore=AgenticArxiv/tests/test_format_control.py`，1069 passed、6 skipped、704 subtests passed。Native v2 专项测试单跑 2 passed。另验证了 v1 训练栈 SFTTrainer 导入、Shell 入口的 Bash 语法，以及历史 93 次配对结果读取。真实 CUDA 训练、Adapter 加载、配对结果和 Frozen Probe 将由 GPU 阶段验证。
