# Native Tool Calling 第一阶段验收

目标分支为 `feat/qwen35-base-native-sft`，从 GitHub 快进到 `d1f3129a3a873cbe9be439ef33bd95e3279d3961`。修复位于该分支已有的独立工作树。GPU 基线使用经 SHA256 校验的源码包 `3ac4521cc1a4b5b4a5b2be1222c79a147c88006730a33ab71bfcff9084836a41`。

- 对齐了 10 个工具的模型 Schema 与默认值，统一省略和 `null` 的论文引用语义，隐藏框架参数，并在类别归一化前后保留证据。
- Benchmark 和 GRPO 的参数评分共用生效参数比较逻辑。显式版本号、改变工具行为的额外参数和未知字段仍具有评分影响。协议解析失败来自结构化事件。
- Native Setup 使用实际工具回执和会话状态。翻译提交的 PENDING 与 Snapshot 中已有译文的 READY 状态分别核验。允许按任务要求在 PENDING 后查询一次缓存。
- 远端回归测试共 216 项通过、1 项跳过、63 个子测试通过。测试未调用真实 arXiv 或翻译服务。86 个 TaskSpec 的工具轮数预算均可容纳最终回答。

## 历史 Base 轨迹重评分

使用原始逐轮轨迹与本次评分器复算。旧基线的训练权重和输出保持原样。

| 切分 | 原严格成功 | 重评分 |
| --- | ---: | ---: |
| Dev | 7/24 | 10/24 |
| IID Test | 20/57 | 21/57 |
| OOD Test | 6/12 | 3/12 |
| 合计 | 33/93 | 34/93 |

3 次默认参数判分和 2 次自然语言解析误报得到纠正。另有 4 次因非默认参数改变行为而扣回严格成功。复算结果保存在 `outputs/native_sft_phase1_rescore/`。

## Snapshot 覆盖

Snapshot 路径为 `<repo-root>/data/mock_arxiv_snapshot.json`，SHA256 为 `d97035db326ca11935e882fd2aca7fe932e9e76cf1bc8a21078be7bec1ae8794`。规范切分 `data/splits/v4_86.json` 的 SHA256 为 `0a176b0f67b8b61d4f84e27a40071ab81a5be3d2040c2a58aa695432f5a6eacc`。

| Train 任务 | 缺失的 `get_paper_content` Snapshot 键 |
| --- | --- |
| `read_cl5_result` | `{"paper_id":"2611.18514v1","section":"result"}` |
| `read_cv5_method` | `{"paper_id":"2611.18550v1","section":"method"}` |
| `read_lg5_conclusion` | `{"paper_id":"2611.18563v1","section":"conclusion"}` |

55 个 Train 父任务中，52 个通过专家工具回放。31 个留出任务全部通过预检。
