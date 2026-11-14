# ArxivFetch-RL 数据契约

本文记录 2026-11-14 在 Git 基线 `533f4f2` 上能够由代码、测试和已提交 Manifest 直接验证的数据契约。未实现的设计单列在“P1 计划”，不得当作当前能力。

## 1. 文件角色与血缘

数据链路如下：

```text
派生任务 seed Manifest
  └─ tasks[].derived_task_id / task / setup / steps
       ↓ input_manifest + input_manifest_sha256
语言扩增 JSONL 及 Manifest
       ↓ derived_task_id + trajectory_step
多轮 SFT JSONL 及 Manifest
       ↓ parent_sha256（可选长度过滤）
训练输入 JSONL
```

- `data/sft/sft_v3_t5_parametric_seed.jsonl.manifest.json` 是当前派生任务声明来源，包含 86 个 `tasks`。实际 seed JSONL 未提交。
- 语言扩增 Manifest 必须声明 `input_manifest`；若同时声明 `input_manifest_sha256`，构建器必须先校验哈希。
- 多轮构建器不得凭任务文本猜测 setup；必须按 `derived_task_id` 回查任务 Manifest。
- 历史 `data/sft/sft_v4_multiturn.jsonl` 及其 v1 Manifest 是只读证据，不因构建器修复而原地重写。

## 2. 派生任务 Manifest

当前构建器实际消费的最小结构为：

```json
{
  "kind": "train_only_parametric_expert_seed",
  "tasks": [
    {
      "derived_task_id": "augp1_...",
      "parent_task_id": "...",
      "task": "任务文本",
      "setup": [
        {"name": "工具名", "args": {}}
      ],
      "steps": []
    }
  ]
}
```

约束：

- `tasks` 必须是非空数组。
- `derived_task_id` 必须存在且在 Manifest 内唯一。
- `setup` 缺省时按空数组处理；非数组值是错误。
- setup 动作必须在该链的 `session_id` 内通过 Agent 的既有副作用执行路径执行；工具返回失败 Observation 时构建失败，不得静默替换为空历史。
- `build_visible_setup_context` 生成的可见 setup 上下文必须进入首步 Prompt。

## 3. 语言扩增输入行

多轮构建器依赖以下字段：

| 字段 | 类型 | 语义 |
| --- | --- | --- |
| `messages` | array | 至少包含一个 `user` 和一个 `assistant`；user 中含 `当前任务：...`，assistant 是 gold ReAct 决策。 |
| `derived_task_id` | string | 指向派生任务；某些历史行用 `source_task_id` 承载该血缘，构建器兼容二者。 |
| `source_task_id` / `parent_task_id` | string，可选 | 用于链分组的历史血缘字段。 |
| `trajectory_step` | integer | 同一链内的决策顺序。 |
| `augmentation` | object，可选 | 语言视图；稳定 JSON 序列化后参与链分组。 |
| `sample_sha256` | string，旧格式兜底 | 仅在没有任务血缘字段时用于分组兜底。 |

同一派生任务的不同语言视图是不同链。链内按 `trajectory_step` 升序回放。

## 4. 多轮 SFT 输出行

每个 gold 决策产生一行：

```json
{
  "messages": [
    {"role": "user", "content": "runner 同款 Prompt、setup 与历史 Observation"},
    {"role": "assistant", "content": "Thought: ...\nAction: ..."}
  ],
  "source": "v4_multiturn",
  "derived_task_id": "...",
  "trajectory_step": 0
}
```

约束：

- `messages` 恰为 `user`、`assistant` 两条，内容均为字符串，assistant 不得为空。
- `derived_task_id` 是非空字符串；`trajectory_step` 是非负整数。
- user 内容由现有 `ReActAgent.build_messages` 生成，包含当前步骤之前的 setup、Action 和 Observation；本步 gold 决策只放在 assistant。
- `Action: FINISH` 是正常终止，不执行工具。若它出现在链的最后一步，链计入 `completed_chains`；提前出现计入 `premature_finish`。
- 非 FINISH 动作复用 `rl.grpo_reward.parse_react_action`，并经 Agent 现有副作用路径执行，避免绕开下载/翻译等项目语义。

失败分类固定为：

| 分类 | 含义 |
| --- | --- |
| `parse_error` | gold 决策不是合法工具调用或 FINISH。 |
| `snapshot_miss` | replay Snapshot 缺少本步要求的工具结果。 |
| `tool_error` | setup 后的链步骤出现其他工具或副作用失败。 |
| `premature_finish` | FINISH 后仍有声明的 gold 步骤。 |
| `missing_finish` | 输入链耗尽但未出现 FINISH。 |

setup 失败发生在生成该链样本之前，直接抛出带任务、工具、参数和原始原因的异常，不伪装成截断链。

## 5. 多轮 SFT Manifest v2

修复后的构建器输出：

```json
{
  "kind": "sft_v4_multiturn",
  "version": 2,
  "seed_data": "...",
  "seed_sha256": "...",
  "task_manifest": "...",
  "task_manifest_sha256": "...",
  "snapshot": "...",
  "snapshot_sha256": "...",
  "output": "...",
  "output_sha256": "...",
  "summary": {
    "chains": 1,
    "rows": 1,
    "completed_chains": 1,
    "truncated_chains": 0,
    "failure_counts": {}
  }
}
```

路径用于描述血缘，SHA256 才是内容身份。生成新数据时必须使用新文件名，不覆盖历史 v4。长度过滤后的 Manifest 还必须保存 `parent_dataset`、`parent_sha256`、最大长度、tokenizer、排除文件和长度审计文件。

## 6. Snapshot 当前契约

当前 `MockArxivEnv` 读取的 Snapshot 是无顶层 metadata 的普通 JSON 映射：

```json
{
  "tool_name": {
    "stable_key": {
      "args": {},
      "result": {}
    }
  }
}
```

当前规则：

- 普通工具 key 是参数的稳定 JSON 表示；会话等易变参数不会写入记录的 `args`。
- 关键词搜索 key 使用规范化 query 的 SHA256，并在提供时加入 `days`；`max_results` 只对命中池切片。
- 正文、摘要、图片、图表分析和译文使用解析后的稳定 `paper_id` 构造 key，不能以本地 `ref` 作为跨会话身份。
- replay 模式下未命中的普通 key 抛 `KeyError`，不得联网补齐。
- recent-search 可从同 aspect、通配池或任意现有池派生并按 `max_results` 截断；当前派生逻辑没有按请求的 `days` 重新过滤。
- 未命中的关键词搜索可返回带 `_mock_env.offline_fallback=true` 的确定性回退子集。Agent 会把它呈现为工具失败，但多轮环境仍可能把结果写入自己的 store；这是待修复的语义差异。
- offline download 是本地桩，不证明真实 PDF 可下载。

当前 Snapshot 文件没有 `schema_version`、`as_of`、真实时间窗、Git SHA 或独立 Manifest；部分工具结果可能包含生成机的 `local_path`。因此现有 Snapshot 不能直接视为可移植数据包。

## 7. 本地资源与可移植性

- GitHub 只提交代码、测试、文档、小型 fixture 和 Manifest/索引。
- Snapshot、大型 JSONL、PDF、译文、图片、模型权重和虚拟环境不进入 Git；后续通过带 revision 的 ModelScope Dataset 传递。
- 可移植数据不得依赖 Windows 盘符、WSL `/mnt/...` 或 Notebook `/mnt/workspace/...` 绝对路径。资源路径必须相对显式 bundle root，或在加载时可重映射。
- SHA256 默认针对文件原始字节。当前 Windows Git 的 `core.autocrlf=true` 会把历史 v4 JSONL 的 LF 转成 CRLF，使工作树原始哈希不同于 Manifest；跨环境交付必须保持字节不变或明确记录并校验换行策略。
- 本地 WSL CPU 虚拟环境只是开发工具，不属于数据契约，也不能改变云端 `requirements-qwen35.txt` 或训练启动器。

## 8. P1 计划，不属于当前能力

P1-A 需要新增显式 metadata-only Snapshot 模式，并同时定义：

- 顶层 `schema_version`、`as_of`、查询窗口、Git SHA、构建参数和内容哈希；
- 完全跳过 PDF、正文、图表、翻译和视觉模型的明确开关；
- metadata-only 模式真实支持的任务子集；
- `days`、固定 benchmark 锚点和关键词 miss 的一致语义；
- 断网 replay 的 fail-closed 验证与可移植路径检查。

当前 `--content-max-ref 0` 表示处理全部论文，不表示关闭内容/PDF 阶段，不能作为 metadata-only 使用。
