# pending_task_v1 · 冻结 SFT Probe 就绪说明

本阶段交付 [6 个候选任务](../../data/splits/pending_task_v1_candidates.json) 与 [CPU Gold Replay 记录](gold_audit.json)。候选 SHA-256 为 `9960cbc79ef72bf2cfb0a97896c023d0b3273836139719a868dc475ab6bcba20`。训练起点为 Native SFT Adapter `789af8cf28960892a1fde034d32f9b8ddf230e8450552be70d15024c964ed6c8`，Mock Snapshot 为 `d97035db326ca11935e882fd2aca7fe932e9e76cf1bc8a21078be7bec1ae8794`。正式 Train v2 仍为 18 个任务。现有 Weakness Bank 中 PENDING 的状态仍为 `no_signal`。

| 类型 | 数量 | 模型看到的初始条件 | 正确决策 |
|---|---:|---|---|
| A | 2 | 已下载 PDF，尚未提交翻译 | 提交双语翻译，查一次缓存，说明译文尚未就绪 |
| B | 2 | Setup 已提交翻译，回执和 Agent 均为 PENDING | 查一次缓存，结束并解释状态 |
| C | 2 | Setup 已提交翻译，用户请求中文摘要 | 查一次缓存，解释为何暂时无法读取 |

任务使用不同论文，覆盖序号、显式 arXiv ID、标题和最近操作的论文。生成器读取固定 Snapshot，剔除已有译文的论文，并实际执行 Setup，核对 PDF、目标论文、翻译回执和 Agent 内部状态。派生来源是 Train 父任务。去重检查涵盖 31 个留出任务和冻结 SFT 的 188 条目录记录；目标论文另外排除了留出标准步骤及固定 SFT 留出轨迹中经论文工具实际解析的身份。检索结果列表依然来自所有切分共用的 Snapshot。

每个候选的严格 Gold Replay 通过，首次检查均返回同一目标论文的 PENDING。每个候选独立重置 4 次，核对了同一初始可见上下文与语义状态。在隔离会话中测试提前读取、重复提交及重复轮询，18/18 次返回 `pending_guard`。最长 gold 完整上下文 4,247 tokens，最长 completion 341 tokens，最长单轮输出 52 tokens。参考回答与负例仅用于 CPU 验证。此环境中的新提交保持 PENDING。

CPU 验证涵盖 63 项相关回归，以及完整 AgenticArxiv 套件的 1,067 项通过、704 个通过的子测试、6 项跳过，0 项失败。跳过项依赖可选 TensorBoard、在线服务或人工凭据。实际 GPU 入口已用 CPU dry-run 验证资产与参数。重新构建候选可使用以下命令，路径变量分别指向固定 Base、SFT Adapter、Snapshot、冻结 SFT 数据包与历史 SFT 评测目录。

```bash
python scripts/run_pending_task_v1.py build --base "$BASE_MODEL" --adapter "$SFT_ADAPTER" \
  --snapshot "$SNAPSHOT" --frozen-data "$FROZEN_DATA" --baseline "$SFT_BENCHMARK"
```

## RTX 4090 上的下一步

在已同步本分支的仓库根目录运行：

```bash
bash scripts/run_pending_task_v1_probe.sh --output /root/sj-tmp/pending-task-v1-probe-01
```

入口默认使用前次实例的资产目录，支持通过 `BASE_MODEL`、`SFT_ADAPTER`、`SNAPSHOT`、`PYTHON_BIN` 覆盖。它核对候选文件、模型、Adapter、模板、工具和 Snapshot，使用冻结权重采样，每题 3 组，每组 G=4，共 **18 组、72 条自然轨迹**。使用温度 1、种子 42–59，沿用 5 轮 / 1792 completion tokens / 每轮 704 tokens 的上限。前次同卡 168 组实测约 120 分钟，按组数折算本轮采样约 13 分钟；考虑加载、校验及输出长度差异，建议预留 **15–30 分钟**。

每组保存四条原始轨迹、seed、strict、DFA 和 exact-anchor 原始信用，以及 E4 生效后的局部信用。以自然到达 PENDING 的同一 DFA 状态配对，一边为成功轨迹的正确决策，另一边为失败轨迹的首错。将有效组定义为四条轨迹都可用、起始 anchor 相同、strict 有成有败，且上述同一动作对同时得到预期方向的 DFA 和 E4 信号。另报相同 exact anchor 的可区分动作对数，供 E3 判断覆盖率。每题须至少 **2/3 组有效**才进入候选训练补充集；12/12 成功标为 `ceiling_control`，采样无效或缺失标为 `incomplete`。

结果目录中的 `groups.jsonl` 保存原始证据，`selection.json` 报告逐题判定和六类 PENDING 行为（一次正确检查、正确结束、提前读取、重复提交、重复查询、错误声称就绪），`pending_weakness_bank.json` 汇总暴露。冻结 Probe 的实际优化器更新为零。有入选任务时另存 `proposed_train.json`，其 25 组计划分配为 PENDING 专项 5、其他 P0 8、多步骤 7、控制 5。入选任务超过五个时，依次按有效组数、有 exact-anchor 对照的有效组数优先安排；平局保持候选顺序。采用该提案做正式比较时，E0 也需使用同一新 Split。GPU 入口在 Probe 和数据筛选完成后退出。
