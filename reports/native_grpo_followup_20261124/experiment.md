# Native GRPO 扩充数据复验

## 数据与离线验证

上一轮 Native GRPO 的配对严格成功从冻结 SFT 的 72/93 降到 65/93。其中 3 次回退把全 CS 改成单一子领域，另有 3 次完成第一篇的缓存检查后漏掉第二篇。本轮围绕这两类决策复验，起点为同一份 Native SFT Adapter（SHA256 `789af8cf28960892a1fde034d32f9b8ddf230e8450552be70d15024c964ed6c8`）。

本次 Probe 的 33 个候选包含 24 个新变体和上一轮通过 Probe 的 9 个 Train 任务。两版候选并集有 42 个任务。每个新请求沿用 v4 Train 父任务，并记录请求、初始会话、预期工具、有效参数和论文身份。31 个留出任务和 188 条冻结 SFT 目录记录参与泄漏审计。33/33 条专家轨迹离线回放达到严格成功、Reward 1 和严格前缀进度 1。

额外对 24 个新变体和 31 个留出任务比较仅含完整工具动作与生效参数的签名，去掉 Setup 后仍有 0 个重合。WSL 固定 CPU 环境全量回归 1,017 项通过、7 项跳过、701 个子测试通过。目标决策的回归用例覆盖奖励并列以及截断轨迹。

| 新决策 | 变体数 | 设计 |
|---|---:|---|
| 搜索范围与保存要求 | 12 | 3 种时间窗/篇数组合，各自交叉全 CS 或单一子领域、保存或只返回 |
| 顺序查询两篇缓存 | 6 | 3 组不同的论文引用，每组分别预热第一篇或第二篇 PDF |
| 条件下载，再查询另一篇 | 6 | 3 组不同引用，同一请求分别从已有缓存和缺失缓存开始 |

每条对照都重新初始化会话、执行工具并读取真实 Mock 回包。66/66 条错误轨迹都未通过严格成功，Reward 与前缀进度均低于该请求的正确轨迹。保存行为按离线工具和既有 Benchmark 的参数语义核验。

| 错误轨迹 | 对照数 | 总 Reward 范围 |
|---|---:|---:|
| 提前结束，漏掉最后一个必需动作 | 24 | −0.75 至 −0.25 |
| 搜索范围错误 | 12 | 0.75 |
| 保存要求错误 | 12 | 0.75 |
| 第二次仍查询第一篇 | 12 | 0.575 至 0.666667 |
| 缓存状态对应的下载分支错误 | 6 | −0.25 至 0.445 |

另用原任务重新计算上一轮真实策略的 748 条轨迹，包括 348 条冻结 Probe 和 400 条训练轨迹。上轮有一个包含环境错误的四轨迹 Probe 组，按原协议整组排除。在同时含严格成功和失败的 74 个组中（Probe 24、训练 50），每组最低的成功 Reward 均高于最高的失败 Reward。748 条中没有严格失败且取得满 Reward 的轨迹。逐变体结果见 [离线对照](offline_rankings.json)，历史轨迹汇总见 [复算记录](prior_policy_reward_audit.json)。

## 事先确定的训练判据

沿用五维权重 1/3/2/1/3、`step_credit=progress`、组大小 4、seed 42、每条最多 5 轮、completion 1,792 tokens，每轮 704 tokens。首轮每题 4 组，只有恰好 1 组有信号时才追加 4 组，使用同一 SFT 策略冻结采样。每题至少一半组有 Reward 或 Progress 差异才能入选。

本轮希望修复的两类行为都须在入选任务中出现至少两个独立的有效组。每组含严格成功轨迹，以及自然结束的目标错误轨迹；正确轨迹的最低 Reward 要高于该组目标错误轨迹的最高 Reward。范围错误指用户要求全 CS 而实际检索了一个子领域。缓存错误指已经正确查询指定的第一篇，却未查询指定的第二篇。截断组和只在其他错误上有分差的组不计入。该判据在新的 GPU 采样前提交（`02f86d8`）。

两类均通过后执行 4 步 Smoke，从同一冻结 SFT 起点独立进行 25 步 Dev 试验；若有一类未达到标准，本轮停止更新参数。Dev 以同协议 SFT 的 20/24 严格成功和多步骤完成率为参考，IID/OOD 不参与训练决策。

## 冻结策略实测

2026-11-24 在 RTX 4090 上从冻结 SFT 重新采样，耗时 7,206 秒。168 个独立组产生 672 条轨迹，24/33 项至少出现 1 组有效信号，18/33 项通过正式入选门槛。66 组使用 Reward Credit、2 组使用 Progress fallback，另有 71 组全对、18 组无信号、11 组因截断关闭 Step Credit。环境无效组为 0；最大守恒误差为 `1.33e-15`。逐组 seed 和选择理由见 [本轮 Train Split](../../data/splits/native_grpo_qwen35_train_v2.json)。

| 冻结策略任务 | 入选 / 候选 | G=4 组数 | 严格成功轨迹 |
|---|---:|---:|---:|
| 上轮入选的控制任务 | 6/9 | 40 | 89/160 |
| 搜索范围和文件行为 | 4/12 | 60 | 219/240 |
| 两篇论文缓存 | 5/6 | 32 | 102/128 |
| 条件下载并查另一篇 | 3/6 | 36 | 36/144 |

入选任务中，真正对目标错误产生分差的组有 3 组全 CS 范围对照、17 组漏查第二篇对照，达到进行 25 步 Dev 试验的条件。[逐组证据](probe_summary.json)

条件下载的困难集中在首次查询返回 PDF 缺失的分支。三条冷缓存任务合计 0/80 严格成功，其中 61 条轨迹连续调用两次缓存查询，跳过要求的下载。三条已有 PDF 的对照合计 36/64。正式 Train Split 选入的三个条件变体均为已有 PDF，训练效果的结论须保留这项覆盖缺口。

## 复现与产物

从 `feat/qwen35-native-grpo` 仓库根目录执行。候选数据 SHA256 为 `87750f1ff966f001255b222a0d74869f01210f771d22885c56d032cefa61d939`。GPU 运行归档内保存了独立的启动脚本和阶段日志。

```bash
python scripts/build_native_grpo_tasks.py --followup --snapshot "$SNAPSHOT" \
  --tokenizer "$BASE" --adapter "$NATIVE_SFT" --frozen-data "$FROZEN_DATA" --output "$AUDIT"
python scripts/run_native_grpo.py preflight --cpu --base "$BASE" --adapter "$NATIVE_SFT" \
  --snapshot "$SNAPSHOT" --tasks "$AUDIT/candidates.json" --output "$AUDIT/preflight"
python scripts/run_native_grpo.py probe --base "$BASE" --adapter "$NATIVE_SFT" \
  --snapshot "$SNAPSHOT" --tasks "$AUDIT/candidates.json" --output "$RUN/probe" --micro-batch 1
python scripts/audit_native_grpo_followup_probe.py "$AUDIT/candidates.json" "$RUN/probe"
```

