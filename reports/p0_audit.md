# P0 数据链路审计报告

日期：2026-11-14

状态：`LOCAL_PASS`（CPU 数据链路）；GPU、真实全量 PDF 网络与训练均为 `NOT_RUN`

## 1. 结论

P0 的最小修复已在本地分支完成并通过离线 CPU 回归：多轮 SFT 构建器现在能按派生任务血缘恢复 setup，在同一 Session 中通过 Agent 的既有副作用路径推进动作，把正常 `FINISH` 计为完成，并区分解析、工具、提前结束和缺少结束四类链失败。历史 v4 数据和 Manifest 没有被重建或覆盖。

本地开发环境与目标训练环境保持解耦：未修改 `requirements-qwen35.txt`、SFT/GRPO 启动器或云端模型依赖；未安装 torch、transformers 或 trl；未加载 Qwen3.5 权重。

## 2. Git 与工作目录

| 项目 | 审计结果 |
| --- | --- |
| 仓库 | `E:\2806\ArxivFetch-RL\AgenticArXiv-RL` |
| WSL 路径 | `/mnt/e/2806/ArxivFetch-RL/AgenticArXiv-RL` |
| 来源分支 | `feat/arxivfetch-qwen35-2b` |
| 来源 HEAD | `533f4f27d6dcb74c06887aa10be1c35d06802c9a` |
| 本地工作分支 | `feat/local-data-pipeline` |
| commit / push | 均未执行 |
| 本地代理规则 | `AGENTS.md`，由 `.git/info/exclude` 排除，不进入提交 |

Windows Git 在 E 盘可能报告 dubious ownership；本次只对读取命令使用单命令 `-c safe.directory=...`，没有修改全局 Git 配置。

## 3. 本地环境边界

| 项目 | 结果 |
| --- | --- |
| Windows | Python 3.12.0，仅用于编辑/Git |
| WSL | `Ubuntu-22.04-Recovered`，Ubuntu 22.04.5，WSL2 |
| 隔离环境 | `/home/jfm/.venvs/arxivfetch-rl-p0-py311` |
| Python | 3.11.15 |
| 环境工具 | uv 0.13.8 |
| 安装范围 | arxiv、requests、pydantic、PyMuPDF、Pillow、pytest、loguru 等 CPU 数据/PDF 测试依赖 |
| 明确未安装 | torch、CUDA 包、transformers、trl、Qwen3.5 权重 |

虚拟环境位于 WSL Linux 文件系统，不在仓库内。曾临时用本地 requirements 文件安装依赖，随后删除该文件；仓库不携带“本地 AMD 集显环境”专用依赖声明，避免影响 ModelScope 的正式 GPU 环境。

## 4. 数据与资产盘点

| 资产 | 状态 | 证据/说明 |
| --- | --- | --- |
| `data/sft/sft_v4_multiturn.jsonl` | `PRESENT_WITH_EOL_VARIANCE` | 2,508 行。历史 Manifest 与 LF 归一化内容的 SHA256 均为 `a053590e...24bdec`；当前 Windows 工作树因 `core.autocrlf=true` 含 2,508 个 CRLF，30,477,908 bytes，原始字节 SHA256 为 `ef54be18...a1f39`。Git 状态显示文件未被修改。 |
| v4 历史 Manifest | `PRESENT` | v1，`chains=1032`、`truncated_chains=1032`，保留为缺陷证据。 |
| v4 6144 Manifest | `PRESENT` | 声明 2,496 行、排除 12 行。 |
| v4 6144 JSONL/排除/长度审计 | `MISSING` | Manifest 所指三个 payload 未提交，不能据 Manifest 声称数据可用。 |
| v3 T5 seed Manifest | `PRESENT` | 包含 86 个派生任务及 setup；对应 seed JSONL 未提交。 |
| `data/mock_arxiv_snapshot.json` | `MISSING` | 无法在本机重建完整历史 v4。 |
| `data/splits/qwen35_grpo_train.json` | `MISSING` | Qwen3.5 专属 GRPO split 尚未建立。 |

旧 v4 绑定 Snapshot SHA256 `dcde74d7...34aa3f`。未来新 Snapshot 与新 SFT 数据必须使用新名称和新血缘，禁止覆盖旧资产或混用哈希。

## 5. 已复现问题与最小修复

### 5.1 回归测试先红

在实现辅助函数前运行：

```text
python -m unittest AgenticArxiv.tests.test_build_multiturn_sft_v4
```

结果：`FAIL`，导入 `advance_gold_decision` 失败。这是新回归测试在修复前的失败证据。

### 5.2 根因与修复

1. setup 血缘错误：旧构建器依赖未可靠赋值的任务对象。现在从语言扩增 Manifest 的 `input_manifest` 定位 seed Manifest，校验可用哈希，并按 `derived_task_id` 建立唯一映射。
2. setup 失败被吞：旧逻辑可能把异常替换为空上下文。现在 setup 在同一链 Session 中执行，失败携带任务、工具、参数和原始原因直接抛出。
3. FINISH 被当作解析失败：现在复用训练/rollout 已有的 `parse_react_action`；FINISH 不执行工具，链末 FINISH 计入完成。
4. 执行路径不一致：setup 和 gold 工具动作都经 `ReActAgent._execute_with_side_effects`，保留项目既有下载/翻译副作用与会话语义。
5. 失败不可审计：Manifest v2 新增 `completed_chains` 和按 `parse_error`、`snapshot_miss`、`tool_error`、`premature_finish`、`missing_finish` 汇总的 `failure_counts`。
6. 轻量数据测试被 tokenizer 依赖阻塞：`scripts/prepare_sft_dataset.py` 只把 `AutoTokenizer` 延迟到 CLI `main()` 内导入；正式执行仍要求 transformers，纯 Manifest/schema helper 测试无需安装训练栈。

没有为以上逻辑新增第三方依赖。

## 6. 测试证据

所有有效测试均在 WSL Python 3.11 隔离环境中运行，且不访问真实网络。

### 6.1 构建器专项

工作目录：仓库根目录。

```bash
/home/jfm/.venvs/arxivfetch-rl-p0-py311/bin/python -m unittest \
  AgenticArxiv.tests.test_build_multiturn_sft_v4
```

结果：`PASS`，最终 8 项。覆盖 Manifest 血缘、setup Session/可见上下文、setup 失败、FINISH、工具动作 Session、非法 gold 决策、Snapshot miss 分类和一条完整链的 Manifest 统计。

### 6.2 数据生成回归

工作目录：仓库根目录。模块包括 builder、Snapshot、multi-turn env、offline benchmark、parametric SFT、SFT 准备、关键词搜索、TaskSpec、rollout audit 和 SFT 生成。

结果：`PASS`，`Ran 133 tests`，`OK (skipped=1)`；唯一预期跳过项是缺少真实 `data/mock_arxiv_snapshot.json` 的集成用例。测试中显示的固定锚点查询由 mock 提供，没有访问真实 arXiv。

### 6.3 PDF 与工具链

工作目录：`/mnt/e/2806/ArxivFetch-RL/AgenticArXiv-RL/AgenticArxiv`。

```bash
python -m pytest -q \
  tests/test_pdf_downloader.py \
  tests/test_paper_content_tool.py \
  tests/test_paper_figures_tool.py \
  tests/test_translated_content_tool.py \
  tests/test_figure_analysis_tool.py \
  tests/test_reference_resolution.py \
  tests/test_tool_registry.py \
  tests/test_tool_bootstrap.py \
  tests/test_agent_session_leak.py \
  tests/test_rollout_sandbox.py
```

结果：`PASS`，`100 passed, 1 skipped, 46 subtests passed`。pytest 的 `conftest.py` 将 PDF、译文和图片输出隔离到临时目录。

直接从仓库根目录以 `unittest` 调用这批旧测试会缺少它们要求的 `AgenticArxiv/` 导入根和 pytest fixture，产生 6 个导入错误及 7 个路径隔离误报；该命令不是有效的项目测试入口，不计为产品失败。

### 6.4 训练栈测试边界

一次包含 `test_grpo_reward` 的扩展尝试中，非 TRL 用例通过，6 项依赖 `importlib.metadata.version("trl")` 的训练测试因本地未安装 trl 报错。这些项记为 `NOT_RUN / ENV_MISSING`，不记为业务回归，也不通过安装 GPU/训练依赖来“修复”本地数据环境。

## 7. Snapshot 与论文下载风险审计

以下问题未在 P0 扩张修复，进入 P1：

1. recent-search 的派生池按 aspect 和 `max_results` 工作，但没有按请求 `days` 过滤；窄时间窗可能收到 30 天池中的旧论文。
2. `REFERENCE_POOL_IDS` 把固定 2608.* benchmark 锚点置于 AI/CV 池前部；当前 Snapshot 又没有 `as_of`，固定锚点与“最近 N 天”的语义混在一起。
3. 关键词 miss 会返回显式标记的 deterministic fallback。BaseAgent 不把它同步进会话，但 `AgenticArxivMultiTurnEnv` 仍可能写入自身 store，状态语义不一致。
4. 普通 replay miss 会 fail-closed；关键词 fallback 是有意例外，必须在 P1 明确是否保留。
5. 当前 Snapshot 顶层没有 schema/version/manifest/as_of；结果可包含本机 `local_path`，不能直接打包跨 Windows、WSL、ModelScope 使用。
6. `--content-max-ref 0` 表示选择所有论文，不是 metadata-only。默认构建可能下载并提取全部池论文，P0 禁止实际执行。
7. Git 的自动换行转换会改变大型 JSONL 的原始字节哈希。直接在当前 Windows 工作树上运行严格 SHA256 校验可能失败；P1/P2 必须确定二进制安全的数据交付或明确的 LF 策略，不能手改 Manifest 或历史 payload 掩盖差异。

## 8. 本轮变更文件

- `scripts/build_multiturn_sft_v4.py`：修复 setup、终止、执行路径、失败分类和 Manifest 血缘。
- `scripts/prepare_sft_dataset.py`：仅延迟 tokenizer 导入，保持正式 CLI 行为不变。
- `AgenticArxiv/tests/test_build_multiturn_sft_v4.py`：新增 7 项确定性回归。
- `docs/data_contract.md`：记录已验证数据契约及 P1 边界。
- `reports/p0_audit.md`：本报告。

## 9. P1-A 进入条件

P0 代码可提交到本地数据分支，但在进入 P1-A 前仍需用户确认是否继续。P1-A 的最小目标是 metadata-only Snapshot：不下载 PDF、不提取正文/图表、不翻译，带 schema、as_of、查询窗口、构建参数和哈希，并明确可支持的任务子集与完全断网回放结果。

本报告不证明真实 arXiv 完整 PDF GET 可用、不证明 ModelScope 镜像版本、不证明 Qwen3.5 加载、CUDA、LoRA 更新、SFT/GRPO 收敛或指标提升。
