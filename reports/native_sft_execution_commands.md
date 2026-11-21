# Qwen3.5-2B-Base Native SFT 执行命令与参数记录

本次在 Windows 编辑独立 Git 工作树，在算家云 RTX 4090 上执行 Base、LoRA 和 SFT Benchmark。以下是实际执行的阶段入口、展开命令和故障重试。重复的目录检查、任务状态读取与 loss 同步命令各列一次。所有时间采用北京时间。

## 1. Git、Snapshot 和模型检查

本地工作目录

```powershell
Set-Location -LiteralPath '<repo-root>/outputs/qwen35_2b_base_tool_calling_worktree'
$safe = 'safe.directory=<repo-root>/outputs/qwen35_2b_base_tool_calling_worktree'
git -c $safe status --short --branch
git -c $safe worktree list
git -c $safe fetch origin feat/qwen35-base-native-sft
git -c $safe log --left-right --oneline HEAD...origin/feat/qwen35-base-native-sft
git -c $safe merge --ff-only origin/feat/qwen35-base-native-sft
git -c $safe rev-parse HEAD
Get-Content -Raw -Encoding UTF8 docs/native_tool_agent_design_review.md
Get-Content -Raw -Encoding UTF8 docs/native_sft_badcase_task_plan.md
Get-FileHash -Algorithm SHA256 ../../data/mock_arxiv_snapshot.json,../../data/mock_arxiv_snapshot_d970_backup.json,data/splits/v4_86.json
git -c $safe diff --check
```

目标工作树快进到 `d1f3129a3a873cbe9be439ef33bd95e3279d3961`。GPU 源码包通过 `git ls-files -z`选取项目代码和版本化 Manifest，再用 Python `tarfile`打包当前工作树内容。包内保存 HEAD 和逐文件 SHA256。未提交的 Native 修复也包含在包内。Base 源码包 SHA256 为 `3ac4521cc1a4b5b4a5b2be1222c79a147c88006730a33ab71bfcff9084836a41`。

```powershell
ssh -o BatchMode=yes -o ConnectTimeout=12 -o HostKeyAlias='[GPU_HOST]:GPU_HOSTKEY_PORT' -o HostKeyAlgorithms=ssh-ed25519 -p GPU_SSH_PORT GPU_USER@GPU_HOST
scp -o BatchMode=yes -o HostKeyAlias='[GPU_HOST]:GPU_HOSTKEY_PORT' -o HostKeyAlgorithms=ssh-ed25519 -P GPU_SSH_PORT outputs/native_sft_execution_20261121/base_source.tar.gz GPU_USER@GPU_HOST:/root/sj-tmp/arxiv-native-sft-20261121-5qlJGz/
```

`HostKeyAlias`用于复用本机已验证的实例主机公钥。连接端口由 GPU 实例配置提供。首次公钥认证失败后由用户在网页终端授权本机公钥，随后重新连接。运行目录由 `mktemp -d /root/sj-tmp/arxiv-native-sft-20261121-XXXXXX`分配。

以下变量用于展开后面的远端命令。

```bash
RUN=/root/sj-tmp/arxiv-native-sft-20261121-5qlJGz
MODEL=/root/sj-tmp/models/Qwen3.5-2B-Base
SNAPSHOT=/root/sj-tmp/AgenticArXiv-RL/data/mock_arxiv_snapshot.json
PY=/root/sj-tmp/venvs/qwen35/bin/python
OUT="$RUN/pipeline_aligned"
DATA="$OUT/data/native_sft_train.jsonl"
cd "$RUN"
nvidia-smi -L
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader
df -h /root/sj-tmp
sha256sum "$SNAPSHOT"
(cd "$MODEL"; sha256sum -c /root/sj-tmp/arxiv-qwen35-2b-base-20261121/weights-sha256.txt)
"$PY" -m pip show torch transformers trl peft datasets tensorboard
mkdir source
tar -xzf base_source.tar.gz -C source
cd source
sha256sum -c native_source_sha256.txt
```

权重、模型配置和 tokenizer 的 5 个文件均按历史 Base 权重清单校验。实际运行采用 PyTorch 2.8.0、Transformers 5.16.1、TRL 0.29.1、PEFT 0.20.0 和 Datasets 4.8.4。

## 2. 第一阶段测试与历史轨迹重评分

在 Windows 的 Python 3.11 环境使用临时 `pip --target`安装测试辅助包，先完成 125 项聚焦回归。扩大测试时，本地环境分别缺少 Torch 或 PyMuPDF 的对应导入，随后在 GPU 实例的现有运行环境做完整定向回归。回归过程禁用 CUDA，避免占用 Base 评测显存。

```bash
cd "$RUN/source"
deps=$(mktemp -d "$RUN/pytest-XXXXXX")
"$PY" -m pip install --disable-pip-version-check --target "$deps" pytest
STORE_BACKEND=memory CUDA_VISIBLE_DEVICES="" PYTHONPATH="$deps" "$PY" -m pytest -q \
    AgenticArxiv/tests/test_native_tool_calling.py \
    AgenticArxiv/tests/test_benchmark_offline.py \
    AgenticArxiv/tests/test_multigranular_reward.py \
    AgenticArxiv/tests/test_reference_resolution.py \
    AgenticArxiv/tests/test_mock_env.py \
    AgenticArxiv/tests/test_multiturn_env.py \
    AgenticArxiv/tests/test_constraint_tasks.py \
    AgenticArxiv/tests/test_paper_content_tool.py \
    AgenticArxiv/tests/test_paper_figures_tool.py \
    AgenticArxiv/tests/test_paper_summary_tool.py \
    AgenticArxiv/tests/test_translated_content_tool.py \
    AgenticArxiv/tests/test_figure_analysis_tool.py \
    AgenticArxiv/tests/test_keyword_search.py \
    AgenticArxiv/tests/test_tool_registry.py \
    AgenticArxiv/tests/test_tool_sequence_strict.py --tb=short
```

结果为 216 项通过、1 项跳过、63 个子测试通过。同步用 Python 对 v4_86 的 86 个专家任务做 Snapshot 覆盖预检，记录在 `outputs/native_sft_phase1_rescore/expert_preflight.json`。历史轨迹由下列命令重评分，33/93 更新为 34/93。

```powershell
$py = '../../../.venv-arxivfetch-web/Scripts/python.exe'
$env:STORE_BACKEND = 'memory'
$prior = '../../outputs/qwen35_2b_base_benchmark_20261121/full'
foreach ($part in @('dev','iid_test','ood_test')) {
    & $py -m AgenticArxiv.benchmark.rescore_traces `
        --traces "$prior/$part/traces.jsonl" `
        --summary "$prior/$part/summary.json" `
        --output "outputs/native_sft_phase1_rescore/$part" `
        --split "data/splits/v4_86.json:$part" `
        --snapshot '../../data/mock_arxiv_snapshot.json'
}
```

## 3. 新 Base Benchmark

```bash
cd "$RUN/source"
bash -n scripts/run_qwen35_native_benchmark.sh
nohup env PYTHON_BIN="$PY" bash scripts/run_qwen35_native_benchmark.sh \
  "$MODEL" "$SNAPSHOT" "$RUN/base" > "$RUN/base.log" 2>&1 < /dev/null &
```

启动器用独立目录保存每个切分的工具缓存、报告和逐轮轨迹。三个切分共享下列入口。SFT 评测用同一启动器，通过第四个参数加载正式 Adapter。

```bash
export STORE_BACKEND=memory TOOL_ROUTER=policy PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
unset AA_EVAL_GREEDY VLM_ADAPTER_PATH
for part in dev iid_test ood_test; do
  assets="$RUN/base/$part-assets"
  PDF_RAW_PATH="$assets/pdf_raw" PDF_TRANSLATED_PATH="$assets/pdf_translated" \
  PDF_TRANSLATED_LOG_PATH="$assets/pdf_logs" PDF_FIGURES_PATH="$assets/pdf_figures" \
  PDF_CACHE_PATH="$assets/pdf_cache" TRANSLATE_CACHE_PATH="$assets/translate_cache" \
    "$PY" -u -m AgenticArxiv.benchmark.run_benchmark \
      --backend transformers --model "$MODEL" --local-device cuda --local-dtype bfloat16 \
      --agents native_tool_calling --offline --snapshot "$SNAPSHOT" --task-set expanded \
      --split "data/splits/v4_86.json:$part" --repeat 3 --seed 42 --no-thinking \
      --save-traces --output "$RUN/base/$part"
done
```

`--repeat 3 --seed 42`使 Base 和 SFT 按同一任务、试次及派生种子配对。通过 `--no-thinking`固定模板生成空思考前缀。Native loop 的温度为 0.1，单轮最多生成 1000 token，两个评测共用配置。Snapshot 和留出切分通过 SHA256 绑定。新 Base 在 93 次试验中严格成功 57 次，保存了 306 轮响应。

## 4. 专家回放、Chat Template 与训练数据审计

```bash
cd "$RUN/source"
"$PY" -m py_compile scripts/build_native_sft_data.py scripts/audit_native_sft_data.py \
  scripts/compare_native_sft_benchmark.py AgenticArxiv/rl/train_native_sft.py
bash -n scripts/run_qwen35_native_sft.sh scripts/run_qwen35_native_sft_pipeline.sh
"$PY" -u scripts/build_native_sft_data.py \
  --model "$MODEL" --snapshot "$SNAPSHOT" --base-results "$RUN/base" --output "$OUT/data"
"$PY" -u scripts/audit_native_sft_data.py \
  --model "$MODEL" --data "$OUT/data" --report "$OUT/native_sft_data_audit.md"
```

真实 Base Tokenizer 的输出用 `apply_chat_template(..., tools=..., enable_thinking=False)`渲染。工具参数以字典传入模板，再走相同 Native XML 解码器回解。首次审计发现译文中的 CJK 兼容汉字会由 tokenizer 归一化。审计改用 tokenizer 自身的 normalizer 比对，并保留逐 token 边界和 TRL labels 校验。首个数据版本排除了 6 条 Snapshot 缺键轨迹。按用户修正，终止该版本刚启动的正式训练，保留日志。6 条任务改用同一 Snapshot 里有完整证据的论文引用，记录了原引用、修正引用、paper_id 和 Snapshot key。正式数据重新回放 154 条完整轨迹，共 429 条决策，轨迹排除为 0。六个超出 6144 token 的决策保存在完整轨迹中，训练审计记录其排除原因。

整理期间对本地和远端的旧 V3 seed 路径执行了文件检查，并通过原生成器重建 86 个定义，与原 Manifest 的 `task_manifest_sha256`逐字段核对。训练用的专家回执由当前 Snapshot 回放重新获得。

## 5. Smoke 和正式 LoRA 命令

实际入口如下。第一次运行的是相同参数的 Smoke，得到 2 个 optimizer step，再保存并加载 Smoke Adapter。用户确认 Snapshot 对齐后，Smoke 使用对齐版 429 条决策重跑。

```bash
cd "$RUN/source"
PYTHON_BIN="$PY" SFT_MAX_STEPS=2 bash scripts/run_qwen35_native_sft.sh \
  "$MODEL" "$DATA" "$OUT/smoke" 2>&1 | tee "$OUT/smoke_train.log"

PYTHON_BIN="$PY" SFT_MAX_STEPS=-1 bash scripts/run_qwen35_native_sft.sh \
  "$MODEL" "$DATA" "$OUT/full" 2>&1 | tee "$OUT/full_train.log"
```

启动器展开为以下 Python 命令。Smoke 使用 `--max-steps 2`，正式训练使用 `--max-steps -1`。

```bash
"$PY" -u -m AgenticArxiv.rl.train_native_sft \
  --model "$MODEL" --data "$DATA" --output "$OUT/full" \
  --max-steps -1 --epochs 1 --grad-accum 8
```

| 参数 | 实际值 | 选择依据 |
| --- | --- | --- |
| `--model` | `Qwen3.5-2B-Base` | 训练从纯 Base 权重开始，加载文本 CausalLM。已逐文件核对权重与 tokenizer。 |
| `--data` | 对齐版 `native_sft_train.jsonl` | 训练前再核对 JSONL、Manifest、Tokenizer、模板和 TRL 版本的 SHA 或版本号。 |
| `--output` | `$OUT/smoke`、`$OUT/full` | 两次训练各自写入独立目录。正式目录保留 Adapter、日志、事件文件和训练 Manifest。 |
| `--max-steps` | Smoke 2，正式 -1 | 两个 optimizer step 足以确认有限 loss 和 Adapter 参数变化。正式按一个数据 epoch 运行，429 条样本与梯度累积 8 对应 54 个优化步。 |
| `--epochs` | 1 | 沿用项目第一轮 SFT 的一轮设置，得到可比较的实验结果。 |
| micro batch / `--grad-accum` | 1 / 8 | 长对话序列按单条上 GPU，累积 8 条样本后更新一次。末尾按剩余数据更新。 |
| `max_length` | 6144 | 与本次训练数据审计的长度上限一致。正式训练依然逐样本核对长度。 |
| `completion_only_loss` | true | 当前 Assistant Content、Native Tool Call 和轮结束符参与 Loss。工具定义、用户消息、历史和 Tool Observation 全部 mask。 |
| 学习率 | `5e-5` | 沿用项目 BF16 LoRA 入口的默认学习率。Smoke 无 warmup，正式 warmup 为 2 步，之后使用 cosine 调度。 |
| LoRA | rank 16、alpha 32、dropout 0.05 | 沿用现有 BF16 LoRA 配置。rank 控制更新矩阵容量，alpha 除以 rank 为 2，dropout 为 0.05。训练前检查可训练参数均为 LoRA。 |
| 精度与显存 | BF16、gradient checkpointing、关闭训练 KV cache | 单卡 RTX 4090 用 BF16 加载文本权重。Gradient checkpointing 用重算减少激活驻留显存。 |
| Optimizer | `adamw_torch_fused` | 使用已安装的 CUDA PyTorch 实现。 |
| `packing` / seed | false / 42 | 按审计过的单轮监督样本保留各自边界。固定模型初始化和数据顺序的随机种子。 |

Smoke Adapter 的推理验证命令如下。三个任务来自 Train，覆盖检索、翻译和读取译文。Smoke 通过条件包括 LoRA 参数更新、有限 loss、Adapter 成功加载和至少一次有效 Native Tool Call。

```bash
assets="$OUT/smoke_eval_assets"
VLM_ADAPTER_PATH="$OUT/smoke/adapter" PDF_RAW_PATH="$assets/pdf_raw" \
  PDF_TRANSLATED_PATH="$assets/pdf_translated" PDF_TRANSLATED_LOG_PATH="$assets/pdf_logs" \
  PDF_FIGURES_PATH="$assets/pdf_figures" PDF_CACHE_PATH="$assets/pdf_cache" \
  TRANSLATE_CACHE_PATH="$assets/translate_cache" \
  "$PY" -u -m AgenticArxiv.benchmark.run_benchmark \
    --backend transformers --model "$MODEL" --local-device cuda --local-dtype bfloat16 \
    --agents native_tool_calling --offline --snapshot "$SNAPSHOT" --task-set expanded \
    --task-ids search_CL_7d_5 opt_keep_dual trread_cv5_page2 \
    --repeat 1 --seed 42 --no-thinking --save-traces --output "$OUT/smoke_eval"
```

对齐版 Smoke 训练 loss 为 0.490860，Native 调用 1/1 通过 Schema，严格成功 1/3。用户追加实时记录要求后，正式训练首次启动遇到当前 SFTConfig 不接收 `logging_dir`，在第一个 optimizer step 前退出。该次目录和日志保存为 `full_init_failed` 和 `full_train_init_failed.log`。验证了环境中的 `torch.utils.tensorboard.SummaryWriter` 能写入 scalar 后，使用回调直接写 TensorBoard 事件与 JSONL，并由远端 `nohup bash resume_aligned.sh` 重新启动正式训练。恢复脚本先跑上面的正式训练入口，再进入下面的评测与汇总。

## 6. 正式训练显存异常与分配器重跑

首轮正式训练完成 8 步后在反向传播时 OOM。保留 `full_oom_first_attempt` 和 `full_train_oom_first_attempt.log` 供复核，从同一 Base 权重用同一数据重新训练。远端 `resume_aligned_expandable.sh` 中正式训练这一段执行为以下命令。

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  PYTHON_BIN="$PY" SFT_MAX_STEPS=-1 bash scripts/run_qwen35_native_sft.sh \
  "$MODEL" "$DATA" "$OUT/full" 2>&1 | tee "$OUT/full_train.log"
```

可扩展分段用于降低不同序列长度引起的 CUDA 缓存碎片。此次申请 4.49 GiB 时设备空闲 4.27 GiB，进程另有 4.05 GiB PyTorch 已预留但未使用内存。重跑通过原报错位置，最终完成 54/54 步，训练平均 loss 0.130971，PyTorch 已分配显存峰值 19.1145 GiB。观察、修改与验证记录见 `reports/native_sft_error_fixes.md`。重启本地监控前把失败尝试的 loss JSONL、PNG 和监控日志移入 `live/oom_first_attempt/`。

## 7. 实时 loss、最终评测和导出

正式训练每个 optimizer step 写入 `$OUT/full/metrics.jsonl`，包括 loss、grad_norm、learning_rate、epoch 等字段。TensorBoard 兼容事件写入 `$OUT/full/tensorboard/events.out.tfevents.*`，终端日志写入 `$OUT/full_train.log`。

在本地工作树用已有 Matplotlib 环境运行监控。实际监控按 15 秒读取一次远端 JSONL，数据变化后原子刷新 PNG，读取失败时重试。正式训练结束后保存全量 loss 曲线。

```powershell
$plotPython = '../../outputs/qwen35_training_curves_20261118/.venv/Scripts/python.exe'
& $plotPython -u scripts/watch_native_sft_loss.py --host GPU_USER@GPU_HOST `
  --port GPU_SSH_PORT --host-key-alias '[GPU_HOST]:GPU_HOSTKEY_PORT' `
  --remote-run '/root/sj-tmp/arxiv-native-sft-20261121-5qlJGz/pipeline_aligned' `
  --output outputs/native_sft_execution_20261121/live
```

监控自动调用以下绘图入口。正式评测与配对汇总接续运行。

```powershell
& $plotPython scripts/plot_native_sft_loss.py outputs/native_sft_execution_20261121/live/metrics.jsonl reports/native_sft_loss.png
```

```bash
cd "$RUN/source"
PYTHON_BIN="$PY" bash scripts/run_qwen35_native_benchmark.sh \
  "$MODEL" "$SNAPSHOT" "$OUT/sft" "$OUT/full/adapter"

"$PY" -u scripts/compare_native_sft_benchmark.py \
  --base "$RUN/base" --sft "$OUT/sft" --training "$OUT/full" \
  --data-manifest "$DATA.manifest.json" --output "$OUT/comparison"
```

远端运行期间通过 `date -Is`、`cat phase`、`cat status`、`tail full_train.log`、`tail full/metrics.jsonl`、`ps` 和 `nvidia-smi`检查进度、退出状态与 GPU 是否仍在工作。导出时使用 `tar -czf`打包完整逐轮轨迹、审计、训练曲线原始日志及 Adapter，用 `scp -P GPU_SSH_PORT`回传，并通过 `sha256sum`与 PowerShell `Get-FileHash -Algorithm SHA256`复核。

成功运行的 54 条 TensorBoard loss 标量与 54 行 JSONL 已逐步核对。JSONL SHA256 为 `32ec19ccc4aa7596723730506b02bf4a3cdb314e6010d908a9e747c1b87a60e4`。

## 8. 全量错误分析

```powershell
$run = 'outputs/native_sft_execution_20261121'
& $py scripts/analyze_native_sft_errors.py --base (Join-Path $run 'base') --sft (Join-Path $run 'sft') --train data/sft/native_sft_train.jsonl --output reports/native_sft_error_analysis.md
```

最终结果为 Base 57/93、SFT 64/93，24 个试次改善、17 个试次回退。配对结果与分切分逐轮轨迹 SHA256 已核验。
