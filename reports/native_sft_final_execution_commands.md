# 最终 Native SFT 执行命令与参数

## 分支与输入

本轮在已有的 `qwen35_2b_base_tool_calling_worktree` 执行。拉取并快进到 `f3558864c2e5b4068c28692b31b435ad9071740c`，按最终计划直接更新 Prompt。GitHub 分支只进行了读取和拉取。

```powershell
$worktree = '<repo-root>/outputs/qwen35_2b_base_tool_calling_worktree'
git -c "safe.directory=$worktree" fetch origin feat/qwen35-base-native-sft
git -c "safe.directory=$worktree" show FETCH_HEAD:docs/native_sft_final_execution_plan.md
git -c "safe.directory=$worktree" merge --ff-only FETCH_HEAD
```

使用 `git ls-files -z`、Python `tarfile` 和当前工作树内容生成源码包。包内有 Git HEAD 和逐文件 SHA256，包含本地已有 Native Agent 修复及本轮脚本。在 Windows 和 Linux 间传输时将 Shell 脚本转换为 LF。

```powershell
$sshOptions = @('-o','BatchMode=yes','-o','ConnectTimeout=12',
    '-o','HostKeyAlias=[GPU_HOST]:GPU_HOSTKEY_PORT','-o','HostKeyAlgorithms=ssh-ed25519')
$remote = '/root/sj-tmp/arxiv-native-final-20261121-BN7isO'
scp @sshOptions -P GPU_SSH_PORT outputs/native_sft_final_20261121/source.tar.gz "GPU_USER@GPU_HOST:$remote/"
scp @sshOptions -P GPU_SSH_PORT scripts/audit_native_sft_data.py "GPU_USER@GPU_HOST:$remote/"
```

第二个 SCP 是审计对回合结束标记的修正。第一次审计在任何 GPU 训练发生前停止，失败现场位于远端 `failed_eos_audit/`。

```bash
RUN=/root/sj-tmp/arxiv-native-final-20261121-BN7isO
MODEL=/root/sj-tmp/models/Qwen3.5-2B-Base
SNAPSHOT=/root/sj-tmp/AgenticArXiv-RL/data/mock_arxiv_snapshot.json
PY=/root/sj-tmp/venvs/qwen35/bin/python
OUT="$RUN/pipeline"
cd "$RUN"
printf '%s  source.tar.gz\n' 51e0a69a48cb0bd975778adf5b5533c8bb83e3f97a20e4802eb000a5cbab2df0 | sha256sum -c -
mkdir source
tar -xzf source.tar.gz -C source
(cd source && sha256sum -c native_source_files.sha256) > verified_source.txt
printf '%s  audit_native_sft_data.py\n' dc78bb7d5d559602a00c86f020f0955b56f13fa4c2fab1813f30786383e6d32f | sha256sum -c -
cp audit_native_sft_data.py source/scripts/
printf '%s  %s\n' d97035db326ca11935e882fd2aca7fe932e9e76cf1bc8a21078be7bec1ae8794 "$SNAPSHOT" | sha256sum -c -
cd source
export PYTHON_BIN="$PY" PYTHONDONTWRITEBYTECODE=1
nohup bash scripts/run_qwen35_native_sft_pipeline.sh \
  "$MODEL" "$SNAPSHOT" "$OUT/base" "$OUT" </dev/null >../pipeline.log 2>&1 &
```

## 回归、专家回放和审计

```bash
STORE_BACKEND=memory CUDA_VISIBLE_DEVICES="" \
PYTHONPATH=/root/sj-tmp/arxiv-native-sft-20261121-5qlJGz/pytest-UDieqR "$PY" -m pytest -q \
  AgenticArxiv/tests/test_native_tool_calling.py AgenticArxiv/tests/test_benchmark_offline.py \
  AgenticArxiv/tests/test_multigranular_reward.py AgenticArxiv/tests/test_reference_resolution.py \
  AgenticArxiv/tests/test_mock_env.py AgenticArxiv/tests/test_multiturn_env.py \
  AgenticArxiv/tests/test_constraint_tasks.py AgenticArxiv/tests/test_paper_content_tool.py \
  AgenticArxiv/tests/test_paper_figures_tool.py AgenticArxiv/tests/test_paper_summary_tool.py \
  AgenticArxiv/tests/test_translated_content_tool.py AgenticArxiv/tests/test_figure_analysis_tool.py \
  AgenticArxiv/tests/test_keyword_search.py AgenticArxiv/tests/test_tool_registry.py \
  AgenticArxiv/tests/test_tool_sequence_strict.py --tb=short

"$PY" -u scripts/build_native_sft_data.py --model "$MODEL" --snapshot "$SNAPSHOT" --output "$OUT/data"
"$PY" -u scripts/audit_native_sft_data.py --model "$MODEL" --data "$OUT/data" --report "$OUT/native_sft_data_audit.md"
```

后两条由 pipeline 调用。v4_86 固定 SHA256 为 `0a176b0f67b8b61d4f84e27a40071ab81a5be3d2040c2a58aa695432f5a6eacc`。本轮先审计并冻结数据，再进行 Smoke。冻结训练集 SHA256 为 `42002233e1fe03cd4e4fbb31b44f9565f588ec1d0de5e42995f540aa87c58e68`，训练 Manifest SHA256 为 `34196a2639203ca290a43c1012daa1f82f4621a4f8beb2203c597713d013bf70`。远端和本地通过打包、SCP、SHA256 对照及逐文件 Manifest 完成校验。

旧 V3 成品 `data/sft/sft_v3_t5_parametric_seed.jsonl` 在当前工作树缺失，归档 Manifest 给出的预期 SHA256 为 `1d4d2831225e68ecf773ac9314fb6be14e085ab104dbe48266db68f80882011c`。Builder 按本仓库 V3 生成器重建 86 个任务定义，逐项匹配归档 Manifest，任务定义规范 SHA256 为 `6d5b26fe521a34837b98e4cd986c54e9a9bb64710f5b0a63e0f9c1b266c2fa32`。这 86 个定义和 55 个基础 Train 任务的监督均重新执行固定 Snapshot 获得。

CPU 回归结果为 218 项通过、1 项跳过、63 个子测试通过。首轮 GPU Smoke 两步训练 loss 为 0.463314，确认 LoRA 权重指纹发生变化，推理时 2/2 条 Native 调用通过 Schema，严格成功 1/3。权重、索引、模型配置及 tokenizer 共 5 个文件通过已有 Qwen3.5-2B-Base 权重清单的 SHA256 校验。模型权重 SHA256 为 `928acbf11878c32185bbd863514d191769285065ab9ea14fbfe431303f5fdf2d`。

## Smoke 和正式训练

```bash
DATA="$OUT/data/native_sft_train.jsonl"
PYTHON_BIN="$PY" SFT_MAX_STEPS=2 bash scripts/run_qwen35_native_sft.sh "$MODEL" "$DATA" "$OUT/smoke"
PYTHON_BIN="$PY" SFT_MAX_STEPS=-1 bash scripts/run_qwen35_native_sft.sh "$MODEL" "$DATA" "$OUT/full"
```

正式训练入口展开如下，项目脚本默认启用 CUDA 可扩展分段。训练和基准运行使用离线环境及内存会话存储。

```bash
STORE_BACKEND=memory TOOL_ROUTER=policy HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
"$PY" -u -m AgenticArxiv.rl.train_native_sft \
  --model "$MODEL" --data "$DATA" --output "$OUT/full" --max-steps -1 --epochs 1 --grad-accum 8
```

| 参数 | 实际值 | 选择依据 |
| --- | --- | --- |
| Base | Qwen3.5-2B-Base | 使用同一 Base 文本权重和 tokenizer，比对时只增减本轮 Adapter。 |
| Smoke 步数 | 2 | 在真实 GPU 上验证有限 loss 和 LoRA 权重变化。随后加载 Adapter 跑 3 个 Train 任务，验证 Native 工具调用可解析并符合 Schema。 |
| 正式步数 / epoch | `-1` / 1 | `-1` 由 Trainer 按一轮完整数据计算步数。505 条决策、梯度累积 8，预计 64 个 optimizer step。 |
| micro batch / 累积 | 1 / 8 | 长对话逐条进显存，累计梯度后更新一次，尾批使用剩余样本。 |
| LoRA | rank 16、alpha 32、dropout 0.05 | 沿用上轮；只训练 LoRA 参数，保持优化配置可比较。 |
| 精度 / 显存 | BF16、gradient checkpointing、训练 KV cache 关闭 | BF16 降低权重和激活的常驻显存。梯度检查点以重算换显存。CUDA 可扩展分段处理可变样本长度造成的碎片。 |
| 学习率 / 调度 | `5e-5`、cosine、正式 warmup 2 step | 沿用先前验证配置。Smoke 的 2 step 内 warmup 为 0，直接验证实际更新。 |
| optimizer / seed | `adamw_torch_fused` / 42 | 使用 PyTorch CUDA 实现，固定 LoRA 初始化和训练顺序。 |
| 最大训练长度 | 6144 token，实际 6138 | 沿用单条训练序列上限。5 条长样本只缩短历史 Observation，保留完整轨迹和完整监督目标。 |
| labels / packing | completion-only / false | 当前 Assistant 内容、Native call 及 `<\|im_end\|>` 参与 loss。逐样本保留原边界。 |
| checkpoint | 每 25 step，保留 2 个 | 定期保存训练状态，最终 Adapter 另存为 `full/adapter/`。 |

Smoke 推理由 pipeline 以 `--task-ids search_CL_7d_5 opt_keep_dual trread_cv5_page2 --repeat 1 --seed 42 --no-thinking --save-traces` 运行。评测用 `VLM_ADAPTER_PATH` 指向 Smoke Adapter，PDF、译文和缓存写入 Smoke 独立临时目录。正式训练由 Smoke 的有限 loss、LoRA 更新、Adapter 加载、Native 格式与 Schema 校验通过后自动触发。

### 显存峰值修复后的重试

首轮正式训练在第 13 步反传 OOM。保留整个首次运行，再重启 pipeline。导入相同 SHA256 的冻结数据，Smoke 增加实际最长输入的覆盖，正式训练参数保持上表配置。模型在算交叉熵时只生成受监督位置的 logits。

```bash
cd "$RUN"
mv pipeline pipeline_dense_oom
mv pipeline.log pipeline_dense_oom.log
mv pipeline.pid pipeline_dense_oom.pid
cd source
export NATIVE_SFT_FROZEN_RUN="$RUN/pipeline_dense_oom"
nohup bash scripts/run_qwen35_native_sft_pipeline.sh \
  "$MODEL" "$SNAPSHOT" "$OUT/base" "$OUT" </dev/null >../pipeline.log 2>&1 &
```

重试前用小型 Qwen3.5 验证对应位置 logits、loss 和全部参数梯度的一致性，保存在远端 `completion_projection_parity.log`。重试上传脚本 `train_native_sft.py` SHA256 为 `70a950e7285ed16526800f98edc3e5e5e0644a9a7da0a50a16fc5f406ea1456a`，Launcher SHA256 为 `42fe86c4d70e0df9066c55dfd6b1c95825cd5ec40f4249508cf1b8cc9efc6302`。本地已保留第一次尝试的 loss 曲线，自动监控随后跟踪重试。

重试 Smoke 两步 loss 为 0.158186，PyTorch 已分配峰值为 6.9038 GiB，发出的两个 Native 调用均通过 Schema，严格成功 2/3。正式训练于 19:19:50 进入 Base Benchmark，已完成 1 epoch、64 个 optimizer step。训练平均 loss 为 0.144628，PyTorch 已分配峰值 7.0291 GiB。正式 Adapter SHA256 为 `789af8cf28960892a1fde034d32f9b8ddf230e8450552be70d15024c964ed6c8`。

```bash
cd "$OUT/full"
tar -cf "$RUN/native_sft_checkpoints.tar" checkpoint-25 checkpoint-50
sha256sum "$RUN/native_sft_checkpoints.tar"
```

两个 checkpoint 的归档已同步本地，SHA256 为 `b556fe097c352b5989c1690205fc7d0a6317cab7ec65212fe2930326b41441a1`。

Trainer 在最后一步另外保存 `checkpoint-64`，远端按保留上限留下 50、64 两步，本地归档保留了 25、50 和 64。第 64 步的 Adapter 与导出的最终 Adapter SHA256 一致，并含 optimizer、scheduler、RNG 和 Trainer 状态。`native_sft_checkpoint_epoch1.tar` 的 SHA256 为 `d08be38d17819515ee1b77aba1d7ab7904d2e44493faf40384054d7ab5693132`。

## 实时日志与最终配对评测

训练进程每个 optimizer step 写入 `$OUT/full/metrics.jsonl` 和 `$OUT/full/tensorboard/events.out.tfevents.*`，日志保存在 `$OUT/full_train.log`。启动下列自动监控后，每 15 秒同步 loss JSONL 和刷新本地 PNG。正式结束时用完整 JSONL 生成最终 PNG。

```powershell
$plotPython = '../../outputs/qwen35_training_curves_20261118/.venv/Scripts/python.exe'
& $plotPython scripts/watch_native_sft_loss.py --host GPU_USER@GPU_HOST `
  --port GPU_SSH_PORT --host-key-alias '[GPU_HOST]:GPU_HOSTKEY_PORT' `
  --remote-run '/root/sj-tmp/arxiv-native-final-20261121-BN7isO/pipeline' `
  --output outputs/native_sft_final_20261121/loss_live

& $plotPython scripts/plot_native_sft_loss.py `
  outputs/native_sft_final_20261121/loss_live/metrics.jsonl `
  outputs/native_sft_final_20261121/native_sft_loss.png
```

```bash
PYTHON_BIN="$PY" bash scripts/run_qwen35_native_benchmark.sh "$MODEL" "$SNAPSHOT" "$OUT/base"
PYTHON_BIN="$PY" bash scripts/run_qwen35_native_benchmark.sh "$MODEL" "$SNAPSHOT" "$OUT/sft" "$OUT/full/adapter"
"$PY" -u scripts/compare_native_sft_benchmark.py --base "$OUT/base" --sft "$OUT/sft" \
  --training "$OUT/full" --data-manifest "$DATA.manifest.json" --output "$OUT/comparison"
```

同一脚本对 Dev、IID Test、OOD Test 分别使用 `--backend transformers --local-device cuda --local-dtype bfloat16 --agents native_tool_calling --offline --task-set expanded --repeat 3 --seed 42 --no-thinking --save-traces`。每组 31 道题、93 个试次，单回合生成上限为 1000 token。完整逐轮输出和工具回执由 `traces.jsonl` 保存。运行轮询读取 `phase`、`status`、`metrics.jsonl`，用 `nvidia-smi` 核对实际运行状态。

远端同一运行目录中的 `export_results.sh` 在 pipeline 退出码为 0 后运行逐试次行为分析和打包。打包记录保留 SHA256，原始 benchmark 工具缓存留在各自远端目录。

```bash
"$PY" -u scripts/analyze_native_sft_errors.py --base "$OUT/base" --sft "$OUT/sft" \
  --train "$DATA" --output "$OUT/comparison/final_error_analysis.md"
sha256sum "$RUN/training_and_benchmark.tar.gz"
```

逐试次分析计入工具序列、生效参数、论文引用、多步骤覆盖、只检索不写文件、下载后再查、PENDING、拒绝、重复调用、最终回答重复和截断、工具错误与 Snapshot 缺键。最终文本重复采用可复算的阈值，细节记录在配对错误分析中。

## 第 64 步 Dev Loss 复核

用完整性已校验的 checkpoint 归档恢复第 25 步只读副本。第 50 和 64 步直接加载原运行的 checkpoint。复核只读取 Dev 和对应模型，在独立目录写报告。

```bash
mkdir "$RUN/dev_loss"
tar -xf "$RUN/native_sft_checkpoints.tar" -C "$RUN/dev_loss" checkpoint-25
cd "$RUN/source"
STORE_BACKEND=memory TOOL_ROUTER=policy HF_HUB_OFFLINE=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
"$PY" -u scripts/evaluate_native_dev_loss.py \
  --model "$MODEL" --snapshot "$SNAPSHOT" --train-manifest "$DATA.manifest.json" \
  --checkpoints "$RUN/dev_loss/checkpoint-25" "$OUT/full/checkpoint-50" "$OUT/full/checkpoint-64" \
  --output "$RUN/dev_loss/native_sft_dev_loss.json"
```

Dev 中的邮件任务使用符合原请求的拒绝措辞，即说明无法发送约稿邮件。第一次复核借用了训练里的无法发送 PDF 附件措辞，现已在独立评估脚本纠正并重算所有 checkpoint。训练 JSONL 和 Adapter 均延续此前校验的 SHA256。短时 Dev Loss GPU 推理与 Base Benchmark 重叠，模型间对照以任务质量指标为准。

## 最终结果校验

pipeline 在 20:38:28 以状态 0 完成。将 `training_and_benchmark.tar.gz`、`training_and_benchmark.sha256` 与 `benchmark_gpu_samples.csv` 通过同一 SCP 配置同步本地。结果归档 SHA256 为 `6194194feff6a20fd73f1c3dde1d8e5ddada3041d6df338235ffb79deabe158b`。本地解压后逐份比较 Base / SFT 三个切分的轨迹文件 SHA256，验证每组 93 条且 `(task_id, trial)` 唯一。Adapter 与训练 Manifest 记录的 SHA256 一致。训练 TensorBoard 的 loss、梯度范数、学习率和 epoch 均覆盖第 1–64 步。
