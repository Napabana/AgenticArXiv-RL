# Native SFT 错误与修复记录

## 正式 LoRA 训练在第 9 步显存分配失败

首轮正式训练已完成 8 个 optimizer step。第 9 步反向传播触发 `torch.OutOfMemoryError`，现场记录如下。

| 显存项 | 数值 |
| --- | ---: |
| RTX 4090 进程可见容量 | 22.03 GiB |
| 当时设备可用 | 4.27 GiB |
| 本次额外申请 | 4.49 GiB |
| PyTorch 已分配 | 13.24 GiB |
| PyTorch 已预留且未分配 | 4.05 GiB |
| 进程含 PyTorch 外显存占用 | 17.75 GiB |

本次冻结了 BF16 Base 权重，仅更新 LoRA 参数。训练输入实际最大长度 5,337 token，P95 为 4,425。当前 Transformers 的 `ForCausalLMLoss` 在计算交叉熵时调用 `logits.float()`，因此 LoRA 与 BF16 配置之外仍有激活和 FP32 logits 的峰值显存需求。旧 A10 运行同样使用 BF16 LoRA，624 步完成时报告 PyTorch 已分配峰值 19.11 GiB，软件环境为 PyTorch 2.13.0+cu130。本次算家云环境为 PyTorch 2.8.0+cu128，两个运行的峰值口径、数据和软件环境需分别核对。

按照 PyTorch 的 OOM 诊断，重跑为正式训练进程设置了可扩展显存分段。

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  PYTHON_BIN=/root/sj-tmp/venvs/qwen35/bin/python SFT_MAX_STEPS=-1 \
  bash scripts/run_qwen35_native_sft.sh "$MODEL" "$DATA" "$OUT/full"
```

这一项只改变 CUDA 缓存分配器的行为。正式训练重跑使用相同的 Base 权重、429 条审计样本、随机种子、BF16 LoRA、学习率和梯度累积配置。启动器采用同样的默认设置，并允许调用方按机器环境提供已有的 allocator 配置。重跑通过了原来的第 9 步，并在 17:03 完成 54/54 步。成功运行的 PyTorch 已分配峰值为 19.1145 GiB，训练平均 loss 为 0.130971。

初次 OOM 的远端文件保存在 `pipeline_aligned/full_oom_first_attempt`、`pipeline_aligned/full_train_oom_first_attempt.log`。本地首次尝试的原始 loss 和图保存在 `outputs/native_sft_execution_20261121/live/oom_first_attempt`，实时曲线重新从完整重跑的第 1 步开始采集。

## TensorBoard 路径配置与实时曲线

算家云的 TRL 0.29.1 配合 Transformers 5.16.1 时，`SFTConfig(logging_dir=...)`在初始化阶段抛出 `TypeError`。该次退出发生在 optimizer 运行之前。

使用 `torch.utils.tensorboard.SummaryWriter`直接写入指定输出的 `full/tensorboard/`，每步写入并 flush loss、学习率、梯度范数等标量。Trainer 回调同时追加到 `full/metrics.jsonl`。本地监控通过 SSH 读取这份 JSONL，并生成定期刷新的 PNG。Windows 隐藏后台进程未持续运行后，监控改用保活的 PowerShell 命令会话。已实际验证远端文件产生、标量持续写入和本地 PNG 刷新。浏览器页面每 15 秒读取最新 PNG。

## Native 数据设计与 Snapshot 覆盖

第一次对齐前的专家回放发现 6 条轨迹的原论文引用对应的 Snapshot 键缺失。数据设计改为在入 Catalog 时校验确有相应工具回执的论文，并对原请求、下载引用和后续工具引用同时更新。

| 训练任务 | 原序号 | 新序号和论文 | 保留的目标 |
| --- | ---: | --- | --- |
| `read_cl5_result` | 3 | 1，`2611.18533v1` | CL 实验结果章节 |
| `read_cv5_method` | 5 | 4，`2611.18559v1` | CV 方法章节 |
| `read_lg5_conclusion` | 1 | 3，`2611.18562v1` | LG 结论章节 |
| `native_augp1_read_ai5_default_cl_ref4` | 4 | 2，`2611.18527v1` | CL 标题与摘要 |
| `native_augp1_summary_ai5_tldr60_ro_ref4_bullet250` | 4 | 2，`2611.18546v1` | RO 的 bullet、250 词摘要 |
| `native_augp1_figure_cv5_ref1_ro_ref3` | 3 | 2，`2611.18546v1` | RO PDF 图表提取 |

Snapshot SHA256 为 `d97035db326ca11935e882fd2aca7fe932e9e76cf1bc8a21078be7bec1ae8794`。对齐后的 154 条轨迹全部完成专家回放。修正后的数据 Manifest 保留原引用、修正引用、论文 ID 和精确 Snapshot 键。Builder 对任何后续未覆盖的专家步骤直接报错并停止产数。

## Qwen Tokenizer 的兼容汉字归一化

第一次真实模板审计发现，译文原始文本含 `不`、`里` 等兼容汉字，经过真实 tokenizer 解码分别成为 `不`、`里`。旧审计按原始 Unicode 字面逐字检查，报出不匹配。

改用同一个 tokenizer 的 normalizer 先处理预期内容，再比对实际监督片段。Prompt token 前缀、Native Tool Call 回解与 TRL labels 都继续逐样本核验。最终 429 个决策样本通过审计。

## 最终版 SFT 的输出截断与收尾监督

上轮 SFT 的 6 个 `finish_reason:length` 全部出现在最后一轮，覆盖 `state_dl_then_cache` 的 1 个试次、`chain_cl5_dl3_tr3` 的 3 个试次和 `summary_lg5_tldr120` 的 2 个试次。逐轮轨迹显示，`state_dl_then_cache` 试次 1 已完成下载、缓存查询，随后在终答中重复 `2608.14539v1` 共 150 次。其他 5 次终答也反复输出相同论文编号和 PDF 状态。翻译链另外存在插入用户未要求的缓存查询。

系统 Prompt 增加「每个结果汇报一次，回答完即结束」。专家监督删去执行链中与终答无关的候选列表和重复的下载状态。新增缓存、翻译 PENDING、译文 READY、实际摘要等任务，确保最终回答包含用户所需信息。最终评测沿用每轮最多生成 1000 token，训练上限仍为 6144 token。两者分别控制生成和训练序列长度。

## 最终版审计发现回合结束标记与 Base EOS 不同

初版审计把 Base tokenizer 的 `<|endoftext|>`（248044）当作训练回合结束符，导致首个样本被拒绝，GPU 训练尚未启动。复核模板发现 Assistant 回合实际以 `<|im_end|>`（248046）及换行结尾，Native 推理已把该标记加入停止条件。现在逐样本验证监督区恰有一个 `<|im_end|>`、其后只有空白，且该 token 参与 TRL loss。

冻结版有 188 条完整 Snapshot 轨迹、505 条训练决策。5 条历史 Observation 在训练视图中保留前 2400 字符，完整轨迹和完整监督目标维持原值。最大训练样本 6138 token，轨迹和决策排除数均为 0。固定 Snapshot 初始缺键的 6 条旧轨迹继续使用上述已经校验的论文引用。

冻结后复查发现，`native_final_title` 的 `source_badcase_id=ref_ctrl_title_download` 是对设计计划 §6.2 E 的行为标记，未对应 v4_86 的实际任务 ID。该样本沿用 Train 父任务 `constraint_download_only`，按标题片段命中的是 CL 候选里的第 2 篇 WaferTetris（`2611.18527v1`）。相关 Train 标题引用任务为 `ref_ctrl_word_handover` 和 `ref_ctrl_word_marionette`。此处记录标记释义和实际样本，保留冻结训练集的哈希。

## 最终版继续使用已修复的显存分配和日志配置

Smoke 和正式训练都由启动器默认设置 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。正式运行依然是 Qwen3.5-2B-Base BF16 加 LoRA，使用单条 micro batch 和梯度累积 8。每步追加并 flush JSONL 与 TensorBoard 事件。自动监控通过 SSH 同步 loss 曲线，最终从完整 JSONL 生成 PNG。

### 最终版长度 6138 token 的训练在第 13 步触及物理显存峰值

第一轮正式训练已完成 12/64 步。下一步反传需额外分配 5.68 GiB，设备当时空闲 5.63 GiB，PyTorch 已分配 15.74 GiB、预留未分配约 193 MiB。可扩展分段已生效，这次峰值的主因是对包括无监督前缀在内的全部位置生成大词表 logits，交叉熵又将其转换为 FP32。

修复使用 Qwen 已支持的 `logits_to_keep`，在输出层只投影能预测非 `-100` 标签的前一 token 位置，把一致的 `shift_labels` 交给原生 CausalLM 交叉熵。Attention 仍接收完整输入，归一化仍按当前梯度累积窗口的监督 token 数。小型真实 Qwen 模型以带掩码空洞的样本对照了全部 logits 和选择后的对应 logits，loss 为 `2.2984979153` 与 `2.2984981537`，所有参数梯度通过数值一致性检查。Smoke 使用冻结集最长的 16 条样本，执行两个更新步，再加载 Adapter 测试 Native 工具调用。

原尝试停止于第一个 25 步 checkpoint 之前，因此正式重跑从同一 Base 开始。重跑复用完全相同的冻结数据、Prompt、LoRA 和训练计划。原始日志位于远端 `pipeline_dense_oom/`，实时曲线和已记录的 12 步 loss 保存在本地 `outputs/native_sft_final_20261121/loss_dense_oom/`。

最长样本组 Smoke 完成两步，PyTorch 已分配峰值 6.9038 GiB。正式训练完成 64/64 步，平均 loss 0.144628，PyTorch 已分配峰值 7.0291 GiB。上次成功的全序列投影运行记录为 19.1145 GiB，两次使用同一个 `torch.cuda.max_memory_allocated()` 统计口径。本轮每步 loss 全量回传并核对 SHA256，checkpoint 25 和 50 分别保留了 LoRA、优化器、调度器和随机状态。
