# Qwen3.5-2B-Base Native SFT · 2026-11-21

第 64 步完成 1 个 epoch。与使用同一 Native Agent、Prompt、Snapshot 和解码设置的 Base 相比，严格成功从 **46/93** 提升到 **69/93**。Dev 为 17/24 → 17/24，IID 为 20/57 → 45/57，OOD 为 9/12 → 7/12。保留第 64 步作为本轮最终模型。

## 报告与复现

- [配对 Benchmark](native_sft_benchmark.md)与[逐试次错误分析](final_error_analysis.md)
- [Dev Loss、Content 与 Final Answer 复核](../native_sft_final_dev_checkpoint.md)
- [冻结数据审计](native_sft_data_audit.md)与[训练数据 Manifest](../../data/sft/native_sft_final/native_sft_train.jsonl.manifest.json)
- [训练曲线](native_sft_loss.png)、[逐步 Loss](train_metrics.jsonl)、[训练记录](training.json)和[Smoke 结果](smoke_pass.json)
- [执行命令与参数依据](../native_sft_final_execution_commands.md)及[显存、终止和数据修复记录](../native_sft_error_fixes.md)

冻结数据含 55 个 v4_86 Train 父任务、188 条轨迹和 505 条监督决策。Dev、IID Test 和 OOD Test 任务用于评估。训练 JSONL 的 SHA256 为 `42002233e1fe03cd4e4fbb31b44f9565f588ec1d0de5e42995f540aa87c58e68`。

| 本地归档 | SHA256 |
| --- | --- |
| `frozen_data.tar.gz` | `776da5eec24985ae3d75aa11f5ba5abe2e693a197adac15322d11d5449dffc34` |
| `training_and_benchmark.tar.gz` | `6194194feff6a20fd73f1c3dde1d8e5ddada3041d6df338235ffb79deabe158b` |
| `native_sft_checkpoint_epoch1.tar` | `d08be38d17819515ee1b77aba1d7ab7904d2e44493faf40384054d7ab5693132` |
| `native_sft_checkpoints.tar` | `b556fe097c352b5989c1690205fc7d0a6317cab7ec65212fe2930326b41441a1` |
| `adapter_model.safetensors` | `789af8cf28960892a1fde034d32f9b8ddf230e8450552be70d15024c964ed6c8` |

本地归档保留冻结训练集、完整 Snapshot、两组共 186 条逐轮评测轨迹、TensorBoard events 和 25、50、64 步检查点。第 64 步包含 LoRA、优化器、调度器及随机状态。Base 可按[模型文件清单](base_model_modelscope_files.json)中的文件修订重新获取，使用[训练时校验清单](base_model.sha256)核对。 [ModelScope Base 模型](https://modelscope.cn/models/Qwen/Qwen3.5-2B-Base)
