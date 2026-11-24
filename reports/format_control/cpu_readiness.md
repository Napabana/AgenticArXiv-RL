# Format-RL Control 的 CPU 验收

2026-11-24 已完成 Phase 0 和第一轮 Format-RL 的 GPU 前检查。真实模型的 Probe、训练和留出成绩待测。

| 检查 | 结果 |
|---|---|
| 起点 | 从干净的 `feat/grpo-credit-v2` 提交 `2f878f07` 建立实验分支。核对 Base 全部五个文件、v1 Adapter 权重和配置、实际 tokenizer、模板、原训练清单及四个数据文件。505 条决策，188 条轨迹。 |
| 历史资产 | 核验 E1–E4 四个 25 步 Adapter 和训练配置，以及冻结 SFT 的 Dev/IID/OOD 协议。设备与显存来自历史运行记录；当前实例待开机实测。路径和 SHA 见[基线清单](baseline_manifest.json)。 |
| 格式与数据 | 24/30/120 个 Train/Dev/Test 输入无重复。174 条标准输出全部通过 parser、独立字符串校验与实际 Qwen 编解码。含结束标记的最长标准答案为 27 tokens，生成上限为 128。 |
| 真实训练入口 CPU 集成 | 微型 Qwen + LoRA，25 个 optimizer steps，4 条/组，100 条受控回答。核对逐步 KL、非零梯度、冻结参考副本、有效组数、原始回答落盘和 adapter SHA 变化。这项测试给定有成有败的生成样本，仅用于验证奖励到更新的链路。 |
| 回归 | 专项 19 项通过。完整 CPU 测试 1,086 项通过，704 个子测试通过，6 项跳过，11 条警告。 |

GPU 先执行[实验协议与启动命令](task_spec.md)。初轮 Probe 未达标时入口会停止并保留原始组数据；调整难度前检查原文。该启动脚本不执行 SFT 或 Agent GRPO。
