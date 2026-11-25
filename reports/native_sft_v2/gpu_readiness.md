# Native SFT v2 GPU 交接

本轮从 `exp/format-rl-control-sft-v2` 的 `16414ebc56afdd27cd757b6f7db83ef8a0eb4e3d` 创建独立分支 `exp/native-sft-v2`。冻结数据的校验结果见 [数据审计](data_audit.md)。

## 设备与预算

单张 RTX 4090 24 GB，BF16。v1 同类训练实测峰值为 7.03 GiB；v2 用最长样本进行两步 Smoke 后读取实际峰值。预计连续预留 **4–6 小时**。此为排期估计。主要耗时来自 33 个 Train 候选各 8 组、每组 4 条的自回归采样，共 1056 条；生成长度和新实例环境安装会影响实际时间。实例开通后按实测吞吐更新。

## GPU 执行

在分支根目录执行以下入口。环境变量分别指向具有 BF16 CUDA 支持的 v1 SFT 环境、用于重放归档评测协议的推理环境、经过 SHA 核验的 Base 目录、原始 Snapshot、已解包且含 status 的 v1 基线，以及一个新的输出目录。

```bash
export SFT_PYTHON=/path/to/v1-sft/bin/python
export EVAL_PYTHON=/path/to/native-eval/bin/python
export BASE_MODEL=/path/to/Qwen3.5-2B-Base
export SNAPSHOT="$PWD/data/mock_arxiv_snapshot.json"
export V1_BENCHMARK=/path/to/sft_benchmark
export SFT_V2_OUT="/path/to/runs/native-sft-v2-$(date +%Y%m%d-%H%M%S)"
bash scripts/run_native_sft_v2.sh
```

训练采用 TRL 0.29.1、Transformers 5.16.1。首先核对 Base 文件和数据 SHA，再从 Base 新建 LoRA。Smoke 和正式训练分别初始化，只要 Smoke 无法产生参数更新、有限 loss、完整日志和可加载的 Native Adapter，流水线就停止。正式模式使用 epoch 驱动、seed 42、微批 1、累积 8、学习率 5e-5、cosine 调度、warmup 2 步、LoRA r=16、alpha=32、dropout=0.05、最大长度 6144。逐项计数实际消费的决策，核对全部曝光及更新步数后保存权重和 SHA。

依次运行 Dev 24、IID 57、OOD 12 次，使用 v1 原提示词、工具、快照、seed 42、三次试验和相同评分器，保留所有轨迹；先校验 93 组配对再运行 Frozen Probe。有效组需要 4 条均可判定，组内同时出现 Strict 成功和失败才计入 Strict informative；按题达到 2/8 的候选单独列出，另报 shaped reward 方差。运行结束后归档 Adapter、训练账本、评测和 Probe，并在本地复算文件 SHA 后结算 GPU 时间。

## 当前基线

已回放归档 v1 评测记录的配对读取和评分。Dev 20/24、IID 44/57、OOD 8/12。v2 的实际训练曝光、Adapter SHA、三组配对结果及是否继续 Agent GRPO 的结论均依赖本轮 GPU 结果。
