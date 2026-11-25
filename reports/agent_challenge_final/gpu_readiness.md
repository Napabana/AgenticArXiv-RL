# Challenge GPU 准备记录

| 资产 | 固定标识 |
|---|---|
| Native SFT v2 | f0c33a4d32a175bcaf1e83c55231a0108a84859cb748c318db49988a52f7911f |
| 原生产提示词 | 31674f2183d642a9035ecf8360c138135c5ef4645904017eadf4ec7bdf4abf05 |
| Snapshot | d97035db326ca11935e882fd2aca7fe932e9e76cf1bc8a21078be7bec1ae8794 |
| 新训练候选 | e9ad62fdd0401bb6cf27aab89119e2b9ffc8e088f2568b5d62b3c7125434cd82 |
| Dev | 125fb28bfc62920c633b80cb00cb794ca96a8f87ae5afbf7fcb33c50160e9dfd |
| Sealed | 413011d68c948688f3d935c0712c73c3528803b0e447dba4ac966ae9e0eea327 |

原环境 TRL 1.17.1，历史默认损失归一化为 dapo。冻结最新 SFT v2 的实际提示词、工具 Schema、Chat Template 和 Tokenizer，详见 asset_identity.json。CPU 回归运行 test_grpo_credit_v2.py 和 test_native_grpo_rollout.py，46 项通过。所有 44 道 Gold 任务重复回放，243 个决策对齐。全部数据文件和审计文件 SHA 见 data/agent_challenge_final/manifest.json。

从部署仓库目录执行：

新门槛、原始与有效信号区分、固定计划以及原 Credit/Rollout 回归合计 48 项通过（7 条已有 Pydantic 弃用提示）。4090 长任务首个 G=4 实测为 29.8 秒，CUDA 峰值分配 5.47 GiB。四条采样的 completion 长度为 203、211、212、211，此结果仅验证初始推理链路。双进程可按前后半池分别传入 `--partition front` 和 `--partition back`，原任务序号决定 Seed，两半写各自的任务目录。两半完成后用 `analyze_probe.py` 汇总。

```bash
export PYTHONPATH=AgenticArxiv:scripts HF_HUB_OFFLINE=1 STORE_BACKEND=memory OMP_NUM_THREADS=1
/root/sj-tmp/grpo-credit-56517-bfaaf59/grpo-env/bin/python -u scripts/agent_challenge_final/probe.py \
  --base /root/sj-tmp/models/Qwen3.5-2B-Base \
  --adapter /root/sj-tmp/native-sft-v2-56780/run/formal/adapter \
  --snapshot data/mock_arxiv_snapshot.json --output ../probe
```

每题 8 个独立组，每组 4 条。首个采样组用于长训练任务的推理 Smoke，保留原定 Seed。限时预算以实测为准，任务书预估 Probe 2–4 小时，全部 GPU 阶段 6–9 小时。高截断在采样前操作化为单题超过 8/32 条截断；分母含全部实际采样。入选与训练门槛按任务书 §5.3 执行。DFA 和 Anchor 非零覆盖率以有效组中可判定的模型决策为分母，包含首错之后的已采样动作。每种消融使用相同的 25-group 固定计划和独立的 SFT v2 初始化。
