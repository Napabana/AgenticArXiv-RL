# 24 GB RTX 4090 执行记录

首次正式 Probe 在第 6 个训练题的长上下文注意力预填充时显存不足。GPU 的 23.52 GiB 可分配容量中当时剩余 6.20 GiB，该调用还需分配 6.21 GiB。前五个完整组已单独落盘，第六组失败的生成没有产生训练样本。

恢复执行时沿用 CPU 冻结的模型、数据、代码、评分与 Token 预算。附加入口仅在批量生成抛出 CUDA OOM 时生效。它释放失败调用、恢复调用前的 CPU 和 CUDA 随机数状态，把同一轮的输入列表二分，按原位置拼回结果。若单条轨迹仍然 OOM，向上抛出异常。分批会改变随机数分配给不同序列的次序，因此恢复后的采样与假设有足够显存的一次性采样可能不同。独立统计单位仍为原任务的一个 G=4 组，包含四条新轨迹。

第六题使用原 seed 重启，预填充的四个上下文长度为 28,866、28,866、28,868、3,499。两个物理批次各 2 条，最终收齐 4 条。该组有一条环境无效，按冻结规则整组零更新。资源恢复和评分结果分别记录。

恢复器本地 CPU 回归 2/2 通过。在实际 4090 上还注入嵌套内存异常，逐项核对了行序、行数和恢复后的 CPU/CUDA 随机序列。运行时追加文件 SHA-256 如下。

| 文件 | SHA-256 |
|---|---|
| repair_4090.py | `3c1eb217324775ba900fe18ea4d71131d98cafc7ede824d7600ace932a734b45` |
| repair_4090_python.sh | `d4599611879aa849013a86c2cad1ab4187c3829a802e6575f013465b76c470b4` |

使用训练就绪报告里的常规参数，并将原始训练环境解释器置于 `REPAIR_REAL_PYTHON`，指定恢复器路径为 `PYTHON`。

```bash
export REPAIR_REAL_PYTHON="$PYTHON"
export PYTHON="$PWD/scripts/agent_challenge_final/repair_4090_python.sh"
bash scripts/agent_challenge_final/repair_pipeline.sh
```
