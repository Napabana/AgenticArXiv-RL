# 4090 内核切换与速度比较

原参考实现 GRPO 在第 14 次优化后按用户要求终止，没有保存可恢复的中间 Adapter。参考组 14 步的 TRL step_time 合计 3,528.97 秒。原始配置、轨迹、日志和关停时的 GPU 读数见 [reference14.tgz](reference14.tgz)，SHA-256 为 `7bc5218674deee0fc439c47449b957e714e16b268d02792a2e1243eff05450b2`。参考端没有连续显存采样，关停读数不能当作峰值。

加速组从相同 SFT v2 独立初始化，在线生成新的 25 个 G=4 组。保留相同的任务顺序、随机种子、配置、奖励、Token 预算和学习率时间表。两种浮点实现可能产生不同采样，所以同任务的端到端耗时也包含实际轨迹长度和工具选择的变化。

在原环境外单独安装 `causal-conv1d==1.7.0` 与 `flash-linear-attention==0.4.2`、`fla-core==0.4.2`。原 Torch 2.8.0 + CUDA 12.8、Triton 3.4.0、Transformers 5.22.2、TRL 1.17.1 均保持一致。FLA 0.6.0 在此 Triton 上导入失败，最终实验未使用。官方 CUDA 卷积 wheel 的 SHA-256 是 `0bc392a4f205a99a8a2ee801d135ab63441f5ad0e270b470ccf8e34b3676954c`。FLA Core 和上层包 SHA-256 分别为 `cba3db29380002da3cbfc0db94d6efac19aaf528900d19c05c2765e8f3cc485b` 和 `c08be006ce4dbe1be81f54938ee8e6fc7968cfba397c8d06c7669e97b8c44c0d`。

已在真实 RTX 4090 验证 Qwen 包装层的四个内核实际分派位置，涵盖分块预填充、缓存解码，以及预填充的反向梯度。数值检查用 BF16，记录相对 L2 误差。结果与软件版本见[验证回执](kernel-verification.json)。

新运行按步记录端到端时间、Rollout 时间、助手与工具反馈 Token、轮数及 PyTorch 分配器高水位，每秒采 GPU 显存、利用率、功耗。速度报告同时列各后端相同任务序号的耗时及不同工作量。固定输入测试单独在空闲 GPU 串行执行两个后端，从相同原始轨迹重建短、长输入及接下来的 16 个真实助手 Token，固定同一 SFT Adapter；每个长度热身一次后测两次，分别统计预填充、16 个 Token 缓存解码以及前向加反向。该测试不更新 Adapter，不产生任务成功率。
