# Format-RL Control 实验协议

以冻结 Native SFT v1 为策略起点，只检验它在无工具、单轮、可精确判分的输出格式上能否通过 GRPO 提升留出成绩。[资产核验](baseline_manifest.json) · [输出语法](../../data/format_control/spec.md) · [数据清单](dataset_manifest.json)

## 数据与回报

Train 24 题用于 Probe 和训练，Dev 30 题用于开发检查，Test 120 题冻结到最终配对评测。Test 含两种未在 Train 或 Dev 出现的字段类型组合。数据生成 seed 为 42，输入实例在三个切分间去重。使用 v1 的 Qwen 模板，关闭 thinking，只给用户格式指令、字段顺序和输入记录，零工具定义、零演示。严格 parser 和独立的规范字符串比较必须一致。每次正常输出仅删除生成器的终止 token，保留回答中的前后空白与特殊符号。

## GPU 阶段

1. 冻结 v1，在每道 Train 题上采 8 组，每组 4 次独立多项式抽样，温度 1。保留组 seed、组内抽样编号、回答、token 数、结束状态和二元奖励。8 组中至少 2 组有成有败、单题成功率在 (5%, 95%) 才入选。整体有效组占比须至少 30%。若首轮不通过，查看 Train/Dev 失败原文后只调整难度，保存版本和已跑结果，最多校准两轮。Test 不用于校准。
2. 通过后从原 v1 独立起步，用入选 Train 题跑 25 optimizer steps。BF16 LoRA，单卡微批 1，梯度累积 4，G=4，LR 5e-6，KL β=0.04，seed 42，completion 上限 128 tokens。保存参考模型身份、每步优化指标、全部原始训练回答和新 adapter。训练时严格回报由同一个 parser 计算，同时由规范字符串独立核对。字段级统计不参与训练。
3. v1 和新 adapter 在相同 Train、Dev、Test 上按 3 个固定种子配对，每个模型共 522 条，保留全部原文。按任务 bootstrap 5,000 次，重复种子随同一道题一起重采样。Train 严格成功提高，Test 至少 +5 个百分点、区间下界为正、三个种子方向相同，且没有字段类型准确率下降超过 5 个百分点，才按本轮规则报告通过。解析失败的输出在字段统计中按未恢复任何字段处理。规则组合分别列出改善和回退，再结合实测轨迹审阅。单个训练种子限制了结论范围。
4. 训练结果、不同回合和适配器使用独立目录。Format 控制没有通过前保留 SFT v2 的阶段门槛。本入口在控制评测后退出。

### 单卡启动

在仓库根目录，环境变量取实际已同步且与清单一致的文件。

~~~bash
export PYTHON_BIN=/path/to/pinned-venv/bin/python
export BASE_MODEL=/path/to/Qwen3.5-2B-Base
export SFT_ADAPTER=/path/to/native-sft-v1/adapter
export FORMAT_OUT=outputs/format_control/run1
bash scripts/format_control/run_control.sh
~~~

租用预算按单张 24 GB RTX 4090 预估。初轮 Probe 192 组，按组并行生成 4 条。预留 20–60 分钟。若通过，训练和两模型三种子的配对评测另预留约 45–120 分钟。冷启动、传模型和难度校准另计。首批实测后按实际 token/s 更新 ETA。这些是运行前的规划值。
