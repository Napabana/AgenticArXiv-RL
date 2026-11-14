# V5 多轮 SFT 数据重建 Runbook

更新日期：2026-11-14

## 目标与边界

使用 P0 修复后的多轮 Builder，基于 **与历史 V4 相同的专家输入、任务 Manifest 和 Mock Snapshot**，生成一份独立的 V5 多轮数据：

- `data/sft/sft_v5_multiturn.jsonl`
- `data/sft/sft_v5_multiturn.jsonl.manifest.json`（Manifest schema `version=2`）

V4 原始数据、V4 的 6144-token 数据、历史 Manifest 和旧 SFT Adapter 均不覆盖。

这里的 V5 指多轮数据集命名版本；Manifest 的 `version=2` 指 Manifest schema 版本，两者不是同一个版本号。

**本任务只需要 CPU，不加载 Qwen 权重，不需要 CUDA、torch、transformers 或 TRL。** Builder 通过项目的 `MockArxivEnv(mode="replay")` 重放工具调用，不会在线访问 arXiv。运行环境仍需满足项目工具模块的 Python 依赖。

## 必需输入

从 Git 仓库当前分支可取得两个 Manifest，但大数据资产没有全部提交到 GitHub。运行前必须确认执行环境中存在：

| 文件 | 用途 |
|---|---|
| `data/sft/sft_v3_t5_parametric_linguistic.jsonl` | Builder 的实际专家决策输入，历史 Manifest 声明 2,988 行 |
| `data/sft/sft_v3_t5_parametric_linguistic.jsonl.manifest.json` | 输入 JSONL 的 SHA256，以及下游任务 Manifest 的路径和 SHA256 |
| `data/sft/sft_v3_t5_parametric_seed.jsonl.manifest.json` | 86 个派生任务的 task/setup 声明；Builder 按 `derived_task_id` 查找 setup |
| `data/mock_arxiv_snapshot.json` | 重放专家 Action 并生成 Observation |

注意：Builder 直接需要的是 linguistic JSONL 和 seed Manifest；原始 parametric seed JSONL 本体不是该 Builder 的直接输入。若 linguistic JSONL 缺失，不能拿 V4 输出 JSONL 冒充它。

为了让 V4/V5 对比只反映 Builder 修复，优先使用历史 V4 对应的 Snapshot。历史 seed Manifest 记录的 Snapshot SHA256 为：

`dcde74d7f86da03d544b6a7e27fee3e07df078be85f1b1c39d8f47e81c34aa3f`

如果当前 Snapshot 的 SHA256 不同，应先停止并确认是否有意更换环境数据；更换 Snapshot 后，V4/V5 差异就不再只由 Builder 修复造成。

## 1. 拉取代码并检查输入

在仓库根目录执行：

```bash
git pull --ff-only origin feat/arxivfetch-qwen35-2b
git rev-parse --short HEAD
```

检查输入文件和血缘哈希：

```bash
python - <<'PY'
import hashlib
import json
from pathlib import Path

def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

seed = Path("data/sft/sft_v3_t5_parametric_linguistic.jsonl")
ling_manifest_path = Path(str(seed) + ".manifest.json")
snapshot = Path("data/mock_arxiv_snapshot.json")

for path in (seed, ling_manifest_path, snapshot):
    if not path.is_file():
        raise SystemExit(f"MISSING REQUIRED INPUT: {path}")

ling = json.loads(ling_manifest_path.read_text(encoding="utf-8"))
task_manifest = Path(ling["input_manifest"])
if not task_manifest.is_file():
    raise SystemExit(f"MISSING TASK MANIFEST: {task_manifest}")

checks = [
    ("linguistic JSONL", seed, ling.get("output_sha256")),
    ("task manifest", task_manifest, ling.get("input_manifest_sha256")),
]
for name, path, expected in checks:
    actual = sha256(path)
    print(f"{name}: {path}\n  actual={actual}\n  expected={expected}")
    if expected and actual != expected:
        raise SystemExit(f"SHA256 MISMATCH: {name}")

task = json.loads(task_manifest.read_text(encoding="utf-8"))
expected_snapshot = task.get("snapshot_sha256")
actual_snapshot = sha256(snapshot)
print(f"snapshot: {snapshot}\n  actual={actual_snapshot}\n  expected={expected_snapshot}")
if expected_snapshot and actual_snapshot != expected_snapshot:
    raise SystemExit("SNAPSHOT SHA256 MISMATCH: do not mix a new snapshot into the V4/V5 comparison")

print("V5_INPUT_PREFLIGHT_OK")
PY
```

如果预检失败，不要跳过哈希检查，也不要修改旧 Manifest 来让检查通过。先恢复正确的源文件或 Snapshot。

## 2. 执行 V5 Builder（CPU）

先确认目标输出不存在，避免覆盖已有版本：

```bash
test ! -e data/sft/sft_v5_multiturn.jsonl
test ! -e data/sft/sft_v5_multiturn.jsonl.manifest.json
```

执行：

```bash
set -o pipefail
mkdir -p outputs
python -m scripts.build_multiturn_sft_v5 \
  --seed-data data/sft/sft_v3_t5_parametric_linguistic.jsonl \
  --snapshot data/mock_arxiv_snapshot.json \
  --output data/sft/sft_v5_multiturn.jsonl \
  2>&1 | tee outputs/build_sft_v5.log
```

成功标志：

`SFT_V5_MULTITURN_OK`

该命令不运行 SFT，不会加载模型。Builder 输出的 Manifest 应包含：

- `kind=sft_v5_multiturn`
- `version=2`（Manifest schema）
- `dataset_version=5`
- `chains`、`rows`
- `completed_chains`、`truncated_chains`
- `failure_counts`：`parse_error`、`snapshot_miss`、`tool_error`、`premature_finish`、`missing_finish`
- 输入、任务 Manifest、Snapshot 和输出文件的 SHA256

## 3. 校验 V5 输出

```bash
python - <<'PY'
import hashlib
import json
from pathlib import Path

data = Path("data/sft/sft_v5_multiturn.jsonl")
manifest_path = Path(str(data) + ".manifest.json")
m = json.loads(manifest_path.read_text(encoding="utf-8"))
s = m["summary"]

def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

rows = sum(1 for line in data.open(encoding="utf-8") if line.strip())
assert m["kind"] == "sft_v5_multiturn"
assert m["version"] == 2
assert m["dataset_version"] == 5
assert rows == s["rows"]
assert sha256(data) == m["output_sha256"]
assert s["chains"] == s["completed_chains"] + s["truncated_chains"]
assert sum(s.get("failure_counts", {}).values()) == s["truncated_chains"]

print(json.dumps({
    "kind": m["kind"],
    "manifest_schema_version": m["version"],
    "rows": rows,
    "chains": s["chains"],
    "completed_chains": s["completed_chains"],
    "truncated_chains": s["truncated_chains"],
    "failure_counts": s.get("failure_counts", {}),
    "output_sha256": m["output_sha256"],
    "V5_MANIFEST_VALID": True,
}, ensure_ascii=False, indent=2))
PY
```

如果上述断言全部通过，说明 V5 数据已成功生成，Manifest 的数量关系和输出哈希一致。它不等同于“所有链都完成”：仍需根据 `completed_chains` 和 `failure_counts` 判断实际轨迹质量。

## 4. 后续处理

1. 先比较 V4/V5 的链数、样本数、完成率和失败分类；抽查 setup、首步 Prompt、Action/Observation 历史及 FINISH。
2. 只有在 V5 的链级统计和样本抽查通过后，再用 Qwen3.5 Tokenizer 对 V5 做 6144-token 过滤，生成独立的 `sft_v5_multiturn_6144.jsonl` 及 Manifest。
3. 这一步仍然是 CPU 工作；不要复用 V4 的长度 Manifest，也不要覆盖 V4 文件。
4. 暂不重新训练 30-step SFT。V4 的 smoke 结果保留为旧数据上的工程验证；后续若 V5 通过，再用 V5 重新跑 smoke/正式 SFT。
5. 大型 JSONL 和 Snapshot 建议保存在持久化数据存储中。GitHub 提交代码、Manifest 和审计报告即可；不要提交模型权重、PDF、Snapshot 或运行环境目录。
