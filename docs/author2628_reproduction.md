# 作者 2628 条 SFT 数据复现（Qwen3.5-2B）

目标：按原仓库公开的数据生成代码复现作者 Qwen2.5-1.5B SFT 使用的 2628 条 linguistic SFT 数据规模，再仅替换基座为 Qwen3.5-2B。

历史公开 Manifest 记录：
- seed rows: 219
- derived tasks: 80
- parent train tasks: 29
- linguistic rows: 2628
- task variants: 6
- thought variants: 2

注意：原始 JSONL 与 Mock Snapshot 被上游 .gitignore 排除，因此无法从 GitHub 恢复字节级相同数据。重新构建时 Snapshot 会随 arXiv 当前内容变化，所以这是“公开代码与任务拓扑复现”，不是“历史数据文件 SHA256 复现”。

## 1. 优先恢复历史 split v3_77

历史 Manifest 中 split SHA256：
`48bd91063c0b935e9c1f9eacab92be4f1a98f044e4ac42db826d867c47d366b9`

在完整 Git 历史中查找并导出：

```bash
EXPECTED=48bd91063c0b935e9c1f9eacab92be4f1a98f044e4ac42db826d867c47d366b9
FOUND=""
for c in $(git rev-list --all -- data/splits/v3_77.json); do
  if git cat-file -e "$c:data/splits/v3_77.json" 2>/dev/null; then
    h=$(git show "$c:data/splits/v3_77.json" | sha256sum | awk '{print $1}')
    if [ "$h" = "$EXPECTED" ]; then
      git show "$c:data/splits/v3_77.json" > data/splits/v3_77_author2628.json
      FOUND="$c"
      break
    fi
  fi
done
echo "found_commit=$FOUND"
sha256sum data/splits/v3_77_author2628.json 2>/dev/null || true
```

如果本地 Git 历史找不到该 blob，可退而使用当前 `data/splits/v3_81.json`；它会降低与历史实验的严格可比性，但不改变 80 个 no-T5 参数化派生任务本身的生成代码。

## 2. 构建用于 2628 数据生成的 Snapshot

不覆盖任何历史文件：

```bash
python -m AgenticArxiv.rl.build_snapshot \
  --snapshot data/mock_arxiv_snapshot_author2628_regen.json \
  --aspects AI LG CL CV RO CR \
  --keyword-queries 'all:large language model' \
  --max_results 50 \
  --days 30 \
  --content-max-ref 5 \
  --translate-max-ref 0
```

这里关闭译文页生成，因为 2628 的 no-T5 parametric seed 使用 `LocalSideEffectManager` 对 `translate_arxiv_pdf` 做确定性 mock；生成该 SFT 数据不需要在线 PDF 翻译。该 Snapshot 不等于后续完整 GRPO Snapshot。

## 3. 生成 219 条 parametric expert seed

若恢复了历史 split：

```bash
python scripts/generate_parametric_sft_data.py \
  --split-file data/splits/v3_77_author2628.json \
  --snapshot data/mock_arxiv_snapshot_author2628_regen.json \
  --output data/sft/sft_author2628_regen_seed.jsonl
```

不要传 `--include-t5`。预期：
- derived tasks = 80
- sample rows = 219
- parent tasks = 29
- heldout overlap = 0

## 4. 语言扩增到 2628 条

```bash
python scripts/augment_parametric_sft_data.py \
  --input data/sft/sft_author2628_regen_seed.jsonl \
  --manifest data/sft/sft_author2628_regen_seed.jsonl.manifest.json \
  --split-file data/splits/v3_77_author2628.json \
  --output data/sft/sft_author2628_regen_linguistic.jsonl \
  --wrapper-set base
```

预期：
- seed rows = 219
- output rows = 2628
- derived tasks = 80
- parent tasks = 29
- unique samples = 2628
- heldout overlap = 0

## 5. Qwen3.5 Token 长度与 Manifest 预检

```bash
python -m AgenticArxiv.rl.train_sft \
  --model /mnt/workspace/models/Qwen3.5-2B \
  --data data/sft/sft_author2628_regen_linguistic.jsonl \
  --data_manifest data/sft/sft_author2628_regen_linguistic.jsonl.manifest.json \
  --max_length 6144 \
  --no-qlora \
  --bf16_lora \
  --inspect_only
```

若没有超长错误，就直接保留全部 2628 条，不再做长度过滤。

## 6. 30-step Smoke（仅工程检查）

```bash
SFT_DATA=data/sft/sft_author2628_regen_linguistic.jsonl \
SFT_MANIFEST=data/sft/sft_author2628_regen_linguistic.jsonl.manifest.json \
SFT_MAX_STEPS=30 \
SFT_EPOCHS=2 \
OUTPUT_DIR=outputs/qwen35_author2628_smoke \
bash scripts/run_qwen35_sft.sh
```

## 7. 正式 2 epoch SFT

Smoke 通过后：

```bash
SFT_DATA=data/sft/sft_author2628_regen_linguistic.jsonl \
SFT_MANIFEST=data/sft/sft_author2628_regen_linguistic.jsonl.manifest.json \
SFT_MAX_STEPS=-1 \
SFT_EPOCHS=2 \
OUTPUT_DIR=outputs/qwen35_author2628_full \
bash scripts/run_qwen35_sft.sh
```

作者公开 Qwen2.5-1.5B SFT 是全参数 2 epoch；当前 Qwen3.5-2B 路径为单卡 BF16 LoRA。因此这版首先控制“数据生成协议/任务拓扑/epoch”，训练参数化方式仍是一个明确的实验差异。若后续有 A100 显存预算，再单独补全参数 SFT 对照。
