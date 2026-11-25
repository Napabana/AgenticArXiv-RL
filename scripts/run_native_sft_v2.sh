#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="AgenticArxiv:scripts:${PYTHONPATH:-}" STORE_BACKEND=memory TOOL_ROUTER=policy HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset VLM_ADAPTER_PATH AA_EVAL_GREEDY
sft=${SFT_PYTHON:?set SFT_PYTHON to the v1 SFT environment}
eval_py=${EVAL_PYTHON:?set EVAL_PYTHON to the verified Native benchmark environment}
base=$(realpath "${BASE_MODEL:?}")
snapshot=$(realpath "${SNAPSHOT:?}")
baseline=$(realpath "${V1_BENCHMARK:?verified v1 benchmark directory required}")
out=$(realpath -m "${SFT_V2_OUT:?new run directory required}")
mkdir -p "$(dirname "$out")"
mkdir "$out"
train=data/sft/native_sft_v2/native_sft_v2_train.jsonl
train_mode() {
    "$sft" -u -m AgenticArxiv.rl.train_native_sft --mode "$1" --model "$base" --data "$train" --output "$out/$1" --epochs 1 --grad-accum 8 2>&1 | tee "$out/$1.log"
}
train_mode smoke
assets="$out/smoke-assets"
VLM_ADAPTER_PATH="$out/smoke/adapter" PDF_RAW_PATH="$assets/raw" PDF_TRANSLATED_PATH="$assets/translated" PDF_TRANSLATED_LOG_PATH="$assets/log" PDF_FIGURES_PATH="$assets/figures" PDF_CACHE_PATH="$assets/cache" TRANSLATE_CACHE_PATH="$assets/tcache" \
    "$eval_py" scripts/native_sft_v2_gpu.py benchmark --backend transformers --model "$base" --local-device cuda --local-dtype bfloat16 --agents native_tool_calling --offline --snapshot "$snapshot" --task-set expanded --task-ids constraint_cache_only constraint_download_only chain_ai5_read_then_summary --repeat 1 --seed 42 --no-thinking --save-traces --output "$out/smoke-eval" >"$out/smoke-eval.log" 2>&1
"$eval_py" - "$out" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
report = json.loads((root / "smoke-eval/protocol_comparison.json").read_text())
metrics = report["agents"]["native_tool_calling"]
if (report["native_protocol"]["adapter"] != str(root / "smoke/adapter")
        or metrics["benchmark_errors"] or metrics["request_errors"]
        or min(metrics[key]["count"] for key in ("format_parse_success", "schema_compliance")) < 1):
    raise SystemExit("Smoke failed to exercise an adapter-backed native call")
PY
train_mode formal
export VLM_ADAPTER_PATH="$out/formal/adapter"
mkdir "$out/benchmark"
for part in dev iid_test ood_test; do
    assets="$out/benchmark/$part-assets"
    PDF_RAW_PATH="$assets/raw" PDF_TRANSLATED_PATH="$assets/translated" PDF_TRANSLATED_LOG_PATH="$assets/log" PDF_FIGURES_PATH="$assets/figures" PDF_CACHE_PATH="$assets/cache" TRANSLATE_CACHE_PATH="$assets/tcache" \
      "$eval_py" scripts/native_sft_v2_gpu.py benchmark --backend transformers --model "$base" --local-device cuda --local-dtype bfloat16 --agents native_tool_calling --offline --snapshot "$snapshot" --task-set expanded --split "data/splits/v4_86.json:$part" --repeat 3 --seed 42 --no-thinking --save-traces --output "$out/benchmark/$part" >"$out/$part.log" 2>&1
    "$eval_py" -c 'import json,sys; p=json.load(open(sys.argv[1])); n=p["agents"]["native_tool_calling"]; assert n["benchmark_errors"]==n["request_errors"]==0 and p["native_protocol"]["system_prompt_sha256"]==sys.argv[2]' "$out/benchmark/$part/protocol_comparison.json" 31674f2183d642a9035ecf8360c138135c5ef4645904017eadf4ec7bdf4abf05
done
echo 0 > "$out/benchmark/status"
"$eval_py" scripts/report_native_sft_v2.py --baseline "$baseline" --evaluation "$out/benchmark" --adapter "$VLM_ADAPTER_PATH" --output "$out/paired_evaluation.json"
"$eval_py" scripts/native_sft_v2_gpu.py probe --base "$base" --adapter "$VLM_ADAPTER_PATH" --snapshot "$snapshot" --output "$out/probe" 2>&1 | tee "$out/probe.log"
