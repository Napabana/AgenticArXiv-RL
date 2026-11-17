#!/usr/bin/env bash
# Qwen3.5-2B BF16-LoRA GRPO launcher.
# Thin wrapper around AgenticArxiv.rl.train_grpo: rollout, environment,
# RewardCalculator, TRL optimization and checkpointing stay in the Python pipeline.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

BASE_MODEL="${MODEL_PATH:-/mnt/workspace/models/Qwen3.5-2B}"
SFT_ADAPTER="${SFT_ADAPTER:-outputs/qwen35_sft_full_e2/final}"
SNAPSHOT="${SNAPSHOT:-data/mock_arxiv_snapshot.json}"
SPLIT="${GRPO_SPLIT:-}"

# off      = original scalar-reward GRPO baseline
# reward   = prefix-tree credit from non-flat final rewards
# progress = reward credit plus strict-progress fallback for flat reward groups
STEP_CREDIT="${GRPO_STEP_CREDIT:-progress}"
STEP_CREDIT_WEIGHT="${GRPO_STEP_CREDIT_WEIGHT:-0.5}"
STEP_CREDIT_BETA="${GRPO_STEP_CREDIT_BETA:-1.0}"

OUTPUT_DIR="${OUTPUT_DIR:-outputs/qwen35_grpo_${STEP_CREDIT}_smoke}"
MAX_STEPS="${GRPO_MAX_STEPS:-20}"
BATCH_SIZE="${GRPO_BATCH_SIZE:-4}"
GRAD_ACCUM="${GRPO_GRAD_ACCUM:-1}"
LEARNING_RATE="${GRPO_LR:-5e-6}"
KL_BETA="${GRPO_BETA:-0.04}"
MAX_COMPLETION_LENGTH="${GRPO_MAX_COMPLETION_LENGTH:-512}"
MAX_TURNS="${GRPO_MAX_TURNS:-4}"
OPTIM="${GRPO_OPTIM:-adamw_torch_fused}"
SEED="${SEED:-42}"
RUN_NAME="${RUN_NAME:-qwen35_grpo_${STEP_CREDIT}_$(date +%Y%m%d_%H%M%S)}"

case "$STEP_CREDIT" in
  off|reward|progress) ;;
  *)
    echo "GRPO_STEP_CREDIT must be one of: off, reward, progress (got: $STEP_CREDIT)" >&2
    exit 2
    ;;
esac

[[ -d "$BASE_MODEL" ]] || { echo "Base model directory not found: $BASE_MODEL" >&2; exit 2; }
[[ -f "$SFT_ADAPTER/adapter_config.json" ]] || { echo "SFT LoRA adapter not found: $SFT_ADAPTER" >&2; exit 2; }
[[ -f "$SNAPSHOT" ]] || { echo "Offline environment snapshot not found: $SNAPSHOT. Build it before starting GRPO." >&2; exit 2; }
if [[ -z "$SPLIT" ]]; then
  echo "GRPO_SPLIT is required: create a model-specific split from the frozen Qwen3.5 SFT reward-variance probe first." >&2
  echo "Do not reuse invalidated or older-policy splits as formal Qwen3.5 training data." >&2
  exit 2
fi
[[ -f "${SPLIT%%:*}" ]] || { echo "GRPO split file not found: ${SPLIT%%:*}" >&2; exit 2; }

python - "$SPLIT" "$SFT_ADAPTER" <<'PY'
import json
import sys
from pathlib import Path

split_spec, adapter_arg = sys.argv[1:]
split_path, _, split_name = split_spec.partition(":")
if not split_name:
    raise SystemExit("GRPO_SPLIT must use the form path/to/split.json:rl_train")
payload = json.loads(Path(split_path).read_text(encoding="utf-8"))
status = str(payload.get("status", ""))
if status.startswith("invalidated"):
    raise SystemExit(f"Refusing invalidated GRPO split: status={status}")
if payload.get("task_set") != "expanded":
    raise SystemExit("GRPO split must declare task_set=expanded")
selected = (payload.get("split") or {}).get(split_name)
if not isinstance(selected, list) or not selected:
    raise SystemExit(f"GRPO split has no non-empty split named {split_name!r}")

source = payload.get("source") or {}
if source.get("policy_frozen") is not True or float(source.get("learning_rate", -1)) != 0.0:
    raise SystemExit("GRPO split must come from a frozen-policy probe (policy_frozen=true, learning_rate=0)")
source_model = source.get("model")
if not source_model:
    raise SystemExit("GRPO split must record source.model for model-specific provenance")

def resolved(value):
    path = Path(value)
    return (Path.cwd() / path).resolve() if not path.is_absolute() else path.resolve()

if resolved(source_model) != resolved(adapter_arg):
    raise SystemExit(
        "GRPO split was selected with a different policy: "
        f"split source.model={source_model!r}, current SFT_ADAPTER={adapter_arg!r}"
    )
print(f"GRPO split validated: {split_path}:{split_name} ({len(selected)} tasks; source={source_model})")
PY

mkdir -p "$OUTPUT_DIR"
git rev-parse HEAD > "$OUTPUT_DIR/git_commit.txt"
python --version > "$OUTPUT_DIR/python_version.txt" 2>&1
python -m pip freeze > "$OUTPUT_DIR/environment.freeze.txt"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi -q > "$OUTPUT_DIR/gpu_info.txt"
elif command -v amd-smi >/dev/null 2>&1; then
  amd-smi static > "$OUTPUT_DIR/gpu_info.txt" 2>&1 || true
elif command -v rocm-smi >/dev/null 2>&1; then
  rocm-smi > "$OUTPUT_DIR/gpu_info.txt" 2>&1 || true
fi

python -c 'import torch; print("torch", torch.__version__); print("cuda", torch.version.cuda); print("hip", getattr(torch.version, "hip", None)); print("gpu_available", torch.cuda.is_available()); print("device_count", torch.cuda.device_count()); print("device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")' > "$OUTPUT_DIR/device_info.txt"

cmd=(
  python -m AgenticArxiv.rl.train_grpo
  --model "$SFT_ADAPTER"
  --base_model "$BASE_MODEL"
  --snapshot "$SNAPSHOT"
  --task_set expanded
  --split "$SPLIT"
  --output_dir "$OUTPUT_DIR"
  --max_steps "$MAX_STEPS"
  --batch_size "$BATCH_SIZE"
  --grad_accum "$GRAD_ACCUM"
  --lr "$LEARNING_RATE"
  --beta "$KL_BETA"
  --reward_curriculum_steps 0
  --num_generations 4
  --step_credit "$STEP_CREDIT"
  --step_credit_weight "$STEP_CREDIT_WEIGHT"
  --step_credit_beta "$STEP_CREDIT_BETA"
  --max_completion_length "$MAX_COMPLETION_LENGTH"
  --max_turns "$MAX_TURNS"
  --no-qlora
  --bf16_lora
  --optim "$OPTIM"
  --lora_r "${LORA_R:-16}"
  --lora_alpha "${LORA_ALPHA:-32}"
  --lora_dropout "${LORA_DROPOUT:-0.05}"
  --seed "$SEED"
  --report_to tensorboard
  --run_name "$RUN_NAME"
  --save_rollout_traces
  --save_steps "${GRPO_SAVE_STEPS:-10}"
  --save_total_limit "${GRPO_SAVE_TOTAL_LIMIT:-2}"
)

if [[ "${GRPO_VERIFY:-1}" == "1" ]]; then
  cmd+=(--verify)
else
  cmd+=(--no-verify)
fi

printf 'Running:'
printf ' %q' "${cmd[@]}"
printf '\n'

if [[ "${GRPO_DRY_RUN:-0}" == "1" ]]; then
  echo "GRPO_DRY_RUN=1: command prepared; training was not started."
  exit 0
fi

if ! python -c 'import tensorboard' >/dev/null 2>&1; then
  echo "TensorBoard is required by this launcher. Install it in your environment: pip install tensorboard" >&2
  exit 2
fi

"${cmd[@]}" 2>&1 | tee "$OUTPUT_DIR/console.log"
