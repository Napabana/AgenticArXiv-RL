#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
py=${TRAIN_PYTHON:?pinned TRL environment}
eval_py=${EVAL_PYTHON:?production benchmark environment}
base=$(realpath "${BASE_MODEL:?}")
adapter=$(realpath "${SFT_ADAPTER:?}")
snapshot=$(realpath "${SNAPSHOT:?}")
probe=$(realpath "${CHALLENGE_PROBE_OUT:?}")
out=$(realpath -m "${CHALLENGE_RUN_OUT:?new run directory}")
export PYTHONPATH=AgenticArxiv:scripts HF_HUB_OFFLINE=1 STORE_BACKEND=memory TOOL_ROUTER=policy OMP_NUM_THREADS=1
"$py" scripts/agent_challenge_final/analyze_probe.py "$probe"
if ! "$py" -c 'import json,sys;sys.exit(json.load(open(sys.argv[1]))["decision"]!="GO")' "$probe/probe.json"; then
    echo "NO_GO: see probe.json"
    exit 0
fi
mkdir -p "$out"
assets=(--base "$base" --snapshot "$snapshot")
train_stage() {
    local mode=$1 stage=$2
    local opts=()
    if test "$stage" = smoke; then opts=(--smoke); fi
    "$py" scripts/agent_challenge_final/train.py "${assets[@]}" --adapter "$adapter" --probe "$probe" \
        --experiment "$mode" --output "$out/$mode/$stage" "${opts[@]}"
}
for pair in "E1 E2" "E3 E4"; do
    read -r first second <<< "$pair"
    parallel=${ABLATION_WORKERS:-2}
    for stage in smoke pilot; do
        if test "$parallel" = 1; then
            for mode in "$first" "$second"; do
                train_stage "$mode" "$stage" > "$out/$mode-$stage.log" 2>&1
            done
            continue
        fi
        pids=()
        for mode in "$first" "$second"; do
            train_stage "$mode" "$stage" > "$out/$mode-$stage.log" 2>&1 &
            pids+=("$!")
        done
        failed=()
        wait "${pids[0]}" || failed+=("$first")
        wait "${pids[1]}" || failed+=("$second")
        for mode in "${failed[@]}"; do
            log="$out/$mode-$stage.log"
            if ! grep -Eq 'CUDA out of memory|torch.OutOfMemoryError' "$log"; then
                tail -n 25 "$log" >&2
                exit 1
            fi
            # Preserve every artifact of the failed attempt before replaying from its fixed start.
            fallback=$(mktemp -d "$out/$mode-$stage-oom.XXXXXX")
            mv "$log" "$fallback/output.log"
            if test -d "$out/$mode/$stage"; then mv "$out/$mode/$stage" "$fallback/attempt"; fi
            parallel=1
            train_stage "$mode" "$stage" > "$log" 2>&1
        done
    done
done
for split in dev sealed_test; do
    for mode in SFT E1 E2 E3 E4; do
        checkpoint=$adapter
        if test "$mode" != SFT; then checkpoint="$out/$mode/pilot/checkpoint-25"; fi
        "$eval_py" scripts/agent_challenge_final/evaluate_policy.py "${assets[@]}" --adapter "$checkpoint" \
            --policy "$mode" --split "$split" --dev-freeze "$out/dev_freeze.json" --output "$out/$mode/$split"
    done
    "$py" scripts/agent_challenge_final/paired.py "$out" "$split"
done
best=$("$py" -c 'import sys,json;print(json.load(open(sys.argv[1]))["winner"])' "$out/dev_freeze.json")
for split in dev iid_test ood_test; do
    "$eval_py" scripts/agent_challenge_final/evaluate_policy.py "${assets[@]}" --adapter "$out/$best/pilot/checkpoint-25" \
        --policy "$best" --split "production_$split" --dev-freeze "$out/dev_freeze.json" --output "$out/production/$split"
done
echo 0 > "$out/production/status"
"$eval_py" scripts/agent_challenge_final/production.py --baseline "${SFT_V2_BENCHMARK:?archived v2 benchmark}" \
    --after "$out/production" --adapter "$out/$best/pilot/checkpoint-25"
