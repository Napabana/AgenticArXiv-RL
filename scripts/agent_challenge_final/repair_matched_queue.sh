#!/usr/bin/env bash
# This queue resumes the existing probe, then evaluates the preregistered checkpoints.
set -euo pipefail
cd "$(dirname "$0")/../.."
: "${REPAIR_OUT:?}" "${GRPO25_ROOT:?}" "${REPAIR_REAL_PYTHON:?}" "${SFT_ADAPTER:?}" "${BASE_MODEL:?}"
export HF_HUB_OFFLINE=1 STORE_BACKEND=memory OMP_NUM_THREADS=1 PYTHONPATH=AgenticArxiv:scripts
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
out=$REPAIR_OUT
exec 9>"$out/matched_queue.lock"
flock -n 9
trap 'echo "$?" >"$out/matched_queue.exit"' EXIT
run() {
    "$REPAIR_REAL_PYTHON" -u scripts/agent_challenge_final/repair_matched_checkpoints.py "$@" \
        --base "$BASE_MODEL" --adapter "$SFT_ADAPTER" --snapshot data/mock_arxiv_snapshot.json \
        --output "$out" --grpo "$GRPO25_ROOT"
}
run register
if [[ ! -s "$out/checkpoints.json" ]]; then
    while [[ ! -s "$out/probe.json" ]] && kill -0 "${PROBE_PID:?}" 2>/dev/null; do
        [[ ! -r /proc/$PROBE_PID/stat ]] || [[ $(awk '{print $3}' "/proc/$PROBE_PID/stat") != Z ]] || break
        sleep 5
    done
    test -s "$out/probe.json"
    run train >"$out/train.log" 2>&1
fi
mkdir -p "$out/dev"
if [[ ! -d "$out/dev/GRPO25" ]]; then cp -a "$GRPO25_ROOT/dev/25" "$out/dev/GRPO25"; fi
if [[ ! -d "$out/dev/SFT" ]]; then cp -a "$GRPO25_ROOT/dev/SFT" "$out/dev/SFT"; fi
job() {
    local phase=$1 policy=$2 split=${3:-dev}
    run "$phase" --policy "$policy" --split "$split" >>"$out/matched-$phase-$policy-$split.log" 2>&1
}
parallel_limit=4
batch() {
    local phase=$1
    shift
    local policy split log index had_oom
    local -a policies splits pids offsets
    while (( $# )); do
        policies=() splits=() pids=() offsets=()
        while (( $# && ${#pids[@]} < parallel_limit )); do
            policy=$1 split=$2
            shift 2
            log="$out/matched-$phase-$policy-$split.log"
            touch "$log"
            offsets+=("$(stat -c %s "$log")")
            policies+=("$policy") splits+=("$split")
            job "$phase" "$policy" "$split" & pids+=("$!")
        done
        echo "$(date -Is) $phase policies=${policies[*]} splits=${splits[*]} concurrency=${#pids[@]}"
        local -a failed=()
        for index in "${!pids[@]}"; do
            wait "${pids[index]}" || failed+=("$index")
        done
        had_oom=0
        for index in "${failed[@]}"; do
            policy=${policies[index]} split=${splits[index]}
            log="$out/matched-$phase-$policy-$split.log"
            if ! tail -c "+$((offsets[index]+1))" "$log" | grep -Ei 'OutOfMemoryError|CUDA out of memory' >/dev/null; then
                return 1
            fi
            had_oom=1
            if [[ $phase == production && -d "$out/production/$policy/$split" && ! -e "$out/production/$policy/$split/identity.json" ]]; then
                mv "$out/production/$policy/$split" "$out/production/$policy/$split-oom-$(date +%s)"
            fi
            job "$phase" "$policy" "$split" || return 1
        done
        if (( had_oom && parallel_limit > 1 )); then parallel_limit=$((parallel_limit-1)); fi
    done
}
batch dev 72 dev 25 dev 48 dev
run freeze
batch final_holdout 72 dev SFT dev GRPO25 dev 25 dev 48 dev
# Reuse the archived SFT production trials; benchmark all four RL adapters.
for split in dev iid_test ood_test; do
    batch production 72 "$split" GRPO25 "$split" 25 "$split" 48 "$split"
done
for policy in GRPO25 25 48 72; do echo 0 >"$out/production/$policy/status"; done
echo 0 >"$out/matched_evaluations_complete"
