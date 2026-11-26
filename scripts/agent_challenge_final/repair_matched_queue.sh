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
pair() {
    local phase=$1 a=$2 b=$3 split=${4:-dev} failed=()
    job "$phase" "$a" "$split" & local x=$!
    job "$phase" "$b" "$split" & local y=$!
    wait "$x" || failed+=("$a")
    wait "$y" || failed+=("$b")
    for policy in "${failed[@]}"; do
        local log="$out/matched-$phase-$policy-$split.log"
        grep -Eqi 'outofmemoryerror|CUDA out of memory' "$log" || return 1
        if [[ $phase == production && -d "$out/production/$policy/$split" && ! -e "$out/production/$policy/$split/identity.json" ]]; then
            mv "$out/production/$policy/$split" "$out/production/$policy/$split-oom-$(date +%s)"
        fi
        job "$phase" "$policy" "$split" || return 1
    done
}
job dev 72
pair dev 25 48
run freeze
pair final_holdout 72 SFT
for split in dev iid_test ood_test; do
    job production 72 "$split"
done
pair final_holdout GRPO25 25
job final_holdout 48
for split in dev iid_test ood_test; do
    pair production 25 48 "$split"
done
for policy in 25 48 72; do echo 0 >"$out/production/$policy/status"; done
echo 0 >"$out/matched_evaluations_complete"
