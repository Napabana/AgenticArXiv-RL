#!/usr/bin/env bash
# Run once on the provisioned GPU after copying the frozen checkout and assets.
set -euo pipefail
cd "$(dirname "$0")/../.."
export HF_HUB_OFFLINE=1 STORE_BACKEND=memory OMP_NUM_THREADS=1 PYTHONPATH=AgenticArxiv:scripts
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
py=${PYTHON:?training environment python}
out=$(realpath -m "${REPAIR_OUT:?new run directory}")
mkdir -p "$out"
exec 8>"$out/pipeline.lock"
flock -n 8
args=(--base "${BASE_MODEL:?}" --adapter "${SFT_ADAPTER:?}" --snapshot data/mock_arxiv_snapshot.json --output "$out")
run() { "$py" -u scripts/agent_challenge_final/repair_run.py "$@" "${args[@]}"; }
if [[ ! -f "$out/checkpoints.json" ]]; then
    test ! -e "$out/train"
    run probe >"$out/probe.log" 2>&1
    run train >"$out/train.log" 2>&1
fi
pair() {
    local phase=$1 a=$2 b=$3 result=0
    shift 3
    run "$phase" --policy "$a" "$@" >>"$out/$phase-$a.log" 2>&1 & local first=$!
    run "$phase" --policy "$b" "$@" >>"$out/$phase-$b.log" 2>&1 & local second=$!
    wait "$first" || result=1
    wait "$second" || result=1
    return "$result"
}
for phase in dev final_holdout; do
    if [[ $phase == final_holdout ]]; then run freeze; fi
    if [[ ${EVAL_PARALLEL:-1} == 2 ]]; then
        pair "$phase" SFT 24
        pair "$phase" 48 72
    else
        for policy in SFT 24 48 72; do run "$phase" --policy "$policy" >"$out/$phase-$policy.log" 2>&1; done
    fi
done
# The locked 83/93 baseline is already published. Run all three fresh checkpoints against the same 93 trials.
for split in dev iid_test ood_test; do
    if [[ ${PRODUCTION_PARALLEL:-1} == 2 ]]; then
        pair production 24 48 --split "$split"
    else
        for policy in 24 48; do
            run production --policy "$policy" --split "$split" >"$out/production-$policy-$split.log" 2>&1
        done
    fi
    run production --policy 72 --split "$split" >"$out/production-72-$split.log" 2>&1
done
for policy in 24 48 72; do
    echo 0 >"$out/production/$policy/status"
done
"$py" scripts/agent_challenge_final/repair_results.py "$out"
echo 0 >"$out/status"
