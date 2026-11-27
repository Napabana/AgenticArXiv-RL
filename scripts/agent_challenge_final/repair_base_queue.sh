#!/usr/bin/env bash
set -euo pipefail
: "${BASE_REPO:?}" "${BASE_OUT:?}" "${BASE_MODEL:?}" "${SFT_REFERENCE:?}" "${BASE_PYTHON:?}"
runner=$(dirname "$(realpath "$0")")/repair_base_benchmark.py
cd "$BASE_REPO"
mkdir -p "$BASE_OUT"
exec 9>"$BASE_OUT/queue.lock"
flock -n 9
trap 'echo "$?" >"$BASE_OUT/queue.exit"' EXIT
export PYTHONPATH="scripts/agent_challenge_final:AgenticArxiv:scripts"
export HF_HUB_OFFLINE=1 STORE_BACKEND=memory OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset VLM_ADAPTER_PATH AA_EVAL_GREEDY
kind=${BASE_KIND:-Base}
run() {
    if [[ -n ${AFTER_BASE_OUT:-} ]]; then
        local marker="$AFTER_BASE_OUT/$2/Base/summary.json"
        if [[ $1 == production ]]; then
            marker="$AFTER_BASE_OUT/production/Base/$2/identity.json"
            if [[ $2 == dev ]]; then marker="$AFTER_BASE_OUT/production/Base/ood_test/identity.json"; fi
        fi
        while [[ ! -s $marker && ! -s "$AFTER_BASE_OUT/queue.exit" ]]; do sleep 5; done
        test -s "$marker" || return 1
    fi
    "$BASE_PYTHON" -u "$runner" "$1" "$2" --base "$BASE_MODEL" --reference-adapter "$SFT_REFERENCE" \
        --model-kind "$kind" --reference-base "${REFERENCE_BASE_MODEL:-$BASE_MODEL}" \
        --snapshot data/mock_arxiv_snapshot.json --output "$BASE_OUT" >>"$BASE_OUT/$1-$2.log" 2>&1
}
retry() {
    local suite=$1 split=$2 output="$BASE_OUT/production/$kind/$2"
    grep -Eqi 'OutOfMemoryError|CUDA out of memory' "$BASE_OUT/$suite-$split.log" || return 1
    if [[ $suite == production && -d "$output" && ! -f "$output/identity.json" ]]; then
        mv "$output" "$output.oom-$(date +%s)"
    fi
    run "$suite" "$split"
}
# Two new-suite splits and two production splits share the card. Start the short
# production OOD split as soon as the Dev worker finishes.
run challenge dev & a=$!
run challenge final_holdout & b=$!
run production iid_test & c=$!
( run production dev && run production ood_test ) & d=$!
pids=("$a" "$b" "$c" "$d")
suites=(challenge challenge production production)
splits=(dev final_holdout iid_test dev)
failed=()
for i in 0 1 2 3; do wait "${pids[i]}" || failed+=("$i"); done
for i in "${failed[@]}"; do
    if [[ $i == 3 ]]; then
        if [[ ! -f "$BASE_OUT/production/$kind/dev/identity.json" ]]; then
            retry production dev
            run production ood_test
        else
            retry production ood_test
        fi
    else
        retry "${suites[i]}" "${splits[i]}"
    fi
done
for s in dev iid_test ood_test; do test -s "$BASE_OUT/production/$kind/$s/identity.json"; done
for s in dev final_holdout; do test -s "$BASE_OUT/$s/$kind/summary.json"; done
echo 0 >"$BASE_OUT/production/$kind/status"
