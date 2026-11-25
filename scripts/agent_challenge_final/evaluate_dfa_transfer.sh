#!/usr/bin/env bash
# Use the archived SFT Dev if available; reserve a single eval slot while the Probe is still running.
set -euo pipefail
cd "$(dirname "$0")/../.."
py=${EVAL_PYTHON:?production benchmark environment}
out=$(realpath "${CHALLENGE_RUN_OUT:?E2 transfer directory}")
probe=$(realpath "${CHALLENGE_PROBE_OUT:?frozen probe directory}")
base=$(realpath "${BASE_MODEL:?}")
sft=$(realpath "${SFT_ADAPTER:?}")
snapshot=$(realpath "${SNAPSHOT:?}")
export OMP_NUM_THREADS=1 PYTHONPATH=AgenticArxiv:scripts STORE_BACKEND=memory HF_HUB_OFFLINE=1 TOOL_ROUTER=policy
steps=$("$py" -c 'import sys,json; print(len(json.load(open(sys.argv[1]))["plan"]))' "$out/pilot_inputs.json")
eval_one() {
    local mode=$1 split=$2 target=$3 adapter=$sft
    if test "$mode" = E2; then adapter="$out/E2/pilot/checkpoint-$steps"; fi
    "$py" -u scripts/agent_challenge_final/evaluate_policy.py --base "$base" --adapter "$adapter" \
        --snapshot "$snapshot" --policy "$mode" --split "$split" --output "$target" --dev-freeze "$out/dev_freeze.json" \
        > "$out/$mode-$split.log" 2>&1
}
test -s "$out/E2/pilot/update_tokens.json"
if ! test -s "$out/SFT/dev/evaluation_identity.json"; then
    eval_one SFT dev "$out/SFT/dev"
fi
eval_one E2 dev "$out/E2/dev"
"$py" scripts/agent_challenge_final/paired.py "$out" dev --dfa-only
if test -s "$probe/probe.json"; then
    eval_one SFT sealed_test "$out/SFT/sealed_test" & a=$!
    eval_one E2 sealed_test "$out/E2/sealed_test" & b=$!
    result=0
    wait "$a" || result=1
    wait "$b" || result=1
    test "$result" = 0
else
    eval_one SFT sealed_test "$out/SFT/sealed_test"
    eval_one E2 sealed_test "$out/E2/sealed_test"
fi
"$py" scripts/agent_challenge_final/paired.py "$out" sealed_test --dfa-only
for split in dev iid_test ood_test; do
    eval_one E2 "production_$split" "$out/production/$split"
done
echo 0 > "$out/production/status"
"$py" scripts/agent_challenge_final/production.py --baseline "${SFT_V2_BENCHMARK:?archived SFT v2 benchmark}" \
    --after "$out/production" --adapter "$out/E2/pilot/checkpoint-$steps"
