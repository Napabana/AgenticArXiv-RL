#!/usr/bin/env bash
# Each shard owns twelve immutable tasks. Both use the original task-index seeds.
set -eu
cd "$(dirname "$0")/../.."
py=${PROBE_PYTHON:?path to the pinned TRL environment}
out=$(realpath -m "${CHALLENGE_PROBE_OUT:?output directory}")
base=$(realpath "${BASE_MODEL:?base model}")
adapter=$(realpath "${SFT_ADAPTER:?frozen Native SFT v2 adapter}")
snapshot=$(realpath "${SNAPSHOT:?snapshot}")
mkdir -p "$out"
export OMP_NUM_THREADS=1 STORE_BACKEND=memory HF_HUB_OFFLINE=1 PYTHONPATH=AgenticArxiv:scripts
args=(--base "$base" --adapter "$adapter" --snapshot "$snapshot" --output "$out")
"$py" -u scripts/agent_challenge_final/probe.py "${args[@]}" --partition front > "$out/front.log" 2>&1 &
front=$!
"$py" -u scripts/agent_challenge_final/probe.py "${args[@]}" --partition back > "$out/back.log" 2>&1 &
back=$!
failed=0
wait "$front" || failed=1
wait "$back" || failed=1
if test "$failed" != 0; then
    echo "Incomplete parallel probe; completed groups remain resumable." >&2
    exit 1
fi
"$py" scripts/agent_challenge_final/analyze_probe.py "$out"
