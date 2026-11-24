#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="AgenticArxiv:scripts" HF_HUB_OFFLINE=1
py=${PYTHON_BIN:-python}
out=${FORMAT_OUT:-outputs/format_control/run1}
common=(--base "${BASE_MODEL:?Set BASE_MODEL}" --adapter "${SFT_ADAPTER:?Set SFT_ADAPTER}")
"$py" scripts/format_control/audit_dataset.py
"$py" scripts/format_control/run_probe.py "${common[@]}" --output "$out/probe"
if ! "$py" -c 'import json,sys; sys.exit(not json.load(open(sys.argv[1]))["passed"])' "$out/probe/probe.json"; then
    echo "Probe below threshold. Keep this run for difficulty calibration."
    exit 2
fi
"$py" -m rl.format_control.train_format_grpo "${common[@]}" --probe "$out/probe/probe.json" --output "$out/training"
"$py" -m rl.format_control.evaluate "${common[@]}" --training "$out/training" --output "$out/evaluation"
