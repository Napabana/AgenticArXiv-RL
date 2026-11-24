#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
previous=${PREVIOUS_GRPO_ROOT:-/root/sj-tmp/arxiv-native-grpo-20261123-jm6bPq}
export PYTHON_BIN=${PYTHON_BIN:-"$previous/grpo-venv/bin/python"} HF_HUB_OFFLINE=1 STORE_BACKEND=memory
exec "$PYTHON_BIN" scripts/grpo_credit_v2_ablation.py \
  --base "${BASE_MODEL:-/root/sj-tmp/models/Qwen3.5-2B-Base}" \
  --adapter "${SFT_ADAPTER:-/root/sj-tmp/arxiv-native-final-20261121-BN7isO/pipeline/full/adapter}" \
  --snapshot "${SNAPSHOT:-/root/sj-tmp/AgenticArXiv-RL/data/mock_arxiv_snapshot.json}" \
  --baseline "${SFT_BENCHMARK:-$previous/run/sft_benchmark}" "$@"
