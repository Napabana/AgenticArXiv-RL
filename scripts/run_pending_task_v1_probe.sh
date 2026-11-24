#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
previous=${PREVIOUS_GRPO_ROOT:-/root/sj-tmp/arxiv-native-grpo-20261123-jm6bPq}
export HF_HUB_OFFLINE=1 STORE_BACKEND=memory
exec "${PYTHON_BIN:-$previous/grpo-venv/bin/python}" scripts/run_pending_task_v1.py probe \
  --base "${BASE_MODEL:-/root/sj-tmp/models/Qwen3.5-2B-Base}" \
  --adapter "${SFT_ADAPTER:-/root/sj-tmp/arxiv-native-final-20261121-BN7isO/pipeline/full/adapter}" \
  --snapshot "${SNAPSHOT:-/root/sj-tmp/AgenticArXiv-RL/data/mock_arxiv_snapshot.json}" "$@"
