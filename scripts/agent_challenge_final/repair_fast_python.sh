#!/usr/bin/env bash
# Use the verified Qwen CUDA extensions in this process, retaining the pinned trainer environment.
set -eu
test -s "${QWEN_FAST_RECEIPT:?CUDA verification receipt}"
test -d "${QWEN_FAST_SITE:?isolated kernel site}"
export PYTHONPATH="$QWEN_FAST_SITE${PYTHONPATH:+:$PYTHONPATH}"
exec "${QWEN_BASE_PYTHON:?original pinned Python}" "$@"
