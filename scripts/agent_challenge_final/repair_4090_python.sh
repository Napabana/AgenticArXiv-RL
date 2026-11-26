#!/usr/bin/env bash
set -eu
if [[ ${1:-} == -u && ${2:-} == scripts/agent_challenge_final/repair_run.py ]]; then
    shift 2
    exec "${REPAIR_REAL_PYTHON:?}" -u scripts/agent_challenge_final/repair_4090.py "$@"
fi
exec "${REPAIR_REAL_PYTHON:?}" "$@"
