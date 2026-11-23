#!/usr/bin/env bash
# Run exp 2_3 (exp_2_3_gate_analysis) on the given dataset.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${ROOT}/.venv/bin/python"; [ -x "${PY}" ] || PY=python3

DATASET=""
EXTRA=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset) DATASET="$2"; shift 2 ;;
    *) EXTRA+=("$1"); shift ;;
  esac
done
[ -n "${DATASET}" ] || { echo "usage: $0 --dataset {1k|27k} [--seeds ... --device ...]"; exit 1; }

cd "${ROOT}"
${PY} -m src.experiments.run_experiment --exp 2_3 --dataset "${DATASET}" "${EXTRA[@]+"${EXTRA[@]}"}"
