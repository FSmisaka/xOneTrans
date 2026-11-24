#!/usr/bin/env bash
# Run exp 4_2 (low-alpha continuous spectral OneTrans, all alpha nodes <= 1).
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
[ -n "${DATASET}" ] || { echo "usage: $0 --dataset {1k|27k} [--seeds ... --conditions ... --device ...]"; exit 1; }

cd "${ROOT}"
${PY} -m src.experiments.run_experiment --exp 4_2 --dataset "${DATASET}" "${EXTRA[@]+"${EXTRA[@]}"}"
