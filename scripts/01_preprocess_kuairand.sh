#!/usr/bin/env bash
# Preprocess KuaiRand logs into parquet splits (add --debug for small sample caps).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${ROOT}/.venv/bin/python"; [ -x "${PY}" ] || PY=python3

DATASET=""
DEBUG=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset) DATASET="$2"; shift 2 ;;
    --debug) DEBUG="--debug"; shift ;;
    *) echo "unknown arg: $1"; exit 1 ;;
  esac
done
[ -n "${DATASET}" ] || { echo "usage: $0 --dataset {1k|27k} [--debug]"; exit 1; }

cd "${ROOT}"
${PY} -m src.data.kuairand_preprocess --dataset "${DATASET}" ${DEBUG}
