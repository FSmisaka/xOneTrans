#!/usr/bin/env bash
# Aggregate all per-seed metric JSONs into data/metrics/summary.csv.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${ROOT}/.venv/bin/python"; [ -x "${PY}" ] || PY=python3
cd "${ROOT}"
${PY} -m src.experiments.summarize
