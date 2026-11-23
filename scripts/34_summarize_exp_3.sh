#!/usr/bin/env bash
# Rebuild exp 3 multi-seed summary.json / summary.csv / paired_deltas.csv.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${ROOT}/.venv/bin/python"; [ -x "${PY}" ] || PY=python3
cd "${ROOT}"
${PY} -m src.experiments.dual_branch --summarize-only
