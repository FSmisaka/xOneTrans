#!/usr/bin/env bash
# Config and wrapper checks for exp 4_2.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${ROOT}/.venv/bin/python"; [ -x "${PY}" ] || PY=python3
cd "${ROOT}"
${PY} -m tests.test_exp_4_2
