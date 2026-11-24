#!/usr/bin/env bash
# Numerical and structural checks for exp 4.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${ROOT}/.venv/bin/python"; [ -x "${PY}" ] || PY=python3
cd "${ROOT}"
${PY} -m tests.test_exp_4
