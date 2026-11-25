#!/usr/bin/env bash
# Config, wrapper and continuous-representation model checks for exp 4_3.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${ROOT}/.venv/bin/python"; [ -x "${PY}" ] || PY=python3
cd "${ROOT}"
${PY} -m tests.test_exp_4_3
