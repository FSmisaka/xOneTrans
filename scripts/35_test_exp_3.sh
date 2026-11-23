#!/usr/bin/env bash
# Structural and numerical checks for the exp 3 dual-branch implementation.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${ROOT}/.venv/bin/python"; [ -x "${PY}" ] || PY=python3
cd "${ROOT}"
${PY} -m tests.test_exp_3
