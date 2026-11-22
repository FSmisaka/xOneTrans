#!/usr/bin/env bash
# Link KuaiRand raw data directories into data/raw (no copying).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET="${1:-}"

RAW_27K="/Users/york/Documents/GitHub/hand-roll/data/raw_data/KuaiRand-27K"
RAW_1K="/Users/york/Documents/GitHub/hand-roll/data/raw_data/KuaiRand-1K"

mkdir -p "${ROOT}/data/raw"
ln -sfn "${RAW_1K}" "${ROOT}/data/raw/KuaiRand-1K"
ln -sfn "${RAW_27K}" "${ROOT}/data/raw/KuaiRand-27K"
echo "linked data/raw/KuaiRand-1K -> ${RAW_1K}"
echo "linked data/raw/KuaiRand-27K -> ${RAW_27K}"
