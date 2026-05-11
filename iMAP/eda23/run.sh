#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <input.aig> <output.seq>" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAP_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

INPUT_AIG="$1"
OUTPUT_SEQ="$2"

PYTHON_BIN="${PYTHON_BIN:-python3}"
IMAP_BIN="${IMAP_BIN:-${IMAP_ROOT}/bin/imap}"
BF_MAX_STEPS="${BF_MAX_STEPS:-3}"
SEARCH_ENGINE="${SEARCH_ENGINE:-heuristic}"

if [[ "${SEARCH_ENGINE}" == "bruteforce" ]]; then
  exec "${PYTHON_BIN}" "${SCRIPT_DIR}/bruteforce_search.py" \
    "${INPUT_AIG}" \
    "${OUTPUT_SEQ}" \
    --imap-bin "${IMAP_BIN}" \
    --max-steps "${BF_MAX_STEPS}"
fi

exec "${PYTHON_BIN}" "${SCRIPT_DIR}/heuristic_search.py" \
  "${INPUT_AIG}" \
  "${OUTPUT_SEQ}" \
  --imap-bin "${IMAP_BIN}" \
  --max-steps "${SEARCH_MAX_STEPS:-5}" \
  --beam-width "${SEARCH_BEAM_WIDTH:-8}"
