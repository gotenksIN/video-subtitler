#!/usr/bin/env bash
set -euo pipefail

usage() {
    printf 'Usage: %s VIDEO\n' "${0##*/}" >&2
}

if [ "$#" -ne 1 ]; then
    usage
    exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

VIDEO="$1"
OUTPUT="${VIDEO}.vtt"

uv run --project "$REPO_ROOT" "${REPO_ROOT}/gemini_subs.py" "$VIDEO" \
    --workers 8 \
    --output "$OUTPUT"
