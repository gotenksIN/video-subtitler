#!/usr/bin/env bash
set -euo pipefail

usage() {
    printf 'Usage: %s VIDEO\n' "${0##*/}" >&2
}

if [ "$#" -ne 1 ]; then
    usage
    exit 2
fi

VIDEO="$1"
OUTPUT="${VIDEO}.vtt"

uv run gemini_subs.py "$VIDEO" \
    --clip-workers 3 \
    --workers 8 \
    --output "$OUTPUT"
