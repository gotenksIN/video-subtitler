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

CONTEXT_ARGS=()
if [ -t 0 ]; then
    printf 'Optional context URL for grounded refinement (blank to skip): ' >&2
    read -r CONTEXT_URL || true
    if [[ "${CONTEXT_URL:-}" =~ [^[:space:]] ]]; then
        CONTEXT_ARGS+=(--context-url "$CONTEXT_URL")
    fi
fi

"${REPO_ROOT}/bin/video-subtitler" "$VIDEO" \
    --output "$OUTPUT" "${CONTEXT_ARGS[@]}"
