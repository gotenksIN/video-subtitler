#!/usr/bin/env bash
set -euo pipefail

usage() {
    printf 'Usage: %s VIDEO [VTT_FILE] [OUTPUT]\n' "${0##*/}" >&2
    exit 1
}

if [ "$#" -lt 1 ] || [ "$#" -gt 3 ]; then
    usage
fi

VIDEO="$1"
VTT="${2:-${VIDEO}.vtt}"

if [ "$#" -ge 3 ]; then
    OUTPUT="$3"
else
    BASENAME="${VIDEO%.*}"
    EXT="${VIDEO##*.}"
    OUTPUT="${BASENAME}.subs.${EXT}"
fi

[ -f "$VIDEO" ] || { printf 'Error: video not found: %s\n' "$VIDEO" >&2; exit 1; }
[ -f "$VTT" ]   || { printf 'Error: VTT not found: %s\n' "$VTT" >&2; exit 1; }

ffmpeg -n -i "$VIDEO" -i "$VTT" \
    -c copy \
    -map 0 -map 1 \
    -metadata:s:s:0 language=eng \
    -disposition:s:0 default \
    "$OUTPUT"
