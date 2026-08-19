#!/usr/bin/env bash
set -euo pipefail

usage() {
    printf 'Usage: %s YOUTUBE_URL [OUTPUT_BASENAME]\n' "${0##*/}" >&2
}

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
    usage
    exit 2
fi

URL="$1"
OUTPUT="${2:-%(title)s}.webm"

uvx --prerelease allow yt-dlp \
    --no-playlist \
    --format "bestvideo[vcodec^=vp9]+bestaudio/best[ext=webm]" \
    --merge-output-format webm \
    --output "$OUTPUT" \
    "$URL"
