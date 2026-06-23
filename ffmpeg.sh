#!/usr/bin/env bash
set -euo pipefail

usage() {
    printf 'Usage: %s\n' "${0##*/}" >&2
}

if [ "$#" -ne 0 ]; then
    usage
    exit 2
fi

need_cmd() {
    if ! command -v "$1" >/dev/null 2>&1; then
        printf 'Error: required command not found: %s\n' "$1" >&2
        exit 1
    fi
}

case "$(uname -m)" in
    x86_64|amd64)
        PLATFORM="linux64"
        ;;
    aarch64|arm64)
        PLATFORM="linuxarm64"
        ;;
    *)
        printf 'Error: unsupported architecture: %s\n' "$(uname -m)" >&2
        exit 1
        ;;
esac

need_cmd tar
need_cmd mktemp

if command -v curl >/dev/null 2>&1; then
    DOWNLOAD_CMD=(curl -fL --retry 3 --output)
elif command -v wget >/dev/null 2>&1; then
    DOWNLOAD_CMD=(wget -O)
else
    printf 'Error: curl or wget is required to download FFmpeg\n' >&2
    exit 1
fi

ARCHIVE="ffmpeg-master-latest-${PLATFORM}-gpl.tar.xz"
URL="https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/${ARCHIVE}"
BIN_DIR="${HOME}/.local/bin"
TMP_DIR="$(mktemp -d)"

cleanup() {
    rm -rf "$TMP_DIR"
}
trap cleanup EXIT

mkdir -p "$BIN_DIR"

printf 'Downloading %s...\n' "$URL"
"${DOWNLOAD_CMD[@]}" "${TMP_DIR}/${ARCHIVE}" "$URL"

printf 'Extracting %s...\n' "$ARCHIVE"
tar -xf "${TMP_DIR}/${ARCHIVE}" -C "$TMP_DIR"

EXTRACTED_DIR="${TMP_DIR}/${ARCHIVE%.tar.xz}"
install -m 0755 "${EXTRACTED_DIR}/bin/ffmpeg" "${BIN_DIR}/ffmpeg"
install -m 0755 "${EXTRACTED_DIR}/bin/ffprobe" "${BIN_DIR}/ffprobe"

printf 'Installed FFmpeg tools to %s\n' "$BIN_DIR"
"${BIN_DIR}/ffmpeg" -version | sed -n '1p'
"${BIN_DIR}/ffprobe" -version | sed -n '1p'

case ":${PATH}:" in
    *":${BIN_DIR}:"*) ;;
    *)
        printf 'Note: add %s to PATH to use these binaries from any shell.\n' "$BIN_DIR"
        printf "For example: export PATH=\"\$HOME/.local/bin:\$PATH\"\n"
        ;;
esac
