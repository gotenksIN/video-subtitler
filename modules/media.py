"""FFmpeg and FFprobe operations for complete song audio."""

import json
import math
import os
import subprocess
from pathlib import Path

AUDIO_MIME_TYPE = "audio/ogg"
EXTRACTED_AUDIO_NAME = "complete_audio.ogg"
AUDIO_DURATION_TOLERANCE_SECONDS = 2.0


def probe_duration(path: str | Path) -> float:
    """Return the positive finite container duration."""
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        duration = float(result.stdout.strip())
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        raise RuntimeError(f"Failed to probe audio duration: {error}") from error
    if not math.isfinite(duration) or duration <= 0:
        raise RuntimeError("Audio duration must be finite and greater than zero")
    return duration


def probe_audio_streams(path: str | Path) -> list[dict]:
    """Return audio stream metadata from FFprobe."""
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a",
        "-show_entries",
        "stream=codec_name,sample_rate,channels",
        "-of",
        "json",
        str(path),
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        streams = json.loads(result.stdout).get("streams", [])
    except OSError, ValueError, AttributeError, subprocess.CalledProcessError:
        return []
    return [stream for stream in streams if isinstance(stream, dict)]


def extracted_audio_is_valid(path: str | Path, source_duration: float) -> bool:
    """Check the normalized mono Opus audio and its source duration."""
    try:
        duration = probe_duration(path)
    except RuntimeError:
        return False
    streams = probe_audio_streams(path)
    return (
        len(streams) == 1
        and streams[0].get("codec_name") == "opus"
        and str(streams[0].get("sample_rate")) == "48000"
        and streams[0].get("channels") == 1
        and abs(duration - source_duration) <= AUDIO_DURATION_TOLERANCE_SECONDS
    )


def extract_complete_audio(
    source: str | Path, work_dir: str | Path
) -> tuple[Path, float]:
    """Normalize the complete first audio stream to mono Ogg Opus atomically."""
    source = Path(source)
    source_duration = probe_duration(source)
    if not probe_audio_streams(source):
        raise RuntimeError(f"Source has no audio stream: {source}")
    target = Path(work_dir) / EXTRACTED_AUDIO_NAME
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-map",
        "0:a:0",
        "-vn",
        "-sn",
        "-c:a",
        "libopus",
        "-b:a",
        "64k",
        "-ac",
        "1",
        "-ar",
        "48000",
        "-f",
        "ogg",
        str(temporary),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True)
        if not extracted_audio_is_valid(temporary, source_duration):
            raise RuntimeError("Normalized audio failed validation")
        os.replace(temporary, target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return target, source_duration
