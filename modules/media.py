"""FFmpeg and FFprobe operations for probing, splitting, and audio."""

import json
import math
import os
import re
import subprocess
from pathlib import Path

from modules import io

SPLIT_COMPLETE_MARKER = ".split_complete"
EXTRACTED_AUDIO_NAME = "extracted_audio.ogg"
AUDIO_DURATION_TOLERANCE_SECONDS = 2.0
AUDIO_MIME_TYPE = "audio/ogg"


def probe_video_format(path):
    """Probe video codec and return (extension, MIME type, codec name)."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        path,
    ]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        codec = result.stdout.strip().lower()
        if codec == "vp9":
            return ".webm", "video/webm", "vp9"
        if codec == "h264":
            return ".mp4", "video/mp4", "h264"
        if codec in ("hevc", "h265"):
            return ".mp4", "video/mp4", "hevc"
        raise RuntimeError(f"Video format not supported: {path}")
    except RuntimeError:
        raise
    except Exception as e:  # noqa: BLE001 - Wrap all probe failures consistently.
        raise RuntimeError(f"Failed to probe video format: {e}")


def ffprobe_format_duration(path: str | Path) -> float:
    """Return container duration in seconds, or 0.0 on error."""
    cmd = [
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
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            return 0.0
        parsed = float(result.stdout.strip())
        return parsed if math.isfinite(parsed) and parsed > 0 else 0.0
    except OSError, ValueError:
        return 0.0


def ffprobe_audio_streams(path: str | Path) -> list[dict]:
    """Return audio stream metadata dictionaries from FFprobe."""
    cmd = [
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
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            return []
        streams = json.loads(result.stdout).get("streams", [])
    except OSError, ValueError, AttributeError:
        return []
    return [stream for stream in streams if isinstance(stream, dict)]


def has_audio_stream(video_file: str | Path) -> bool:
    """Return True when the source contains at least one audio stream."""
    return bool(ffprobe_audio_streams(video_file))


def extracted_audio_is_valid(path: str | Path) -> bool:
    """Return True when path contains a single 48 kHz mono Opus stream."""
    if ffprobe_format_duration(path) <= 0:
        return False
    streams = ffprobe_audio_streams(path)
    if len(streams) != 1:
        return False
    stream = streams[0]
    return (
        stream.get("codec_name") == "opus"
        and str(stream.get("sample_rate")) == "48000"
        and stream.get("channels") == 1
    )


def audio_duration_consistent(audio_duration: float, source_duration: float) -> bool:
    """Return True when audio duration is within tolerance of source duration."""
    return (
        math.isfinite(audio_duration)
        and audio_duration > 0
        and math.isfinite(source_duration)
        and source_duration > 0
        and abs(audio_duration - source_duration) <= AUDIO_DURATION_TOLERANCE_SECONDS
    )


def extract_complete_audio(
    video_file: str | Path, work_dir: str | Path
) -> tuple[str, float, float, bool]:
    """Extract complete mono Opus audio from video_file into work_dir."""
    source_duration = ffprobe_format_duration(video_file)
    target = Path(work_dir) / EXTRACTED_AUDIO_NAME

    if target.exists():
        if extracted_audio_is_valid(target):
            cached_duration = ffprobe_format_duration(target)
            if audio_duration_consistent(cached_duration, source_duration):
                print("Complete audio already exists, skipping extraction.")
                return str(target), cached_duration, source_duration, True
            print(
                "Cached extracted audio is inconsistent with the source; removing it."
            )
        else:
            print("Cached extracted audio is invalid; removing it.")
        target.unlink(missing_ok=True)

    tmp_path = Path(f"{target}.tmp")
    tmp_path.unlink(missing_ok=True)
    print("Extracting complete mono Ogg Opus audio...")
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_file),
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
        str(tmp_path),
    ]
    try:
        subprocess.run(
            cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
    except Exception as e:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(
            "Failed to extract complete audio. "
            "The source may not contain an audio stream."
        ) from e

    try:
        if not extracted_audio_is_valid(tmp_path):
            raise RuntimeError(
                "Extracted audio validation failed: expected one mono Opus audio "
                "stream at 48 kHz with a positive duration."
            )

        duration = ffprobe_format_duration(tmp_path)
        if not audio_duration_consistent(duration, source_duration):
            raise RuntimeError(
                f"Extracted audio duration {duration:.3f}s is not consistent with "
                f"the source duration {source_duration:.3f}s."
            )

        os.replace(tmp_path, target)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise

    print("Complete audio extraction finished.")
    return str(target), duration, source_duration, False


def clean_incomplete_split(chunk_dir):
    """Remove split chunk files and segment index from chunk_dir."""
    for name in os.listdir(chunk_dir):
        if (
            re.fullmatch(r"chunk_\d+\.(mp4|webm)", name)
            or re.fullmatch(r"subtitle_chunk_\d+\.json(\.tmp)?", name)
            or name == "segments.csv"
        ):
            os.remove(os.path.join(chunk_dir, name))


def split_video(video_file, chunk_dir, chunk_dur_sec, manifest):
    """Split video into stream-copy chunks and record segments.csv."""
    print(f"Splitting video into {chunk_dur_sec}-second chunks (stream copy mode)...")
    os.makedirs(chunk_dir, exist_ok=True)
    io.atomic_write_json(os.path.join(chunk_dir, io.MANIFEST_NAME), manifest)

    complete_marker = os.path.join(chunk_dir, SPLIT_COMPLETE_MARKER)
    chunks = list_chunks(chunk_dir)
    listed_chunks = {chunk["name"] for chunk in chunks}
    stored_chunks = {
        name
        for name in os.listdir(chunk_dir)
        if re.fullmatch(r"chunk_\d+\.(mp4|webm)", name)
        and os.path.isfile(os.path.join(chunk_dir, name))
    }
    split_is_valid = (
        bool(chunks)
        and listed_chunks == stored_chunks
        and all(
            os.path.getsize(os.path.join(chunk_dir, chunk["name"])) > 0
            for chunk in chunks
        )
    )
    if os.path.exists(complete_marker) and split_is_valid:
        print("Chunks already exist, skipping splitting.")
        return

    if os.path.exists(complete_marker):
        os.remove(complete_marker)
    clean_incomplete_split(chunk_dir)
    ext = manifest["chunk_ext"]

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        video_file,
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-sn",
        "-c",
        "copy",
        "-f",
        "segment",
        "-segment_time",
        str(chunk_dur_sec),
        "-segment_list",
        os.path.join(chunk_dir, "segments.csv"),
        "-reset_timestamps",
        "1",
        os.path.join(chunk_dir, f"chunk_%03d{ext}"),
    ]
    subprocess.run(
        cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    Path(complete_marker).write_text("ok\n", encoding="utf-8")
    print("Splitting complete.")


def list_chunks(chunk_dir):
    """Read segments.csv and return validated chunk metadata list."""
    csv_path = os.path.join(chunk_dir, "segments.csv")
    if not os.path.exists(csv_path):
        return []

    chunks = []
    seen_names = set()
    with open(csv_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            row = line.strip()
            if not row:
                continue
            parts = row.split(",")
            name = parts[0]
            if (
                len(parts) < 3
                or not re.fullmatch(r"chunk_\d+\.(mp4|webm)", name)
                or name in seen_names
            ):
                return []
            try:
                start = float(parts[1])
                end = float(parts[2])
            except ValueError:
                return []
            if (
                not math.isfinite(start)
                or not math.isfinite(end)
                or start < 0
                or end <= start
            ):
                return []
            chunks.append(
                {
                    "idx": i,
                    "name": name,
                    "start": start,
                    "end": end,
                    "duration": end - start,
                }
            )
            seen_names.add(name)
    return chunks
