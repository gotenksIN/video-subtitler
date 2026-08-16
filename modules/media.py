"""FFmpeg and FFprobe operations for probing, splitting, and overlap clips."""

import os
import re
import subprocess
from pathlib import Path

from modules import core, io

SPLIT_COMPLETE_MARKER = ".split_complete"


def probe_video_format(path):
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


def clean_incomplete_split(chunk_dir):
    for name in os.listdir(chunk_dir):
        if (
            re.fullmatch(r"chunk_\d+\.(mp4|webm)", name)
            or re.fullmatch(r"context_chunk_\d+\.(mp4|webm)(\.tmp)?", name)
            or re.fullmatch(r"subtitle_chunk_\d+\.json(\.tmp)?", name)
            or name == "segments.csv"
        ):
            os.remove(os.path.join(chunk_dir, name))


def split_video(video_file, chunk_dir, chunk_dur_sec, manifest):
    print(f"Splitting video into {chunk_dur_sec}-second chunks (stream copy mode)...")
    os.makedirs(chunk_dir, exist_ok=True)
    io.atomic_write_json(os.path.join(chunk_dir, io.MANIFEST_NAME), manifest)

    complete_marker = os.path.join(chunk_dir, SPLIT_COMPLETE_MARKER)
    chunks = list_chunks(chunk_dir)
    split_is_valid = chunks and all(
        os.path.isfile(path := os.path.join(chunk_dir, chunk["name"]))
        and os.path.getsize(path) > 0
        for chunk in chunks
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
    csv_path = os.path.join(chunk_dir, "segments.csv")
    if not os.path.exists(csv_path):
        return []

    chunks = []
    with open(csv_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            parts = line.strip().split(",")
            if len(parts) >= 3:
                name = parts[0]
                start = float(parts[1])
                end = float(parts[2])
                chunks.append(
                    {
                        "idx": i,
                        "name": name,
                        "start": start,
                        "end": end,
                        "duration": end - start,
                    }
                )
    return chunks


def get_processing_windows(chunks, overlap_sec):
    if not chunks:
        return []

    video_end = chunks[-1]["end"]
    windows = []
    for chunk in chunks:
        owner_start = chunk["start"]
        owner_end = chunk["end"]
        clip_start = max(0.0, owner_start - overlap_sec)
        clip_end = min(video_end, owner_end + overlap_sec)
        windows.append(
            {
                **chunk,
                "clip_start": clip_start,
                "clip_end": clip_end,
                "clip_duration": clip_end - clip_start,
                "owner_start": owner_start,
                "owner_end": owner_end,
                "owner_start_rel": owner_start - clip_start,
                "owner_end_rel": owner_end - clip_start,
            }
        )
    return windows


def available_cpu_count():
    return os.process_cpu_count() or 1


def suggested_clip_workers(api_workers):
    return min(api_workers, available_cpu_count())


def ffmpeg_threads_for_workers(clip_workers):
    if clip_workers <= 1:
        return 0
    return max(1, available_cpu_count() // clip_workers)


def overlap_codec_args(ext, codec, threads=None):
    threads = available_cpu_count() if threads is None else threads
    if codec == "vp9":
        if ext != ".webm":
            raise ValueError("VP9 input requires WebM overlap clips")
        return [
            "-c:v",
            "libvpx-vp9",
            "-crf",
            "32",
            "-b:v",
            "0",
            "-deadline",
            "good",
            "-cpu-used",
            "4",
            "-threads",
            str(threads),
            "-tile-columns",
            "2",
            "-row-mt",
            "1",
            "-frame-parallel",
            "1",
            "-c:a",
            "libopus",
            "-b:a",
            "128k",
        ]

    if codec == "h264":
        if ext != ".mp4":
            raise ValueError("H.264 input requires MP4 overlap clips")
        return [
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "32",
            "-b:v",
            "0",
            "-threads",
            str(threads),
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-movflags",
            "+faststart",
        ]

    if codec == "hevc":
        if ext != ".mp4":
            raise ValueError("HEVC input requires MP4 overlap clips")
        return [
            "-c:v",
            "libx265",
            "-preset",
            "veryfast",
            "-crf",
            "32",
            "-threads",
            "8",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-movflags",
            "+faststart",
        ]

    raise ValueError(f"Overlap format not supported: {ext}")


def create_overlap_clip(
    video_file,
    chunk_dir,
    chunk_idx,
    clip_start,
    clip_end,
    clip_ext,
    ffmpeg_threads=None,
):
    clip_name = f"context_chunk_{chunk_idx:03d}{clip_ext}"
    clip_path = os.path.join(chunk_dir, clip_name)
    if os.path.exists(clip_path):
        probe = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                clip_path,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        try:
            cached_duration = float(probe.stdout.strip())
        except ValueError:
            cached_duration = 0.0
        if probe.returncode == 0 and cached_duration > 0:
            return clip_name
        os.remove(clip_path)
    tmp_path = f"{clip_path}.tmp"
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    duration = clip_end - clip_start
    if duration <= 0:
        raise ValueError(
            f"Invalid overlap clip duration for chunk {chunk_idx}: {duration}"
        )

    manifest = io.load_manifest(chunk_dir)
    video_codec = manifest.get("video_codec")

    print(
        f"Creating overlap clip {clip_name} "
        f"({core.format_time(clip_start)} to {core.format_time(clip_end)})..."
    )
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        core.format_time(clip_start),
        "-i",
        video_file,
        "-t",
        f"{duration:.3f}",
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-sn",
        *overlap_codec_args(clip_ext, video_codec, ffmpeg_threads),
        "-f",
        "webm" if clip_ext == ".webm" else "mp4",
        tmp_path,
    ]
    try:
        subprocess.run(
            cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        os.replace(tmp_path, clip_path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
    return clip_name


def attach_overlap_clip(
    video_file, chunk_dir, chunk, overlap_sec, clip_ext, ffmpeg_threads=None
):
    if overlap_sec > 0:
        clip_name = create_overlap_clip(
            video_file,
            chunk_dir,
            chunk["idx"],
            chunk["clip_start"],
            chunk["clip_end"],
            clip_ext,
            ffmpeg_threads,
        )
    else:
        clip_name = chunk["name"]

    return {
        **chunk,
        "clip_name": clip_name,
    }
