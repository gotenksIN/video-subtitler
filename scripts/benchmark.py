#!/usr/bin/env -S uv run python
import argparse
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from google.genai import types  # noqa: E402

from gemini_subs import (  # noqa: E402
    SubtitleResponse,
    create_client,
    default_chunk_thinking_level,
    format_time,
    generate_content_config,
    overlap_codec_args,
    probe_video_format,
    suggested_clip_workers,
    validate_captions,
    validate_thinking_level_for_model,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark FFmpeg overlap clip creation and Gemini processing."
    )
    parser.add_argument("video_file", help="Path to the source video file")
    parser.add_argument(
        "--start",
        type=float,
        default=0.0,
        help="Clip start time in seconds (default: 0)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=70.0,
        help="Benchmark clip duration in seconds (default: 70)",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("GEMINI_API_KEY"),
        help="Gemini API key",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("GEMINI_API_BASE"),
        help="Base URL for Gemini API (optional)",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("GEMINI_MODEL", "gemini-3.1-pro-preview"),
        help="Gemini model to use",
    )
    parser.add_argument(
        "--thinking-level",
        choices=("minimal", "low", "medium", "high"),
        default=None,
        help="Gemini thinking level. Default: minimal for Flash models, low otherwise.",
    )
    return parser.parse_args()


def run_ffmpeg(video_file, clip_path, start, duration, ext, codec):
    cmd = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-nostats",
        "-i",
        video_file,
        "-ss",
        format_time(start),
        "-t",
        f"{duration:.3f}",
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-sn",
        *overlap_codec_args(ext, codec),
        "-f",
        "webm" if ext == ".webm" else "mp4",
        "-progress",
        "pipe:1",
        str(clip_path),
    ]
    started = time.perf_counter()
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert process.stdout is not None
    for raw_line in process.stdout:
        key, sep, value = raw_line.strip().partition("=")
        if sep and key == "out_time_ms":
            if not value.isdecimal():
                continue
            elapsed = max(0.0, int(value) / 1_000_000)
            percent = min(100.0, elapsed / duration * 100)
            print(
                f"\rFFmpeg progress: {percent:5.1f}% "
                f"({elapsed:.1f}s / {duration:.1f}s)",
                end="",
                flush=True,
            )

    _stdout, stderr = process.communicate()
    print("\rFFmpeg progress: 100.0%" + " " * 24)
    if process.returncode != 0:
        raise RuntimeError(f"FFmpeg benchmark failed: {stderr.strip()}")
    return time.perf_counter() - started


def benchmark_gemini(args, clip_path, mime_type):
    video_data = clip_path.read_bytes()
    prompt = f"""
    You are benchmarking subtitle generation latency.
    Watch this {args.duration:.3f}-second video clip and generate concise English subtitles.
    Create accurate timestamps relative to the start of this clip, ranging from 00:00:00.000 to {format_time(args.duration)}.
    Use sequential integer IDs starting at 0. Keep captions sorted by start time and do not overlap them.
    Return ONLY the valid JSON object matching the required schema with a 'captions' array.
    """
    started = time.perf_counter()
    print("Sending benchmark clip to Gemini...")
    with create_client(args.api_key, args.base_url) as client:
        response_stream = client.models.generate_content_stream(
            model=args.model,
            contents=[
                types.Part.from_bytes(data=video_data, mime_type=mime_type),
                prompt,
            ],
            config=generate_content_config(args.thinking_level),
        )
        full_json_text = ""
        chunk_count = 0
        for chunk in response_stream:
            chunk_count += 1
            print(
                f"\rGemini stream chunks received: {chunk_count}",
                end="",
                flush=True,
            )
            if chunk.text:
                full_json_text += chunk.text
    print()

    print("Validating Gemini response...")
    parsed = SubtitleResponse.model_validate_json(full_json_text)
    validate_captions(parsed.captions, args.duration)
    return time.perf_counter() - started, len(parsed.captions)


def recommended_workers(ffmpeg_seconds, api_seconds, clip_workers):
    if ffmpeg_seconds <= 0:
        return 1
    return max(1, math.ceil(clip_workers * api_seconds / ffmpeg_seconds))


def print_summary(
    video_file,
    clip_path,
    ext,
    mime_type,
    codec,
    ffmpeg_seconds,
    api_seconds,
    caption_count,
):
    clip_size_mb = clip_path.stat().st_size / 1024 / 1024
    clip_workers = suggested_clip_workers()
    workers = recommended_workers(ffmpeg_seconds, api_seconds, clip_workers)

    print()
    print("Benchmark results")
    print(f"  Source: {video_file}")
    print(f"  Codec: {codec}")
    print(f"  Clip container: {ext} ({mime_type})")
    print("  Clip path: temporary file removed after benchmark")
    print(f"  Clip size: {clip_size_mb:.1f} MB")
    print(f"  FFmpeg clip generation: {ffmpeg_seconds:.2f}s")
    print(f"  Gemini processing: {api_seconds:.2f}s")
    print(f"  Captions returned: {caption_count}")
    print()
    print("Worker guidance")
    print(f"  Default clip workers: {clip_workers}")
    print(f"  Suggested scripts/subtitle.sh workers: {workers}")
    print(
        "  Rationale: ceil(default clip workers * Gemini seconds / FFmpeg seconds), "
        "so API processing keeps pace with local clip generation throughput."
    )


def main():
    args = parse_args()
    args.thinking_level = args.thinking_level or default_chunk_thinking_level(
        args.model
    )

    if args.start < 0:
        print("Error: --start must be greater than or equal to 0", file=sys.stderr)
        sys.exit(2)
    if args.duration <= 0:
        print("Error: --duration must be greater than 0", file=sys.stderr)
        sys.exit(2)
    if not args.api_key:
        print(
            "Error: Gemini API key not configured. Set GEMINI_API_KEY in .env or the environment, or pass --api-key.",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        validate_thinking_level_for_model(args.model, args.thinking_level)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(2)

    video_file = Path(args.video_file)
    if not video_file.exists():
        print(f"Error: video file not found: {video_file}", file=sys.stderr)
        sys.exit(1)

    print("Probing video format...")
    ext, mime_type, codec = probe_video_format(str(video_file))
    with tempfile.TemporaryDirectory(prefix="video-subtitler-benchmark-") as tmp_dir:
        clip_path = Path(tmp_dir) / f"benchmark_clip{ext}"
        print("Creating benchmark clip with FFmpeg...")
        ffmpeg_seconds = run_ffmpeg(
            str(video_file), clip_path, args.start, args.duration, ext, codec
        )
        api_seconds, caption_count = benchmark_gemini(args, clip_path, mime_type)
        print_summary(
            video_file,
            clip_path,
            ext,
            mime_type,
            codec,
            ffmpeg_seconds,
            api_seconds,
            caption_count,
        )


if __name__ == "__main__":
    main()
