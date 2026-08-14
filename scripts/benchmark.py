#!/usr/bin/env -S uv run python
import argparse
import difflib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import webvtt

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from google.genai import types

from gemini_subs import (
    DEFAULT_CHUNK_MODEL,
    SubtitleResponse,
    acquire_lock,
    build_generation_prompt,
    build_manifest,
    clean_completed_work,
    create_client,
    default_chunk_thinking_level,
    format_time,
    generate_content_config,
    global_refine_subtitles,
    list_chunks,
    overlap_codec_args,
    parse_time,
    probe_video_format,
    process_chunks,
    release_lock,
    split_video,
    stitch,
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
        action="append",
        default=None,
        help="Generation model; repeat for independent runs",
    )
    parser.add_argument(
        "--case",
        action="append",
        help="Generation/refinement pair as GEN_MODEL:REFINE_MODEL",
    )
    parser.add_argument(
        "--reference-vtt",
        type=Path,
        help="Reference VTT used for final subtitle comparison",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for generated VTT files and benchmark-results.json",
    )
    parser.add_argument("--chunk-dur", type=int, default=60)
    parser.add_argument("--overlap", type=float, default=5.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--thinking-level",
        choices=("minimal", "low", "medium", "high"),
        default=None,
        help=(
            "Gemini thinking level. Default: high. "
            "Lowest supported: minimal for Flash models, low otherwise."
        ),
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
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as stderr_file:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=stderr_file,
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

        process.wait()
        stderr_file.seek(0)
        stderr = stderr_file.read()
    print("\rFFmpeg progress: 100.0%" + " " * 24)
    if process.returncode != 0:
        raise RuntimeError(f"FFmpeg benchmark failed: {stderr.strip()}")
    return time.perf_counter() - started


def probe_clip_duration(clip_path):
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(clip_path),
    ]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError as e:
        raise RuntimeError("FFprobe is required to measure the benchmark clip") from e
    except subprocess.CalledProcessError as e:
        detail = e.stderr.strip() if e.stderr else "unknown FFprobe error"
        raise RuntimeError(f"Could not probe benchmark clip duration: {detail}") from e

    try:
        duration = float(result.stdout.strip())
    except ValueError as e:
        raise RuntimeError(
            f"FFprobe returned an invalid benchmark clip duration: {result.stdout.strip()!r}"
        ) from e
    if not math.isfinite(duration) or duration <= 0:
        raise RuntimeError(
            f"FFprobe returned a non-positive benchmark clip duration: {duration}"
        )
    return duration


def benchmark_gemini(args, clip_path, mime_type, clip_duration):
    video_data = clip_path.read_bytes()
    prompt = build_generation_prompt(clip_duration, 0.0, clip_duration)
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
        for chunk_count, chunk in enumerate(response_stream, start=1):
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
    return time.perf_counter() - started, validate_captions(
        parsed.captions, clip_duration
    )


def save_vtt(captions, output):
    value = webvtt.WebVTT()
    for caption in captions:
        value.captions.append(
            webvtt.Caption(caption["start"], caption["end"], caption["text"])
        )
    value.save(str(output))


def normalize_text(text):
    text = re.sub(r"^\s*[A-Z][\w' -]{1,30}:\s*", "", text)
    text = re.sub(r"[^\w\s]", " ", text.lower())
    return text.split()


def merge_intervals(intervals):
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def interval_intersection(left, right):
    total = 0.0
    left_index = right_index = 0
    while left_index < len(left) and right_index < len(right):
        start = max(left[left_index][0], right[right_index][0])
        end = min(left[left_index][1], right[right_index][1])
        total += max(0.0, end - start)
        if left[left_index][1] < right[right_index][1]:
            left_index += 1
        else:
            right_index += 1
    return total


def compare_reference(output_vtt, reference_vtt, start):
    generated = webvtt.read(str(output_vtt))
    reference = webvtt.read(str(reference_vtt))
    generated_words = [
        word for caption in generated for word in normalize_text(caption.text)
    ]
    reference_words = [
        word for caption in reference for word in normalize_text(caption.text)
    ]
    generated_intervals = merge_intervals(
        [(parse_time(c.start), parse_time(c.end)) for c in generated]
    )
    reference_intervals = merge_intervals(
        [(parse_time(c.start) - start, parse_time(c.end) - start) for c in reference]
    )
    generated_seconds = sum(end - begin for begin, end in generated_intervals)
    reference_seconds = sum(end - begin for begin, end in reference_intervals)
    overlap_seconds = interval_intersection(generated_intervals, reference_intervals)
    text_similarity = difflib.SequenceMatcher(
        None, reference_words, generated_words
    ).ratio()
    return {
        "reference_cues": len(reference),
        "generated_cues": len(generated),
        "text_similarity": text_similarity,
        "reference_active_seconds": reference_seconds,
        "generated_active_seconds": generated_seconds,
        "temporal_overlap_seconds": overlap_seconds,
        "temporal_recall": overlap_seconds / reference_seconds
        if reference_seconds
        else 0.0,
        "temporal_precision": overlap_seconds / generated_seconds
        if generated_seconds
        else 0.0,
        "temporal_iou": overlap_seconds
        / (reference_seconds + generated_seconds - overlap_seconds)
        if reference_seconds + generated_seconds - overlap_seconds
        else 0.0,
    }


def safe_model_name(model):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", model)


def run_full_generation(args, model, output):
    pipeline_args = argparse.Namespace(
        video_file=str(args.video_file),
        model=model,
        chunk_dur=args.chunk_dur,
        overlap=args.overlap,
        chunk_thinking_level=args.thinking_level or default_chunk_thinking_level(model),
    )
    manifest, chunk_dir = build_manifest(pipeline_args)
    os.makedirs(chunk_dir, exist_ok=True)
    lock_file = acquire_lock(chunk_dir)
    started = time.perf_counter()
    try:
        split_video(str(args.video_file), chunk_dir, args.chunk_dur, manifest)
        chunks = list_chunks(chunk_dir)
        if not chunks:
            raise RuntimeError("No video chunks were created")
        failed = process_chunks(
            args.api_key,
            args.base_url,
            str(args.video_file),
            chunk_dir,
            chunks,
            args.overlap,
            manifest["process_ext"],
            suggested_clip_workers(),
            args.workers,
            model,
            manifest["process_mime"],
            pipeline_args.chunk_thinking_level,
        )
        if failed:
            raise RuntimeError(f"Failed chunks: {', '.join(sorted(failed))}")
        stitch(chunk_dir, output)
        clean_completed_work(chunk_dir)
    finally:
        release_lock(lock_file)
    return time.perf_counter() - started


def run_full_matrix(args):
    args.video_file = Path(args.video_file)
    if not args.api_key:
        print("Error: Gemini API key not configured.", file=sys.stderr)
        sys.exit(1)
    if args.workers <= 0 or args.chunk_dur <= 0 or args.overlap < 0:
        print("Error: invalid chunk or worker settings", file=sys.stderr)
        sys.exit(2)
    if args.overlap >= args.chunk_dur:
        print("Error: --overlap must be smaller than --chunk-dur", file=sys.stderr)
        sys.exit(2)
    if not args.video_file.exists():
        print(f"Error: video file not found: {args.video_file}", file=sys.stderr)
        sys.exit(1)

    default_model = os.environ.get("GEMINI_MODEL", DEFAULT_CHUNK_MODEL)
    if args.case:
        cases = []
        for value in args.case:
            generation, separator, refinement = value.partition(":")
            if not separator or not generation or not refinement:
                print(f"Error: invalid --case {value!r}", file=sys.stderr)
                sys.exit(2)
            cases.append((generation, refinement))
    else:
        cases = [(model, None) for model in (args.model or [default_model])]

    output_dir = args.output_dir or Path("benchmark_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    generated = {}
    results = []
    for generation_model in dict.fromkeys(model for model, _ in cases):
        validate_thinking_level_for_model(
            generation_model,
            args.thinking_level or default_chunk_thinking_level(generation_model),
        )
        generated[generation_model] = output_dir / (
            f"{safe_model_name(generation_model)}.generated.vtt"
        )
        generation_seconds = run_full_generation(
            args, generation_model, generated[generation_model]
        )
        for case_generation, refinement_model in cases:
            if case_generation == generation_model and refinement_model:
                results.append(
                    run_refinement(
                        args,
                        generation_model,
                        refinement_model,
                        generated[generation_model],
                        output_dir,
                        generation_seconds,
                    )
                )

    for generation_model, refinement_model in cases:
        if refinement_model:
            continue
        results.append(
            {
                "generation_model": generation_model,
                "refinement_model": None,
                "generated_vtt": str(generated[generation_model]),
            }
        )

    (output_dir / "benchmark-results.json").write_text(
        json.dumps(results, indent=2, default=str) + "\n", encoding="utf-8"
    )


def run_refinement(
    args,
    generation_model,
    refinement_model,
    generated_vtt,
    output_dir,
    generation_seconds,
):
    result = {
        "generation_model": generation_model,
        "refinement_model": refinement_model,
        "generation_seconds": generation_seconds,
        "generated_vtt": str(generated_vtt),
    }
    final_vtt = output_dir / (
        f"{safe_model_name(generation_model)}_to_{safe_model_name(refinement_model)}.final.vtt"
    )
    started = time.perf_counter()
    global_refine_subtitles(
        str(generated_vtt),
        str(final_vtt),
        args.api_key,
        args.base_url,
        refinement_model,
        "medium",
    )
    result["refinement_seconds"] = time.perf_counter() - started
    result["final_vtt"] = str(final_vtt)
    if args.reference_vtt:
        result["comparison"] = compare_reference(final_vtt, args.reference_vtt, 0)
    print(json.dumps(result, indent=2, default=str))
    return result


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
    clip_duration,
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
    print(f"  Clip size: {clip_size_mb:.1f} MB")
    print(f"  Gemini request clip duration: {clip_duration:.3f}s")
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
    default_model = os.environ.get("GEMINI_MODEL", DEFAULT_CHUNK_MODEL)
    if args.case and args.model:
        print("Error: use --case or --model, not both", file=sys.stderr)
        sys.exit(2)
    if args.case:
        cases = []
        for value in args.case:
            generation_model, separator, refinement_model = value.partition(":")
            if not separator or not generation_model or not refinement_model:
                print(
                    f"Error: invalid --case {value!r}; use GEN_MODEL:REFINE_MODEL",
                    file=sys.stderr,
                )
                sys.exit(2)
            cases.append((generation_model, refinement_model))
    else:
        cases = [(model, None) for model in (args.model or [default_model])]

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

    generation_thinking = args.thinking_level
    for generation_model, _ in cases:
        thinking = generation_thinking or default_chunk_thinking_level(generation_model)
        try:
            validate_thinking_level_for_model(generation_model, thinking)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(2)

    video_file = Path(args.video_file)
    if not video_file.exists():
        print(f"Error: video file not found: {video_file}", file=sys.stderr)
        sys.exit(1)
    if args.reference_vtt and not args.reference_vtt.exists():
        print(f"Error: reference VTT not found: {args.reference_vtt}", file=sys.stderr)
        sys.exit(1)

    output_dir = args.output_dir
    if output_dir is None and (args.case or args.reference_vtt):
        output_dir = Path("benchmark_results")
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    print("Probing video format...")
    ext, mime_type, codec = probe_video_format(str(video_file))
    with tempfile.TemporaryDirectory(prefix="video-subtitler-benchmark-") as tmp_dir:
        clip_path = Path(tmp_dir) / f"benchmark_clip{ext}"
        print("Creating benchmark clip with FFmpeg...")
        ffmpeg_seconds = run_ffmpeg(
            str(video_file), clip_path, args.start, args.duration, ext, codec
        )
        print("Probing benchmark clip duration...")
        clip_duration = probe_clip_duration(clip_path)
        results = []
        for generation_model, refinement_model in cases:
            generation_started = time.perf_counter()
            generation_args = argparse.Namespace(
                api_key=args.api_key,
                base_url=args.base_url,
                model=generation_model,
                thinking_level=generation_thinking
                or default_chunk_thinking_level(generation_model),
            )
            result = {
                "generation_model": generation_model,
                "refinement_model": refinement_model,
                "generation_thinking_level": generation_args.thinking_level,
                "ffmpeg_seconds": ffmpeg_seconds,
            }
            try:
                generation_seconds, captions = benchmark_gemini(
                    generation_args, clip_path, mime_type, clip_duration
                )
                result["generation_seconds"] = generation_seconds
                result["generation_caption_count"] = len(captions)
                result["suggested_workers"] = recommended_workers(
                    ffmpeg_seconds, generation_seconds, suggested_clip_workers()
                )
                generated_vtt = None
                final_vtt = None
                if output_dir:
                    generated_vtt = output_dir / (
                        f"{safe_model_name(generation_model)}"
                        f"_to_{safe_model_name(refinement_model or 'none')}.generated.vtt"
                    )
                    save_vtt(captions, generated_vtt)
                    final_vtt = generated_vtt
                    result["generated_vtt"] = str(generated_vtt)

                if refinement_model:
                    if final_vtt is None:
                        raise RuntimeError(
                            "--output-dir is required for refinement cases"
                        )
                    refined_vtt = output_dir / (
                        f"{safe_model_name(generation_model)}"
                        f"_to_{safe_model_name(refinement_model)}.final.vtt"
                    )
                    refine_started = time.perf_counter()
                    global_refine_subtitles(
                        str(final_vtt),
                        str(refined_vtt),
                        args.api_key,
                        args.base_url,
                        refinement_model,
                        "medium",
                    )
                    result["refinement_seconds"] = time.perf_counter() - refine_started
                    result["refinement_thinking_level"] = "medium"
                    result["final_vtt"] = str(refined_vtt)
                    final_vtt = refined_vtt
                if args.reference_vtt:
                    result["comparison"] = compare_reference(
                        final_vtt or generated_vtt, args.reference_vtt, args.start
                    )
            except (Exception, SystemExit) as e:  # noqa: BLE001 - Continue the matrix.
                result["error"] = f"{type(e).__name__}: {e}"
            result["elapsed_seconds"] = time.perf_counter() - generation_started
            results.append(result)
            print(json.dumps(result, indent=2, default=str))

        if output_dir:
            (output_dir / "benchmark-results.json").write_text(
                json.dumps(results, indent=2, default=str) + "\n", encoding="utf-8"
            )
            print(f"Saved matrix results to {output_dir / 'benchmark-results.json'}")


if __name__ == "__main__":
    run_full_matrix(parse_args())
