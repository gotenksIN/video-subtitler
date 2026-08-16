#!/usr/bin/env -S uv run python
import argparse
import difflib
import json
import os
import re
import sys
import time
from pathlib import Path

import webvtt

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from modules import core, gemini, pipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark subtitle generation and refinement models across full video runs."
    )
    parser.add_argument("video_file", help="Path to the source video file")
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
    parser.add_argument("--workers", type=int, default=pipeline.DEFAULT_API_WORKERS)
    parser.add_argument(
        "--thinking-level",
        choices=gemini.THINKING_LEVELS,
        default=None,
        help=(
            "Gemini thinking level. Default: high. "
            "Lowest supported: minimal for Flash models, low otherwise."
        ),
    )
    return parser.parse_args()


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
        [(core.parse_time(c.start), core.parse_time(c.end)) for c in generated]
    )
    reference_intervals = merge_intervals(
        [
            (core.parse_time(c.start) - start, core.parse_time(c.end) - start)
            for c in reference
        ]
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
    config = pipeline.GenerationConfig(
        video_path=args.video_file,
        output_path=Path(output),
        model=model,
        api_key=args.api_key,
        base_url=args.base_url,
        chunk_dur=args.chunk_dur,
        overlap=args.overlap,
        workers=args.workers,
        thinking_level=args.thinking_level,
        refine_text=False,
    )
    started = time.perf_counter()
    pipeline.run_generation(config)
    return time.perf_counter() - started


def run_full_matrix(args):
    args.video_file = Path(args.video_file)
    if args.case:
        cases = []
        for value in args.case:
            generation, separator, refinement = value.partition(":")
            if not separator or not generation or not refinement:
                print(f"Error: invalid --case {value!r}", file=sys.stderr)
                sys.exit(2)
            cases.append((generation, refinement))
    else:
        default_model = os.environ.get("GEMINI_MODEL", gemini.DEFAULT_CHUNK_MODEL)
        cases = [(model, None) for model in (args.model or [default_model])]

    output_dir = args.output_dir or Path("benchmark_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    generated = {}
    results = []
    for generation_model in dict.fromkeys(model for model, _ in cases):
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
    gemini.global_refine_subtitles(
        str(generated_vtt),
        str(final_vtt),
        args.api_key,
        args.base_url,
        refinement_model,
        "medium",
        source_title=core.derive_source_title(args.video_file),
    )
    result["refinement_seconds"] = time.perf_counter() - started
    result["final_vtt"] = str(final_vtt)
    if args.reference_vtt:
        result["comparison"] = compare_reference(final_vtt, args.reference_vtt, 0)
    print(json.dumps(result, indent=2, default=str))
    return result


if __name__ == "__main__":
    try:
        run_full_matrix(parse_args())
    except Exception as e:  # noqa: BLE001 - Convert pipeline failures to CLI errors.
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
