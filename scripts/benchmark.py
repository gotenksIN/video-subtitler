#!/usr/bin/env -S uv run python
import argparse
import difflib
import hashlib
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

load_dotenv()

from modules import core, gemini, io, media, pipeline


def parse_case(value):
    """Parse one GEN:AUDIO:REFINE benchmark case."""
    parts = value.split(":")
    if len(parts) != 3 or not all(parts):
        raise argparse.ArgumentTypeError(
            f"invalid case {value!r}; expected GEN:AUDIO:REFINE"
        )
    return tuple(parts)


def parse_args():
    """Parse command-line arguments for the benchmark runner."""
    parser = argparse.ArgumentParser(
        description="Benchmark subtitle generation, audio refinement, and text refinement models across full video runs."
    )
    parser.add_argument("video_file", help="Path to the source video file.")
    parser.add_argument(
        "--api-key",
        default=os.environ.get("GEMINI_API_KEY"),
        help="Gemini API key override.",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("GEMINI_API_BASE"),
        help="Optional Gemini-compatible proxy base URL.",
    )
    parser.add_argument(
        "--model",
        action="append",
        default=None,
        help="Generation model. Repeat for independent chunk-only runs.",
    )
    parser.add_argument(
        "--case",
        action="append",
        type=parse_case,
        help="Generation, audio refinement, and text refinement model tuple as GEN:AUDIO:REFINE.",
    )
    parser.add_argument(
        "--context-url",
        action="append",
        default=None,
        help="Optional grounding context URL for preflight research, chunk generation, and text refinement. Repeatable.",
    )
    parser.add_argument(
        "--reference-vtt",
        type=Path,
        help="Reference WebVTT file for comparison.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for generated WebVTT files and benchmark-results.json.",
    )
    parser.add_argument(
        "--chunk-dur",
        type=int,
        default=60,
        help="Video chunk duration in seconds.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=pipeline.DEFAULT_API_WORKERS,
        help="Maximum concurrent API workers.",
    )
    parser.add_argument(
        "--thinking-level",
        choices=gemini.THINKING_LEVELS,
        default=None,
        help=(
            "Gemini thinking level for chunk video requests (low, medium, high, xhigh)."
        ),
    )
    return parser.parse_args()


def normalize_text(text):
    """Normalize subtitle text by removing speaker labels and punctuation."""
    text = re.sub(core.SPEAKER_LABEL_RE.pattern, "", text, flags=re.MULTILINE)
    text = re.sub(r"[^\w\s]", " ", text.lower())
    return text.split()


def interval_intersection(left, right):
    """Calculate total temporal intersection duration between two interval lists."""
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


def compare_reference(output_vtt, reference_vtt):
    """Compare generated subtitles against reference VTT and return metrics."""
    generated = io.read_plain_vtt(output_vtt)
    reference = io.read_plain_vtt(reference_vtt)
    generated_words = [
        word for caption in generated for word in normalize_text(caption.raw_text)
    ]
    reference_words = [
        word for caption in reference for word in normalize_text(caption.raw_text)
    ]
    generated_intervals = core.merge_intervals(
        [(core.parse_time(c.start), core.parse_time(c.end)) for c in generated]
    )
    reference_intervals = core.merge_intervals(
        [(core.parse_time(c.start), core.parse_time(c.end)) for c in reference]
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
    """Return a sanitized model name safe for filenames."""
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", model).strip(".") or "model"
    digest = hashlib.sha256(model.encode("utf-8")).hexdigest()[:8]
    return f"{stem[:48]}-{digest}"


def link_or_copy(src, dst):
    """Hard-link a file when possible, falling back to copy."""
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def run_chunk_generation(
    args, model, output_vtt, work_dir, chunks, mime, source_title, preflight_context
):
    """Run chunk subtitle generation and stitch into one VTT."""
    started = time.perf_counter()
    cached = all(
        gemini.load_cached_captions(
            work_dir / f"subtitle_chunk_{chunk['idx']:03d}.json", chunk["duration"]
        )
        is not None
        for chunk in chunks
    )
    failed = pipeline.process_chunks(
        args.api_key,
        args.base_url,
        str(work_dir),
        chunks,
        args.workers,
        model,
        mime,
        args.thinking_level,
        source_title,
        candidate_names=preflight_context.grounded_names,
    )
    if failed:
        raise RuntimeError(
            f"Failed to process {len(failed)} chunk(s): {', '.join(sorted(failed))}"
        )
    pipeline.stitch(str(work_dir), str(output_vtt))
    io.atomic_save_vtt(
        core.canonicalize_speaker_casing(io.read_plain_vtt(output_vtt)), output_vtt
    )
    return time.perf_counter() - started, cached


def run_audio_refinement(
    args,
    audio_model,
    stitched_vtt,
    output_vtt,
    audio_path,
    audio_duration,
    source_duration,
    boundaries,
    work_dir,
    source_title,
):
    """Run boundary audio refinement on stitched subtitles."""
    started = time.perf_counter()
    if not boundaries:
        io.atomic_save_vtt(io.read_plain_vtt(stitched_vtt), output_vtt)
        return time.perf_counter() - started, True
    cached = gemini.boundary_audio_refine_subtitles(
        stitched_vtt=str(stitched_vtt),
        audio_path=str(audio_path),
        audio_duration=audio_duration,
        source_duration=source_duration,
        boundaries=boundaries,
        work_dir=str(work_dir),
        output_vtt=str(output_vtt),
        api_key=args.api_key,
        base_url=args.base_url,
        model_name=audio_model,
        source_title=source_title,
    )
    io.atomic_save_vtt(
        core.canonicalize_speaker_casing(io.read_plain_vtt(output_vtt)), output_vtt
    )
    return time.perf_counter() - started, cached


def run_text_refinement(
    args,
    refine_model,
    input_vtt,
    output_vtt,
    source_title,
    context_urls,
    preflight_context,
):
    """Run global text refinement on input subtitles."""
    started = time.perf_counter()
    gemini.global_refine_subtitles(
        str(input_vtt),
        str(output_vtt),
        args.api_key,
        args.base_url,
        refine_model,
        gemini.REFINEMENT_THINKING_LEVEL,
        source_title=source_title,
        context_urls=context_urls,
        preflight_context=preflight_context,
    )
    return time.perf_counter() - started


def run_full_matrix(args):
    """Lock the output directory and run the configured benchmark matrix."""
    output_dir = args.output_dir or Path("benchmark_results")
    args.video_file = Path(args.video_file)
    if not args.video_file.is_file():
        raise RuntimeError(f"Source video not found: {args.video_file}")

    if args.reference_vtt and not args.reference_vtt.is_file():
        raise RuntimeError(f"Reference VTT not found: {args.reference_vtt}")

    cases = list(args.case or [])
    if not cases and not args.model:
        cases = [
            (
                gemini.DEFAULT_CHUNK_MODEL,
                gemini.DEFAULT_AUDIO_REFINE_MODEL,
                gemini.DEFAULT_REFINE_MODEL,
            )
        ]

    generation_models = list(
        dict.fromkeys([gen for gen, _, _ in cases] + (args.model or []))
    )
    selected_models = [*generation_models, *(model for case in cases for model in case)]
    if any(not model.strip() for model in selected_models):
        raise ValueError("Benchmark model names must not be empty")
    thinking_level = args.thinking_level or gemini.DEFAULT_CHUNK_THINKING_LEVEL
    args.thinking_level = thinking_level
    context_urls = pipeline.validate_generation_config(
        pipeline.GenerationConfig(
            video_path=args.video_file,
            output_path=output_dir / "benchmark-results.json",
            model=generation_models[0],
            api_key=args.api_key,
            chunk_dur=args.chunk_dur,
            workers=args.workers,
            thinking_level=thinking_level,
            context_urls=tuple(args.context_url or ()),
        ),
        require_output_parent=False,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    lock_root = output_dir / "work"
    lock_root.mkdir(parents=True, exist_ok=True)
    lock_file = pipeline.acquire_lock(lock_root)
    try:
        _run_full_matrix(args, output_dir, cases, generation_models, context_urls)
    finally:
        pipeline.release_lock(lock_file)


def _run_full_matrix(args, output_dir, cases, generation_models, context_urls):
    """Run benchmark matrix across validated generation and refinement models."""
    thinking_level = args.thinking_level
    source_title = core.derive_source_title(args.video_file)

    # 1. Probe video and split once.
    source_inspection = media.inspect_media(args.video_file)
    ext, mime, video_codec = source_inspection.video_format
    manifest = {
        "video": io.file_fingerprint(args.video_file),
        "chunk_dur": args.chunk_dur,
        "format": "stream-copy-v1",
        "mode": "benchmark",
        "chunk_ext": ext,
        "chunk_mime": mime,
        "video_codec": video_codec,
    }
    work_key = pipeline.manifest_digest(manifest)
    work_dir = output_dir / "work" / work_key
    work_dir.mkdir(parents=True, exist_ok=True)
    media.split_video(str(args.video_file), str(work_dir), args.chunk_dur, manifest)
    chunks = media.list_chunks(str(work_dir))
    if not chunks:
        raise RuntimeError("No video chunks were created")
    boundaries = [chunk["start"] for chunk in chunks[1:]]
    split_identity = {
        "segments": (work_dir / "segments.csv").read_text(encoding="utf-8"),
        "chunks": [
            {
                "name": chunk["name"],
                "size": (work_dir / chunk["name"]).stat().st_size,
                "mtime_ns": (work_dir / chunk["name"]).stat().st_mtime_ns,
            }
            for chunk in chunks
        ],
    }

    # 2. Extract complete audio when audio refinement is required.
    audio_path = None
    audio_duration = 0.0
    source_duration = 0.0
    needs_audio = any(audio_model is not None for _, audio_model, _ in cases)
    if needs_audio:
        if not source_inspection.audio_streams:
            raise RuntimeError(
                "Source video does not contain an audio stream for audio refinement."
            )
        source_duration = source_inspection.duration
        if source_duration <= 0:
            raise RuntimeError("Source video duration must be positive and finite")
        if boundaries:
            audio_path, audio_duration, source_duration, _reused = (
                media.extract_complete_audio(
                    str(args.video_file),
                    str(work_dir),
                    source_inspection=source_inspection,
                )
            )

    # 3. Chunk generation per unique generation model.
    generated_vtts = {}
    generation_timings = {}
    generation_cached = {}
    preflight_timings = {}
    preflight_cached = {}
    preflight_contexts = {}
    for gen_model in generation_models:
        gen_output = output_dir / f"{safe_model_name(gen_model)}.generated.vtt"
        generation_identity = {
            "model": gen_model,
            "thinking_level": thinking_level,
            "context_urls": context_urls,
            "split": split_identity,
        }
        generation_key = pipeline.manifest_digest(generation_identity)
        gen_work = work_dir / f"generation-{generation_key}"
        gen_work.mkdir(parents=True, exist_ok=True)
        preflight_identity = {
            "source_title": source_title,
            "context_urls": context_urls,
            "model_name": gen_model,
            "thinking_level": gemini.REFINEMENT_THINKING_LEVEL,
        }
        preflight_started = time.perf_counter()
        preflight_path = gen_work / "preflight_context.json"
        preflight_stat = preflight_path.stat() if preflight_path.exists() else None
        preflight_context = gemini.load_or_run_preflight_context(
            gen_work,
            api_key=args.api_key,
            base_url=args.base_url,
            **preflight_identity,
        )
        preflight_timings[gen_model] = time.perf_counter() - preflight_started
        # Cache misses publish a replacement file atomically.
        published_stat = preflight_path.stat()
        preflight_cached[gen_model] = preflight_stat is not None and (
            preflight_stat.st_ino,
            preflight_stat.st_mtime_ns,
        ) == (published_stat.st_ino, published_stat.st_mtime_ns)
        preflight_contexts[gen_model] = preflight_context
        for chunk in chunks:
            src = work_dir / chunk["name"]
            dst = gen_work / chunk["name"]
            dst.unlink(missing_ok=True)
            link_or_copy(src, dst)
        seg_src = work_dir / "segments.csv"
        seg_dst = gen_work / "segments.csv"
        seg_dst.unlink(missing_ok=True)
        link_or_copy(seg_src, seg_dst)

        gen_seconds, cached = run_chunk_generation(
            args,
            gen_model,
            gen_output,
            gen_work,
            chunks,
            mime,
            source_title,
            preflight_contexts[gen_model],
        )
        generated_vtts[gen_model] = gen_output
        generation_timings[gen_model] = gen_seconds
        generation_cached[gen_model] = cached

    # 4. Audio refinement per unique (gen_model, audio_model).
    audio_refined_vtts = {}
    audio_refine_timings = {}
    audio_refine_cached = {}
    unique_audio_pairs = dict.fromkeys(
        (gen, audio) for gen, audio, _ in cases if audio is not None
    ).keys()
    for gen_model, audio_model in unique_audio_pairs:
        audio_output = (
            output_dir
            / f"{safe_model_name(gen_model)}_audio_{safe_model_name(audio_model)}.vtt"
        )
        audio_key = pipeline.manifest_digest(
            {"generation_model": gen_model, "audio_model": audio_model}
        )
        audio_work = work_dir / f"audio-{audio_key}"
        audio_work.mkdir(parents=True, exist_ok=True)
        audio_seconds, cached = run_audio_refinement(
            args,
            audio_model,
            generated_vtts[gen_model],
            audio_output,
            audio_path,
            audio_duration,
            source_duration,
            boundaries,
            audio_work,
            source_title,
        )
        audio_refined_vtts[(gen_model, audio_model)] = audio_output
        audio_refine_timings[(gen_model, audio_model)] = audio_seconds
        audio_refine_cached[(gen_model, audio_model)] = cached

    # 5. Global text refinement and final reporting.
    results = []
    for gen_model, audio_model, refine_model in cases:
        input_vtt = audio_refined_vtts.get((gen_model, audio_model))
        final_vtt = (
            output_dir
            / f"{safe_model_name(gen_model)}_{safe_model_name(audio_model)}_to_{safe_model_name(refine_model)}.final.vtt"
        )
        refine_seconds = run_text_refinement(
            args,
            refine_model,
            input_vtt,
            final_vtt,
            source_title,
            context_urls,
            preflight_contexts[gen_model],
        )
        gen_seconds = generation_timings[gen_model]
        audio_seconds = audio_refine_timings[(gen_model, audio_model)]
        case_result = {
            "generation_model": gen_model,
            "audio_refine_model": audio_model,
            "refinement_model": refine_model,
            "preflight_seconds": preflight_timings[gen_model],
            "preflight_cached": preflight_cached[gen_model],
            "generation_seconds": gen_seconds,
            "generation_cached": generation_cached[gen_model],
            "audio_refine_seconds": audio_seconds,
            "audio_refine_cached": audio_refine_cached[(gen_model, audio_model)],
            "refinement_seconds": refine_seconds,
            "total_seconds": preflight_timings[gen_model]
            + gen_seconds
            + audio_seconds
            + refine_seconds,
            "final_vtt": str(final_vtt),
        }
        if args.reference_vtt:
            case_result["comparison"] = compare_reference(final_vtt, args.reference_vtt)
        results.append(case_result)
        print(json.dumps(case_result, indent=2, default=str))

    # Independent chunk-only model entries when requested.
    if args.model:
        for model in args.model:
            model_result = {
                "generation_model": model,
                "audio_refine_model": None,
                "refinement_model": None,
                "preflight_seconds": preflight_timings[model],
                "preflight_cached": preflight_cached[model],
                "generation_seconds": generation_timings[model],
                "generation_cached": generation_cached[model],
                "audio_refine_seconds": 0.0,
                "audio_refine_cached": True,
                "refinement_seconds": 0.0,
                "total_seconds": preflight_timings[model] + generation_timings[model],
                "generated_vtt": str(generated_vtts[model]),
            }
            if args.reference_vtt:
                model_result["comparison"] = compare_reference(
                    generated_vtts[model], args.reference_vtt
                )
            results.append(model_result)
            print(json.dumps(model_result, indent=2, default=str))

    results_file = output_dir / "benchmark-results.json"
    io.atomic_write_json(results_file, results)
    print(f"Saved benchmark results to {results_file}")


if __name__ == "__main__":
    try:
        run_full_matrix(parse_args())
    except Exception as e:  # noqa: BLE001 - Convert pipeline failures to CLI errors.
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
