"""Resumable generation pipeline: configuration, locking, scheduling, stitching."""

import concurrent.futures
import fcntl
import hashlib
import itertools
import json
import os
import shutil
import traceback
from dataclasses import dataclass
from pathlib import Path

import webvtt
from google.genai import errors

from modules import core, gemini, io, media

CHUNK_ROOT = "temp_video_chunks"
LOCK_NAME = ".lock"
DEFAULT_API_WORKERS = 7


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    """Immutable inputs for one generation run."""

    video_path: Path
    output_path: Path
    model: str
    api_key: str | None = None
    base_url: str | None = None
    refine_model: str | None = None
    audio_refine_model: str | None = None
    chunk_dur: int = 60
    workers: int = DEFAULT_API_WORKERS
    thinking_level: str | None = None
    audio_refine: bool = True
    refine_text: bool = True
    context_urls: tuple[str, ...] = ()

    @property
    def chunk_thinking_level(self) -> str:
        """Resolved chunk thinking level used by the manifest and API calls."""
        return self.thinking_level or gemini.DEFAULT_CHUNK_THINKING_LEVEL


def _build_manifest(config: GenerationConfig, source_inspection: media.MediaInspection):
    """Build the manifest dictionary and resolved work directory path."""
    ext, mime, video_codec = source_inspection.video_format

    manifest = {
        "video": io.file_fingerprint(config.video_path),
        "chunk_dur": config.chunk_dur,
        "format": "stream-copy-v1",
        "mode": "generate",
        "model": config.model,
        "chunk_thinking_level": config.chunk_thinking_level,
        "chunk_ext": ext,
        "chunk_mime": mime,
        "video_codec": video_codec,
    }
    return manifest, os.path.join(CHUNK_ROOT, manifest_digest(manifest))


def manifest_digest(manifest):
    """Return the stable work-directory key for a JSON manifest."""
    return hashlib.sha256(
        json.dumps(manifest, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]


def acquire_lock(chunk_dir):
    """Acquire an exclusive advisory POSIX lock on the work directory."""
    lock_path = os.path.join(chunk_dir, LOCK_NAME)
    lock_file = open(  # noqa: SIM115 - The caller keeps the lock file open.
        lock_path, "a+", encoding="utf-8"
    )
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock_file.seek(0)
        pid = lock_file.read().strip()
        lock_file.close()
        detail = f" (PID {pid})" if pid.isdigit() else ""
        raise RuntimeError(
            f"Another run{detail} is already using {chunk_dir}"
        ) from None

    lock_file.seek(0)
    lock_file.truncate()
    lock_file.write(str(os.getpid()))
    lock_file.flush()
    return lock_file


def release_lock(lock_file):
    """Release the advisory lock and close the lock file descriptor."""
    if lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        finally:
            lock_file.close()


def _clean_completed_work(chunk_dir):
    """Remove all work directory contents except the active lock file."""
    for entry in os.scandir(chunk_dir):
        if entry.name == LOCK_NAME:
            continue
        if entry.is_dir(follow_symlinks=False):
            shutil.rmtree(entry.path)
        else:
            os.remove(entry.path)


def _collect_api_results(futures):
    """Collect completed futures and return names of failed chunks."""
    failed = []
    for future in concurrent.futures.as_completed(futures):
        chunk_name = futures[future]
        try:
            if not future.result():
                failed.append(chunk_name)
        except errors.ClientError as error:
            if error.code in {400, 401, 403}:
                raise
            io.log(f"ERROR processing {chunk_name}: {error}")
            failed.append(chunk_name)
        except Exception as e:  # noqa: BLE001 - Convert worker failures to results.
            io.log(f"ERROR processing {chunk_name}: {e}")
            failed.append(chunk_name)
    return failed


def process_chunks(
    api_key,
    base_url,
    chunk_dir,
    chunks,
    api_workers,
    model_name,
    chunk_mime,
    thinking_level,
    source_title=None,
    candidate_names=None,
):
    """Process video chunks concurrently with a thread pool and collect failures."""
    io.log(f"Processing {len(chunks)} chunks using {api_workers} workers...")
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=api_workers)
    try:
        futures = {
            executor.submit(
                gemini.process_chunk,
                api_key,
                base_url,
                chunk,
                chunk_dir,
                model_name,
                chunk_mime,
                thinking_level,
                source_title,
                candidate_names,
            ): chunk["name"]
            for chunk in chunks
        }
        return _collect_api_results(futures)
    finally:
        executor.shutdown(wait=True, cancel_futures=True)


def stitch(chunk_dir, output_vtt):
    """Stitch stream-copy chunk results into one VTT at segment offsets."""
    io.log("Stitching chunks into final VTT...")
    chunks = media.list_chunks(chunk_dir)
    boundary_starts = [chunk["start"] for chunk in chunks[1:]]
    chunk_by_index = {chunk["idx"]: chunk for chunk in chunks}
    next_start_by_index = {
        left["idx"]: right["start"] for left, right in itertools.pairwise(chunks)
    }

    json_files = sorted(
        [
            f
            for f in os.listdir(chunk_dir)
            if f.startswith("subtitle_chunk_") and f.endswith(".json")
        ]
    )
    result_indices = {
        int(name.removeprefix("subtitle_chunk_").removesuffix(".json"))
        for name in json_files
    }
    expected_indices = set(chunk_by_index)
    missing_indices = sorted(expected_indices - result_indices)
    unexpected_indices = sorted(result_indices - expected_indices)
    if missing_indices or unexpected_indices:
        problems = []
        if missing_indices:
            problems.append(f"missing chunk indices: {missing_indices}")
        if unexpected_indices:
            problems.append(f"unexpected chunk indices: {unexpected_indices}")
        raise ValueError(f"Invalid subtitle results: {'; '.join(problems)}")

    entries = []
    for json_name in json_files:
        chunk_idx = int(json_name.removeprefix("subtitle_chunk_").removesuffix(".json"))
        chunk = chunk_by_index[chunk_idx]
        offset_sec = chunk["start"]
        captions = gemini.load_cached_captions(
            os.path.join(chunk_dir, json_name), chunk["duration"]
        )
        if captions is None:
            raise ValueError(f"Invalid subtitle results: cannot load {json_name}")

        for cap in captions:
            abs_start = offset_sec + core.parse_time(cap["start"])
            abs_end = offset_sec + core.parse_time(cap["end"])
            if chunk_idx in next_start_by_index:
                abs_end = min(abs_end, next_start_by_index[chunk_idx])
            abs_start = core.parse_time(core.format_time(abs_start))
            abs_end = core.parse_time(core.format_time(abs_end))
            if abs_end <= abs_start:
                raise ValueError(
                    f"Invalid subtitle results: non-positive stitched interval "
                    f"in {json_name} for id={cap['id']}"
                )
            entries.append(
                {
                    "start": abs_start,
                    "end": abs_end,
                    "text": cap["text"],
                    "chunk_idx": chunk_idx,
                }
            )

    entries.sort(key=lambda item: item["start"])
    entries = core.merge_visual_boundary_fragments(entries, boundary_starts)

    final_vtt = webvtt.WebVTT()
    for entry in entries:
        final_vtt.captions.append(
            webvtt.Caption(
                core.format_time(entry["start"]),
                core.format_time(entry["end"]),
                entry["text"],
            )
        )

    io.atomic_save_vtt(final_vtt, output_vtt)
    io.log(
        f"Successfully saved to {output_vtt} with {len(final_vtt.captions)} total captions."
    )
    return output_vtt


def validate_generation_config(
    config: GenerationConfig, *, require_output_parent: bool = True
) -> list[str]:
    """Validate pipeline generation configuration inputs."""
    context_urls = core.validate_context_urls(config.context_urls)

    if config.chunk_dur <= 0:
        raise ValueError("--chunk-dur must be greater than 0")

    if config.workers <= 0:
        raise ValueError("--workers must be greater than 0")

    if config.chunk_thinking_level not in gemini.THINKING_LEVELS:
        raise ValueError("--thinking-level must be low, medium, high, or xhigh")

    if not config.video_path.exists():
        raise RuntimeError(f"Video file not found: {config.video_path}")

    if config.video_path.resolve() == config.output_path.resolve():
        raise RuntimeError("--output must not resolve to the source video")

    if require_output_parent and not config.output_path.parent.is_dir():
        raise RuntimeError(
            f"Output parent is not a directory: {config.output_path.parent}"
        )

    if not config.api_key:
        raise RuntimeError(
            "Gemini API key not configured. Set GEMINI_API_KEY in .env or the environment, or pass --api-key."
        )
    return context_urls


def run_generation(config: GenerationConfig) -> None:
    """Run the complete resumable generation lifecycle for one config."""
    context_urls = validate_generation_config(config)

    source_title = core.derive_source_title(config.video_path)
    source_inspection = media.inspect_media(config.video_path)
    manifest, chunk_dir = _build_manifest(config, source_inspection)
    os.makedirs(chunk_dir, exist_ok=True)
    lock_file = None
    completed = False

    try:
        lock_file = acquire_lock(chunk_dir)
        with io.work_log(chunk_dir):
            try:
                io.log(f"Using work directory: {chunk_dir}")

                if config.audio_refine:
                    if not source_inspection.audio_streams:
                        raise RuntimeError(
                            "Failed to extract complete audio. "
                            "The source may not contain an audio stream."
                        )
                    source_duration = source_inspection.duration
                    if source_duration <= 0:
                        raise RuntimeError(
                            "Source video duration must be positive and finite"
                        )

                # Pass 0: Preflight identity and research context
                preflight_identity = {
                    "source_title": source_title,
                    "context_urls": context_urls,
                    "model_name": config.model,
                    "thinking_level": gemini.REFINEMENT_THINKING_LEVEL,
                }

                # Settle preflight workers before closing the log and releasing the lock.
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    preflight_future = executor.submit(
                        gemini.load_or_run_preflight_context,
                        chunk_dir,
                        api_key=config.api_key,
                        base_url=config.base_url,
                        **preflight_identity,
                    )
                    media.split_video(
                        str(config.video_path), chunk_dir, config.chunk_dur, manifest
                    )

                    chunks = media.list_chunks(chunk_dir)
                    if not chunks:
                        raise RuntimeError("No video chunks were created")
                    boundaries = [chunk["start"] for chunk in chunks[1:]]
                    if config.audio_refine and boundaries:
                        audio_path, audio_duration, source_duration, _cached = (
                            media.extract_complete_audio(
                                config.video_path,
                                chunk_dir,
                                source_inspection=source_inspection,
                            )
                        )
                    preflight_context = preflight_future.result()

                # 2. Process chunks concurrently through one API worker pool.
                failed = process_chunks(
                    config.api_key,
                    config.base_url,
                    chunk_dir,
                    chunks,
                    config.workers,
                    config.model,
                    manifest["chunk_mime"],
                    config.chunk_thinking_level,
                    source_title,
                    candidate_names=preflight_context.grounded_names,
                )
                if failed:
                    raise RuntimeError(
                        f"Failed to process {len(failed)} chunk(s): {', '.join(sorted(failed))}. "
                        f"Keeping {chunk_dir} so you can retry."
                    )

                # 3. Stitch chunks at actual segment offsets into a work artifact.
                stitched_vtt = Path(chunk_dir) / "stitched.vtt"
                stitch(chunk_dir, stitched_vtt)
                current_vtt = stitched_vtt

                # 4. Boundary-limited audio refinement over the complete audio.
                if config.audio_refine and boundaries:
                    audio_refined_vtt = Path(chunk_dir) / "audio_refined.vtt"
                    gemini.boundary_audio_refine_subtitles(
                        stitched_vtt=stitched_vtt,
                        audio_path=audio_path,
                        audio_duration=audio_duration,
                        source_duration=source_duration,
                        boundaries=boundaries,
                        work_dir=chunk_dir,
                        output_vtt=audio_refined_vtt,
                        api_key=config.api_key,
                        base_url=config.base_url,
                        model_name=config.audio_refine_model or config.model,
                        source_title=source_title,
                    )
                    current_vtt = audio_refined_vtt

                # 5. Global text refinement or direct atomic publication.
                if config.refine_text:
                    gemini.global_refine_subtitles(
                        input_vtt=str(current_vtt),
                        output_vtt=str(config.output_path),
                        api_key=config.api_key,
                        base_url=config.base_url,
                        model_name=config.refine_model or config.model,
                        thinking_level=gemini.REFINEMENT_THINKING_LEVEL,
                        source_title=source_title,
                        context_urls=context_urls,
                        preflight_context=preflight_context,
                    )
                else:
                    published = core.canonicalize_speaker_casing(
                        io.read_plain_vtt(current_vtt)
                    )
                    io.atomic_save_vtt(published, config.output_path)

                completed = True
            except BaseException as error:
                io.log(f"ERROR: {error}")
                io.log_file(traceback.format_exc())
                raise

    finally:
        try:
            # 6. Cleanup.
            if completed and os.path.exists(chunk_dir):
                io.log(f"Cleaning up temporary directory: {chunk_dir}")
                _clean_completed_work(chunk_dir)
        finally:
            release_lock(lock_file)
