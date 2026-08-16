"""Resumable generation pipeline: configuration, locking, scheduling, stitching."""

import concurrent.futures
import fcntl
import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import webvtt

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
    chunk_dur: int = 60
    overlap: float = 5.0
    workers: int = DEFAULT_API_WORKERS
    thinking_level: str | None = None
    refine_text: bool = True
    context_urls: tuple[str, ...] = ()

    @property
    def chunk_thinking_level(self) -> str:
        """Resolved chunk thinking level used by the manifest and API calls."""
        return self.thinking_level or gemini.default_chunk_thinking_level(self.model)


def build_manifest(config: GenerationConfig):
    ext, mime, video_codec = media.probe_video_format(str(config.video_path))
    process_ext, process_mime = ext, mime

    manifest = {
        "video": io.file_fingerprint(config.video_path),
        "chunk_dur": config.chunk_dur,
        "format": "stream-copy-v1",
        "mode": "generate",
        "model": config.model,
        "chunk_thinking_level": config.chunk_thinking_level,
        "overlap": config.overlap,
        "chunk_ext": ext,
        "chunk_mime": mime,
        "process_ext": process_ext,
        "process_mime": process_mime,
        "video_codec": video_codec,
    }
    digest = hashlib.sha256(
        json.dumps(manifest, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    return manifest, os.path.join(CHUNK_ROOT, digest)


def acquire_lock(chunk_dir):
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
    if lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        finally:
            lock_file.close()


def clean_completed_work(chunk_dir):
    for entry in os.scandir(chunk_dir):
        if entry.name == LOCK_NAME:
            continue
        if entry.is_dir(follow_symlinks=False):
            shutil.rmtree(entry.path)
        else:
            os.remove(entry.path)


def collect_api_results(futures):
    failed = []
    for future in concurrent.futures.as_completed(futures):
        chunk_name = futures[future]
        try:
            if not future.result():
                failed.append(chunk_name)
        except Exception as e:  # noqa: BLE001 - Convert worker failures to results.
            print(f"ERROR processing {chunk_name}: {e}")
            failed.append(chunk_name)
    return failed


def process_chunks(
    api_key,
    base_url,
    video_file,
    chunk_dir,
    chunks,
    overlap_sec,
    clip_ext,
    clip_workers,
    api_workers,
    model_name,
    chunk_mime,
    thinking_level,
    source_title=None,
):
    windows = media.get_processing_windows(chunks, overlap_sec)
    if overlap_sec <= 0 or len(windows) <= 1:
        ffmpeg_threads = media.ffmpeg_threads_for_workers(1)
        processing_chunks = [
            media.attach_overlap_clip(
                video_file, chunk_dir, chunk, overlap_sec, clip_ext, ffmpeg_threads
            )
            for chunk in windows
        ]
        print(
            f"Processing {len(processing_chunks)} chunks using {api_workers} workers..."
        )
        with concurrent.futures.ThreadPoolExecutor(max_workers=api_workers) as executor:
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
                ): chunk["clip_name"]
                for chunk in processing_chunks
            }
            return collect_api_results(futures)

    clip_workers = min(clip_workers, len(windows))
    ffmpeg_threads = media.ffmpeg_threads_for_workers(clip_workers)
    print(
        f"Creating {len(windows)} overlap clips using {clip_workers} workers "
        f"({ffmpeg_threads or 'auto'} FFmpeg threads each) and processing them "
        f"using {api_workers} API workers..."
    )
    failed = []
    api_futures = {}
    with (
        concurrent.futures.ThreadPoolExecutor(
            max_workers=clip_workers
        ) as clip_executor,
        concurrent.futures.ThreadPoolExecutor(max_workers=api_workers) as api_executor,
    ):
        clip_futures = {
            clip_executor.submit(
                media.attach_overlap_clip,
                video_file,
                chunk_dir,
                chunk,
                overlap_sec,
                clip_ext,
                ffmpeg_threads,
            ): chunk
            for chunk in windows
        }
        for future in concurrent.futures.as_completed(clip_futures):
            chunk = clip_futures[future]
            chunk_name = f"context_chunk_{chunk['idx']:03d}{clip_ext}"
            try:
                processing_chunk = future.result()
            except Exception as e:  # noqa: BLE001 - Record clip failures and continue.
                print(f"ERROR creating {chunk_name}: {e}")
                failed.append(chunk_name)
                continue

            api_futures[
                api_executor.submit(
                    gemini.process_chunk,
                    api_key,
                    base_url,
                    processing_chunk,
                    chunk_dir,
                    model_name,
                    chunk_mime,
                    thinking_level,
                    source_title,
                )
            ] = processing_chunk["clip_name"]

        failed.extend(collect_api_results(api_futures))

    return failed


def stitch(chunk_dir, output_vtt):
    """Stitch chunk results into one VTT.

    Returns the surviving per-caption owner chunk indices when generated
    overlap filtering applies boundary dedup, else None.
    """
    print("Stitching chunks into final VTT...")
    final_vtt = webvtt.WebVTT()
    captions_to_write = []

    manifest = io.load_manifest(chunk_dir)
    chunks = media.list_chunks(chunk_dir)
    windows = media.get_processing_windows(
        chunks, float(manifest.get("overlap") or 0.0)
    )
    window_map = {c["idx"]: c for c in windows}
    filter_generated_context = (
        manifest.get("mode") == "generate" and float(manifest.get("overlap") or 0.0) > 0
    )

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
    expected_indices = set(window_map)
    missing_indices = sorted(expected_indices - result_indices)
    unexpected_indices = sorted(result_indices - expected_indices)
    if missing_indices or unexpected_indices:
        problems = []
        if missing_indices:
            problems.append(f"missing chunk indices: {missing_indices}")
        if unexpected_indices:
            problems.append(f"unexpected chunk indices: {unexpected_indices}")
        raise ValueError(f"Invalid subtitle results: {'; '.join(problems)}")

    for json_name in json_files:
        chunk_idx = int(json_name.replace("subtitle_chunk_", "").replace(".json", ""))
        window = window_map[chunk_idx]
        offset_sec = window["clip_start"]

        with open(os.path.join(chunk_dir, json_name), "r") as f:
            captions = json.load(f)

        for cap in captions:
            rel_start = core.parse_time(cap["start"])
            rel_end = core.parse_time(cap["end"])
            if filter_generated_context:
                midpoint = (rel_start + rel_end) / 2
                if not (
                    window["owner_start_rel"] <= midpoint < window["owner_end_rel"]
                ):
                    continue

            abs_start = rel_start + offset_sec
            abs_end = rel_end + offset_sec
            if abs_end <= abs_start:
                raise ValueError(f"Invalid caption timing in {json_name}: {cap}")

            captions_to_write.append(
                {
                    "start": abs_start,
                    "end": abs_end,
                    "text": cap["text"],
                    "chunk_idx": chunk_idx,
                }
            )

    captions_to_write.sort(key=lambda item: item["start"])
    chunk_indices = [cap["chunk_idx"] for cap in captions_to_write]
    timings = [(cap["start"], cap["end"]) for cap in captions_to_write]
    for cap in captions_to_write:
        final_vtt.captions.append(
            webvtt.Caption(
                core.format_time(cap["start"]),
                core.format_time(cap["end"]),
                cap["text"],
            )
        )

    provenance = None
    if filter_generated_context:
        provenance = core.dedup_boundary_overlap(final_vtt, chunk_indices, timings)

    io.atomic_save_vtt(final_vtt, output_vtt)
    print(
        f"Successfully saved to {output_vtt} with {len(final_vtt.captions)} total captions."
    )
    return provenance


def validate_generation_config(config: GenerationConfig) -> None:
    core.validate_context_urls(config.context_urls)

    if config.chunk_dur <= 0:
        raise ValueError("--chunk-dur must be greater than 0")

    if config.workers <= 0:
        raise ValueError("--workers must be greater than 0")

    gemini.validate_thinking_level_for_model(config.model, config.chunk_thinking_level)

    if config.overlap < 0:
        raise ValueError("--overlap must be greater than or equal to 0")

    if config.overlap >= config.chunk_dur:
        raise ValueError("--overlap must be smaller than --chunk-dur")

    if not config.video_path.exists():
        raise RuntimeError(f"Video file not found: {config.video_path}")

    if config.video_path.resolve() == config.output_path.resolve():
        raise RuntimeError("--output must not resolve to the source video")

    if not config.api_key:
        raise RuntimeError(
            "Gemini API key not configured. Set GEMINI_API_KEY in .env or the environment, or pass --api-key."
        )


def run_generation(config: GenerationConfig) -> None:
    """Run the complete resumable generation lifecycle for one config."""
    validate_generation_config(config)

    source_title = core.derive_source_title(config.video_path)
    clip_workers = media.suggested_clip_workers(config.workers)
    manifest, chunk_dir = build_manifest(config)
    os.makedirs(chunk_dir, exist_ok=True)
    lock_file = None
    staging_vtt = None
    completed = False

    try:
        lock_file = acquire_lock(chunk_dir)
        print(f"Using work directory: {chunk_dir}")

        # 1. Split Video
        media.split_video(str(config.video_path), chunk_dir, config.chunk_dur, manifest)

        chunks = media.list_chunks(chunk_dir)
        if not chunks:
            raise RuntimeError("No video chunks were created")

        # 2. Process chunks concurrently. Overlap runs pipeline clip creation into API calls.
        failed = process_chunks(
            config.api_key,
            config.base_url,
            str(config.video_path),
            chunk_dir,
            chunks,
            config.overlap,
            manifest["process_ext"],
            clip_workers,
            config.workers,
            config.model,
            manifest["process_mime"],
            config.chunk_thinking_level,
            source_title,
        )
        if failed:
            raise RuntimeError(
                f"Failed to process {len(failed)} chunk(s): {', '.join(sorted(failed))}. "
                f"Keeping {chunk_dir} so you can retry."
            )

        # 3. Stitch chunks together and optionally refine before publication
        if config.refine_text:
            output_path = config.output_path
            fd, staging_name = tempfile.mkstemp(
                prefix=f".{output_path.name}.",
                suffix=".staging.vtt",
                dir=output_path.parent,
            )
            staging_vtt = Path(staging_name)
            os.close(fd)
            provenance = stitch(chunk_dir, staging_vtt)
            gemini.global_refine_subtitles(
                staging_vtt,
                str(output_path),
                config.api_key,
                config.base_url,
                config.refine_model or config.model,
                gemini.REFINEMENT_THINKING_LEVEL,
                source_title=source_title,
                context_urls=list(config.context_urls),
                boundary_provenance=provenance,
            )
        else:
            stitch(chunk_dir, str(config.output_path))

        completed = True

    finally:
        try:
            if staging_vtt is not None:
                staging_vtt.unlink(missing_ok=True)

            # 4. Cleanup
            if completed and os.path.exists(chunk_dir):
                print(f"Cleaning up temporary directory: {chunk_dir}")
                clean_completed_work(chunk_dir)
        finally:
            release_lock(lock_file)
