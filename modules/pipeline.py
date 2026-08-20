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


def build_manifest(config: GenerationConfig):
    """Build the manifest dictionary and resolved work directory path."""
    ext, mime, video_codec = media.probe_video_format(str(config.video_path))

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
    digest = hashlib.sha256(
        json.dumps(manifest, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    return manifest, os.path.join(CHUNK_ROOT, digest)


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


def clean_completed_work(chunk_dir):
    """Remove all work directory contents except the active lock file."""
    for entry in os.scandir(chunk_dir):
        if entry.name == LOCK_NAME:
            continue
        if entry.is_dir(follow_symlinks=False):
            shutil.rmtree(entry.path)
        else:
            os.remove(entry.path)


def collect_api_results(futures):
    """Collect completed futures and return names of failed chunks."""
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
    chunk_dir,
    chunks,
    api_workers,
    model_name,
    chunk_mime,
    thinking_level,
    source_title=None,
):
    """Process video chunks concurrently with a thread pool and collect failures."""
    print(f"Processing {len(chunks)} chunks using {api_workers} workers...")
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
            ): chunk["name"]
            for chunk in chunks
        }
        return collect_api_results(futures)


def stitch(chunk_dir, output_vtt):
    """Stitch stream-copy chunk results into one VTT at segment offsets."""
    print("Stitching chunks into final VTT...")
    chunks = media.list_chunks(chunk_dir)
    boundary_starts = [chunk["start"] for chunk in chunks[1:]]
    chunk_by_index = {chunk["idx"]: chunk for chunk in chunks}

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
        offset_sec = chunk_by_index[chunk_idx]["start"]

        with open(os.path.join(chunk_dir, json_name), "r", encoding="utf-8") as f:
            captions = json.load(f)

        for cap in captions:
            abs_start = offset_sec + core.parse_time(cap["start"])
            abs_end = offset_sec + core.parse_time(cap["end"])
            if abs_end <= abs_start:
                raise ValueError(f"Invalid caption timing in {json_name}: {cap}")

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
    print(
        f"Successfully saved to {output_vtt} with {len(final_vtt.captions)} total captions."
    )
    return output_vtt


def validate_generation_config(config: GenerationConfig) -> list[str]:
    """Validate pipeline generation configuration inputs."""
    context_urls = core.validate_context_urls(config.context_urls)

    if config.chunk_dur <= 0:
        raise ValueError("--chunk-dur must be greater than 0")

    if config.workers <= 0:
        raise ValueError("--workers must be greater than 0")

    gemini.validate_thinking_level_for_model(config.model, config.chunk_thinking_level)

    if not config.video_path.exists():
        raise RuntimeError(f"Video file not found: {config.video_path}")

    if config.video_path.resolve() == config.output_path.resolve():
        raise RuntimeError("--output must not resolve to the source video")

    if not config.api_key:
        raise RuntimeError(
            "Gemini API key not configured. Set GEMINI_API_KEY in .env or the environment, or pass --api-key."
        )
    return context_urls


def run_generation(config: GenerationConfig) -> None:
    """Run the complete resumable generation lifecycle for one config."""
    context_urls = validate_generation_config(config)

    source_title = core.derive_source_title(config.video_path)
    manifest, chunk_dir = build_manifest(config)
    os.makedirs(chunk_dir, exist_ok=True)
    lock_file = None
    staging_vtt = None
    completed = False

    try:
        lock_file = acquire_lock(chunk_dir)
        print(f"Using work directory: {chunk_dir}")

        if config.audio_refine:
            if not media.has_audio_stream(config.video_path):
                raise RuntimeError(
                    "Failed to extract complete audio. "
                    "The source may not contain an audio stream."
                )
            audio_path, audio_duration, _source_duration, _cached = (
                media.extract_complete_audio(config.video_path, chunk_dir)
            )

        # 1. Split video.
        media.split_video(str(config.video_path), chunk_dir, config.chunk_dur, manifest)

        chunks = media.list_chunks(chunk_dir)
        if not chunks:
            raise RuntimeError("No video chunks were created")

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
        if config.audio_refine:
            audio_refined_vtt = Path(chunk_dir) / "audio_refined.vtt"
            gemini.boundary_audio_refine_subtitles(
                stitched_vtt=stitched_vtt,
                audio_path=audio_path,
                audio_duration=audio_duration,
                boundaries=[chunk["start"] for chunk in chunks[1:]],
                work_dir=chunk_dir,
                output_vtt=audio_refined_vtt,
                api_key=config.api_key,
                base_url=config.base_url,
                model_name=config.audio_refine_model
                or gemini.DEFAULT_AUDIO_REFINE_MODEL,
                source_title=source_title,
            )
            current_vtt = audio_refined_vtt

        # 5. Global text refinement or direct atomic publication.
        if config.refine_text:
            output_path = config.output_path
            fd, staging_name = tempfile.mkstemp(
                prefix=f".{output_path.name}.",
                suffix=".staging.vtt",
                dir=output_path.parent,
            )
            staging_vtt = Path(staging_name)
            os.close(fd)
            gemini.global_refine_subtitles(
                input_vtt=str(current_vtt),
                output_vtt=str(staging_vtt),
                api_key=config.api_key,
                base_url=config.base_url,
                model_name=config.refine_model or config.model,
                thinking_level=gemini.REFINEMENT_THINKING_LEVEL,
                source_title=source_title,
                context_urls=context_urls,
            )
            os.replace(staging_vtt, config.output_path)
        else:
            published = core.canonicalize_speaker_casing(webvtt.read(str(current_vtt)))
            io.atomic_save_vtt(published, config.output_path)

        completed = True

    finally:
        try:
            if staging_vtt is not None:
                staging_vtt.unlink(missing_ok=True)

            # 6. Cleanup.
            if completed and os.path.exists(chunk_dir):
                print(f"Cleaning up temporary directory: {chunk_dir}")
                clean_completed_work(chunk_dir)
        finally:
            release_lock(lock_file)
