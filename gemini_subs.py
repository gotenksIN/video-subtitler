import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.parse
from dataclasses import dataclass
from pathlib import Path

import webvtt
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

# Load environment variables from .env file
load_dotenv()


# Define Structured Output Schema
class Caption(BaseModel):
    id: int
    start: str = Field(description="Start time in HH:MM:SS.mmm format")
    end: str = Field(description="End time in HH:MM:SS.mmm format")
    text: str = Field(description="The subtitle text")


class SubtitleResponse(BaseModel):
    captions: list[Caption]


class RefinedCaption(BaseModel):
    id: int = Field(description="The integer ID of the subtitle to change")
    text: str = Field(description="The corrected text")


class RefinementResponse(BaseModel):
    changes: list[RefinedCaption] = Field(
        description="List of subtitles to change. Only include ones that need changes."
    )


CHUNK_ROOT = "temp_video_chunks"
SPLIT_COMPLETE_MARKER = ".split_complete"
MANIFEST_NAME = "manifest.json"
LOCK_NAME = ".lock"
INLINE_VIDEO_WARNING_BYTES = 20 * 1024 * 1024
THINKING_LEVELS = ("minimal", "low", "medium", "high")
DEFAULT_CHUNK_MODEL = "gemini-3.7-flash"
DEFAULT_REFINE_MODEL = "gemini-3.1-pro-preview"
DEFAULT_API_WORKERS = 7
REFINEMENT_THINKING_LEVEL = "medium"
MEDIA_SUFFIXES = (".webm", ".mp4", ".mkv", ".mov", ".avi", ".m4v")
SUBTITLE_SUFFIXES = (".vtt", ".srt", ".sub", ".sbv")
LANGUAGE_TAG_RE = re.compile(r"^[a-z]{2,3}(-[A-Za-z0-9]{2,4})?$")
SPEAKER_LABEL_RE = re.compile(r"^([^:\[\]]+): ")


def derive_source_title(path):
    """Return a human-readable source title from a video or subtitle filename."""
    name = Path(path).name
    for suffix in SUBTITLE_SUFFIXES:
        if name.lower().endswith(suffix) and len(name) > len(suffix):
            name = name[: -len(suffix)]
            if "." in name and LANGUAGE_TAG_RE.fullmatch(name.rsplit(".", 1)[1]):
                name = name.rsplit(".", 1)[0]
            break
    for suffix in MEDIA_SUFFIXES:
        if name.lower().endswith(suffix) and len(name) > len(suffix):
            name = name[: -len(suffix)]
            break
    return name.strip()


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
        return self.thinking_level or default_chunk_thinking_level(self.model)


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


def parse_time(time_str):
    value = str(time_str).strip().replace(",", ".")
    if value.startswith("-"):
        raise ValueError(f"Negative timestamp: {time_str}")

    parts = value.split(":")
    if len(parts) == 3:
        h, m, s_ms = parts
    elif len(parts) == 2:
        h = "0"
        m, s_ms = parts
    elif len(parts) == 1:
        h, m = "0", "0"
        s_ms = parts[0]
    else:
        raise ValueError(f"Invalid timestamp: {time_str}")

    if "." in s_ms:
        s, frac = s_ms.split(".", 1)
        frac_seconds = int(frac) / (10 ** len(frac)) if frac else 0
    else:
        s = s_ms
        frac_seconds = 0

    return int(h) * 3600 + int(m) * 60 + int(s) + frac_seconds


def format_time(seconds):
    if seconds < 0:
        raise ValueError(f"Negative timestamp: {seconds}")

    total_ms = round(seconds * 1000)
    h, rem = divmod(total_ms, 3600 * 1000)
    m, rem = divmod(rem, 60 * 1000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def atomic_write_json(path, data):
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def atomic_save_vtt(vtt, path):
    path = Path(path)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp.vtt", dir=path.parent
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        vtt.save(str(tmp_path))
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)


def remove_boundary_duplicate_prefix(previous_text, current_text):
    def normalize_words(line):
        return tuple(re.findall(r"\w+", SPEAKER_LABEL_RE.sub("", line).casefold()))

    def normalized_elements(text):
        elements = []
        active_turn = None
        lines = text.splitlines()

        def flush_turn(end):
            nonlocal active_turn
            if active_turn is not None:
                label, words, start = active_turn
                elements.append((label, tuple(words), start, end))
                active_turn = None

        for position, line in enumerate(lines):
            match = SPEAKER_LABEL_RE.match(line)
            if match:
                flush_turn(position)
                active_turn = (
                    match.group(1).casefold(),
                    list(normalize_words(line)),
                    position,
                )
            elif line.lstrip().startswith("["):
                flush_turn(position)
                elements.append((None, (), position, position + 1))
            elif active_turn is not None:
                active_turn[1].extend(normalize_words(line))
            else:
                elements.append((None, (), position, position + 1))

        flush_turn(len(lines))
        return elements

    previous_turns = []
    for element in reversed(normalized_elements(previous_text)):
        if element[0] is None:
            break
        previous_turns.append(element)
    previous_turns.reverse()

    current_turns = []
    for element in normalized_elements(current_text):
        if element[0] is None:
            break
        current_turns.append(element)

    current_lines = current_text.splitlines()
    for count in range(min(len(previous_turns), len(current_turns)), 0, -1):
        previous_suffix = previous_turns[-count:]
        current_prefix = current_turns[:count]
        exact_turns = count > 1
        if all(
            len(current_words) >= 2
            and previous_label == current_label
            and (
                previous_words == current_words
                if exact_turns
                else len(previous_words) >= len(current_words)
                and previous_words[-len(current_words) :] == current_words
            )
            for (previous_label, previous_words, *_), (
                current_label,
                current_words,
                *_,
            ) in zip(previous_suffix, current_prefix)
        ):
            del current_lines[: current_prefix[-1][3]]
            break
    return "\n".join(current_lines)


def dedup_boundary_overlap(vtt, chunk_indices, timings=None):
    """Remove exact boundary echoes between consecutive owner chunks.

    Captions must be sorted by start time. Each element of chunk_indices is
    the owner chunk index of the caption at the same position. When a
    caption belongs to the owner chunk directly after the previous
    surviving caption and overlaps it in time, exact same-speaker
    word-suffix echoes are removed from the start of its text. Captions
    whose text becomes empty are removed. Returns the surviving chunk
    indices aligned with the surviving captions.
    """
    if len(vtt.captions) != len(chunk_indices):
        raise ValueError(
            "boundary dedup requires one chunk index per caption: "
            f"{len(vtt.captions)} captions, {len(chunk_indices)} indices"
        )
    if timings is not None and len(vtt.captions) != len(timings):
        raise ValueError("boundary dedup requires one timing per caption")

    survivors = []
    surviving_indices = []
    surviving_ends = []
    for position, (caption, chunk_idx) in enumerate(zip(vtt.captions, chunk_indices)):
        if timings is None:
            start = parse_time(caption.start)
            end = parse_time(caption.end)
        else:
            start, end = timings[position]
        if (
            survivors
            and chunk_idx == surviving_indices[-1] + 1
            and start < surviving_ends[-1]
        ):
            text = remove_boundary_duplicate_prefix(survivors[-1].text, caption.text)
            if not text:
                continue
            caption.text = text
        survivors.append(caption)
        surviving_indices.append(chunk_idx)
        surviving_ends.append(end)
    vtt.captions = survivors
    return surviving_indices


def file_fingerprint(path):
    stat = os.stat(path)
    return {
        "path": str(Path(path).resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def build_manifest(config: GenerationConfig):
    ext, mime, video_codec = probe_video_format(str(config.video_path))
    process_ext, process_mime = ext, mime

    manifest = {
        "video": file_fingerprint(config.video_path),
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
    atomic_write_json(os.path.join(chunk_dir, MANIFEST_NAME), manifest)

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

    manifest = load_manifest(chunk_dir)
    video_codec = manifest.get("video_codec")

    print(
        f"Creating overlap clip {clip_name} ({format_time(clip_start)} to {format_time(clip_end)})..."
    )
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        format_time(clip_start),
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
    windows = get_processing_windows(chunks, overlap_sec)
    if overlap_sec <= 0 or len(windows) <= 1:
        ffmpeg_threads = ffmpeg_threads_for_workers(1)
        processing_chunks = [
            attach_overlap_clip(
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
                    process_chunk,
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
    ffmpeg_threads = ffmpeg_threads_for_workers(clip_workers)
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
                attach_overlap_clip,
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
                    process_chunk,
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


def validate_captions(captions, chunk_duration):
    validated = []

    seen_ids = set()
    duplicate_ids = set()
    for cap in captions:
        if cap.id in seen_ids:
            duplicate_ids.add(cap.id)
        seen_ids.add(cap.id)
    if duplicate_ids:
        raise ValueError(f"Duplicate caption IDs: {sorted(duplicate_ids)}")

    for cap in captions:
        start = parse_time(cap.start)
        end = parse_time(cap.end)
        if start < 0 or end <= start:
            raise ValueError(
                f"Invalid caption timing for id={cap.id}: {cap.start} --> {cap.end}"
            )

        if end > chunk_duration:
            if end - chunk_duration > 0.5:
                raise ValueError(
                    f"Caption timing exceeds chunk duration for id={cap.id}: "
                    f"{cap.start} --> {cap.end}"
                )
            end = chunk_duration
            if end <= start:
                raise ValueError(
                    f"Caption timing exceeds chunk duration for id={cap.id}: "
                    f"{cap.start} --> {cap.end}"
                )

        canonical_start = format_time(start)
        canonical_end = format_time(end)
        if parse_time(canonical_end) <= parse_time(canonical_start):
            raise ValueError(
                f"Caption timing rounds to a non-positive interval for id={cap.id}: "
                f"{cap.start} --> {cap.end}"
            )

        validated.append(
            {
                "id": cap.id,
                "start": canonical_start,
                "end": canonical_end,
                "text": cap.text,
            }
        )

    validated = sorted(
        validated, key=lambda item: (parse_time(item["start"]), item["id"])
    )

    return validated


def load_cached_captions(out_json, chunk_duration):
    if not os.path.exists(out_json):
        return None
    try:
        with open(out_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        response = SubtitleResponse(captions=data)
        return validate_captions(response.captions, chunk_duration)
    except Exception as e:  # noqa: BLE001 - Invalid cache data must be regenerated.
        print(f"Ignoring invalid cached output {out_json}: {e}")
        os.remove(out_json)
        return None


def create_client(api_key, base_url):
    kwargs = {}
    if api_key:
        kwargs["api_key"] = api_key
    if base_url:
        kwargs["http_options"] = {"base_url": base_url}
    return genai.Client(**kwargs)


def default_chunk_thinking_level(model_name):
    return "high"


def validate_thinking_level_for_model(model_name, thinking_level):
    if thinking_level == "minimal" and "flash" not in model_name.lower():
        raise ValueError(
            "--thinking-level minimal is only supported by Flash models. Use low, medium, or high for this model."
        )


def validate_context_urls(urls):
    """Return deduplicated absolute HTTP(S) URLs or raise with a clear message."""
    validated = []
    for raw in urls or []:
        value = str(raw).strip()
        if any(character.isspace() for character in value):
            raise ValueError(
                f"Invalid --context-url {value!r}: URL must not contain whitespace"
            )
        try:
            parsed = urllib.parse.urlsplit(value)
            hostname = parsed.hostname
            _ = parsed.port
        except ValueError as e:
            raise ValueError(f"Invalid --context-url {value!r}: {e}") from None
        if parsed.scheme.lower() not in ("http", "https") or not hostname:
            raise ValueError(
                f"Invalid --context-url {value!r}: "
                "expected an absolute HTTP or HTTPS URL with a host"
            )
        validated.append(value)
    return list(dict.fromkeys(validated))


def url_identity(url):
    """Normalize a URL for retrieval matching while preserving its query."""
    parsed = urllib.parse.urlsplit(url)
    return (
        parsed.scheme.lower(),
        parsed.netloc.lower(),
        parsed.path.rstrip("/"),
        parsed.query,
    )


def is_youtube_video_url(url):
    """Return True for a public YouTube watch or share URL."""
    parsed = urllib.parse.urlsplit(url)
    host = (parsed.hostname or "").lower()
    if host == "youtu.be":
        return bool(parsed.path.strip("/"))
    return (
        host == "youtube.com" or host.endswith(".youtube.com")
    ) and parsed.path == "/watch"


def classify_context_urls(urls):
    """Split validated context URLs into YouTube video inputs and URL Context inputs."""
    youtube_urls = []
    ordinary_urls = []
    for url in urls:
        if is_youtube_video_url(url):
            youtube_urls.append(url)
        else:
            ordinary_urls.append(url)
    return youtube_urls, ordinary_urls


def generate_content_config(thinking_level):
    kwargs = {
        "temperature": 0.0,
        "response_mime_type": "application/json",
        "response_schema": SubtitleResponse,
        "automatic_function_calling": types.AutomaticFunctionCallingConfig(
            disable=True
        ),
    }
    if thinking_level is not None:
        kwargs["thinking_config"] = types.ThinkingConfig(
            thinking_level=thinking_level.upper()
        )
    return types.GenerateContentConfig(**kwargs)


def build_generation_prompt(
    clip_duration, owner_start_rel, owner_end_rel, source_title=None
):
    source_block = ""
    if source_title:
        source_block = (
            "SOURCE CONTEXT\n\n"
            f"Source title: {source_title}\n"
            "Names in the source title are candidate identities only. "
            "They do not prove which speaker said a specific line.\n\n"
        )
    return f"""You are an expert subtitle generator and translator.

Watch this {clip_duration:.3f}-second video clip.

The main chunk window is {format_time(owner_start_rel)} to {format_time(owner_end_rel)} in this clip. Video before or after that window is context only.

Generate accurate, natural English subtitles for dialogue and meaningful on-screen text throughout the entire clip, including the context windows. Captions outside the main window will be filtered later.

{source_block}TIMING

1. Create timestamps relative to the beginning of the full clip, ranging from 00:00:00.000 to {format_time(clip_duration)}.
2. For spoken dialogue, start at the exact first audible syllable and end at the exact end of the last audible syllable.
3. For on-screen text, start when the text becomes visible and end when it disappears.
4. Preserve real silent gaps. Do not stretch captions through silence, reaction shots, or scene changes.
5. Keep captions sorted by start time and do not overlap them.
6. Avoid cues shorter than 500 milliseconds. If a meaningful short utterance cannot fit naturally, combine it with an adjacent utterance from the same speaker only when doing so preserves meaning and timing.

TRANSLATION

7. Translate all spoken dialogue and meaningful on-screen text from the source language into natural English. Never return a source-language transcription instead of an English translation.
8. Prefer faithful, clear English over punchy paraphrases.
9. Preserve every meaningful question, answer, joke, reaction, and product detail. Do not summarize or omit meaningful content.
10. Do not infer missing dialogue or invent facts, product claims, jokes, or cultural explanations.
11. Preserve established names, brands, foods, products, titles, and recurring terms consistently.
12. Preserve useful source-language cultural terms when they express a relationship or concept that English cannot express as precisely.
13. Do not replace understandable English with unexplained romanized source-language terms.
14. Transliterate uncertain proper nouns conservatively instead of inventing a nickname, joke, or English equivalent.
15. Preserve wordplay naturally in English whenever possible. Do not silently replace a pun with unrelated dialogue.

SPEAKER LABELS

16. Use a person's name only when the clip itself establishes attribution: a visible name label or title card, a spoken introduction, or other direct in-clip evidence.
17. Never identify a speaker from appearance alone.
18. When a name cannot be established, prefer a stable descriptive role such as "Host:", "Resident:", "Shop Owner:", or "Producer:" when the role is clear from the clip.
19. Leave dialogue unlabeled when neither a name nor a stable role can be distinguished.
20. Do not use generic numbered labels such as "Speaker 1:".
21. Use the exact format "Name: Dialogue".
22. When multiple identifiable speakers share a cue, place each attributed turn on a separate line.
23. Do not assign speaker labels to on-screen text.

ON-SCREEN TEXT

24. Include meaningful on-screen editorial text when it contributes information, context, humor, branding, or narrative meaning.
25. Ignore decorative text, logos, persistent watermarks, repeated UI, and text unrelated to understanding the video.
26. Keep on-screen text distinct from spoken dialogue.
27. Render on-screen text in square brackets, without mechanical prefixes such as "On-screen text:".
28. Do not combine unrelated dialogue and on-screen text in one caption.
29. Do not describe visible actions such as "(walks)", "(rings bell)", or "(sprays product)" unless corresponding written editorial text actually appears in the video.
30. Translate source-language editorial idioms and visual-caption metaphors into understandable English rather than preserving an incomprehensible literal translation.
31. Do not wrap ordinary spoken dialogue in quotation marks.

FORMATTING

32. Use sequential integer IDs starting at 0.
33. Follow standard subtitle readability rules: no more than 42 characters per line and no more than two lines per caption.
34. Split long speech into readable, natural phrases without changing meaning.
35. Do not use markdown or include explanations outside subtitle captions.
36. Return only a valid JSON object matching the required schema with a "captions" array.
"""


def process_chunk(
    api_key,
    base_url,
    chunk,
    chunk_dir,
    model_name,
    chunk_mime,
    thinking_level,
    source_title=None,
):
    chunk_idx = chunk["idx"]
    clip_name = chunk["clip_name"]
    clip_duration = chunk["clip_duration"]
    owner_start_rel = chunk["owner_start_rel"]
    owner_end_rel = chunk["owner_end_rel"]
    out_json = os.path.join(chunk_dir, f"subtitle_chunk_{chunk_idx:03d}.json")
    chunk_path = os.path.join(chunk_dir, clip_name)

    cached = load_cached_captions(out_json, clip_duration)
    if cached is not None:
        print(f"Skipping {clip_name} - already processed.")
        return True

    prompt = build_generation_prompt(
        clip_duration, owner_start_rel, owner_end_rel, source_title
    )

    try:
        with open(chunk_path, "rb") as f:
            video_data = f.read()
        if len(video_data) > INLINE_VIDEO_WARNING_BYTES:
            print(
                f"[Worker-{chunk_idx:03d}] Warning: {clip_name} is {len(video_data) / 1024 / 1024:.1f} MB. "
                "Gemini docs recommend inline video below 20 MB; reduce --chunk-dur if requests fail."
            )

        print(f"[Worker-{chunk_idx:03d}] Generating {clip_name} using Gemini API...")

        with create_client(api_key, base_url) as client:
            response_stream = client.models.generate_content_stream(
                model=model_name,
                contents=[
                    types.Part.from_bytes(data=video_data, mime_type=chunk_mime),
                    prompt,
                ],
                config=generate_content_config(thinking_level),
            )
            full_json_text = ""
            for response_chunk in response_stream:
                if response_chunk.text:
                    full_json_text += response_chunk.text

        parsed_response = SubtitleResponse.model_validate_json(full_json_text)
        validated = validate_captions(parsed_response.captions, clip_duration)
        atomic_write_json(out_json, validated)

        print(f"[Worker-{chunk_idx:03d}] Finished {clip_name}.")
        return True
    except Exception as e:  # noqa: BLE001 - A chunk failure must keep the run resumable.
        print(f"[Worker-{chunk_idx:03d}] ERROR processing {clip_name}: {e}")
        return False


def load_manifest(chunk_dir):
    manifest_path = os.path.join(chunk_dir, MANIFEST_NAME)
    if not os.path.exists(manifest_path):
        return {}
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def stitch(chunk_dir, output_vtt):
    """Stitch chunk results into one VTT.

    Returns the surviving per-caption owner chunk indices when generated
    overlap filtering applies boundary dedup, else None.
    """
    print("Stitching chunks into final VTT...")
    final_vtt = webvtt.WebVTT()
    captions_to_write = []

    manifest = load_manifest(chunk_dir)
    chunks = list_chunks(chunk_dir)
    windows = get_processing_windows(chunks, float(manifest.get("overlap") or 0.0))
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
            rel_start = parse_time(cap["start"])
            rel_end = parse_time(cap["end"])
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
                format_time(cap["start"]), format_time(cap["end"]), cap["text"]
            )
        )

    provenance = None
    if filter_generated_context:
        provenance = dedup_boundary_overlap(final_vtt, chunk_indices, timings)

    atomic_save_vtt(final_vtt, output_vtt)
    print(
        f"Successfully saved to {output_vtt} with {len(final_vtt.captions)} total captions."
    )
    return provenance


def build_identity_research_prompt(source_title=None, context_urls=(), youtube_urls=()):
    """Build the plain-text prompt for the grounded web identity research pass."""
    title_block = ""
    if source_title:
        title_block = f"\nSOURCE TITLE\n\n{source_title}\n"
    url_block = ""
    if context_urls:
        url_lines = "\n".join(f"- {url}" for url in context_urls)
        url_block = (
            "\nCONTEXT URLS\n\n"
            f"{url_lines}\n"
            "Read the content at these URLs. They may identify the participants.\n"
        )
    youtube_block = ""
    if youtube_urls:
        youtube_lines = "\n".join(f"- {url}" for url in youtube_urls)
        youtube_block = (
            "\nYOUTUBE VIDEO URLS\n\n"
            f"{youtube_lines}\n"
            "Do not open these URLs. Their video content is analyzed in a "
            "separate pass. Treat the URLs as identifiers only.\n"
        )
    return f"""You research speaker identities for an English subtitle localization pass.

Return a concise plain-text summary of the participants who speak in this video.
For each participant return their name in official English styling, their role, and the evidence for that attribution.
Evidence must come from reputable web sources.
{title_block}{url_block}{youtube_block}REQUIREMENTS

1. Use Google Search at least once and rely on reputable evidence.
2. Cite the source for each attribution so the evidence can be reviewed.
3. Rank identity evidence: reputable grounded web evidence first, the source title last.
4. Web evidence may establish speaker identity and canonical proper-name spelling only. It must never infer or change dialogue content, meaning, or events.
5. When identity cannot be established, state one stable descriptive role such as Host, Resident, Shop Owner, or Producer when the role is clear; otherwise state that the speaker stays unlabeled.
6. Return plain text only, with no markdown formatting.
"""


def build_youtube_analysis_prompt(source_title=None):
    """Build the plain-text prompt for the direct YouTube analysis pass."""
    title_block = ""
    if source_title:
        title_block = f"\nSOURCE TITLE\n\n{source_title}\n"
    return f"""You analyze public YouTube videos for an English subtitle localization pass.

Watch the attached video content.
Return concise plain text with:
- Each participant's name in official English styling and their role.
- Timestamped speaker-identification observations: when a visible label, title card, or spoken introduction establishes attribution, give the video timestamp and the evidence.

These observations may establish speaker identity and canonical proper-name spelling only.
They must never infer or change dialogue content, meaning, or events.
{title_block}Return plain text only, with no markdown formatting.
"""


def build_refinement_prompt(
    full_script, source_title=None, identity_context=None, youtube_context=None
):
    source_block = ""
    if source_title:
        source_block = (
            "\nSource title: "
            f"{source_title}\n"
            "A name in the source title is a candidate identity, not proof "
            "that a specific line was spoken by that person.\n"
        )
    identity_block = ""
    if identity_context:
        identity_block = (
            "\nGROUNDED IDENTITY CONTEXT\n\n"
            f"{identity_context}\n"
            "The identity context above was researched with grounded web "
            "evidence. It ranks below explicit script introductions and title "
            "cards. It may establish speaker identity and canonical "
            "proper-name spelling only. It must never change dialogue "
            "meaning, events, or facts.\n"
        )
    youtube_block = ""
    if youtube_context:
        youtube_block = (
            "\nDIRECT VIDEO IDENTITY ANALYSIS\n\n"
            f"{youtube_context}\n"
            "The analysis above was produced by a separate pass that watched "
            "the source video content. It ranks below explicit script "
            "introductions and title cards. It may establish speaker identity "
            "and canonical proper-name spelling only. It must never change "
            "dialogue meaning, events, or facts.\n"
        )
    return f"""You are an expert English subtitle localization editor.

Below is the complete subtitle script for a video.

You do not have access to the source video or audio. Never infer or reconstruct source content that is not established by the provided script.
{source_block}{identity_block}{youtube_block}Use the complete script as global context and correct only lines with a clear problem involving:

1. Speaker labels that are missing, inconsistent, conflicting, or attached to on-screen text. Audit speaker labels first, before polishing any text.
2. Inconsistent character names, brands, foods, products, program titles, or recurring terms.
3. Unnatural or ungrammatical English.
4. Literal translations of source-language idioms, slang, or editorial captions that are incomprehensible in English.
5. Clear continuity errors that can be resolved confidently from the script.
6. Formatting artifacts such as stray quotation marks, raw OCR debris, or inconsistent punctuation.

Do not rewrite the entire script. If a line is acceptable, leave it unchanged.

SEMANTIC PRESERVATION

7. Preserve each line's distinct semantic content.
8. Never delete a question, answer, joke, reaction, product detail, qualification, or meaningful on-screen caption.
9. Never replace a line with a duplicate or paraphrase of an adjacent line.
10. Never add dialogue, facts, product qualities, marketing claims, relationships, jokes, or events.
11. Do not infer what the original audio or on-screen text might have said.
12. If a proposed correction is uncertain, leave the line unchanged.
13. Do not merge, split, reorder, add, or remove subtitle entries.
14. Do not alter IDs or timestamps.

TERMINOLOGY AND LOCALIZATION

15. Preserve established names, brands, foods, products, program titles, and recurring terminology consistently.
16. Do not change proper-name romanization unless needed to correct an inconsistency clearly established within the script.
17. Do not replace understandable English with unexplained romanized source-language terms.
18. Preserve useful source-language cultural terms when they communicate a relationship or concept that ordinary English does not express as precisely.
19. Localize source-language idioms and editorial-caption metaphors into understandable English without inventing new meaning.
20. Preserve visible footnote markers such as "*".
21. Preserve meaningful vocalizations when they carry humor or characterization. Clarify them only when their meaning is unambiguous from the script.

SPEAKER LABELS

22. Rank speaker identity evidence in this order: an explicit introduction or title card within the script; the grounded identity context and the direct video analysis; the source title.
23. Use each confidently established person's official English name styling consistently.
24. Normalize labels when the evidence confidently establishes the identity.
25. Treat an abrupt label change near a chunk boundary as a likely generation error and normalize it to the established identity.
26. When conflicting identities are attached to one speaker and no identity is confidently established, replace them all with one stable descriptive role when the role is established in the script; otherwise remove the uncertain label.
27. Preserve each speaker's turn when multiple speakers occur in one caption.
28. When consecutive lines within one caption have the same speaker label and form one continuous turn, keep the label only once. Preserve every sentence, its order, and readable line breaks. Do not merge separate captions or alternating speaker turns.
29. Never infer identity from appearance.
30. Never add speaker labels to on-screen text.
31. The grounded identity context and the direct video analysis may establish speaker identity and canonical proper-name spelling only. They must never change dialogue meaning, events, or facts.

ON-SCREEN TEXT

32. Preserve square brackets around on-screen editorial text.
33. Keep on-screen text distinct from dialogue.
34. Do not convert on-screen text into spoken dialogue or accessibility-style action descriptions.
35. Remove mechanical prefixes such as "On-screen text:" while preserving the translated text itself.
36. Correct incomprehensible literal caption idioms only when the intended meaning can be established from the full script.

FORMATTING AND OUTPUT

37. Preserve line breaks when they distinguish multiple speakers.
38. Keep each subtitle to no more than 42 characters per line and two lines where possible without deleting meaning.
39. Return a JSON object containing a "changes" list with only entries that genuinely require correction.
40. Each change must contain the existing numeric subtitle "id" and the complete corrected "text".
41. Do not return unchanged entries.
42. Do not return timestamps, markdown, or explanations.

SCRIPT

{full_script}
"""


def validate_refinement_changes(changes, caption_count):
    seen_ids = set()
    for change in changes:
        if not 0 <= change.id < caption_count:
            raise ValueError(f"subtitle ID {change.id} is out of range")
        if change.id in seen_ids:
            raise ValueError(f"subtitle ID {change.id} is duplicated")
        if not change.text.strip():
            raise ValueError(f"subtitle ID {change.id} has empty text")
        seen_ids.add(change.id)


def build_research_config(thinking_level, ordinary_urls):
    """Plain-text config that always enables Google Search grounding."""
    tools = [types.Tool(google_search=types.GoogleSearch())]
    if ordinary_urls:
        tools.append(types.Tool(url_context=types.UrlContext()))
    kwargs = {
        "temperature": 0.0,
        "tools": tools,
    }
    if thinking_level is not None:
        kwargs["thinking_config"] = types.ThinkingConfig(
            thinking_level=thinking_level.upper()
        )
    return types.GenerateContentConfig(**kwargs)


def build_youtube_analysis_config(thinking_level):
    """Plain-text config for the direct YouTube analysis pass. No tools."""
    kwargs = {"temperature": 0.0}
    if thinking_level is not None:
        kwargs["thinking_config"] = types.ThinkingConfig(
            thinking_level=thinking_level.upper()
        )
    return types.GenerateContentConfig(**kwargs)


def build_refinement_config(thinking_level):
    """Structured config for the script refinement pass. No tools."""
    kwargs = {
        "temperature": 0.0,
        "response_mime_type": "application/json",
        "response_schema": RefinementResponse,
        "automatic_function_calling": types.AutomaticFunctionCallingConfig(
            disable=True
        ),
    }
    if thinking_level is not None:
        kwargs["thinking_config"] = types.ThinkingConfig(
            thinking_level=thinking_level.upper()
        )
    return types.GenerateContentConfig(**kwargs)


def collect_stream_metadata(response_stream):
    """Collect response text and grounding metadata in one stream pass."""
    full_text = ""
    search_queries = []
    grounded_sources = []
    retrieved_urls = {}
    for chunk in response_stream:
        if chunk.text:
            full_text += chunk.text
        for candidate in chunk.candidates or []:
            metadata = getattr(candidate, "grounding_metadata", None)
            if metadata:
                search_queries.extend(
                    query for query in (metadata.web_search_queries or []) if query
                )
                for grounding_chunk in metadata.grounding_chunks or []:
                    web = getattr(grounding_chunk, "web", None)
                    uri = web and getattr(web, "uri", None)
                    if uri:
                        grounded_sources.append((getattr(web, "title", None), uri))
            url_context = getattr(candidate, "url_context_metadata", None)
            if url_context:
                for entry in url_context.url_metadata or []:
                    url = entry.retrieved_url
                    if url:
                        retrieved_urls[url] = entry.url_retrieval_status
    return full_text, search_queries, grounded_sources, retrieved_urls


def retrieval_status_value(status):
    """Return the plain status string for a enum or raw value."""
    return getattr(status, "value", status)


def verify_refinement_grounding(
    search_queries, grounded_sources, retrieved_urls, context_urls
):
    """Fail refinement before publication when grounding requirements are unmet."""
    if not search_queries and not grounded_sources:
        print(
            "Error: The identity research response has no Google Search grounding. "
            "Failing without publishing output."
        )
        sys.exit(1)

    retrieved_by_identity = {
        url_identity(url): status for url, status in retrieved_urls.items()
    }
    for url in context_urls:
        status = retrieved_by_identity.get(url_identity(url))
        if status is None:
            print(
                f"Error: Context URL {url} was not retrieved. "
                "Failing without publishing output."
            )
            sys.exit(1)
        if (
            str(retrieval_status_value(status)).upper()
            != "URL_RETRIEVAL_STATUS_SUCCESS"
        ):
            print(
                f"Error: Context URL {url} retrieval failed with "
                f"{retrieval_status_value(status)}. "
                "Failing without publishing output."
            )
            sys.exit(1)


def print_refinement_grounding(
    search_queries, grounded_sources, retrieved_urls, context_urls
):
    unique_queries = list(dict.fromkeys(search_queries))
    if unique_queries:
        print("Search queries:")
        for query in unique_queries:
            print(f"  - {query}")
    unique_sources = list(dict.fromkeys(grounded_sources))
    if unique_sources:
        print("Grounded sources:")
        for title, uri in unique_sources:
            print(f"  - {title or 'Untitled'}: {uri}")
    if context_urls:
        print("Context URL retrieval:")
        retrieved_by_identity = {
            url_identity(url): (url, status) for url, status in retrieved_urls.items()
        }
        for url in context_urls:
            entry = retrieved_by_identity.get(url_identity(url))
            status = retrieval_status_value(entry[1]) if entry else "NOT RETRIEVED"
            print(f"  - {url}: {status}")


def global_refine_subtitles(
    input_vtt,
    output_vtt,
    api_key,
    base_url,
    model_name,
    thinking_level,
    source_title=None,
    context_urls=None,
    boundary_provenance=None,
):
    context_urls = validate_context_urls(context_urls)
    youtube_urls, ordinary_urls = classify_context_urls(context_urls)
    print(f"Loading {input_vtt} for global refinement...")
    vtt = webvtt.read(input_vtt)
    if boundary_provenance is not None and len(vtt) != len(boundary_provenance):
        raise ValueError(
            "boundary dedup requires one chunk index per caption: "
            f"{len(vtt)} captions, {len(boundary_provenance)} indices"
        )

    script_lines = []
    for i, caption in enumerate(vtt):
        script_lines.append(f"[{i}] {caption.start} --> {caption.end}: {caption.text}")

    full_script = "\n".join(script_lines)

    # 1. Grounded web identity research pass. Plain text with Google Search.
    # No video Parts: YouTube content is analyzed in a separate request.
    research_prompt = build_identity_research_prompt(
        source_title, ordinary_urls, youtube_urls
    )
    with create_client(api_key, base_url) as client:
        print(
            "Researching speaker identities with Google Search "
            "(this may take a minute)..."
        )
        research_stream = client.models.generate_content_stream(
            model=model_name,
            contents=research_prompt,
            config=build_research_config(thinking_level, ordinary_urls),
        )
        (
            research_text,
            search_queries,
            grounded_sources,
            retrieved_urls,
        ) = collect_stream_metadata(research_stream)

    verify_refinement_grounding(
        search_queries, grounded_sources, retrieved_urls, ordinary_urls
    )
    print_refinement_grounding(
        search_queries, grounded_sources, retrieved_urls, ordinary_urls
    )

    # 2. Direct YouTube identity analysis. Only when YouTube context URLs
    # exist. Plain text with video Parts and no tools; request completion is
    # the success signal for public video retrieval.
    youtube_analysis_text = ""
    if youtube_urls:
        print("YouTube video context (direct video input):")
        for url in youtube_urls:
            print(f"  - {url}")
        with create_client(api_key, base_url) as client:
            print(
                "Analyzing YouTube videos for speaker identities "
                "(this may take a minute)..."
            )
            youtube_contents = [
                types.Part.from_uri(file_uri=url, mime_type="video/*")
                for url in youtube_urls
            ]
            youtube_contents.append(build_youtube_analysis_prompt(source_title))
            youtube_stream = client.models.generate_content_stream(
                model=model_name,
                contents=youtube_contents,
                config=build_youtube_analysis_config(thinking_level),
            )
            youtube_analysis_text, *_ = collect_stream_metadata(youtube_stream)

    # 3. Structured refinement pass. No tools; the identity sections supply
    # context.
    prompt = build_refinement_prompt(
        full_script, source_title, research_text, youtube_analysis_text
    )

    with create_client(api_key, base_url) as client:
        print(
            "Sending script to Gemini for global refinement (this may take a minute)..."
        )
        response_stream = client.models.generate_content_stream(
            model=model_name,
            contents=prompt,
            config=build_refinement_config(thinking_level),
        )
        full_json_text = ""
        for response_chunk in response_stream:
            if response_chunk.text:
                full_json_text += response_chunk.text

    try:
        refinements = RefinementResponse.model_validate_json(full_json_text)
        validate_refinement_changes(refinements.changes, len(vtt))
    except ValueError as e:
        print(f"Error parsing or validating model response: {e}")
        print("Raw response:")
        print(full_json_text)
        sys.exit(1)

    changes = refinements.changes
    print(f"Model proposed changes to {len(changes)} lines out of {len(vtt)}.")

    for change in changes:
        vtt[change.id].text = change.text

    if boundary_provenance is not None:
        dedup_boundary_overlap(vtt, boundary_provenance)

    atomic_save_vtt(vtt, output_vtt)
    print(f"Saved refined subtitles to {output_vtt}")


def validate_generation_config(config: GenerationConfig) -> None:
    validate_context_urls(config.context_urls)

    if config.chunk_dur <= 0:
        raise ValueError("--chunk-dur must be greater than 0")

    if config.workers <= 0:
        raise ValueError("--workers must be greater than 0")

    validate_thinking_level_for_model(config.model, config.chunk_thinking_level)

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

    source_title = derive_source_title(config.video_path)
    clip_workers = suggested_clip_workers(config.workers)
    manifest, chunk_dir = build_manifest(config)
    os.makedirs(chunk_dir, exist_ok=True)
    lock_file = None
    staging_vtt = None
    completed = False

    try:
        lock_file = acquire_lock(chunk_dir)
        print(f"Using work directory: {chunk_dir}")

        # 1. Split Video
        split_video(str(config.video_path), chunk_dir, config.chunk_dur, manifest)

        chunks = list_chunks(chunk_dir)
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
            global_refine_subtitles(
                staging_vtt,
                str(output_path),
                config.api_key,
                config.base_url,
                config.refine_model or config.model,
                REFINEMENT_THINKING_LEVEL,
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


def main():
    parser = argparse.ArgumentParser(
        description="Generate VTT subtitles for a video using Gemini API."
    )
    parser.add_argument(
        "video_file_or_vtt",
        help="Path to the original video file (OR path to input VTT if --refine-only is used)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="output_subtitles.vtt",
        help="Output path for the generated VTT file",
    )
    parser.add_argument(
        "--api-key", default=os.environ.get("GEMINI_API_KEY"), help="Gemini API Key"
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("GEMINI_API_BASE"),
        help="Base URL for Gemini API (optional)",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("GEMINI_MODEL", DEFAULT_CHUNK_MODEL),
        help="Gemini model to use for chunk video generation",
    )
    parser.add_argument(
        "--refine-model",
        default=os.environ.get("GEMINI_REFINE_MODEL", DEFAULT_REFINE_MODEL),
        help="Gemini model to use for the global refinement pass",
    )
    parser.add_argument(
        "--disable-text-refine",
        action="store_true",
        help="Disable the global text refinement pass after generation",
    )
    parser.add_argument(
        "--refine-only",
        action="store_true",
        help="Skip video processing entirely; only run global text refinement on the input VTT file",
    )
    parser.add_argument(
        "--chunk-dur",
        type=int,
        default=60,
        help="Chunk duration in seconds (default: 60)",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=5.0,
        help="Seconds of context to add before and after each chunk (default: 5)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_API_WORKERS,
        help="Max concurrent API workers",
    )
    parser.add_argument(
        "--thinking-level",
        choices=THINKING_LEVELS,
        default=None,
        help=(
            "Chunk Gemini thinking level. Default: high. "
            "Lowest supported: minimal for Flash models, low otherwise."
        ),
    )
    parser.add_argument(
        "--context-url",
        action="append",
        default=None,
        help=(
            "Absolute HTTP(S) URL used as grounding context for global "
            "refinement. Repeatable. Public YouTube watch or share URLs are "
            "analyzed in a separate direct-video pass. Other URLs use the "
            "URL Context tool and refinement fails if one is not retrieved "
            "successfully."
        ),
    )

    args = parser.parse_args()

    try:
        context_urls = validate_context_urls(args.context_url)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    if args.refine_only:
        if not os.path.exists(args.video_file_or_vtt):
            print(f"Error: Input VTT file not found: {args.video_file_or_vtt}")
            sys.exit(1)
        if not args.api_key:
            print(
                "Error: Gemini API key not configured. Set GEMINI_API_KEY in .env or the environment, or pass --api-key."
            )
            sys.exit(1)
        global_refine_subtitles(
            args.video_file_or_vtt,
            args.output,
            args.api_key,
            args.base_url,
            args.refine_model or args.model,
            REFINEMENT_THINKING_LEVEL,
            source_title=derive_source_title(Path(args.video_file_or_vtt)),
            context_urls=context_urls,
        )
        sys.exit(0)

    config = GenerationConfig(
        video_path=Path(args.video_file_or_vtt),
        output_path=Path(args.output),
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
        refine_model=args.refine_model,
        chunk_dur=args.chunk_dur,
        overlap=args.overlap,
        workers=args.workers,
        thinking_level=args.thinking_level,
        refine_text=not args.disable_text_refine,
        context_urls=tuple(context_urls),
    )
    try:
        run_generation(config)
    except Exception as e:  # noqa: BLE001 - Convert pipeline failures to CLI errors.
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
