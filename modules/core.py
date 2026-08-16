"""Core schemas, timestamp handling, source titles, URL policy, and dedup."""

import re
import urllib.parse
from pathlib import Path

from pydantic import BaseModel, Field

SUBTITLE_SUFFIXES = (".vtt", ".srt", ".sub", ".sbv")
MEDIA_SUFFIXES = (".webm", ".mp4", ".mkv", ".mov", ".avi", ".m4v")
LANGUAGE_TAG_RE = re.compile(r"^[a-z]{2,3}(-[A-Za-z0-9]{2,4})?$")
SPEAKER_LABEL_RE = re.compile(r"^([^:\[\]]+): ")


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
