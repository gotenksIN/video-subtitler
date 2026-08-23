"""Core schemas, timestamp handling, source titles, URL policy, and audio refinement."""

import itertools
import re
import urllib.parse
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

SUBTITLE_SUFFIXES = (".vtt", ".srt", ".sub", ".sbv")
MEDIA_SUFFIXES = (".webm", ".mp4", ".mkv", ".mov", ".avi", ".m4v")
LANGUAGE_TAG_RE = re.compile(r"^[a-z]{2,3}(-[A-Za-z0-9]{2,4})?$")
SPEAKER_LABEL_RE = re.compile(r"^([ \t]*)([A-Z][\w' -]{1,30})(:[ \t]*)")
AUDIO_REFINE_RESPONSE_CONTRACT = "sparse-patch-v1"
REPAIR_WINDOW_SECONDS = 10.0


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


class AudioRefinedCue(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    source_ids: list[int] = Field(
        alias="sourceIds", description="Request-only source IDs this cue replaces"
    )
    start: str = Field(description="Source-relative start in HH:MM:SS.mmm format")
    end: str = Field(description="Source-relative end in HH:MM:SS.mmm format")
    text: str = Field(description="The complete replacement cue text")


class AudioRefinementResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    contract_version: Literal["sparse-patch-v1"] = Field(
        alias="contractVersion",
        description="Sparse audio-refinement response contract version",
    )
    deleted_source_ids: list[int] = Field(
        default_factory=list,
        alias="deletedSourceIds",
        description="Source IDs removed from the script",
    )
    cues: list[AudioRefinedCue] = Field(
        description="Only changed, replacement, split, merged, or recovered cues"
    )


class PreflightContext(BaseModel):
    contract_version: Literal["preflight-v1"] = Field(
        default="preflight-v1", alias="contractVersion"
    )
    identity_context: str = ""
    terminology_context: str = ""
    youtube_context: str | None = None
    grounded_names: list[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


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
        return len([part for part in parsed.path.split("/") if part]) == 1
    if host not in {"youtube.com", "www.youtube.com", "m.youtube.com"}:
        return False
    video_ids = urllib.parse.parse_qs(parsed.query).get("v", [])
    return parsed.path == "/watch" and bool(video_ids and video_ids[0])


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


def _parse_bracketed_text(text):
    """Return outside text, complete outer fragments, and balance state."""
    outside = []
    fragments = []
    depth = 0
    fragment_start = None
    unmatched = False

    for index, character in enumerate(text):
        if character == "[":
            if depth == 0:
                fragment_start = index
            depth += 1
        elif character == "]":
            if depth == 0:
                unmatched = True
                outside.append(character)
                continue
            depth -= 1
            if depth == 0:
                fragments.append(text[fragment_start : index + 1])
                fragment_start = None
        elif depth == 0:
            outside.append(character)

    if depth:
        unmatched = True
        outside.extend(text[fragment_start:])

    return "".join(outside), fragments, unmatched


def classify_cue_text(text):
    """Classify one cue as dialogue, editorial, or mixed."""
    outside, fragments, _unmatched = _parse_bracketed_text(text)
    if not fragments:
        return "dialogue"
    if re.search(r"\w", outside):
        return "mixed"
    return "editorial"


def visual_fragment_strings(text):
    """Return exact complete outer bracketed fragment strings."""
    _outside, fragments, _unmatched = _parse_bracketed_text(text)
    return fragments


def has_unmatched_brackets(text):
    _outside, _fragments, unmatched = _parse_bracketed_text(text)
    return unmatched


def dialogue_turns(text):
    """Return per-line normalized dialogue word tuples, skipping visual lines."""
    turns = []
    for line in text.splitlines():
        dialogue = re.sub(
            r"^([^:\[\]]+):\s*", "", re.sub(r"\[[^\]]*\]", "", line)
        ).strip()
        words = tuple(re.findall(r"\w+", dialogue.lower()))
        if words:
            turns.append(words)
    return turns


def canonicalize_speaker_casing(vtt, grounded_names=None):
    """Rewrite speaker label spellings to one canonical casing per speaker.

    A grounded name whose casefold matches a label group overrides script
    frequency. Ungrounded groups keep their most frequent script spelling;
    exact frequency ties keep the first spelling seen in the script.
    """
    captions = getattr(vtt, "captions", vtt)
    grounded_lookup = {
        name.casefold(): name
        for name in (grounded_names or ())
        if name and name.strip()
    }

    spelling_counts = {}
    for caption in captions:
        if classify_cue_text(caption.text) == "editorial":
            continue
        for line in caption.text.splitlines():
            match = SPEAKER_LABEL_RE.match(line)
            if not match:
                continue
            label = match.group(2)
            counts = spelling_counts.setdefault(label.casefold(), {})
            counts[label] = counts.get(label, 0) + 1

    targets = {}
    for group_key, counts in spelling_counts.items():
        if group_key in grounded_lookup:
            targets[group_key] = grounded_lookup[group_key]
        else:
            targets[group_key] = max(counts, key=counts.get)

    for caption in captions:
        if classify_cue_text(caption.text) == "editorial":
            continue
        lines = caption.text.splitlines()
        rewritten = []
        for line in lines:
            match = SPEAKER_LABEL_RE.match(line)
            if match and match.group(2).casefold() in targets:
                target = targets[match.group(2).casefold()]
                line = match.group(1) + target + match.group(3) + line[match.end() :]
            rewritten.append(line)
        if rewritten != lines:
            caption.text = "\n".join(rewritten)

    return vtt


def merge_intervals(intervals):
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def build_repair_regions(boundaries, window=REPAIR_WINDOW_SECONDS):
    return merge_intervals(
        [(boundary - window, boundary + window) for boundary in boundaries]
    )


def interval_intersects_any(interval, regions):
    start, end = interval
    return any(
        start < region_end and end > region_start
        for region_start, region_end in regions
    )


def interval_contained_in_any(interval, regions):
    start, end = interval
    return any(
        region_start <= start and end <= region_end
        for region_start, region_end in regions
    )


def canonical_timestamp(value):
    try:
        parsed = parse_time(value)
    except ValueError as error:
        raise ValueError(f"Invalid timestamp {value!r}: {error}") from None
    formatted = format_time(parsed)
    if str(value).strip() != formatted:
        raise ValueError(f"Timestamp {value!r} is not canonical HH:MM:SS.mmm format")
    return formatted


def is_identical_copy(cue, source_by_id):
    if len(cue["source_ids"]) != 1:
        return False
    source = source_by_id[cue["source_ids"][0]]
    return (
        cue["text"] == source["text"]
        and cue["start"] == source["start"]
        and cue["end"] == source["end"]
    )


def filter_audio_refinement_patch(
    response, source_entries, boundaries, window=REPAIR_WINDOW_SECONDS
):
    """Drop patch cues and deletions without repair-region authority.

    Gemini receives the complete script and sometimes edits cues outside
    every repair region. Discarding those edits keeps each cue outside
    repair regions as an exact copy of its source entry.
    """
    regions = build_repair_regions(boundaries, window)
    source_intervals = {
        entry["id"]: (parse_time(entry["start"]), parse_time(entry["end"]))
        for entry in source_entries
    }

    kept_cues = []
    for cue in response.cues:
        cue_ids = cue.source_ids
        if not cue_ids:
            keep = interval_intersects_any(
                (parse_time(cue.start), parse_time(cue.end)), regions
            )
        elif any(source_id not in source_intervals for source_id in cue_ids):
            # Unknown source IDs stay fatal during validation.
            keep = True
        else:
            keep = all(
                interval_intersects_any(source_intervals[source_id], regions)
                for source_id in cue_ids
            )
        if keep:
            kept_cues.append(cue)

    kept_deleted = [
        source_id
        for source_id in response.deleted_source_ids
        if source_id not in source_intervals
        or interval_intersects_any(source_intervals[source_id], regions)
    ]

    return response.model_copy(
        update={"cues": kept_cues, "deleted_source_ids": kept_deleted}
    )


def reconstruct_sparse_audio_candidate(response, source_entries, source_by_id):
    """Expand sparse patches with exact copies of every omitted source cue."""
    referenced = {source_id for cue in response.cues for source_id in cue.source_ids}
    deleted = set(response.deleted_source_ids)

    patch_starts = [parse_time(canonical_timestamp(cue.start)) for cue in response.cues]
    if any(after < before for before, after in itertools.pairwise(patch_starts)):
        raise ValueError("Sparse audio refinement cues are not in script order")
    patch_lineage = [source_id for cue in response.cues for source_id in cue.source_ids]
    if any(after < before for before, after in itertools.pairwise(patch_lineage)):
        raise ValueError(
            "Sparse audio refinement cues reference source IDs out of script order"
        )

    for cue in response.cues:
        if has_unmatched_brackets(cue.text):
            raise ValueError("Sparse audio refinement cue contains unmatched brackets")

    candidate_cues = list(response.cues)
    candidate_cues.extend(
        AudioRefinedCue(
            sourceIds=[entry["id"]],
            start=entry["start"],
            end=entry["end"],
            text=entry["text"],
        )
        for entry in source_entries
        if entry["id"] not in referenced and entry["id"] not in deleted
    )
    candidate_cues.sort(
        key=lambda cue: (
            parse_time(canonical_timestamp(cue.start)),
            cue.source_ids[0] if cue.source_ids else -1,
        )
    )
    return response.model_copy(update={"cues": candidate_cues})


def validate_audio_refinement_response(
    response, source_entries, audio_duration, boundaries
):
    """Reconstruct and validate the complete candidate before writing files.

    The patch is filtered to repair-region authority first, so cues outside
    repair regions survive as exact host-side copies. Lineage and authority
    checks then run on the complete candidate.
    """
    if any(has_unmatched_brackets(entry["text"]) for entry in source_entries):
        raise ValueError("Audio refinement source cue contains unmatched brackets")
    source_by_id = {entry["id"]: entry for entry in source_entries}
    source_ids = set(source_by_id)
    if len(source_ids) != len(source_entries):
        raise ValueError("Audio refinement source IDs are not unique")
    response = filter_audio_refinement_patch(response, source_entries, boundaries)
    response = reconstruct_sparse_audio_candidate(
        response, source_entries, source_by_id
    )
    regions = build_repair_regions(boundaries)
    audio_duration_ms = round(audio_duration * 1000)

    deleted = list(response.deleted_source_ids)
    if len(set(deleted)) != len(deleted):
        raise ValueError("Audio refinement deleted_source_ids contains duplicates")
    unknown_deleted = set(deleted) - source_ids
    if unknown_deleted:
        raise ValueError(
            f"Audio refinement deletes unknown source IDs: {sorted(unknown_deleted)}"
        )

    referenced_counts = Counter()
    response_cues = []
    for cue in response.cues:
        if not cue.text.strip():
            raise ValueError("Audio refinement output cues must contain text")
        cue_ids = list(cue.source_ids)
        if not cue_ids and visual_fragment_strings(cue.text):
            raise ValueError("Recovered cues must not contain bracketed text")
        if not cue_ids and not dialogue_turns(cue.text):
            raise ValueError("Recovered cues must contain spoken dialogue")
        if cue_ids != sorted(set(cue_ids)):
            raise ValueError(
                f"Audio refinement cue source_ids must be strictly "
                f"increasing with no duplicates: {cue_ids}"
            )
        for source_id in cue_ids:
            referenced_counts[source_id] += 1
        start = canonical_timestamp(cue.start)
        end = canonical_timestamp(cue.end)
        start_ms = round(parse_time(start) * 1000)
        end_ms = round(parse_time(end) * 1000)
        if start_ms < 0:
            raise ValueError(f"Audio refinement cue has a negative start: {start}")
        if end_ms > audio_duration_ms:
            if end_ms - audio_duration_ms > 500:
                raise ValueError(
                    f"Audio refinement cue {start} --> {end} exceeds the "
                    "complete audio duration"
                )
            end_ms = audio_duration_ms
            end = format_time(end_ms / 1000)
        if end_ms <= start_ms:
            raise ValueError(
                f"Audio refinement cue {start} --> {end} rounds to a "
                "non-positive interval"
            )
        response_cues.append(
            {
                "source_ids": cue_ids,
                "start": format_time(start_ms / 1000),
                "end": end,
                "text": cue.text,
            }
        )

    unknown_referenced = set(referenced_counts) - source_ids
    if unknown_referenced:
        raise ValueError(
            "Audio refinement references unknown source IDs: "
            f"{sorted(unknown_referenced)}"
        )
    conflicting = set(deleted) & set(referenced_counts)
    if conflicting:
        raise ValueError(
            "Audio refinement both references and deletes source IDs: "
            f"{sorted(conflicting)}"
        )
    unaccounted = source_ids - set(referenced_counts) - set(deleted)
    if unaccounted:
        raise ValueError(
            f"Audio refinement does not account for source IDs: {sorted(unaccounted)}"
        )

    cues_by_source_id = {source_id: [] for source_id in source_ids}
    for cue in response_cues:
        for source_id in cue["source_ids"]:
            cues_by_source_id[source_id].append(cue)

    # Lineage checks run in response order, which must be script order.
    last_cue_index = {}
    for index, cue in enumerate(response_cues):
        for source_id in set(cue["source_ids"]):
            if source_id in last_cue_index and last_cue_index[source_id] != index - 1:
                raise ValueError(
                    f"Source ID {source_id} repeats in non-adjacent output cues"
                )
            last_cue_index[source_id] = index
    for source_id, count in referenced_counts.items():
        if count > 1:
            for cue in response_cues:
                if source_id in cue["source_ids"] and cue["source_ids"] != [source_id]:
                    raise ValueError(
                        f"Source ID {source_id} must split into singleton cues; "
                        "merges cannot overlap splits"
                    )
    for cue in response_cues:
        cue_ids = cue["source_ids"]
        if cue_ids:
            missing_ids = set(range(cue_ids[0], cue_ids[-1] + 1)) - set(cue_ids)
            for mid in missing_ids:
                if source_by_id[mid]["classification"] != "editorial":
                    raise ValueError(
                        f"Output cue {cue_ids} merges source IDs that are not "
                        "contiguous in script order"
                    )

    # Deletions may target dialogue cues only.
    for source_id in deleted:
        if source_by_id[source_id]["classification"] != "dialogue":
            raise ValueError(f"Audio refinement deletes non-dialogue cue {source_id}")

    # Pure editorial cues are preserved one-to-one with identical text and
    # timing. Mixed-cue fragments stay attached to their source lineage:
    # every source fragment appears at least once among that source's
    # descendant cues, and the complete output preserves the exact fragment
    # multiset, so nothing is lost or duplicated.
    for entry in source_entries:
        descendants = cues_by_source_id[entry["id"]]
        matches = [cue for cue in descendants if cue["source_ids"] == [entry["id"]]]
        if entry["classification"] == "editorial" and not (
            len(matches) == 1
            and matches[0]["text"] == entry["text"]
            and matches[0]["start"] == entry["start"]
            and matches[0]["end"] == entry["end"]
        ):
            raise ValueError(
                f"Pure editorial cue {entry['id']} must be preserved with "
                "identical text and timing"
            )
        if entry["classification"] == "mixed":
            expected = Counter(visual_fragment_strings(entry["text"]))
            consumed = Counter(
                fragment
                for cue in descendants
                for fragment in visual_fragment_strings(cue["text"])
            )
            if not all(
                consumed[fragment] >= count for fragment, count in expected.items()
            ):
                raise ValueError(
                    f"Bracketed fragments of mixed cue {entry['id']} must be "
                    "preserved exactly once"
                )

    # A cue may only carry bracketed fragments from its own source
    # lineages, so merged mixed cues keep distinct fragments attached to
    # their origins without exchange or injection.
    for cue in response_cues:
        if not cue["source_ids"]:
            continue
        allowed = {
            fragment
            for source_id in cue["source_ids"]
            for fragment in visual_fragment_strings(source_by_id[source_id]["text"])
        }
        for fragment in visual_fragment_strings(cue["text"]):
            if fragment not in allowed:
                raise ValueError(
                    f"Cue {cue['source_ids']} contains bracketed fragments "
                    "that do not belong to it"
                )

    expected_fragments = Counter(
        fragment
        for entry in source_entries
        if entry["classification"] != "dialogue"
        for fragment in visual_fragment_strings(entry["text"])
    )
    consumed_fragments = Counter(
        fragment
        for cue in response_cues
        for fragment in visual_fragment_strings(cue["text"])
    )
    if consumed_fragments != expected_fragments:
        raise ValueError(
            "The complete output must preserve the exact bracketed fragment multiset"
        )

    ordered_cues = sorted(
        response_cues,
        key=lambda item: parse_time(item["start"]),
    )

    for entry in source_entries:
        start = parse_time(entry["start"])
        end = parse_time(entry["end"])
        if interval_intersects_any((start, end), regions):
            continue
        matches = [
            cue
            for cue in cues_by_source_id[entry["id"]]
            if cue["source_ids"] == [entry["id"]]
        ]
        if not (
            len(matches) == 1
            and matches[0]["text"] == entry["text"]
            and matches[0]["start"] == entry["start"]
            and matches[0]["end"] == entry["end"]
        ):
            raise ValueError(
                f"Cue {entry['id']} lies outside every repair region and "
                "must remain identical"
            )
    for cue in ordered_cues:
        if is_identical_copy(cue, source_by_id):
            continue
        start = parse_time(cue["start"])
        end = parse_time(cue["end"])
        if cue["source_ids"]:
            common_regions = None
            source_intervals = []
            for source_id in cue["source_ids"]:
                source = source_by_id[source_id]
                source_interval = (
                    parse_time(source["start"]),
                    parse_time(source["end"]),
                )
                source_intervals.append(source_interval)
                matching = [
                    region
                    for region in regions
                    if interval_intersects_any(source_interval, [region])
                ]
                common_regions = (
                    set(matching)
                    if common_regions is None
                    else common_regions & set(matching)
                )
            envelopes = [
                (
                    min(region[0], *(interval[0] for interval in source_intervals))
                    - REPAIR_WINDOW_SECONDS,
                    max(region[1], *(interval[1] for interval in source_intervals))
                    + REPAIR_WINDOW_SECONDS,
                )
                for region in common_regions or []
            ]
            if not interval_contained_in_any((start, end), envelopes):
                raise ValueError(
                    "Changed cues must stay inside one shared repair region "
                    "plus the full extents of its referenced source cues"
                )
        elif not interval_contained_in_any(
            (start, end),
            [
                (region[0] - REPAIR_WINDOW_SECONDS, region[1] + REPAIR_WINDOW_SECONDS)
                for region in regions
            ],
        ):
            raise ValueError("Recovered cues must stay inside a repair region")

    seen = set()
    for cue in ordered_cues:
        key = (cue["start"], cue["end"], cue["text"])
        if key in seen:
            raise ValueError("Audio refinement produced duplicate output cues")
        seen.add(key)

    return ordered_cues


def merge_visual_boundary_fragments(entries, boundary_starts):
    """Merge exact pure-editorial cues clipped at adjacent segment boundaries.

    Two cues merge when they come from adjacent owner chunks, classify as
    pure editorial cues, share exactly the same trimmed text, and the first
    ends within 0.5 seconds of the shared boundary while the second starts
    within 0.5 seconds of it. The merged cue keeps the first start, the
    second end, the shared trimmed text, and the later owner index so one
    persistent visual cue can cross several boundaries.
    """
    entries = list(entries)
    merged = True
    while merged:
        merged = False
        for current_index, current in enumerate(entries):
            if current["chunk_idx"] >= len(boundary_starts):
                continue
            boundary = boundary_starts[current["chunk_idx"]]
            for following_index in range(current_index + 1, len(entries)):
                following = entries[following_index]
                if (
                    following["chunk_idx"] == current["chunk_idx"] + 1
                    and classify_cue_text(current["text"]) == "editorial"
                    and classify_cue_text(following["text"]) == "editorial"
                    and current["text"].strip() == following["text"].strip()
                    and abs(boundary - current["end"]) <= 0.5
                    and abs(following["start"] - boundary) <= 0.5
                ):
                    entries[current_index] = {
                        "start": current["start"],
                        "end": following["end"],
                        "text": current["text"].strip(),
                        "chunk_idx": following["chunk_idx"],
                    }
                    del entries[following_index]
                    merged = True
                    break
            if merged:
                break
    return entries
