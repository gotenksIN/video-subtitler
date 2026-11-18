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
_LABEL_TOKEN = r"[A-Z](?:[\w'.-]*[\w.])?"
_TITLE_LABEL_BODY = rf"{_LABEL_TOKEN}(?: {_LABEL_TOKEN}){{0,3}}"
_MIXED_LABEL_BODY = rf"{_LABEL_TOKEN}(?: (?i:{_LABEL_TOKEN})){{0,2}}"
_COMMA_LABEL_BODY = rf"{_LABEL_TOKEN}, (?:(?i:the) )?{_TITLE_LABEL_BODY}"
_SPEAKER_LABEL_BODY = rf"(?:[A-Z][\w'. -]{{1,30}}|{_COMMA_LABEL_BODY})"
_LINE_LABEL_PATTERN = (
    rf"\A\s*(?i:{_LABEL_TOKEN}):(?:[ \t]+(?=[^\n]*\S)|[ \t]*\Z)"
    rf"|^[ \t]*(?:(?:{_TITLE_LABEL_BODY}|{_MIXED_LABEL_BODY}|{_COMMA_LABEL_BODY}):[ \t]+(?=[^\n]*\S)"
    rf"|(?:{_TITLE_LABEL_BODY}|{_COMMA_LABEL_BODY}):[ \t]*$)"
)
_INLINE_LABEL_PATTERN = (
    rf"(?<!\S)(?:{_COMMA_LABEL_BODY}:(?:[ \t]+(?=[^\n]*\w)|[ \t]*$)"
    rf"|{_LABEL_TOKEN}:[ \t]+(?=[^\n]*\w))"
)
SPEAKER_LABEL_RE = re.compile(rf"^([ \t]*)({_SPEAKER_LABEL_BODY})(:[ \t]*)")
REFINEMENT_LABEL_RE = re.compile(rf"(?m){_LINE_LABEL_PATTERN}|{_INLINE_LABEL_PATTERN}")
AUDIO_REFINE_RESPONSE_CONTRACT = "sparse-patch-v1"
REPAIR_WINDOW_SECONDS = 10.0


class Caption(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int
    start: str = Field(description="Start time in HH:MM:SS.mmm format")
    end: str = Field(description="End time in HH:MM:SS.mmm format")
    text: str = Field(description="The subtitle text")


class SubtitleResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    captions: list[Caption]


class RefinedCaption(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int = Field(description="The integer ID of the subtitle to change")
    text: str = Field(description="The corrected text")


class RefinementResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    changes: list[RefinedCaption] = Field(
        description="List of subtitles to change. Only include ones that need changes."
    )
    deleted_ids: list[int] = Field(default_factory=list, alias="deletedIds")


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

    s, decimal, frac = s_ms.partition(".")
    if not (h.isdecimal() and m.isdecimal() and s.isdecimal()) or (
        decimal and not frac.isdecimal()
    ):
        raise ValueError(f"Invalid timestamp: {time_str}")

    frac_seconds = int(frac) / (10 ** len(frac)) if frac else 0
    return int(h) * 3600 + int(m) * 60 + int(s) + frac_seconds


def format_time(seconds):
    if seconds < 0:
        raise ValueError(f"Negative timestamp: {seconds}")

    total_ms = round(seconds * 1000)
    h, rem = divmod(total_ms, 3600 * 1000)
    m, rem = divmod(rem, 60 * 1000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def _validate_caption_text(text):
    """Return cue text that survives WebVTT serialization without cue loss."""
    text = text.removesuffix("\n")
    if not text.strip() or any(not line.strip() for line in text.split("\n")):
        raise ValueError("Caption text must contain nonempty lines")
    if "-->" in text or "\r" in text:
        raise ValueError("Caption text contains a reserved WebVTT separator")
    if _has_unmatched_brackets(text):
        raise ValueError("Caption text contains unmatched brackets")
    return text


def validate_captions(captions, chunk_duration, *, offset_sec=0, next_start=None):
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
        relative_start = parse_time(canonical_start)
        relative_end = parse_time(canonical_end)
        if relative_end <= relative_start:
            raise ValueError(
                f"Caption timing rounds to a non-positive interval for id={cap.id}: "
                f"{cap.start} --> {cap.end}"
            )

        abs_start = offset_sec + relative_start
        abs_end = offset_sec + relative_end
        if next_start is not None:
            abs_end = min(abs_end, next_start)
        if round(abs_end * 1000) <= round(abs_start * 1000):
            raise ValueError(
                f"Caption timing rounds to a non-positive stitched interval "
                f"for id={cap.id}: {cap.start} --> {cap.end}"
            )

        text = _validate_caption_text(cap.text)
        if len(text.splitlines()) > 2 and len(_visual_fragment_strings(text)) <= 2:
            raise ValueError(f"Caption id={cap.id} exceeds two lines")
        validated.append(
            {
                "id": cap.id,
                "start": canonical_start,
                "end": canonical_end,
                "text": text,
            }
        )

    validated = sorted(
        validated, key=lambda item: (parse_time(item["start"]), item["id"])
    )

    return validated


def _source_labels(text):
    return {
        match.group(2)
        for line in text.splitlines()
        if (match := SPEAKER_LABEL_RE.match(line))
        and match.group(3) != ":"
        and line[match.end() :].strip()
    }


def _refinement_label_re(source_labels=()):
    if not source_labels:
        return REFINEMENT_LABEL_RE
    names = "|".join(
        re.escape(name) for name in sorted(source_labels, key=len, reverse=True)
    )
    return re.compile(
        rf"(?m)^[ \t]*(?i:{names}):(?:[ \t]+(?=[^\n]*\S)|[ \t]*$)|"
        rf"{_LINE_LABEL_PATTERN}|"
        rf"(?<!\S)(?i:{names}):[ \t]+(?=[^\n]*\w)|{_INLINE_LABEL_PATTERN}",
    )


def _refinement_dialogue_labels(text, cue_id, source_labels=()):
    labels = []
    outside_text, _fragments, _unmatched = _parse_bracketed_text(text)
    label_re = _refinement_label_re(source_labels)
    previous_end = 0
    for label in label_re.finditer(outside_text):
        before = outside_text[previous_end : label.start()]
        if labels and not re.search(r"\w", before):
            if "\n" not in before:
                previous_end = label.end()
                continue
            raise ValueError(f"subtitle ID {cue_id} has empty dialogue")
        if not labels and re.search(r"\w", before):
            labels.append(False)
        labels.append(True)
        previous_end = label.end()
    if labels and not re.search(r"\w", outside_text[previous_end:]):
        raise ValueError(f"subtitle ID {cue_id} has empty dialogue")
    if not labels and re.search(r"\w", outside_text):
        labels.append(False)
    return labels


def _dialogue_body_key(text, source_labels=()):
    outside_text, _fragments, _unmatched = _parse_bracketed_text(text)
    label_re = _refinement_label_re(_source_labels(text) | set(source_labels))
    pieces = []
    cursor = 0
    previous_end = None
    for label in label_re.finditer(outside_text):
        if previous_end is not None:
            before = outside_text[previous_end : label.start()]
            if "\n" not in before and not re.search(r"\w", before):
                previous_end = label.end()
                continue
        pieces.append(outside_text[cursor : label.start()])
        cursor = previous_end = label.end()
    pieces.append(outside_text[cursor:])
    return re.sub(r"\W+", "", "".join(pieces)).casefold()


def validate_refinement_changes(changes, source_texts):
    """Validate all replacements against the structure of their source cues."""
    _valid_by_id, _invalid_ids, error = partition_refinement_changes(
        changes, source_texts
    )
    if error:
        raise ValueError(error)


def partition_refinement_changes(changes, source_texts, allowed_ids=None):
    """Return valid replacements, failed in-range IDs, and validation feedback."""
    valid_by_id = {}
    invalid_ids = []
    seen_ids = set()
    failures = []
    for change in changes:
        source = source_texts[change.id] if 0 <= change.id < len(source_texts) else None
        proposed = change.text
        try:
            if (
                source is not None
                and allowed_ids is not None
                and change.id not in allowed_ids
            ):
                continue
            if change.id in seen_ids:
                valid_by_id.pop(change.id, None)
                raise ValueError(f"subtitle ID {change.id} is duplicated")
            seen_ids.add(change.id)
            if source is None:
                raise ValueError(f"subtitle ID {change.id} is out of range")
            if not change.text.strip():
                raise ValueError(f"subtitle ID {change.id} has empty text")
            change.text = _validate_caption_text(change.text)
            if len(change.text.splitlines()) > max(
                2, len(_visual_fragment_strings(source))
            ):
                raise ValueError(f"subtitle ID {change.id} exceeds the line limit")
            if _has_unmatched_brackets(change.text):
                raise ValueError(f"subtitle ID {change.id} has unmatched brackets")
            if classify_cue_text(change.text) != classify_cue_text(source):
                raise ValueError(f"subtitle ID {change.id} changes cue class")
            if len(_visual_fragment_strings(change.text)) != len(
                _visual_fragment_strings(source)
            ):
                raise ValueError(
                    f"subtitle ID {change.id} changes visual fragment count"
                )

            source_labels = _source_labels(source)
            if _refinement_dialogue_labels(
                change.text, change.id, source_labels
            ) != _refinement_dialogue_labels(source, change.id, source_labels):
                raise ValueError(
                    f"subtitle ID {change.id} changes dialogue turns or labels"
                )
            shared_labels = source_labels | _source_labels(change.text)
            body_key = _dialogue_body_key(change.text, shared_labels)
            if len(body_key) >= 4 and body_key != _dialogue_body_key(
                source, shared_labels
            ):
                for adjacent in (change.id - 1, change.id + 1):
                    if not 0 <= adjacent < len(source_texts):
                        continue
                    adjacent_text = source_texts[adjacent]
                    adjacent_labels = shared_labels | _source_labels(adjacent_text)
                    adjacent_body = _dialogue_body_key(change.text, adjacent_labels)
                    if (
                        len(adjacent_body) >= 4
                        and adjacent_body
                        == _dialogue_body_key(adjacent_text, adjacent_labels)
                        and adjacent_body != _dialogue_body_key(source, adjacent_labels)
                    ):
                        raise ValueError(
                            f"subtitle ID {change.id} duplicates adjacent cue dialogue"
                        )
            valid_by_id[change.id] = change
        except ValueError as error:
            if (
                source is not None
                and (allowed_ids is None or change.id in allowed_ids)
                and change.id not in invalid_ids
            ):
                invalid_ids.append(change.id)
            failures.append(
                f"subtitle ID {change.id}: {error}\n"
                f"Source text: {source!r}\nProposed text: {proposed!r}"
            )
    return valid_by_id, invalid_ids, "\n\n".join(failures) or None


def partition_refinement_response(
    response, source_texts, *, has_audio=False, retained_deleted_ids=()
):
    """Partition edits and audio-backed duplicate deletions against source cues."""
    valid_changes, invalid_ids, change_error = partition_refinement_changes(
        response.changes, source_texts
    )
    failures = [change_error] if change_error else []
    valid_deleted = set()
    retained_deleted_ids = set(retained_deleted_ids)
    counts = Counter(response.deleted_ids)
    changed_ids = {change.id for change in response.changes}
    kinds = [classify_cue_text(text) for text in source_texts]
    keys = [_dialogue_body_key(text) for text in source_texts]
    for cue_id in counts:
        try:
            if not 0 <= cue_id < len(source_texts):
                raise ValueError("is out of range")
            if counts[cue_id] > 1:
                raise ValueError("is duplicated in deletedIds")
            if cue_id in changed_ids:
                raise ValueError("appears in both changes and deletedIds")
            if not has_audio:
                raise ValueError("requires attached audio for deletion")
            if kinds[cue_id] != "dialogue":
                raise ValueError("cannot delete editorial or mixed cues")
            key = keys[cue_id]
            left = right = cue_id
            while left > 0 and kinds[left - 1] == "dialogue" and keys[left - 1] == key:
                left -= 1
            while (
                right + 1 < len(source_texts)
                and kinds[right + 1] == "dialogue"
                and keys[right + 1] == key
            ):
                right += 1
            if not key or left == right:
                raise ValueError("has no adjacent duplicate dialogue")
            run_ids = range(left, right + 1)
            if all(index in counts for index in run_ids) and (
                cue_id not in retained_deleted_ids
                or all(index in retained_deleted_ids for index in run_ids)
            ):
                raise ValueError("cannot delete every cue in a duplicate run")
            valid_deleted.add(cue_id)
        except ValueError as error:
            valid_changes.pop(cue_id, None)
            if 0 <= cue_id < len(source_texts) and cue_id not in invalid_ids:
                invalid_ids.append(cue_id)
            failures.append(f"subtitle ID {cue_id} in deletedIds: {error}")
    return valid_changes, valid_deleted, invalid_ids, "\n\n".join(failures) or None


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


def _is_youtube_video_url(url):
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
        if _is_youtube_video_url(url):
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


def _visual_fragment_strings(text):
    """Return exact complete outer bracketed fragment strings."""
    _outside, fragments, _unmatched = _parse_bracketed_text(text)
    return fragments


def _has_unmatched_brackets(text):
    _outside, _fragments, unmatched = _parse_bracketed_text(text)
    return unmatched


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

    def labeled_lines(text):
        depth = 0
        for line in text.splitlines():
            match = SPEAKER_LABEL_RE.match(line) if depth == 0 else None
            for character in line:
                if character == "[":
                    depth += 1
                elif character == "]" and depth:
                    depth -= 1
            yield line, match

    for caption in captions:
        if classify_cue_text(caption.raw_text) == "editorial":
            continue
        for _line, match in labeled_lines(caption.raw_text):
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
        if classify_cue_text(caption.raw_text) == "editorial":
            continue
        lines = caption.raw_text.splitlines()
        rewritten = []
        for line, match in labeled_lines(caption.raw_text):
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


def _build_repair_regions(boundaries, window=REPAIR_WINDOW_SECONDS):
    return merge_intervals(
        [(boundary - window, boundary + window) for boundary in boundaries]
    )


def _interval_intersects_any(interval, regions):
    start, end = interval
    return any(
        start < region_end and end > region_start
        for region_start, region_end in regions
    )


def _interval_contained_in_any(interval, regions):
    start, end = interval
    return any(
        region_start <= start and end <= region_end
        for region_start, region_end in regions
    )


def _canonical_timestamp(value):
    try:
        parsed = parse_time(value)
    except ValueError as error:
        raise ValueError(f"Invalid timestamp {value!r}: {error}") from None
    formatted = format_time(parsed)
    if str(value).strip() != formatted:
        raise ValueError(f"Timestamp {value!r} is not canonical HH:MM:SS.mmm format")
    return formatted


def _is_identical_copy(cue, source_by_id):
    if len(cue["source_ids"]) != 1:
        return False
    source = source_by_id[cue["source_ids"][0]]
    return (
        cue["text"] == source["text"]
        and cue["start"] == source["start"]
        and cue["end"] == source["end"]
    )


def _filter_audio_refinement_patch(
    response, source_entries, boundaries, window=REPAIR_WINDOW_SECONDS
):
    """Drop patch cues and deletions without repair-region authority.

    Gemini receives the complete script and sometimes edits cues outside
    every repair region. Discarding those edits keeps each cue outside
    repair regions as an exact copy of its source entry.
    """
    regions = _build_repair_regions(boundaries, window)
    source_intervals = {
        entry["id"]: (parse_time(entry["start"]), parse_time(entry["end"]))
        for entry in source_entries
    }

    kept_cues = []
    for cue in response.cues:
        cue_ids = cue.source_ids
        if not cue_ids:
            keep = _interval_intersects_any(
                (parse_time(cue.start), parse_time(cue.end)), regions
            )
        elif any(source_id not in source_intervals for source_id in cue_ids):
            # Unknown source IDs stay fatal during validation.
            keep = True
        else:
            keep = all(
                _interval_intersects_any(source_intervals[source_id], regions)
                for source_id in cue_ids
            )
        if keep:
            kept_cues.append(cue)

    kept_deleted = [
        source_id
        for source_id in response.deleted_source_ids
        if source_id not in source_intervals
        or _interval_intersects_any(source_intervals[source_id], regions)
    ]

    return response.model_copy(
        update={"cues": kept_cues, "deleted_source_ids": kept_deleted}
    )


def _reconstruct_sparse_audio_candidate(response, source_entries):
    """Expand sparse patches with exact copies of every omitted source cue."""
    referenced = {source_id for cue in response.cues for source_id in cue.source_ids}
    deleted = set(response.deleted_source_ids)

    patch_lineage = [source_id for cue in response.cues for source_id in cue.source_ids]
    if any(after < before for before, after in itertools.pairwise(patch_lineage)):
        raise ValueError(
            "Sparse audio refinement cues reference source IDs out of script order"
        )

    for cue in response.cues:
        if _has_unmatched_brackets(cue.text):
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
            parse_time(_canonical_timestamp(cue.start)),
            cue.source_ids[0] if cue.source_ids else -1,
        )
    )
    return response.model_copy(update={"cues": candidate_cues})


def validate_audio_refinement_response(
    response, source_entries, audio_duration, boundaries, *, source_duration
):
    """Reconstruct and validate the complete candidate before writing files.

    The patch is filtered to repair-region authority first, so cues outside
    repair regions survive as exact host-side copies. Lineage and authority
    checks then run on the complete candidate.
    """
    if any(_has_unmatched_brackets(entry["text"]) for entry in source_entries):
        raise ValueError("Audio refinement source cue contains unmatched brackets")
    source_by_id = {entry["id"]: entry for entry in source_entries}
    source_ids = set(source_by_id)
    if len(source_ids) != len(source_entries):
        raise ValueError("Audio refinement source IDs are not unique")
    response = _filter_audio_refinement_patch(response, source_entries, boundaries)
    response = _reconstruct_sparse_audio_candidate(response, source_entries)
    regions = _build_repair_regions(boundaries)
    audio_duration_ms = round(audio_duration * 1000)
    source_duration_ms = round(source_duration * 1000)

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
        if not cue_ids and _visual_fragment_strings(cue.text):
            raise ValueError("Recovered cues must not contain bracketed text")
        if not cue_ids and not re.search(
            r"\w", REFINEMENT_LABEL_RE.sub("", cue.text.removesuffix("\n"))
        ):
            raise ValueError("Recovered cues must contain spoken dialogue")
        if cue_ids != sorted(set(cue_ids)):
            raise ValueError(
                f"Audio refinement cue source_ids must be strictly "
                f"increasing with no duplicates: {cue_ids}"
            )
        for source_id in cue_ids:
            referenced_counts[source_id] += 1
        start = _canonical_timestamp(cue.start)
        end = _canonical_timestamp(cue.end)
        start_ms = round(parse_time(start) * 1000)
        end_ms = round(parse_time(end) * 1000)
        if start_ms < 0:
            raise ValueError(f"Audio refinement cue has a negative start: {start}")
        cue_text = cue.text
        candidate = {
            "source_ids": cue_ids,
            "start": start,
            "end": end,
            "text": cue_text,
        }
        if not (
            len(cue_ids) == 1
            and cue_ids[0] in source_by_id
            and _is_identical_copy(candidate, source_by_id)
        ):
            cue_text = _validate_caption_text(cue_text)
            candidate["text"] = cue_text
        unchanged_visual = (
            len(cue_ids) == 1
            and cue_ids[0] in source_by_id
            and source_by_id[cue_ids[0]]["classification"] in {"editorial", "mixed"}
            and _is_identical_copy(candidate, source_by_id)
        )
        if unchanged_visual and end_ms > source_duration_ms:
            raise ValueError(
                f"Unchanged visual cue {start} --> {end} exceeds the source video duration"
            )
        if end_ms > audio_duration_ms and not unchanged_visual:
            if end_ms - audio_duration_ms > 500:
                raise ValueError(
                    f"Audio refinement cue {start} --> {end} exceeds the "
                    "complete audio duration"
                )
            unchanged_outside_dialogue = (
                len(cue_ids) == 1
                and cue_ids[0] in source_by_id
                and source_by_id[cue_ids[0]]["classification"] == "dialogue"
                and _is_identical_copy(candidate, source_by_id)
                and not _interval_intersects_any(
                    (parse_time(start), parse_time(end)), regions
                )
            )
            if not unchanged_outside_dialogue:
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
                "text": cue_text,
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
            if source_id in last_cue_index and any(
                len(interposed["source_ids"]) != 1
                or source_by_id[interposed["source_ids"][0]]["classification"]
                != "editorial"
                or not _is_identical_copy(interposed, source_by_id)
                for interposed in response_cues[last_cue_index[source_id] + 1 : index]
            ):
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
            expected = Counter(_visual_fragment_strings(entry["text"]))
            consumed = Counter(
                fragment
                for cue in descendants
                for fragment in _visual_fragment_strings(cue["text"])
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
            for fragment in _visual_fragment_strings(source_by_id[source_id]["text"])
        }
        for fragment in _visual_fragment_strings(cue["text"]):
            if fragment not in allowed:
                raise ValueError(
                    f"Cue {cue['source_ids']} contains bracketed fragments "
                    "that do not belong to it"
                )

    expected_fragments = Counter(
        fragment
        for entry in source_entries
        if entry["classification"] != "dialogue"
        for fragment in _visual_fragment_strings(entry["text"])
    )
    consumed_fragments = Counter(
        fragment
        for cue in response_cues
        for fragment in _visual_fragment_strings(cue["text"])
    )
    if consumed_fragments != expected_fragments:
        raise ValueError(
            "The complete output must preserve the exact bracketed fragment multiset"
        )

    ordered_cues = sorted(
        response_cues,
        key=lambda item: parse_time(item["start"]),
    )

    active = []
    for cue in ordered_cues:
        start = parse_time(cue["start"])
        active = [other for other in active if parse_time(other["end"]) > start]
        for other in active:
            if not cue["source_ids"] or not other["source_ids"]:
                continue
            inherited_overlap = any(
                left != right
                and max(
                    parse_time(source_by_id[left]["start"]),
                    parse_time(source_by_id[right]["start"]),
                )
                < min(
                    parse_time(source_by_id[left]["end"]),
                    parse_time(source_by_id[right]["end"]),
                )
                for left in other["source_ids"]
                for right in cue["source_ids"]
            )
            if not inherited_overlap:
                raise ValueError("Audio refinement introduces a new timestamp overlap")
        active.append(cue)

    for entry in source_entries:
        start = parse_time(entry["start"])
        end = parse_time(entry["end"])
        if _interval_intersects_any((start, end), regions):
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
        if _is_identical_copy(cue, source_by_id):
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
                    if _interval_intersects_any(source_interval, [region])
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
            if not _interval_contained_in_any((start, end), envelopes):
                raise ValueError(
                    "Changed cues must stay inside one shared repair region "
                    "plus the full extents of its referenced source cues"
                )
        elif not _interval_contained_in_any(
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
    """Merge case-insensitive editorial fragments in one script-order pass."""
    result = []
    pending = {}
    for entry in entries:
        if classify_cue_text(entry["text"]) != "editorial":
            result.append(entry)
            continue
        owner = entry["chunk_idx"]
        text_key = entry["text"].strip().casefold()
        previous = pending.pop((owner - 1, text_key), None)
        if (
            previous is not None
            and abs(entry["start"] - boundary_starts[owner - 1]) <= 0.5
        ):
            current = result[previous]
            entry = {
                "start": current["start"],
                "end": entry["end"],
                "text": current["text"].strip(),
                "chunk_idx": owner,
            }
            result[previous] = entry
            index = previous
        else:
            index = len(result)
            result.append(entry)
        if (
            owner < len(boundary_starts)
            and abs(entry["end"] - boundary_starts[owner]) <= 0.5
        ):
            pending[(owner, text_key)] = index
    return result
