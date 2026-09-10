"""Lyrics schemas, timing validation, and output serialization."""

import math
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from xml.etree import ElementTree as ET

from pydantic import BaseModel, ConfigDict, Field

TTML_NS = "http://www.w3.org/ns/ttml"
TTM_NS = "http://www.w3.org/ns/ttml#metadata"
ITUNES_INTERNAL_NS = "http://music.apple.com/lyric-ttml-internal"
ITUNES_NS = ITUNES_INTERNAL_NS
XML_NS = "http://www.w3.org/XML/1998/namespace"


class LyricLine(BaseModel):
    """One source-language lyric line with an audible interval."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(description="Sequential line ID such as L1")
    start: str = Field(description="Start time in HH:MM:SS.mmm format")
    end: str = Field(description="End time in HH:MM:SS.mmm format")
    text: str = Field(description="The sung source-language text")


class LyricsResponse(BaseModel):
    """A complete source-language lyric script."""

    model_config = ConfigDict(extra="forbid")
    lines: list[LyricLine]


class LyricCorrection(BaseModel):
    """Changed text or timing fields for one existing draft line."""

    model_config = ConfigDict(extra="forbid")
    id: str
    text: str | None = None
    start: str | None = None
    end: str | None = None


class RefinementResponse(BaseModel):
    """Only the corrections needed by the draft lyrics."""

    model_config = ConfigDict(extra="forbid")
    changes: list[LyricCorrection]


class TranslationLine(BaseModel):
    """One translation linked to a source line ID."""

    model_config = ConfigDict(extra="forbid")
    id: str
    text: str


class TranslationResponse(BaseModel):
    """A complete translation of one source lyric script."""

    model_config = ConfigDict(extra="forbid")
    translations: list[TranslationLine]


class GroundedSource(BaseModel):
    """One source reported by Gemini grounding metadata."""

    model_config = ConfigDict(extra="forbid")
    title: str = ""
    url: str


class SongContext(BaseModel):
    """Verified research context for one song and artist."""

    model_config = ConfigDict(extra="forbid")
    summary: str
    search_queries: list[str] = Field(default_factory=list)
    sources: list[GroundedSource] = Field(default_factory=list)


def derive_song_title(path: str | Path) -> str:
    """Return the source filename without its final suffix."""
    return Path(path).stem.strip()


def parse_time(value: str) -> float:
    """Parse SS, MM:SS, or HH:MM:SS with an optional decimal fraction."""
    raw = str(value).strip().replace(",", ".")
    if not raw or raw.startswith("-"):
        raise ValueError(f"Invalid timestamp: {value!r}")
    parts = raw.split(":")
    if len(parts) > 3 or any(not part for part in parts):
        raise ValueError(f"Invalid timestamp: {value!r}")
    try:
        numbers = [float(part) for part in parts]
    except ValueError:
        raise ValueError(f"Invalid timestamp: {value!r}") from None
    if any(not math.isfinite(number) or number < 0 for number in numbers):
        raise ValueError(f"Invalid timestamp: {value!r}")
    if len(numbers) > 1 and numbers[-1] >= 60:
        raise ValueError(f"Invalid timestamp: {value!r}")
    if len(numbers) == 3 and numbers[-2] >= 60:
        raise ValueError(f"Invalid timestamp: {value!r}")
    return sum(number * (60**index) for index, number in enumerate(reversed(numbers)))


def format_time(seconds: float) -> str:
    """Format seconds as a millisecond TTML timestamp."""
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError(f"Invalid timestamp: {seconds!r}")
    total_ms = int((Decimal(str(seconds)) * 1000).quantize(Decimal(1), ROUND_HALF_UP))
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds_part, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds_part:02d}.{milliseconds:03d}"


def format_lrc_time(seconds: float) -> str:
    """Format seconds as deterministic LRC minutes and centiseconds."""
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError(f"Invalid timestamp: {seconds!r}")
    total_cs = int((Decimal(str(seconds)) * 100).quantize(Decimal(1), ROUND_HALF_UP))
    minutes, remainder = divmod(total_cs, 6000)
    whole_seconds, centiseconds = divmod(remainder, 100)
    return f"{minutes:02d}:{whole_seconds:02d}.{centiseconds:02d}"


def validate_lyrics(lines: list[LyricLine], duration: float) -> list[LyricLine]:
    """Return canonical, ordered lyric lines within the complete song duration."""
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Song duration must be finite and greater than zero")
    if not lines:
        raise ValueError("Lyrics must contain at least one line")
    expected_ids = [f"L{index}" for index in range(1, len(lines) + 1)]
    if [line.id for line in lines] != expected_ids:
        raise ValueError("Lyric IDs must be unique and sequential from L1")

    validated = []
    previous_start = -1.0
    for line in lines:
        text = " ".join(part.strip() for part in line.text.splitlines()).strip()
        if not text:
            raise ValueError(f"Lyric line {line.id} must not be blank")
        start = parse_time(line.start)
        end = parse_time(line.end)
        if end <= start:
            raise ValueError(f"Lyric line {line.id} must have a positive interval")
        if end > duration:
            raise ValueError(f"Lyric line {line.id} exceeds the song duration")
        if start < previous_start:
            raise ValueError("Lyric lines must be ordered by start time")
        canonical_start = format_time(start)
        canonical_end = format_time(end)
        if parse_time(canonical_end) <= parse_time(canonical_start):
            raise ValueError(f"Lyric line {line.id} collapses at millisecond precision")
        if parse_time(canonical_end) > duration:
            raise ValueError(f"Lyric line {line.id} exceeds the song duration")
        previous_start = start
        validated.append(
            LyricLine(
                id=line.id,
                start=canonical_start,
                end=canonical_end,
                text=text,
            )
        )
    return validated


def apply_corrections(
    source_lines: list[LyricLine],
    changes: list[LyricCorrection],
    duration: float,
) -> list[LyricLine]:
    """Apply sparse text and timing corrections without changing line structure."""
    source_ids = {line.id for line in source_lines}
    by_id = {}
    for change in changes:
        if change.id not in source_ids:
            raise ValueError(f"Correction references unknown line {change.id}")
        if change.id in by_id:
            raise ValueError(f"Duplicate correction for line {change.id}")
        by_id[change.id] = change.model_dump(exclude={"id"}, exclude_none=True)
    corrected = [
        line.model_copy(update=by_id.get(line.id, {})) for line in source_lines
    ]
    return validate_lyrics(corrected, duration)


def validate_translation(
    translation: TranslationResponse, source_lines: list[LyricLine]
) -> list[TranslationLine]:
    """Validate complete, exact source-line coverage by a translation."""
    expected = [line.id for line in source_lines]
    actual = [line.id for line in translation.translations]
    if actual != expected:
        raise ValueError("Translation must cover every source line ID exactly once")
    result = []
    for line in translation.translations:
        text = " ".join(part.strip() for part in line.text.splitlines()).strip()
        if not text:
            raise ValueError(f"Translation for {line.id} must not be blank")
        result.append(TranslationLine(id=line.id, text=text))
    return result


def serialize_lrc(lines: list[LyricLine], duration: float) -> str:
    """Serialize source lyrics as line-timed LRC with explicit gap endings."""
    output = []
    audible_until = 0.0
    for index, line in enumerate(lines):
        output.append(f"[{format_lrc_time(parse_time(line.start))}]{line.text}")
        next_start = (
            parse_time(lines[index + 1].start) if index + 1 < len(lines) else duration
        )
        audible_until = max(audible_until, parse_time(line.end))
        if audible_until < next_start and format_lrc_time(
            audible_until
        ) != format_lrc_time(next_start):
            output.append(f"[{format_lrc_time(audible_until)}]")
    return "\n".join(output) + "\n"


def serialize_ttml(
    lines: list[LyricLine],
    translations: list[TranslationLine],
    duration: float,
    target_language: str,
    title: str,
) -> str:
    """Serialize source lyrics and linked translations as Apple-style TTML."""
    ET.register_namespace("", TTML_NS)
    ET.register_namespace("ttm", TTM_NS)
    ET.register_namespace("itunes", ITUNES_NS)
    root = ET.Element(
        f"{{{TTML_NS}}}tt",
        {
            f"{{{XML_NS}}}lang": "und",
            f"{{{ITUNES_NS}}}timing": "Line",
        },
    )
    head = ET.SubElement(root, f"{{{TTML_NS}}}head")
    metadata = ET.SubElement(head, f"{{{TTML_NS}}}metadata")
    ET.SubElement(metadata, f"{{{TTM_NS}}}title").text = title
    internal = ET.SubElement(metadata, "iTunesMetadata", {"xmlns": ITUNES_INTERNAL_NS})
    translation_set = ET.SubElement(internal, "translations")
    translation = ET.SubElement(
        translation_set,
        "translation",
        {"type": "subtitle", f"{{{XML_NS}}}lang": target_language},
    )
    for item in translations:
        ET.SubElement(translation, "text", {"for": item.id}).text = item.text

    body = ET.SubElement(root, f"{{{TTML_NS}}}body", {"dur": format_time(duration)})
    division = ET.SubElement(body, f"{{{TTML_NS}}}div")
    for line in lines:
        paragraph = ET.SubElement(
            division,
            f"{{{TTML_NS}}}p",
            {
                "begin": line.start,
                "end": line.end,
                f"{{{ITUNES_NS}}}key": line.id,
            },
        )
        paragraph.text = line.text
    ET.indent(root, space="  ")
    payload = ET.tostring(root, encoding="unicode", xml_declaration=False)
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + payload + "\n"
