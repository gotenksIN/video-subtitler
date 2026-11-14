"""Text capability contracts without model or filesystem collaborators."""

import pytest
import webvtt

from modules import core


def test_text_refinement_rejects_structural_mutations():
    for source, replacement in (
        ("[Title] Host: Hello", "Host: Hello"),
        ("[Title] Host: Hello", "[Title] [Extra] Host: Hello"),
        ("[Title] Host: Hello", "[Title Host: Hello"),
        ("Host: Hello\nGuest: Yes", "Host: Hello and yes"),
        ("Host: Hello", "Hello"),
        ("Hello", "host: Hello"),
        ("Host: Hello", "Host:"),
        ("Host: Hello\nGuest: Yes", "Host: Hello\nGuest:"),
        ("Host: Hello\nworld", "Host: Hello\nGuest: world"),
        ("Hello\nHost: Yes", "Host: Hello\nHost: Yes"),
        ("Host: Hello world", "Host: Hello\n\nworld"),
        ("Host: Hello world", "Host: Hello\n \nworld"),
        ("Host: Hello world", "Host: Hello\rworld"),
        ("Host: Hello", "Host: Hello\n\n"),
    ):
        with pytest.raises(ValueError):
            core.validate_refinement_changes(
                [core.RefinedCaption(id=0, text=replacement)], [source]
            )
    for changes in (
        [core.RefinedCaption(id=0, text="One"), core.RefinedCaption(id=0, text="Two")],
        [core.RefinedCaption(id=1, text="Unknown")],
        [core.RefinedCaption(id=-1, text="Negative")],
        [core.RefinedCaption(id=0, text=" ")],
    ):
        with pytest.raises(ValueError):
            core.validate_refinement_changes(changes, ["Original"])

    sources = [
        "Host: This is",
        "Host: a test.",
        "and a continuation.",
        "Original",
        "More text",
    ]
    replacements = ["Host: This is", "the test.", "Host: and a continuation."]
    changes = [
        core.RefinedCaption(id=index, text=text)
        for index, text in enumerate(replacements)
    ] + [
        core.RefinedCaption(id=5, text="Unknown"),
        core.RefinedCaption(id=0, text="Host: Duplicate"),
        core.RefinedCaption(id=3, text=""),
        core.RefinedCaption(id=4, text="Broken\n\ntext"),
    ]
    with pytest.raises(ValueError) as failure:
        core.validate_refinement_changes(changes, sources)
    message = str(failure.value)
    for cue_id in (1, 2):
        assert f"subtitle ID {cue_id} changes dialogue turns or labels" in message
        assert f"Source text: {sources[cue_id]!r}" in message
        assert f"Proposed text: {replacements[cue_id]!r}" in message
    assert "subtitle ID 5 is out of range" in message
    assert "Source text: None" in message
    assert "subtitle ID 0 is duplicated" in message
    assert "subtitle ID 3 has empty text" in message
    assert "subtitle ID 4: Caption text must contain nonempty lines" in message


def test_text_refinement_accepts_corrections_and_same_turn_rewrapping():
    for source, replacement in (
        ("[Bad OCR] host: Helo\nGuest: Yess", "[Correct OCR] Host: Hello\nGuest: Yes"),
        ("[Bad\nOCR]", "[Correct OCR]"),
        ("Host: Hello world", "Host: Hello\nworld"),
        ("Host: Hello\nGuest: Yes", "Host: Hello Guest: Yes"),
        ("well, she asked:", "well, she asks:"),
        ("Host: Hello world", "Host: Hello\nworld\n"),
    ):
        core.validate_refinement_changes(
            [core.RefinedCaption(id=0, text=replacement)], [source]
        )


@pytest.mark.parametrize(
    "texts,names,expected",
    [
        (
            ["Host: One", "HOST: Two", "HOST: Three"],
            [],
            [
                "HOST: One",
                "HOST: Two",
                "HOST: Three",
            ],
        ),
        (["Host: One", "HOST: Two"], [], ["Host: One", "Host: Two"]),
        (
            ["HOST: One", "HOST: Two", "Host: Three"],
            ["Host"],
            [
                "Host: One",
                "Host: Two",
                "Host: Three",
            ],
        ),
    ],
    ids=["frequency", "first-seen-tie", "grounded-override"],
)
def test_speaker_casing_preserves_visual_and_non_label_text(texts, names, expected):
    value = webvtt.WebVTT()
    for index, text in enumerate([*texts, "[HOST: Title]", "No label [HOST: Graphic]"]):
        value.captions.append(
            webvtt.Caption(
                core.format_time(index),
                core.format_time(index + 1),
                text,
            )
        )
    core.canonicalize_speaker_casing(value, names)
    assert [cue.raw_text for cue in value] == [
        *expected,
        "[HOST: Title]",
        "No label [HOST: Graphic]",
    ]
    assert value[-1].end == core.format_time(len(texts) + 2)
    visual = "[Profile [Details]\nHOST: Kim]\nHost: Hi"
    for grounded_names in ([], ["Host"]):
        mixed = [webvtt.Caption("00:00:00.000", "00:00:01.000", visual)]
        core.canonicalize_speaker_casing(mixed, grounded_names)
        assert mixed[0].raw_text == visual


def test_only_matching_editorial_fragments_merge_at_adjacent_boundaries():
    entries = [
        {"start": 18.0, "end": 20.1, "text": "[Banner]", "chunk_idx": 0},
        {"start": 19.5, "end": 20.5, "text": "Host: Talk.", "chunk_idx": 0},
        {"start": 19.9, "end": 22.0, "text": "[BANNER]", "chunk_idx": 1},
        {"start": 21.9, "end": 24.0, "text": "[Banner]", "chunk_idx": 2},
    ]
    assert core.merge_visual_boundary_fragments(entries, [20.0, 22.0]) == [
        {"start": 18.0, "end": 24.0, "text": "[Banner]", "chunk_idx": 2},
        {"start": 19.5, "end": 20.5, "text": "Host: Talk.", "chunk_idx": 0},
    ]
    for left, right in (
        ("Host: Hello", "Host: Hello"),
        ("[Banner]", "[Different]"),
        ("[Banner] Hello", "[Banner] Hello"),
    ):
        values = [
            {"start": 19.0, "end": 20.0, "text": left, "chunk_idx": 0},
            {"start": 20.0, "end": 21.0, "text": right, "chunk_idx": 1},
        ]
        assert core.merge_visual_boundary_fragments(values, [20.0]) == values


def test_caption_validation_clamps_sorts_and_preserves_text():
    captions = [
        core.Caption(id=2, start="0.5", end="2.5", text="  Host: Hello\nworld  \n"),
        core.Caption(id=1, start="0", end="1", text="[[Title] Detail]"),
    ]
    assert core.validate_captions(captions, 2) == [
        {
            "id": 1,
            "start": "00:00:00.000",
            "end": "00:00:01.000",
            "text": "[[Title] Detail]",
        },
        {
            "id": 2,
            "start": "00:00:00.500",
            "end": "00:00:02.000",
            "text": "  Host: Hello\nworld  ",
        },
    ]


def test_caption_validation_rejects_invalid_intervals_ids_and_text():
    for start, end, text in (
        ("-1", "1", "Hello"),
        ("1", "1", "Hello"),
        ("2", "1", "Hello"),
        ("0.0001", "0.0002", "Hello"),
        ("0", "2.501", "Hello"),
        ("0", "1", ""),
        ("0", "1", "A\n\nB"),
        ("0", "1", "A\n\n"),
        ("0", "1", "A --> B"),
        ("0", "1", "[A"),
        ("0", "1", "A]"),
    ):
        with pytest.raises(ValueError):
            core.validate_captions(
                [core.Caption(id=0, start=start, end=end, text=text)], 2
            )
    cue = core.Caption(id=0, start="0", end="1", text="Hello")
    with pytest.raises(ValueError):
        core.validate_captions([cue, cue], 2)


def test_timestamps_accept_supported_shapes_and_round_to_milliseconds():
    for text, seconds in (
        ("1", 1),
        ("01:02.5", 62.5),
        ("01:02:03,125", 3723.125),
    ):
        assert core.parse_time(text) == seconds
    assert core.format_time(59.9996) == "00:01:00.000"
    assert core.format_time(3723.125) == "01:02:03.125"
    for text in ("", "word", "1:2:3:4", "nan", "inf", "1::2"):
        with pytest.raises(ValueError):
            core.parse_time(text)


def test_source_titles_strip_only_known_extensions_and_language_suffixes():
    for filename, title in (
        ("/dir/Program.mp4.en-US.vtt", "Program"),
        (" Program.webm ", "Program.webm"),
        ("Program.mp4", "Program"),
        ("Program.en.vtt", "Program"),
        ("Program.notes", "Program.notes"),
    ):
        assert core.derive_source_title(filename) == title


def test_context_urls_validate_deduplicate_and_route_watch_links_only():
    urls = [
        "https://youtu.be/abc",
        "https://www.youtube.com/watch?v=abc",
        "https://example.com/context",
        "https://youtube.com/channel/name",
    ]
    assert core.validate_context_urls([*urls, urls[0]]) == urls
    assert core.classify_context_urls(urls) == (urls[:2], urls[2:])
    assert core.validate_context_urls(None) == []
    for url in ("relative", "ftp://example.com", "https:///path"):
        with pytest.raises(ValueError, match="Invalid --context-url"):
            core.validate_context_urls([url])
