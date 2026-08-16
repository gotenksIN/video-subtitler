"""Boundary-echo deduplication semantics."""

import pytest
import webvtt

from modules import core


def vtt_with(captions):
    value = webvtt.WebVTT()
    value.captions.extend(
        webvtt.Caption(start, end, text) for start, end, text in captions
    )
    return value


def test_boundary_dedup_removes_exact_echo_caption_and_aligns_indices():
    value = vtt_with(
        [
            ("00:00:00.000", "00:00:05.000", "Host: We repeat this phrase."),
            ("00:00:04.000", "00:00:06.000", "Host: Repeat this phrase."),
        ]
    )

    indices = core.dedup_boundary_overlap(value, [0, 1])

    assert [caption.text for caption in value] == ["Host: We repeat this phrase."]
    assert indices == [0]


def test_boundary_dedup_keeps_fresh_text_after_removing_echo_prefix():
    value = vtt_with(
        [
            (
                "00:00:00.000",
                "00:00:05.000",
                "Host: Intro before the repeated boundary\nphrase",
            ),
            (
                "00:00:04.000",
                "00:00:06.000",
                "Host: Repeated boundary\nphrase\nHost: New detail.",
            ),
        ]
    )

    indices = core.dedup_boundary_overlap(value, [0, 1])

    assert [caption.text for caption in value] == [
        "Host: Intro before the repeated boundary\nphrase",
        "Host: New detail.",
    ]
    assert indices == [0, 1]


def test_boundary_dedup_preserves_surviving_timestamps():
    value = vtt_with(
        [
            ("00:00:01.000", "00:00:05.000", "Host: We repeat this phrase."),
            ("00:00:04.500", "00:00:06.000", "Host: Repeat this phrase."),
        ]
    )

    core.dedup_boundary_overlap(value, [0, 1])

    assert [(caption.start, caption.end) for caption in value] == [
        ("00:00:01.000", "00:00:05.000")
    ]


def test_boundary_dedup_requires_time_overlap():
    value = vtt_with(
        [
            ("00:00:00.000", "00:00:04.000", "Host: Repeat this phrase."),
            ("00:00:04.000", "00:00:06.000", "Host: Repeat this phrase."),
        ]
    )

    indices = core.dedup_boundary_overlap(value, [0, 1])

    assert len(value.captions) == 2
    assert indices == [0, 1]


def test_boundary_dedup_requires_adjacent_owner_chunks():
    value = vtt_with(
        [
            ("00:00:00.000", "00:00:05.000", "Host: Repeat this phrase."),
            ("00:00:04.000", "00:00:06.000", "Host: Repeat this phrase."),
        ]
    )

    indices = core.dedup_boundary_overlap(value, [0, 0])

    assert len(value.captions) == 2
    assert indices == [0, 0]


def test_boundary_dedup_compares_against_the_previous_survivor():
    value = vtt_with(
        [
            ("00:00:00.000", "00:00:05.000", "Host: We repeat this phrase."),
            ("00:00:04.000", "00:00:08.000", "Host: Repeat this phrase."),
            ("00:00:07.000", "00:00:09.000", "Host: Repeat this phrase."),
        ]
    )

    indices = core.dedup_boundary_overlap(value, [0, 1, 2])

    assert [caption.text for caption in value] == [
        "Host: We repeat this phrase.",
        "Host: Repeat this phrase.",
    ]
    assert indices == [0, 2]


def test_boundary_dedup_rejects_mismatched_index_count():
    value = vtt_with([("00:00:00.000", "00:00:01.000", "Host: Repeat this phrase.")])

    with pytest.raises(ValueError, match="one chunk index per caption"):
        core.dedup_boundary_overlap(value, [])


def test_boundary_dedup_rejects_mismatched_timing_count():
    value = vtt_with([("00:00:00.000", "00:00:01.000", "Host: Repeat this phrase.")])

    with pytest.raises(ValueError, match="one timing per caption"):
        core.dedup_boundary_overlap(value, [0], timings=[])
