"""Stitching, overlap filtering, and boundary deduplication outcomes."""

import pytest
import webvtt

from modules import pipeline
from tests.support.workdir import write_chunk_layout, write_chunk_subtitles


def stitch_layout(tmp_path, rows, captions_by_chunk, overlap):
    write_chunk_layout(tmp_path, rows, overlap=overlap)
    for index, captions in captions_by_chunk.items():
        write_chunk_subtitles(tmp_path, index, captions)
    return tmp_path / "output.vtt"


def test_stitch_offsets_captions_and_filters_context_by_midpoint(tmp_path):
    output = stitch_layout(
        tmp_path,
        [("chunk_000.mp4", 0, 10), ("chunk_001.mp4", 10, 20)],
        {
            0: [
                {"start": "9", "end": "11", "text": "Right edge excluded"},
                {"start": "8", "end": "10", "text": "First owner"},
            ],
            1: [
                {"start": "1", "end": "3", "text": "Left edge included"},
                {"start": "4", "end": "6", "text": "Offset caption"},
            ],
        },
        overlap=2,
    )

    provenance = pipeline.stitch(tmp_path, output)

    result = webvtt.read(output)
    assert [(c.start, c.end, c.text) for c in result] == [
        ("00:00:08.000", "00:00:10.000", "First owner"),
        ("00:00:09.000", "00:00:11.000", "Left edge included"),
        ("00:00:12.000", "00:00:14.000", "Offset caption"),
    ]
    assert provenance == [0, 1, 1]


def test_stitch_rejects_missing_or_unexpected_results_without_publishing(tmp_path):
    write_chunk_layout(tmp_path, [("chunk_000.mp4", 0, 5)])
    write_chunk_subtitles(tmp_path, 2, [])
    output = tmp_path / "output.vtt"

    with pytest.raises(ValueError) as error:
        pipeline.stitch(tmp_path, output)

    message = str(error.value)
    assert "missing chunk indices: [0]" in message
    assert "unexpected chunk indices: [2]" in message
    assert not output.exists()


def test_stitch_preserves_multiline_text_without_inserting_breaks(tmp_path):
    text = (
        "Host: This generic turn is deliberately longer than forty-two characters\n"
        "[On-screen banner remains unchanged]\n"
        "Plain unlabeled line remains unchanged\n"
        "Guest: This separate turn remains unchanged"
    )
    output = stitch_layout(
        tmp_path,
        [("chunk_000.mp4", 0, 5)],
        {0: [{"start": "1", "end": "3", "text": text}]},
        overlap=0,
    )

    provenance = pipeline.stitch(tmp_path, output)

    caption = webvtt.read(output)[0]
    assert (caption.start, caption.end) == ("00:00:01.000", "00:00:03.000")
    assert caption.text == text
    assert provenance is None


def test_stitch_keeps_repeated_text_and_overlapping_cues_without_overlap_mode(tmp_path):
    output = stitch_layout(
        tmp_path,
        [("chunk_000.mp4", 0, 5), ("chunk_001.mp4", 5, 10)],
        {
            0: [{"start": "1", "end": "5.5", "text": "Again"}],
            1: [{"start": "0", "end": "1", "text": "Again"}],
        },
        overlap=0,
    )

    provenance = pipeline.stitch(tmp_path, output)

    result = webvtt.read(output)
    assert [c.text for c in result] == ["Again", "Again"]
    assert [(c.start, c.end) for c in result] == [
        ("00:00:01.000", "00:00:05.500"),
        ("00:00:05.000", "00:00:06.000"),
    ]
    assert provenance is None


def test_stitch_removes_exact_boundary_echo_and_keeps_new_text(tmp_path):
    output = stitch_layout(
        tmp_path,
        [("chunk_000.mp4", 0, 5), ("chunk_001.mp4", 5, 10)],
        {
            0: [
                {
                    "start": "1",
                    "end": "5.5",
                    "text": "HOST: Intro repeated phrase!",
                }
            ],
            1: [
                {
                    "start": "1",
                    "end": "2",
                    "text": "host: Repeated phrase.\nHost: Keep this detail.",
                }
            ],
        },
        overlap=1,
    )

    provenance = pipeline.stitch(tmp_path, output)

    result = webvtt.read(output)
    assert [c.text for c in result] == [
        "HOST: Intro repeated phrase!",
        "Host: Keep this detail.",
    ]
    assert provenance == [0, 1]


@pytest.mark.parametrize(
    ("earlier_text", "later_text"),
    [
        ("Host: Repeat this phrase.", "Guest: Repeat this phrase."),
        ("Repeat this phrase.", "Repeat this phrase."),
        ("Host: Again.", "Host: Again."),
        ("Host: Keep this phrase.", "Host: Keep that phrase."),
        (
            "Host: Repeat this phrase.\nGuest: Different reply.",
            "Host: Repeat this phrase.",
        ),
        (
            "Host: Repeat this phrase.",
            "[On-screen card]\nHost: Repeat this phrase.",
        ),
        (
            "Host: Shared opening.\nplain text",
            "Host: Shared opening.",
        ),
    ],
)
def test_stitch_preserves_ambiguous_boundary_repetition(
    tmp_path, earlier_text, later_text
):
    output = stitch_layout(
        tmp_path,
        [("chunk_000.mp4", 0, 5), ("chunk_001.mp4", 5, 10)],
        {
            0: [{"start": "1", "end": "5.5", "text": earlier_text}],
            1: [{"start": "1", "end": "2", "text": later_text}],
        },
        overlap=1,
    )

    provenance = pipeline.stitch(tmp_path, output)

    result = webvtt.read(output)
    assert [c.text for c in result] == [earlier_text, later_text]
    assert provenance == [0, 1]


def test_stitch_removes_a_complete_two_turn_boundary_echo(tmp_path):
    repeated = "Host: Shared opening.\nGuest: Shared response."
    output = stitch_layout(
        tmp_path,
        [("chunk_000.mp4", 0, 5), ("chunk_001.mp4", 5, 10)],
        {
            0: [{"start": "1", "end": "5.5", "text": repeated}],
            1: [
                {
                    "start": "1",
                    "end": "2",
                    "text": f"{repeated}\nNarrator: Fresh detail.",
                }
            ],
        },
        overlap=1,
    )

    provenance = pipeline.stitch(tmp_path, output)

    result = webvtt.read(output)
    assert [caption.text for caption in result] == [
        repeated,
        "Narrator: Fresh detail.",
    ]
    assert provenance == [0, 1]


def test_stitch_publishes_an_empty_vtt_when_all_captions_are_context(tmp_path):
    output = stitch_layout(
        tmp_path,
        [("chunk_000.mp4", 0, 5), ("chunk_001.mp4", 5, 10)],
        {
            0: [{"start": "6", "end": "7", "text": "Context only"}],
            1: [{"start": "0", "end": "1", "text": "Context only"}],
        },
        overlap=2,
    )

    provenance = pipeline.stitch(tmp_path, output)

    result = webvtt.read(output)
    assert len(result) == 0
    assert provenance == []


def test_stitch_works_without_a_manifest_file(tmp_path):
    (tmp_path / "segments.csv").write_text("chunk_000.mp4,0,5\n", encoding="utf-8")
    write_chunk_subtitles(tmp_path, 0, [{"start": "1", "end": "2", "text": "Only"}])
    output = tmp_path / "output.vtt"

    provenance = pipeline.stitch(tmp_path, output)

    result = webvtt.read(output)
    assert [c.text for c in result] == ["Only"]
    assert provenance is None
