"""Stitching outcomes for stream-copy chunk results."""

import pytest
import webvtt

from modules import pipeline
from tests.support.workdir import write_chunk_layout, write_chunk_subtitles


def stitch_layout(tmp_path, rows, captions_by_chunk):
    write_chunk_layout(tmp_path, rows)
    for index, captions in captions_by_chunk.items():
        write_chunk_subtitles(tmp_path, index, captions)
    return tmp_path / "output.vtt"


def captions_of(path):
    return [(caption.start, caption.end, caption.text) for caption in webvtt.read(path)]


def test_stitch_offsets_captions_by_actual_segment_starts(tmp_path):
    output = stitch_layout(
        tmp_path,
        [("chunk_000.mp4", 0, 2), ("chunk_001.mp4", 2, 4), ("chunk_002.mp4", 4, 6)],
        {
            0: [{"start": "1", "end": "2", "text": "First chunk"}],
            1: [{"start": "0.5", "end": "1.5", "text": "Second chunk"}],
            2: [{"start": "1", "end": "2", "text": "Third chunk"}],
        },
    )

    result = pipeline.stitch(tmp_path, output)

    assert result == output
    assert captions_of(output) == [
        ("00:00:01.000", "00:00:02.000", "First chunk"),
        ("00:00:02.500", "00:00:03.500", "Second chunk"),
        ("00:00:05.000", "00:00:06.000", "Third chunk"),
    ]


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
    )

    pipeline.stitch(tmp_path, output)

    caption = webvtt.read(output)[0]
    assert (caption.start, caption.end) == ("00:00:01.000", "00:00:03.000")
    assert caption.text == text


def test_stitch_keeps_repeated_text_and_overlapping_cues(tmp_path):
    output = stitch_layout(
        tmp_path,
        [("chunk_000.mp4", 0, 5), ("chunk_001.mp4", 5, 10)],
        {
            0: [{"start": "1", "end": "5.5", "text": "Again"}],
            1: [{"start": "0", "end": "1", "text": "Again"}],
        },
    )

    pipeline.stitch(tmp_path, output)

    result = webvtt.read(output)
    assert [caption.text for caption in result] == ["Again", "Again"]
    assert [(caption.start, caption.end) for caption in result] == [
        ("00:00:01.000", "00:00:05.500"),
        ("00:00:05.000", "00:00:06.000"),
    ]


def test_stitch_merges_exact_editorial_fragments_at_a_boundary(tmp_path):
    output = stitch_layout(
        tmp_path,
        [("chunk_000.mp4", 0, 2), ("chunk_001.mp4", 2, 4)],
        {
            0: [{"start": "1", "end": "2.1", "text": "[Chapter One]"}],
            1: [{"start": "0", "end": "1", "text": "[Chapter One]"}],
        },
    )

    pipeline.stitch(tmp_path, output)

    assert captions_of(output) == [("00:00:01.000", "00:00:03.000", "[Chapter One]")]


@pytest.mark.parametrize(
    ("earlier", "later"),
    [
        (("1", "2.1", "Host: Spoken words."), ("0", "1", "Host: Spoken words.")),
        (("1", "2.1", "[Chapter One]"), ("0", "1", "[Chapter Two]")),
    ],
    ids=["dialogue repeats across a boundary", "editorial text differs"],
)
def test_stitch_keeps_ineligible_boundary_cues_separate(tmp_path, earlier, later):
    output = stitch_layout(
        tmp_path,
        [("chunk_000.mp4", 0, 2), ("chunk_001.mp4", 2, 4)],
        {
            0: [{"start": earlier[0], "end": earlier[1], "text": earlier[2]}],
            1: [{"start": later[0], "end": later[1], "text": later[2]}],
        },
    )

    pipeline.stitch(tmp_path, output)

    assert [caption.text for caption in webvtt.read(output)] == [
        earlier[2],
        later[2],
    ]


def test_stitch_works_without_a_manifest_file(tmp_path):
    (tmp_path / "segments.csv").write_text("chunk_000.mp4,0,5\n", encoding="utf-8")
    write_chunk_subtitles(tmp_path, 0, [{"start": "1", "end": "2", "text": "Only"}])
    output = tmp_path / "output.vtt"

    result = pipeline.stitch(tmp_path, output)

    assert result == output
    assert [caption.text for caption in webvtt.read(output)] == ["Only"]
