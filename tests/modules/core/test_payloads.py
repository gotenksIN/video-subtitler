"""Caption and refinement payload validation."""

import pytest
from pydantic import ValidationError

from modules import core


def caption(caption_id, start, end, text="Text"):
    return core.Caption(id=caption_id, start=start, end=end, text=text)


def test_subtitle_response_accepts_documented_caption_shape():
    response = core.SubtitleResponse.model_validate(
        {"captions": [{"id": 0, "start": "0", "end": "1", "text": "Hi"}]}
    )

    assert response.captions[0].id == 0
    assert response.captions[0].start == "0"
    assert response.captions[0].end == "1"
    assert response.captions[0].text == "Hi"


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"captions": "not-a-list"},
        {"captions": [{"start": "0", "end": "1", "text": "Hi"}]},
        {"captions": [{"id": "not-an-int", "start": "0", "end": "1", "text": "Hi"}]},
    ],
    ids=["missing captions", "captions not a list", "missing id", "wrong id type"],
)
def test_subtitle_response_rejects_malformed_payloads(payload):
    with pytest.raises(ValidationError):
        core.SubtitleResponse.model_validate(payload)


def test_refinement_response_accepts_documented_change_shape():
    response = core.RefinementResponse.model_validate(
        {"changes": [{"id": 0, "text": "Fixed"}]}
    )

    assert response.changes[0].id == 0
    assert response.changes[0].text == "Fixed"


def test_caption_validation_sorts_and_canonicalizes_timestamps():
    result = core.validate_captions(
        [caption(2, "1", "3", "Later"), caption(1, "00:00:00,250", "2", "Earlier")],
        5,
    )

    assert result == [
        {"id": 1, "start": "00:00:00.250", "end": "00:00:02.000", "text": "Earlier"},
        {"id": 2, "start": "00:00:01.000", "end": "00:00:03.000", "text": "Later"},
    ]


def test_caption_validation_preserves_cues_that_start_at_the_same_time():
    result = core.validate_captions(
        [caption(1, "1", "2", "Shorter"), caption(0, "1", "3", "Longer")], 5
    )

    assert [item["id"] for item in result] == [0, 1]
    assert [item["end"] for item in result] == ["00:00:03.000", "00:00:02.000"]


def test_caption_validation_preserves_multiline_text_verbatim():
    text = "Host: One line\n[On-screen card]\nGuest: Another line"

    result = core.validate_captions([caption(0, "1", "2", text)], 5)

    assert result[0]["text"] == text


def test_caption_validation_rejects_duplicate_ids():
    with pytest.raises(ValueError, match="Duplicate caption IDs"):
        core.validate_captions([caption(0, "0", "1"), caption(0, "2", "3")], 10)


@pytest.mark.parametrize(
    ("start", "end"),
    [("1", "1"), ("2", "1"), ("0", "0")],
    ids=["equal endpoints", "reversed interval", "zero interval"],
)
def test_caption_validation_rejects_non_positive_intervals(start, end):
    with pytest.raises(ValueError, match="Invalid caption timing"):
        core.validate_captions([caption(0, start, end)], 10)


def test_caption_validation_rejects_negative_start_times():
    with pytest.raises(ValueError, match="Negative timestamp"):
        core.validate_captions([caption(0, "-0.5", "1")], 10)


def test_caption_validation_clamps_end_overrun_within_tolerance():
    result = core.validate_captions([caption(0, "9", "10.4")], 10)

    assert result[0]["start"] == "00:00:09.000"
    assert result[0]["end"] == "00:00:10.000"


def test_caption_validation_rejects_end_overrun_beyond_tolerance():
    with pytest.raises(ValueError, match="exceeds chunk duration"):
        core.validate_captions([caption(0, "9", "10.6")], 10)


def test_caption_validation_rejects_clamp_that_invalidates_interval():
    with pytest.raises(ValueError, match="exceeds chunk duration"):
        core.validate_captions([caption(0, "10.2", "10.4")], 10)


def test_caption_validation_rejects_intervals_collapsed_by_rounding():
    with pytest.raises(ValueError, match="non-positive interval"):
        core.validate_captions([caption(0, "9.9996", "10.4")], 10)


def test_caption_validation_accepts_empty_caption_list():
    assert core.validate_captions([], 10) == []


def test_refinement_change_validation_accepts_valid_changes():
    changes = [
        core.RefinedCaption(id=0, text="One"),
        core.RefinedCaption(id=2, text="Two"),
    ]

    core.validate_refinement_changes(changes, 3)


@pytest.mark.parametrize(
    "changes",
    [
        [
            core.RefinedCaption(id=0, text="One"),
            core.RefinedCaption(id=0, text="Two"),
        ],
        [core.RefinedCaption(id=2, text="Out of range")],
        [core.RefinedCaption(id=-1, text="Negative")],
        [core.RefinedCaption(id=0, text="   ")],
    ],
    ids=["duplicate", "out of range", "negative", "empty text"],
)
def test_refinement_change_validation_rejects_invalid_changes(changes):
    with pytest.raises(ValueError):
        core.validate_refinement_changes(changes, 2)
