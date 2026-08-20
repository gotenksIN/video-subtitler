"""Caption, refinement, and sparse audio patch payload validation."""

import pytest
import webvtt
from pydantic import ValidationError

from modules import core


def caption(caption_id, start, end, text="Text"):
    return core.Caption(id=caption_id, start=start, end=end, text=text)


def patch_cue(source_ids, start, end, text):
    return core.AudioRefinedCue(sourceIds=source_ids, start=start, end=end, text=text)


def patch_response(cues=(), deleted=(), version="sparse-patch-v1"):
    return core.AudioRefinementResponse(
        contractVersion=version,
        deletedSourceIds=list(deleted),
        cues=list(cues),
    )


def source_entries():
    return [
        {
            "id": 0,
            "start": "00:00:00.000",
            "end": "00:00:05.000",
            "text": "Host: Welcome.",
            "classification": "dialogue",
        },
        {
            "id": 1,
            "start": "00:00:05.000",
            "end": "00:00:10.000",
            "text": "Host: First topic.",
            "classification": "dialogue",
        },
        {
            "id": 2,
            "start": "00:00:09.500",
            "end": "00:00:15.000",
            "text": "Guest: Hello.",
            "classification": "dialogue",
        },
        {
            "id": 3,
            "start": "00:00:19.500",
            "end": "00:00:20.500",
            "text": "[Title Card]",
            "classification": "editorial",
        },
        {
            "id": 4,
            "start": "00:00:25.000",
            "end": "00:00:30.000",
            "text": "Host: See [Chapter 1] now.",
            "classification": "mixed",
        },
    ]


# --- Subtitle & Text Refinement Response Models ---


def test_subtitle_response_accepts_documented_caption_shape():
    response = core.SubtitleResponse.model_validate(
        {"captions": [{"id": 0, "start": "0", "end": "1", "text": "Hi"}]}
    )
    assert response.captions[0].id == 0
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


# --- Chunk Caption Validation ---


def test_caption_validation_sorts_and_canonicalizes_timestamps():
    result = core.validate_captions(
        [caption(2, "1", "3", "Later"), caption(1, "00:00:00,250", "2", "Earlier")],
        5,
    )
    assert result == [
        {"id": 1, "start": "00:00:00.250", "end": "00:00:02.000", "text": "Earlier"},
        {"id": 2, "start": "00:00:01.000", "end": "00:00:03.000", "text": "Later"},
    ]


def test_caption_validation_preserves_concurrent_and_multiline_cues():
    text = "Host: One line\n[On-screen card]\nGuest: Another line"
    result = core.validate_captions(
        [caption(1, "1", "2", "Shorter"), caption(0, "1", "3", text)], 5
    )
    assert [item["id"] for item in result] == [0, 1]
    assert result[0]["text"] == text


def test_caption_validation_rejects_duplicate_ids():
    with pytest.raises(ValueError, match="Duplicate caption IDs"):
        core.validate_captions([caption(0, "0", "1"), caption(0, "2", "3")], 10)


@pytest.mark.parametrize(
    ("start", "end"),
    [("1", "1"), ("2", "1"), ("0", "0"), ("-0.5", "1")],
    ids=["equal endpoints", "reversed interval", "zero interval", "negative start"],
)
def test_caption_validation_rejects_invalid_intervals(start, end):
    with pytest.raises(ValueError):
        core.validate_captions([caption(0, start, end)], 10)


def test_caption_validation_clamps_end_overrun_within_tolerance():
    result = core.validate_captions([caption(0, "9", "10.4")], 10)
    assert result[0]["start"] == "00:00:09.000"
    assert result[0]["end"] == "00:00:10.000"


@pytest.mark.parametrize(
    ("start", "end"),
    [("9", "10.6"), ("10.2", "10.4"), ("9.9996", "10.4")],
    ids=["beyond tolerance", "clamp collapses interval", "rounding collapses interval"],
)
def test_caption_validation_rejects_unrecoverable_end_overruns(start, end):
    with pytest.raises(ValueError):
        core.validate_captions([caption(0, start, end)], 10)


def test_caption_validation_accepts_empty_caption_list():
    assert core.validate_captions([], 10) == []


# --- Refinement Change Validation ---


def test_refinement_change_validation_accepts_valid_changes():
    changes = [
        core.RefinedCaption(id=0, text="One"),
        core.RefinedCaption(id=2, text="Two"),
    ]
    core.validate_refinement_changes(changes, 3)


@pytest.mark.parametrize(
    "changes",
    [
        [core.RefinedCaption(id=0, text="One"), core.RefinedCaption(id=0, text="Two")],
        [core.RefinedCaption(id=2, text="Out of range")],
        [core.RefinedCaption(id=-1, text="Negative")],
        [core.RefinedCaption(id=0, text="   ")],
    ],
    ids=["duplicate", "out of range", "negative", "empty text"],
)
def test_refinement_change_validation_rejects_invalid_changes(changes):
    with pytest.raises(ValueError):
        core.validate_refinement_changes(changes, 2)


# --- Sparse Audio Refinement Response & Authority ---


def test_audio_response_parses_wire_aliases_and_forbids_extra_fields():
    payload = {
        "contractVersion": "sparse-patch-v1",
        "deletedSourceIds": [1],
        "cues": [
            {
                "sourceIds": [0],
                "start": "00:00:00.000",
                "end": "00:00:04.000",
                "text": "Rewritten",
            }
        ],
    }
    response = core.AudioRefinementResponse.model_validate(payload)
    assert response.contract_version == "sparse-patch-v1"
    assert response.deleted_source_ids == [1]
    assert response.cues[0].source_ids == [0]

    with pytest.raises(ValidationError):
        core.AudioRefinementResponse.model_validate({**payload, "extraField": "bad"})


def test_sparse_audio_refinement_applies_edits_and_preserves_omitted_cues():
    sources = source_entries()
    # Edit cue 1 (rewrite), delete cue 2, recover a new cue at 00:00:07.000
    patch = patch_response(
        cues=[
            patch_cue([1], "00:00:05.000", "00:00:09.000", "Host: Refined topic."),
            patch_cue([], "00:00:07.000", "00:00:09.000", "Host: Recovered remark."),
        ],
        deleted=[2],
    )
    cues = core.validate_audio_refinement_response(patch, sources, 35.0, [10.0, 20.0])

    texts = [c["text"] for c in cues]
    assert "Host: Welcome." in texts  # Omitted cue 0 preserved identically
    assert "Host: Refined topic." in texts
    assert "Host: Recovered remark." in texts
    assert "Guest: Hello." not in texts  # Deleted cue 2 removed
    assert "[Title Card]" in texts  # Editorial cue 3 preserved
    assert "Host: See [Chapter 1] now." in texts  # Mixed cue 4 preserved


def test_audio_refinement_rejects_unchanged_cue_included_in_patch():
    sources = source_entries()
    patch = patch_response(
        cues=[patch_cue([0], "00:00:00.000", "00:00:05.000", "Host: Welcome.")]
    )
    with pytest.raises(ValueError, match="must omit unchanged source cue"):
        core.validate_audio_refinement_response(patch, sources, 35.0, [10.0])


def test_audio_refinement_enforces_boundary_authority_in_boundary_mode():
    sources = source_entries()
    # Cue 0 (0-5s) is outside repair region around boundary 10.0s (region: 5-15s). Modifying it must fail.
    patch = patch_response(
        cues=[patch_cue([0], "00:00:00.000", "00:00:04.000", "Host: Altered.")]
    )
    with pytest.raises(ValueError, match="lies outside every repair region"):
        core.validate_audio_refinement_response(patch, sources, 35.0, [10.0])


def test_audio_refinement_enforces_visual_preservation():
    sources = source_entries()
    # Cannot alter pure editorial cue text
    patch_editorial = patch_response(
        cues=[patch_cue([3], "00:00:19.500", "00:00:20.500", "[Altered Card]")],
    )
    with pytest.raises(ValueError, match="Pure editorial cue 3 must be preserved"):
        core.validate_audio_refinement_response(patch_editorial, sources, 35.0, [20.0])

    # Cannot drop bracketed fragments from mixed cue
    patch_mixed = patch_response(
        cues=[
            patch_cue([4], "00:00:25.000", "00:00:30.000", "Host: See Chapter 1 now.")
        ],
    )
    with pytest.raises(
        ValueError, match="Bracketed fragments of mixed cue 4 must be preserved"
    ):
        core.validate_audio_refinement_response(patch_mixed, sources, 35.0, [20.0])


@pytest.mark.parametrize(
    ("patch", "match"),
    [
        (patch_response(deleted=[3]), "deletes non-dialogue cue"),
        (
            patch_response(
                deleted=[1],
                cues=[patch_cue([1], "00:00:05.000", "00:00:08.000", "Text")],
            ),
            "both references and deletes",
        ),
        (
            patch_response(
                cues=[
                    patch_cue(
                        [], "00:00:07.000", "00:00:09.000", "Host: [Visual] talk."
                    )
                ]
            ),
            "Recovered cues must not contain bracketed text",
        ),
        (
            patch_response(
                cues=[patch_cue([], "00:00:07.000", "00:00:09.000", "Host:")]
            ),
            "must contain spoken dialogue",
        ),
        (
            patch_response(
                cues=[
                    patch_cue(
                        [1], "00:00:05.000", "00:00:08.000", "Host: Bad [ bracket"
                    )
                ]
            ),
            "unmatched brackets",
        ),
    ],
    ids=[
        "delete editorial",
        "reference and delete",
        "brackets in recovered",
        "recovered without dialogue",
        "unmatched bracket",
    ],
)
def test_audio_refinement_rejects_invalid_lineage_and_text(patch, match):
    sources = source_entries()
    with pytest.raises(ValueError, match=match):
        core.validate_audio_refinement_response(patch, sources, 35.0, [10.0, 20.0])


# --- Pure-Editorial Boundary Merging ---


def test_pure_editorial_boundary_merging():
    entries = [
        {"start": 18.0, "end": 20.1, "text": "[Visual Banner]", "chunk_idx": 0},
        {
            "start": 19.5,
            "end": 20.5,
            "text": "Host: Overlapping speech.",
            "chunk_idx": 0,
        },
        {"start": 19.9, "end": 22.0, "text": "[Visual Banner]", "chunk_idx": 1},
        {"start": 21.9, "end": 24.0, "text": "[Visual Banner]", "chunk_idx": 2},
    ]
    boundary_starts = [20.0, 22.0]
    merged = core.merge_visual_boundary_fragments(entries, boundary_starts)

    assert len(merged) == 2
    banner = next(e for e in merged if e["text"] == "[Visual Banner]")
    assert banner["start"] == 18.0
    assert banner["end"] == 24.0
    assert banner["chunk_idx"] == 2


# --- Speaker Label Casing Canonicalization ---


def speaker_vtt(*texts):
    vtt = webvtt.WebVTT()
    for index, text in enumerate(texts):
        vtt.captions.append(
            webvtt.Caption(f"00:00:{index:02d}.000", f"00:00:{index + 1:02d}.000", text)
        )
    return vtt


def test_speaker_casing_normalizes_drift_to_dominant_frequency():
    captions = speaker_vtt(
        "Haewon: One",
        "HAEWON: Two",
        "Haewon: Three",
        "Haewon: Four",
    ).captions

    core.canonicalize_speaker_casing(captions)

    assert [caption.text for caption in captions] == [
        "Haewon: One",
        "Haewon: Two",
        "Haewon: Three",
        "Haewon: Four",
    ]


def test_grounded_name_overrides_dominant_frequency():
    vtt = speaker_vtt("Bae: One", "Bae: Two", "BAE: Three")

    core.canonicalize_speaker_casing(vtt, grounded_names=["BAE"])

    assert [caption.text for caption in vtt.captions] == [
        "BAE: One",
        "BAE: Two",
        "BAE: Three",
    ]


def test_frequency_tie_keeps_first_seen_spelling():
    vtt = speaker_vtt("Sana: One", "SANA: Two")
    core.canonicalize_speaker_casing(vtt)
    assert [caption.text for caption in vtt.captions] == [
        "Sana: One",
        "Sana: Two",
    ]

    vtt = speaker_vtt("SANA: One", "Sana: Two")
    core.canonicalize_speaker_casing(vtt)
    assert [caption.text for caption in vtt.captions] == [
        "SANA: One",
        "SANA: Two",
    ]


def test_speaker_casing_preserves_non_label_structure():
    vtt = speaker_vtt(
        "[Opening title]",
        "Unlabeled dialogue line",
        "Haewon: Named turn\nHAEWON: Second turn",
    )

    result = core.canonicalize_speaker_casing(vtt)

    assert [caption.text for caption in result.captions] == [
        "[Opening title]",
        "Unlabeled dialogue line",
        "Haewon: Named turn\nHaewon: Second turn",
    ]
    assert [(caption.start, caption.end) for caption in result.captions] == [
        ("00:00:00.000", "00:00:01.000"),
        ("00:00:01.000", "00:00:02.000"),
        ("00:00:02.000", "00:00:03.000"),
    ]
