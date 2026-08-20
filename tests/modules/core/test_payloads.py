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


@pytest.mark.parametrize(
    "text",
    [
        "[[Mission Rule] Total of 2 chances!]",
        "[[Narrow margin] By 14 points!]",
        "[First] [Second]",
    ],
)
def test_balanced_brackets_are_accepted(text):
    assert not core.has_unmatched_brackets(text)


@pytest.mark.parametrize("text", ["[text", "text]", "[a[b]c"])
def test_unbalanced_brackets_are_rejected(text):
    assert core.has_unmatched_brackets(text)


def test_nested_visual_fragments_preserve_outer_brackets_and_classification():
    editorial = "[[Narrow margin] By 14 points!]"
    mixed = "Chodan: Do we get two chances?\n[[Mission Rule] Total of 2 chances!]"

    assert core.visual_fragment_strings(editorial) == [editorial]
    assert core.classify_cue_text(editorial) == "editorial"
    assert core.visual_fragment_strings(mixed) == [
        "[[Mission Rule] Total of 2 chances!]"
    ]
    assert core.classify_cue_text(mixed) == "mixed"


def test_multiple_visual_fragments_are_extracted_separately():
    text = "[First] Host: Dialogue [Second]"

    assert core.visual_fragment_strings(text) == ["[First]", "[Second]"]
    assert core.classify_cue_text(text) == "mixed"


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


def test_audio_refinement_accepts_unchanged_nested_visual_fragments():
    sources = [
        {
            "id": 0,
            "start": "00:00:00.000",
            "end": "00:00:01.000",
            "text": "[[Narrow margin] By 14 points!]",
            "classification": "editorial",
        },
        {
            "id": 1,
            "start": "00:00:01.000",
            "end": "00:00:02.000",
            "text": "Host: Ready.\n[[Mission Rule] Total of 2 chances!]",
            "classification": "mixed",
        },
    ]

    cues = core.validate_audio_refinement_response(
        patch_response(), sources, 2.0, [1.0]
    )

    assert [cue["text"] for cue in cues] == [source["text"] for source in sources]


def test_audio_refinement_rejects_changed_nested_visual_fragment():
    sources = [
        {
            "id": 0,
            "start": "00:00:00.000",
            "end": "00:00:02.000",
            "text": "Host: Ready.\n[[Mission Rule] Total of 2 chances!]",
            "classification": "mixed",
        }
    ]
    patch = patch_response(
        cues=[
            patch_cue(
                [0],
                "00:00:00.000",
                "00:00:02.000",
                "Host: Ready.\n[[Mission Rule] Total of 1 chance!]",
            )
        ]
    )

    with pytest.raises(ValueError, match="Bracketed fragments of mixed cue"):
        core.validate_audio_refinement_response(patch, sources, 2.0, [1.0])


@pytest.mark.parametrize("text", ["Missing [close", "Missing open]"])
def test_audio_refinement_rejects_unbalanced_source_brackets(text):
    sources = source_entries()
    sources[0]["text"] = text

    with pytest.raises(ValueError, match="source cue contains unmatched brackets"):
        core.validate_audio_refinement_response(patch_response(), sources, 35.0, [10.0])


def test_audio_refinement_accepts_unchanged_cue_included_in_patch():
    sources = source_entries()
    # Boundary 10.0s keeps cue 0 (0-5s) inside repair region 0-20s, so the
    # identical echo is accepted instead of rejected.
    patch = patch_response(
        cues=[patch_cue([0], "00:00:00.000", "00:00:05.000", "Host: Welcome.")]
    )
    cues = core.validate_audio_refinement_response(patch, sources, 35.0, [10.0])
    assert [(c["start"], c["end"], c["text"]) for c in cues] == [
        (source["start"], source["end"], source["text"]) for source in sources
    ]


def test_audio_refinement_discards_edits_outside_repair_regions():
    sources = source_entries()
    # Cues 0 (0-5s) and 1 (5-10s) lie outside the repair region around
    # boundary 21.0s (region: 11-31s). Their rewrite and deletion are
    # discarded, so every source cue survives verbatim.
    patch = patch_response(
        cues=[patch_cue([0], "00:00:00.000", "00:00:04.000", "Host: Altered.")],
        deleted=[1],
    )
    cues = core.validate_audio_refinement_response(patch, sources, 35.0, [21.0])
    assert [(c["start"], c["end"], c["text"]) for c in cues] == [
        (source["start"], source["end"], source["text"]) for source in sources
    ]


def test_audio_refinement_discards_merge_referencing_outside_sources():
    sources = source_entries()
    # Cue 1 (5-10s) lies outside the repair region 11-31s around boundary
    # 21.0s, so the merge of cues 1 and 2 is discarded completely and both
    # cues survive verbatim.
    patch = patch_response(
        cues=[patch_cue([1, 2], "00:00:05.000", "00:00:15.000", "Host: Merged talk.")]
    )
    cues = core.validate_audio_refinement_response(patch, sources, 35.0, [21.0])
    assert [(c["start"], c["end"], c["text"]) for c in cues] == [
        (source["start"], source["end"], source["text"]) for source in sources
    ]


def test_audio_refinement_discards_recovered_cues_outside_repair_regions():
    sources = source_entries()
    # The recovered cue at 33-35s lies outside the repair region 11-31s
    # around boundary 21.0s, so it is discarded.
    patch = patch_response(
        cues=[patch_cue([], "00:00:33.000", "00:00:35.000", "Host: Recovered.")]
    )
    cues = core.validate_audio_refinement_response(patch, sources, 35.0, [21.0])
    assert [(c["start"], c["end"], c["text"]) for c in cues] == [
        (source["start"], source["end"], source["text"]) for source in sources
    ]


@pytest.mark.parametrize(
    ("patch", "match"),
    [
        (
            patch_response(
                cues=[patch_cue([99], "00:00:12.000", "00:00:14.000", "Text")]
            ),
            "references unknown source IDs",
        ),
        (patch_response(deleted=[99]), "deletes unknown source IDs"),
    ],
    ids=["unknown reference", "unknown deletion"],
)
def test_audio_refinement_rejects_unknown_source_ids_after_filtering(patch, match):
    sources = source_entries()
    with pytest.raises(ValueError, match=match):
        core.validate_audio_refinement_response(patch, sources, 35.0, [21.0])


def test_audio_refinement_accepts_retimed_cue_within_window_tolerance():
    sources = source_entries()
    # Cue 2 (9.5-15s) intersects the repair region 10-30s around boundary
    # 20.0s. Its strict envelope ends at 30s; the retimed cue 31-33s stays
    # inside the envelope expanded by the 10s repair window.
    patch = patch_response(
        cues=[patch_cue([2], "00:00:31.000", "00:00:33.000", "Guest: Hello.")]
    )
    cues = core.validate_audio_refinement_response(patch, sources, 35.0, [20.0])
    assert [(c["start"], c["end"], c["text"]) for c in cues] == [
        ("00:00:00.000", "00:00:05.000", "Host: Welcome."),
        ("00:00:05.000", "00:00:10.000", "Host: First topic."),
        ("00:00:19.500", "00:00:20.500", "[Title Card]"),
        ("00:00:25.000", "00:00:30.000", "Host: See [Chapter 1] now."),
        ("00:00:31.000", "00:00:33.000", "Guest: Hello."),
    ]


def test_audio_refinement_accepts_merge_skipping_intermediate_editorial_cue():
    sources = [
        {
            "id": 0,
            "start": "00:00:00.000",
            "end": "00:00:02.000",
            "text": "Host: Part 1.",
            "classification": "dialogue",
        },
        {
            "id": 1,
            "start": "00:00:01.500",
            "end": "00:00:03.000",
            "text": "[On-screen Card]",
            "classification": "editorial",
        },
        {
            "id": 2,
            "start": "00:00:02.000",
            "end": "00:00:04.000",
            "text": "Host: Part 2.",
            "classification": "dialogue",
        },
    ]
    patch = patch_response(
        cues=[patch_cue([0, 2], "00:00:00.000", "00:00:04.000", "Host: Part 1 and 2.")]
    )
    cues = core.validate_audio_refinement_response(patch, sources, 10.0, [2.0])
    assert any(c["text"] == "Host: Part 1 and 2." for c in cues)
    assert any(c["text"] == "[On-screen Card]" for c in cues)


def test_audio_refinement_rejects_merge_skipping_intermediate_dialogue_cue():
    sources = source_entries()
    # Cues 0 and 2 merge, but cue 1 is dialogue (not editorial).
    patch = patch_response(
        cues=[patch_cue([0, 2], "00:00:00.000", "00:00:15.000", "Host: Merged.")]
    )
    with pytest.raises(ValueError, match="merges source IDs that are not contiguous"):
        core.validate_audio_refinement_response(patch, sources, 35.0, [10.0])


def test_audio_refinement_rejects_retimed_cue_beyond_window_tolerance():
    sources = source_entries()
    # The strict envelope for cue 2 ends at 30s and the repair window adds
    # 10s, so a cue ending at 43s exceeds the tolerated envelope.
    patch = patch_response(
        cues=[patch_cue([2], "00:00:41.000", "00:00:43.000", "Guest: Hello.")]
    )
    with pytest.raises(ValueError, match="Changed cues must stay inside"):
        core.validate_audio_refinement_response(patch, sources, 60.0, [20.0])


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
