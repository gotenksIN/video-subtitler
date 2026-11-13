"""Sparse audio-patch authority and reconstruction contracts."""

import pytest

from modules import core


def sources(*entries):
    return [
        {
            "id": index,
            "start": core.format_time(start),
            "end": core.format_time(end),
            "text": text,
            "classification": core.classify_cue_text(text),
        }
        for index, (start, end, text) in enumerate(entries)
    ]


def cue(ids, start, end, text):
    return core.AudioRefinedCue(
        sourceIds=ids,
        start=core.format_time(start),
        end=core.format_time(end),
        text=text,
    )


def validate(entries, cues=(), deleted=(), *, boundaries=(20,), audio=60, video=60):
    return core.validate_audio_refinement_response(
        core.AudioRefinementResponse(
            contractVersion="sparse-patch-v1",
            cues=list(cues),
            deletedSourceIds=list(deleted),
        ),
        entries,
        audio,
        list(boundaries),
        source_duration=video,
    )


def test_repairs_apply_inside_authority_and_discard_every_outside_operation():
    entries = sources(
        (0, 2, "Outside one"),
        (3, 4, "Outside two"),
        (12, 14, "Inside one"),
        (15, 17, "Inside two"),
    )
    for patch, deleted in (
        ([cue([0], 0, 1, "Altered")], []),
        ([cue([0, 2], 0, 14, "Merged")], []),
        ([], [1]),
        ([cue([], 40, 42, "Recovered")], []),
    ):
        result = validate(entries, patch, deleted)
        assert [item["text"] for item in result] == [
            "Outside one",
            "Outside two",
            "Inside one",
            "Inside two",
        ]
    result = validate(
        entries,
        [
            cue([2], 12, 14, "Repaired"),
            cue([], 18, 19, "Recovered"),
        ],
        [3],
    )
    assert [item["text"] for item in result] == [
        "Outside one",
        "Outside two",
        "Repaired",
        "Recovered",
    ]
    merged = validate(entries, [cue([2, 3], 12, 17, "Merged")])
    assert [item["text"] for item in merged] == ["Outside one", "Outside two", "Merged"]


def test_unknown_references_and_deletions_fail_even_outside_authority():
    entries = sources((0, 1, "Original"))
    for patch, deleted in (([cue([99], 0, 1, "Unknown")], []), ([], [99])):
        with pytest.raises(ValueError, match="unknown source IDs"):
            validate(entries, patch, deleted)


def test_lineage_rejects_reversal_duplicates_gaps_and_conflicting_deletion():
    entries = sources((12, 14, "One"), (15, 17, "Two"), (18, 20, "Three"))
    for patch, deleted in (
        ([cue([1], 15, 17, "Two"), cue([0], 12, 14, "One")], []),
        ([cue([0, 0], 12, 14, "One")], []),
        ([cue([0, 2], 12, 20, "Merged")], []),
        ([cue([0], 12, 14, "One")], [0]),
        ([cue([0], 12, 14, "One")], [1, 1]),
    ):
        with pytest.raises(ValueError):
            validate(entries, patch, deleted)


def test_pure_editorial_cues_preserve_exact_timing_text_and_lineage():
    entries = sources((12, 14, "[[Title] Detail]"))
    assert validate(entries)[0]["text"] == "[[Title] Detail]"
    for patch, deleted in (
        ([cue([0], 12, 14, "[Changed]")], []),
        ([cue([0], 12, 15, "[[Title] Detail]")], []),
        ([], [0]),
    ):
        with pytest.raises(ValueError):
            validate(entries, patch, deleted)


def test_mixed_cues_preserve_nested_multi_fragment_multisets_across_descendants():
    text = "Host: Original [[Title] Detail] [Repeat] [Repeat]"
    entries = sources((12, 16, text))
    result = validate(
        entries,
        [
            cue([0], 12, 14, "Host: Fixed [[Title] Detail] [Repeat]"),
            cue([0], 14, 16, "More [Repeat]"),
        ],
    )
    assert [item["text"] for item in result] == [
        "Host: Fixed [[Title] Detail] [Repeat]",
        "More [Repeat]",
    ]
    for replacement in (
        "Host: Fixed [[Title] Detail] [Repeat]",
        "Host: Fixed [[Title] Changed] [Repeat] [Repeat]",
        "Host: Fixed [[Title] Detail] [Repeat] [Repeat] [Repeat]",
        "Host: Fixed",
    ):
        with pytest.raises(ValueError):
            validate(entries, [cue([0], 12, 16, replacement)])


def test_merges_and_splits_skip_unchanged_editorial_but_not_dialogue():
    entries = sources((0, 2, "Part one"), (1.5, 3, "[Card]"), (2, 4, "Part two"))
    result = validate(entries, [cue([0, 2], 0, 4, "Merged")], boundaries=(2,))
    assert [item["text"] for item in result] == ["Merged", "[Card]"]
    entries = sources((0, 4, "Whole turn"), (1.5, 2.5, "[Card]"))
    result = validate(
        entries,
        [
            cue([0], 0, 1.5, "First part"),
            cue([0], 2.5, 4, "Second part"),
        ],
        boundaries=(2,),
    )
    assert [item["text"] for item in result] == ["First part", "[Card]", "Second part"]
    entries[1]["text"] = "Other speech"
    entries[1]["classification"] = "dialogue"
    with pytest.raises(ValueError):
        validate(
            entries,
            [
                cue([0], 0, 1.5, "First part"),
                cue([0], 2.5, 4, "Second part"),
            ],
            boundaries=(2,),
        )


def test_crossed_retimes_sort_and_reject_new_but_allow_existing_overlaps():
    entries = sources((5, 6, "First"), (7, 8, "Second"))
    result = validate(
        entries,
        [
            cue([0], 8, 9, "First"),
            cue([1], 7, 8, "Second"),
        ],
        boundaries=(10,),
    )
    assert [(item["source_ids"], item["start"]) for item in result] == [
        ([1], "00:00:07.000"),
        ([0], "00:00:08.000"),
    ]
    with pytest.raises(ValueError, match="new timestamp overlap"):
        validate(
            entries,
            [
                cue([0], 8, 9, "First"),
                cue([1], 7, 8.5, "Second"),
            ],
            boundaries=(10,),
        )
    entries = sources((0, 6, "First"), (5, 10, "Second"))
    result = validate(entries, [cue([1], 4.5, 9, "Second")], boundaries=(10,))
    assert result[1]["start"] == "00:00:04.500"


def test_changed_and_recovered_cues_obey_expanded_repair_envelopes():
    entries = sources((12, 14, "Original"))
    result = validate(entries, [cue([0], 31, 33, "Retimed")])
    assert result[0]["start"] == "00:00:31.000"
    with pytest.raises(ValueError, match="Changed cues must stay inside"):
        validate(entries, [cue([0], 41, 43, "Retimed")])
    assert validate([], [cue([], 29, 39, "Recovered")])[0]["end"] == "00:00:39.000"
    with pytest.raises(ValueError, match="Recovered cues must stay inside"):
        validate([], [cue([], 29, 41, "Recovered")])
    distant = sources((12, 14, "First"), (48, 49, "Second"))
    with pytest.raises(ValueError, match="shared repair region"):
        validate(distant, [cue([0, 1], 12, 49, "Merged")], boundaries=(20, 50))


def test_unchanged_visual_cues_keep_video_end_when_audio_ends_earlier():
    for text in ("[End card]", "Host: Goodbye [End card]"):
        entries = sources((9, 10, text))
        for patch in ([], [cue([0], 9, 10, text)], [cue([0], 9, 10, text + "\n")]):
            assert validate(entries, patch, boundaries=(5,), audio=8, video=10) == [
                {
                    "source_ids": [0],
                    "start": "00:00:09.000",
                    "end": "00:00:10.000",
                    "text": text,
                },
            ]
        with pytest.raises(ValueError, match="source video duration"):
            validate(entries, boundaries=(5,), audio=8, video=9.9)


def test_untouched_outside_dialogue_keeps_only_half_second_audio_grace():
    for end in (30.001, 30.5, 30.501):
        entries = sources((29, end, "  Host: Goodbye.  "))
        if end > 30.5:
            with pytest.raises(ValueError, match="complete audio duration"):
                validate(entries, boundaries=(5,), audio=30, video=31)
        else:
            result = validate(entries, boundaries=(5,), audio=30, video=31)
            assert result[0]["end"] == core.format_time(end)
            assert result[0]["text"] == "  Host: Goodbye.  "


def test_changed_or_recovered_cues_clamp_grace_but_cannot_use_video_tail():
    entries = sources((28, 29, "Original"))
    for lineage in ([], [0]):
        result = validate(
            entries,
            [cue(lineage, 29, 30.5, "Changed")],
            boundaries=(25,),
            audio=30,
            video=32,
        )
        assert result[-1]["end"] == "00:00:30.000"
        with pytest.raises(ValueError, match="complete audio duration"):
            validate(
                entries,
                [cue(lineage, 29, 31, "Changed")],
                boundaries=(25,),
                audio=30,
                video=32,
            )


def test_recovered_speech_and_brackets_must_be_well_formed():
    assert (
        validate([], [cue([], 12, 14, "well, she asked:")])[0]["text"]
        == "well, she asked:"
    )
    for text in ("Host:", "[Visual] Talk", "Missing [bracket", "Bad]"):
        with pytest.raises(ValueError):
            validate([], [cue([], 12, 14, text)])
    for text in ("Hello\n\nworld", "Hello\n \nworld", "Hello\rworld", "Hello\n\n"):
        for lineage in ([], [0]):
            with pytest.raises(ValueError):
                validate(sources((12, 14, "Original")), [cue(lineage, 12, 14, text)])
    for lineage in ([], [0]):
        result = validate(
            sources((12, 14, "Original")),
            [cue(lineage, 15, 16, "Hello\nworld\n")],
        )
        assert result[-1]["text"] == "Hello\nworld"
    whitespace = "  Host: Hello\n \nworld\r  "
    entries = sources((12, 14, whitespace))
    for patch in ([], [cue([0], 12, 14, whitespace)]):
        assert validate(entries, patch)[0]["text"] == whitespace
    entries = sources((12, 14, "Bad [bracket"))
    with pytest.raises(ValueError, match="source cue contains unmatched brackets"):
        validate(entries)
