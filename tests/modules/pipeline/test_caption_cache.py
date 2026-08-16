"""Persisted caption cache behavior."""

import json

import pytest

import gemini_subs


def test_cached_captions_load_with_canonical_validation(tmp_path):
    cache = tmp_path / "subtitle_chunk_000.json"
    cache.write_text(
        json.dumps([{"id": 0, "start": "1,5", "end": "2", "text": "Hi"}]),
        encoding="utf-8",
    )

    result = gemini_subs.load_cached_captions(cache, 5)

    assert result == [
        {"id": 0, "start": "00:00:01.500", "end": "00:00:02.000", "text": "Hi"}
    ]


def test_missing_cached_captions_return_none(tmp_path):
    assert gemini_subs.load_cached_captions(tmp_path / "missing.json", 5) is None


@pytest.mark.parametrize(
    "contents",
    [
        "not valid json",
        json.dumps({"captions": "wrong shape"}),
        json.dumps([{"id": 0, "start": "9", "end": "10.6", "text": "Too long"}]),
        json.dumps(
            [
                {"id": 0, "start": "0", "end": "1", "text": "A"},
                {"id": 0, "start": "2", "end": "3", "text": "B"},
            ]
        ),
    ],
    ids=["invalid json", "wrong shape", "invalid timing", "duplicate ids"],
)
def test_invalid_cached_captions_are_deleted_for_regeneration(tmp_path, contents):
    cache = tmp_path / "subtitle_chunk_000.json"
    cache.write_text(contents, encoding="utf-8")

    assert gemini_subs.load_cached_captions(cache, 10) is None
    assert not cache.exists()
