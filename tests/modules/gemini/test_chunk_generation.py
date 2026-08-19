"""Chunk generation through the Gemini adapter."""

import json

import pytest
from google.genai import types

from modules import core, gemini
from tests.support.gemini_fakes import ScriptedGeminiClient, chunk_call, use_client


def chunk_state(idx=0, clip_name="clip.mp4", duration=2.0):
    return {
        "idx": idx,
        "clip_name": clip_name,
        "clip_duration": duration,
        "owner_start_rel": 0.0,
        "owner_end_rel": duration,
    }


def test_chunk_request_streams_clip_bytes_and_publishes_canonical_captions(
    tmp_path, monkeypatch
):
    (tmp_path / "clip.mp4").write_bytes(b"video bytes")
    pieces = ['{"captions": [{"id": 0,', '"start": "0", "end": "1", "text": "Hi"}]}']
    client = ScriptedGeminiClient([chunk_call(pieces)])
    use_client(monkeypatch, client)

    assert gemini.process_chunk(
        "key",
        "base",
        chunk_state(idx=3),
        tmp_path,
        "chunk-model",
        "video/mp4",
        "high",
        "Show Title",
    )

    (request,) = client.requests
    assert request.model == "chunk-model"
    video_part = request.contents[0]
    assert video_part.inline_data.mime_type == "video/mp4"
    assert video_part.inline_data.data == b"video bytes"
    assert request.config.response_mime_type == "application/json"
    assert request.config.response_schema is core.SubtitleResponse
    assert request.config.automatic_function_calling.disable is True
    assert request.config.tools is None
    assert request.config.thinking_config.thinking_level == types.ThinkingLevel.HIGH
    assert request.config.temperature == 0.0

    saved = json.loads(
        (tmp_path / "subtitle_chunk_003.json").read_text(encoding="utf-8")
    )
    assert saved == [
        {
            "id": 0,
            "start": "00:00:00.000",
            "end": "00:00:01.000",
            "text": "Hi",
        }
    ]


def test_valid_caption_cache_preserves_the_published_result(tmp_path, monkeypatch):
    cache = tmp_path / "subtitle_chunk_000.json"
    cache.write_text(
        json.dumps(
            [
                {
                    "id": 0,
                    "start": "00:00:00.000",
                    "end": "00:00:01.000",
                    "text": "Cached",
                }
            ]
        ),
        encoding="utf-8",
    )
    original = cache.read_text(encoding="utf-8")
    (tmp_path / "clip.mp4").write_bytes(b"video")
    client = ScriptedGeminiClient(
        [
            chunk_call(
                ['{"captions": [{"id": 0, "start": "0", "end": "1", "text": "Fresh"}]}']
            )
        ]
    )
    use_client(monkeypatch, client)

    assert gemini.process_chunk(
        "key", None, chunk_state(), tmp_path, "model", "video/mp4", "high"
    )
    assert cache.read_text(encoding="utf-8") == original


def test_invalid_captions_cache_is_regenerated_via_api(tmp_path, monkeypatch):
    cache = tmp_path / "subtitle_chunk_000.json"
    cache.write_text("invalid", encoding="utf-8")
    (tmp_path / "clip.mp4").write_bytes(b"video")
    client = ScriptedGeminiClient(
        [
            chunk_call(
                ['{"captions": [{"id": 0, "start": "0", "end": "1", "text": "Fresh"}]}']
            )
        ]
    )
    use_client(monkeypatch, client)

    assert gemini.process_chunk(
        "key", None, chunk_state(), tmp_path, "model", "video/mp4", "high"
    )

    saved = json.loads(cache.read_text(encoding="utf-8"))
    assert [cap["text"] for cap in saved] == ["Fresh"]


@pytest.mark.parametrize(
    "pieces",
    [
        ["not json"],
        ['{"captions": [{"id": "not-an-int"}]}'],
        ['{"captions": [{"id": 0, "start": "0", "end": "0", "text": "bad"}]}'],
    ],
)
def test_invalid_chunk_response_fails_without_publishing(tmp_path, monkeypatch, pieces):
    (tmp_path / "clip.mp4").write_bytes(b"video")
    client = ScriptedGeminiClient([chunk_call(pieces)])
    use_client(monkeypatch, client)

    assert (
        gemini.process_chunk(
            "key", None, chunk_state(), tmp_path, "model", "video/mp4", "high"
        )
        is False
    )
    assert not (tmp_path / "subtitle_chunk_000.json").exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_chunk_sdk_failure_fails_without_publishing(tmp_path, monkeypatch):
    (tmp_path / "clip.mp4").write_bytes(b"video")
    client = ScriptedGeminiClient(
        [chunk_call([], error=RuntimeError("quota exhausted"))]
    )
    use_client(monkeypatch, client)

    assert (
        gemini.process_chunk(
            "key", None, chunk_state(), tmp_path, "model", "video/mp4", "high"
        )
        is False
    )
    assert not (tmp_path / "subtitle_chunk_000.json").exists()
    assert not list(tmp_path.glob("*.tmp"))
