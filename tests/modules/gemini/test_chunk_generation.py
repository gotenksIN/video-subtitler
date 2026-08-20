"""Chunk generation through the Gemini adapter."""

import json

from google.genai import errors, types

from modules import core, gemini
from tests.support.gemini_fakes import ScriptedGeminiClient, chunk_call, use_client


def chunk_state(idx=0, name="chunk_000.mp4", duration=2.0):
    return {
        "idx": idx,
        "name": name,
        "duration": duration,
    }


def valid_captions_call(text="Hi"):
    return chunk_call(
        [json.dumps({"captions": [{"id": 0, "start": "0", "end": "1", "text": text}]})]
    )


def test_chunk_request_streams_chunk_bytes_and_publishes_canonical_captions(
    tmp_path, monkeypatch
):
    chunk = {"idx": 3, "name": "chunk_003.mp4", "duration": 2.0}
    (tmp_path / chunk["name"]).write_bytes(b"video bytes")
    pieces = ['{"captions": [{"id": 0,', '"start": "0", "end": "1", "text": "Hi"}]}']
    client = ScriptedGeminiClient([chunk_call(pieces)])
    use_client(monkeypatch, client)

    assert gemini.process_chunk(
        "key",
        "base",
        chunk,
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
    (tmp_path / "chunk_000.mp4").write_bytes(b"video")
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
    (tmp_path / "chunk_000.mp4").write_bytes(b"video")
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


def test_transient_chunk_failure_retries_and_publishes_valid_result(
    tmp_path, monkeypatch
):
    (tmp_path / "chunk_000.mp4").write_bytes(b"video")
    invalid_timing = chunk_call(
        ['{"captions": [{"id": 0, "start": "1", "end": "0", "text": "bad"}]}']
    )
    client = ScriptedGeminiClient([invalid_timing, valid_captions_call("Hi")])
    use_client(monkeypatch, client)
    monkeypatch.setattr(gemini.time, "sleep", lambda _seconds: None)

    assert gemini.process_chunk(
        "key", None, chunk_state(), tmp_path, "model", "video/mp4", "high"
    )

    saved = json.loads(
        (tmp_path / "subtitle_chunk_000.json").read_text(encoding="utf-8")
    )
    assert saved == [
        {
            "id": 0,
            "start": "00:00:00.000",
            "end": "00:00:01.000",
            "text": "Hi",
        }
    ]
    assert len(client.requests) == 2


def test_repeated_transient_chunk_failures_exhaust_retries_without_publishing(
    tmp_path, monkeypatch
):
    (tmp_path / "chunk_000.mp4").write_bytes(b"video")
    invalid_timing = chunk_call(
        ['{"captions": [{"id": 0, "start": "1", "end": "0", "text": "bad"}]}']
    )
    client = ScriptedGeminiClient([invalid_timing] * 3)
    use_client(monkeypatch, client)
    monkeypatch.setattr(gemini.time, "sleep", lambda _seconds: None)

    assert (
        gemini.process_chunk(
            "key", None, chunk_state(), tmp_path, "model", "video/mp4", "high"
        )
        is False
    )
    assert not (tmp_path / "subtitle_chunk_000.json").exists()
    assert not list(tmp_path.glob("*.tmp"))
    assert len(client.requests) == 3


def test_permanent_chunk_failure_fails_immediately_without_retry(tmp_path, monkeypatch):
    (tmp_path / "chunk_000.mp4").write_bytes(b"video")
    client = ScriptedGeminiClient(
        [
            chunk_call(
                [],
                error=errors.ClientError(
                    403,
                    {
                        "error": {
                            "code": 403,
                            "message": "Permission denied.",
                            "status": "PERMISSION_DENIED",
                        }
                    },
                ),
            )
        ]
        * 3
    )
    use_client(monkeypatch, client)
    monkeypatch.setattr(gemini.time, "sleep", lambda _seconds: None)

    assert (
        gemini.process_chunk(
            "key", None, chunk_state(), tmp_path, "model", "video/mp4", "high"
        )
        is False
    )
    assert not (tmp_path / "subtitle_chunk_000.json").exists()
    assert not list(tmp_path.glob("*.tmp"))
    assert len(client.requests) == 1
