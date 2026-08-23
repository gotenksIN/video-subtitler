"""Boundary audio refinement through the Gemini adapter."""

import json

import pytest
from google.genai import types

from modules import gemini
from tests.support.gemini_fakes import (
    ScriptedGeminiClient,
    audio_call,
    read_captions,
    stream_chunk,
    use_client,
    write_vtt,
)

BOUNDARY = 30.0
AUDIO_DURATION = 60.0

STITCHED_CAPTIONS = [
    ("00:00:04.000", "00:00:06.000", "Host: Hello there."),
    ("00:00:28.000", "00:00:31.000", "Host: Boundary talk."),
    ("00:00:34.000", "00:00:36.000", "Host: See [Chapter One] now."),
    ("00:00:50.000", "00:00:52.000", "[Applause]"),
]


def boundary_patch():
    return {
        "contractVersion": "sparse-patch-v1",
        "deletedSourceIds": [],
        "cues": [
            {
                "sourceIds": [1],
                "start": "00:00:28.000",
                "end": "00:00:31.500",
                "text": "Host: Boundary talk now.",
            }
        ],
    }


def refined_captions():
    return [
        STITCHED_CAPTIONS[0],
        ("00:00:28.000", "00:00:31.500", "Host: Boundary talk now."),
        STITCHED_CAPTIONS[2],
        STITCHED_CAPTIONS[3],
    ]


def write_scenario(tmp_path):
    stitched = write_vtt(tmp_path / "stitched.vtt", STITCHED_CAPTIONS)
    audio = tmp_path / "extracted_audio.ogg"
    audio.write_bytes(b"ogg audio bytes")
    output = tmp_path / "output.vtt"
    return stitched, audio, output


def run_boundary_refine(monkeypatch, tmp_path, calls=(), model_name="audio-model"):
    stitched, audio, output = write_scenario(tmp_path)
    client = ScriptedGeminiClient(list(calls))
    use_client(monkeypatch, client)
    gemini.boundary_audio_refine_subtitles(
        stitched,
        audio,
        AUDIO_DURATION,
        [BOUNDARY],
        tmp_path,
        output,
        "key",
        None,
        model_name,
        source_title="Show Title",
    )
    return output, client


def test_audio_request_sends_inline_ogg_bytes_with_sparse_config(tmp_path, monkeypatch):
    payload = json.dumps(boundary_patch())
    output, client = run_boundary_refine(monkeypatch, tmp_path, [audio_call([payload])])

    (request,) = client.requests
    audio_part = request.contents[0]
    assert audio_part.inline_data.mime_type == "audio/ogg"
    assert audio_part.inline_data.data == b"ogg audio bytes"
    assert isinstance(request.contents[1], str)
    assert request.config.automatic_function_calling.disable is True
    assert request.config.thinking_config.thinking_level == types.ThinkingLevel.HIGH
    assert request.config.thinking_config.include_thoughts is True
    assert request.config.max_output_tokens == gemini.AUDIO_REFINE_MAX_OUTPUT_TOKENS
    assert read_captions(output) == refined_captions()


def test_boundary_refinement_publishes_candidate_and_reuses_cache(
    tmp_path, monkeypatch
):
    output, _client1 = run_boundary_refine(
        monkeypatch, tmp_path, [audio_call([json.dumps(boundary_patch())])]
    )
    assert read_captions(output) == refined_captions()
    assert (tmp_path / "audio_refinement.json").is_file()

    # Second run with empty client reuses cached response without API calls
    output2, client2 = run_boundary_refine(monkeypatch, tmp_path)
    assert client2.requests == []
    assert read_captions(output2) == refined_captions()


def test_cached_audio_refinement_regenerates_on_mismatch_or_corruption(
    tmp_path, monkeypatch
):
    run_boundary_refine(
        monkeypatch, tmp_path, [audio_call([json.dumps(boundary_patch())])]
    )
    # Model mismatch causes cache invalidation and a new request
    output, client = run_boundary_refine(
        monkeypatch,
        tmp_path,
        [audio_call([json.dumps(boundary_patch())])],
        model_name="other-model",
    )
    assert len(client.requests) == 1
    assert client.requests[0].model == "other-model"
    assert read_captions(output) == refined_captions()


def test_out_of_authority_patch_edits_are_discarded(tmp_path, monkeypatch):
    patch = {
        "contractVersion": "sparse-patch-v1",
        "deletedSourceIds": [],
        "cues": [
            {
                "sourceIds": [0],
                "start": "00:00:04.000",
                "end": "00:00:06.500",
                "text": "Outside authority",
            }
        ],
    }
    output, _client = run_boundary_refine(
        monkeypatch, tmp_path, [audio_call([json.dumps(patch)])]
    )

    # Cue 0 (4-6s) lies outside the repair region 20-40s around boundary
    # 30.0s, so the edit is discarded and the stitched cue survives.
    assert read_captions(output) == STITCHED_CAPTIONS


@pytest.mark.parametrize(
    "call",
    [
        audio_call([stream_chunk(finish_reason=types.FinishReason.MAX_TOKENS)]),
        audio_call(
            [
                json.dumps(
                    {
                        "contractVersion": "sparse-patch-v1",
                        "deletedSourceIds": [],
                        "cues": [
                            {
                                "sourceIds": [99],
                                "start": "00:00:28.000",
                                "end": "00:00:31.000",
                                "text": "Unknown source",
                            }
                        ],
                    }
                )
            ]
        ),
    ],
    ids=["max tokens", "unknown source id"],
)
def test_failed_audio_refinement_preserves_previous_output(tmp_path, monkeypatch, call):
    output = tmp_path / "output.vtt"
    output.write_text("previous", encoding="utf-8")

    with pytest.raises(RuntimeError):
        run_boundary_refine(monkeypatch, tmp_path, [call])

    assert output.read_text(encoding="utf-8") == "previous"
    assert not (tmp_path / "audio_refined.vtt").exists()
    assert not (tmp_path / "audio_refinement.json").exists()
