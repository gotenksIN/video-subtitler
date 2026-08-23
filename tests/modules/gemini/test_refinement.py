"""Global refinement through the Gemini adapter."""

import json

import pytest
from google.genai import errors, types

from modules import core, gemini
from tests.support.gemini_fakes import (
    ScriptedGeminiClient,
    read_captions,
    refinement_call,
    research_call,
    use_client,
    write_vtt,
    youtube_call,
)


@pytest.fixture(autouse=True)
def normalized_stream_metadata(monkeypatch):
    """Keep refinement scenarios at the project-owned metadata seam."""

    def collect(response_stream):
        text = "".join(chunk.text or "" for chunk in response_stream)
        retrieved = {
            "https://example.com/notes": "URL_RETRIEVAL_STATUS_SUCCESS",
        }
        return text, ["grounded query"], [], retrieved

    monkeypatch.setattr(gemini, "collect_stream_metadata", collect)


def test_refinement_changes_text_only_and_preserves_timestamps(tmp_path, monkeypatch):
    source = write_vtt(
        tmp_path / "source.vtt",
        [
            ("00:00:00.000", "00:00:01.000", "Old line"),
            ("00:00:02.000", "00:00:03.000", "Keep"),
            ("00:00:04.000", "00:00:05.000", "Untouched"),
        ],
    )
    output = tmp_path / "output.vtt"
    client = ScriptedGeminiClient(
        [
            research_call(),
            refinement_call(
                [json.dumps({"changes": [{"id": 1, "text": "Rewritten"}]})]
            ),
        ]
    )
    use_client(monkeypatch, client)

    gemini.global_refine_subtitles(source, output, "key", None, "refiner", "medium")

    refinement = client.requests[1]
    assert refinement.config.response_mime_type == "application/json"
    assert refinement.config.response_schema is core.RefinementResponse
    assert refinement.config.automatic_function_calling.disable is True
    assert refinement.config.tools is None
    assert (
        refinement.config.thinking_config.thinking_level == types.ThinkingLevel.MEDIUM
    )
    assert refinement.config.temperature == 0.0
    assert read_captions(output) == [
        ("00:00:00.000", "00:00:01.000", "Old line"),
        ("00:00:02.000", "00:00:03.000", "Rewritten"),
        ("00:00:04.000", "00:00:05.000", "Untouched"),
    ]


def test_refinement_without_changes_publishes_identical_script(tmp_path, monkeypatch):
    source = write_vtt(
        tmp_path / "source.vtt",
        [
            ("00:00:00.000", "00:00:01.000", "First"),
            ("00:00:02.000", "00:00:03.000", "Second"),
        ],
    )
    output = tmp_path / "output.vtt"
    client = ScriptedGeminiClient(
        [
            research_call(),
            refinement_call(['{"changes": []}']),
        ]
    )
    use_client(monkeypatch, client)

    gemini.global_refine_subtitles(source, output, "key", None, "refiner", "medium")

    assert read_captions(output) == [
        ("00:00:00.000", "00:00:01.000", "First"),
        ("00:00:02.000", "00:00:03.000", "Second"),
    ]


def test_refinement_stream_pieces_are_assembled_before_parsing(tmp_path, monkeypatch):
    source = write_vtt(
        tmp_path / "source.vtt", [("00:00:00.000", "00:00:01.000", "Old")]
    )
    output = tmp_path / "output.vtt"
    client = ScriptedGeminiClient(
        [
            research_call(),
            refinement_call(['{"changes": [{"id": 0, "text":', '"Assembled"}]}']),
        ]
    )
    use_client(monkeypatch, client)

    gemini.global_refine_subtitles(source, output, "key", None, "refiner", "high")

    assert read_captions(output)[0][2] == "Assembled"


def test_identity_research_sends_grounded_plain_text_request(tmp_path, monkeypatch):
    identity_body = "Jane Doe: Host. Evidence: official program page."
    terminology_body = "Season Premiere: recurring program title spelling."
    research_text = (
        "PARTICIPANTS AND SPEAKERS:\n"
        f"{identity_body}\n"
        "\n"
        "TOPIC TERMINOLOGY AND PROPER NOUNS:\n"
        f"{terminology_body}\n"
    )
    source = write_vtt(
        tmp_path / "source.vtt",
        [("00:00:00.000", "00:00:01.000", "JANE DOE: Only")],
    )
    output = tmp_path / "output.vtt"
    client = ScriptedGeminiClient(
        [
            research_call(pieces=(research_text,)),
            refinement_call(['{"changes": []}']),
        ]
    )
    use_client(monkeypatch, client)

    gemini.global_refine_subtitles(source, output, "key", None, "refiner", "medium")

    research = client.requests[0]
    assert research.model == "refiner"
    assert isinstance(research.contents, str)
    assert research.config.response_mime_type is None
    assert research.config.response_schema is None
    assert research.config.automatic_function_calling.disable is True
    assert research.config.thinking_config.thinking_level == types.ThinkingLevel.MEDIUM
    assert research.config.temperature == 0.0
    assert any(tool.google_search is not None for tool in research.config.tools)
    assert all(tool.url_context is None for tool in research.config.tools)

    refinement_contents = client.requests[1].contents
    assert identity_body in refinement_contents
    assert terminology_body in refinement_contents
    assert refinement_contents.index(identity_body) < refinement_contents.index(
        terminology_body
    )
    assert "PARTICIPANTS AND SPEAKERS:" not in refinement_contents
    assert "TOPIC TERMINOLOGY AND PROPER NOUNS:" not in refinement_contents
    assert read_captions(output) == [("00:00:00.000", "00:00:01.000", "Jane Doe: Only")]


def test_youtube_context_urls_become_direct_video_analysis(tmp_path, monkeypatch):
    source = write_vtt(
        tmp_path / "source.vtt", [("00:00:00.000", "00:00:01.000", "Only")]
    )
    output = tmp_path / "output.vtt"
    youtube_url = "https://www.youtube.com/watch?v=VIDEO_ID&t=30"
    client = ScriptedGeminiClient(
        [
            research_call(),
            youtube_call(pieces=("Direct video identities",)),
            refinement_call([json.dumps({"changes": [{"id": 0, "text": "Refined"}]})]),
        ]
    )
    use_client(monkeypatch, client)

    gemini.global_refine_subtitles(
        source,
        output,
        "key",
        None,
        "refiner",
        "high",
        context_urls=[youtube_url],
    )

    research = client.requests[0]
    assert isinstance(research.contents, str)
    assert all(tool.url_context is None for tool in research.config.tools)

    analysis = client.requests[1]
    video_parts = [part for part in analysis.contents if not isinstance(part, str)]
    assert [part.file_data.file_uri for part in video_parts] == [youtube_url]
    assert all(part.file_data.mime_type == "video/*" for part in video_parts)
    assert isinstance(analysis.contents[-1], str)
    assert analysis.config.tools is None
    assert analysis.config.response_mime_type is None
    assert analysis.config.response_schema is None
    assert analysis.config.automatic_function_calling.disable is True
    assert analysis.config.thinking_config.thinking_level == types.ThinkingLevel.HIGH
    assert analysis.config.temperature == 0.0

    refinement = client.requests[2]
    assert refinement.config.response_mime_type == "application/json"
    assert read_captions(output)[0][2] == "Refined"


def test_ordinary_and_youtube_context_use_their_separate_retrieval_paths(
    tmp_path, monkeypatch
):
    source = write_vtt(
        tmp_path / "source.vtt", [("00:00:00.000", "00:00:01.000", "Only")]
    )
    output = tmp_path / "output.vtt"
    ordinary_url = "https://example.com/notes"
    youtube_url = "https://youtu.be/VIDEO_ID"
    client = ScriptedGeminiClient(
        [
            research_call(),
            youtube_call(),
            refinement_call(['{"changes": []}']),
        ]
    )
    use_client(monkeypatch, client)

    gemini.global_refine_subtitles(
        source,
        output,
        "key",
        None,
        "refiner",
        "high",
        context_urls=[ordinary_url, youtube_url],
    )

    research_tools = client.requests[0].config.tools
    assert any(tool.google_search is not None for tool in research_tools)
    assert any(tool.url_context is not None for tool in research_tools)
    video_parts = [
        part for part in client.requests[1].contents if not isinstance(part, str)
    ]
    assert [part.file_data.file_uri for part in video_parts] == [youtube_url]
    assert read_captions(output) == [("00:00:00.000", "00:00:01.000", "Only")]


def test_youtube_analysis_is_skipped_without_youtube_urls(tmp_path, monkeypatch):
    source = write_vtt(
        tmp_path / "source.vtt", [("00:00:00.000", "00:00:01.000", "Only")]
    )
    output = tmp_path / "output.vtt"
    client = ScriptedGeminiClient(
        [
            research_call(),
            refinement_call(['{"changes": []}']),
        ]
    )
    use_client(monkeypatch, client)

    gemini.global_refine_subtitles(source, output, "key", None, "refiner", "high")

    assert len(client.requests) == 2
    assert client.requests[1].config.response_mime_type == "application/json"


def test_youtube_analysis_sdk_failure_preserves_previous_output(tmp_path, monkeypatch):
    source = write_vtt(
        tmp_path / "source.vtt", [("00:00:00.000", "00:00:01.000", "First")]
    )
    output = tmp_path / "output.vtt"
    output.write_text("previous", encoding="utf-8")
    client = ScriptedGeminiClient(
        [
            research_call(),
            youtube_call(error=RuntimeError("video unavailable")),
            refinement_call(['{"changes": [{"id": 0, "text": "Changed"}]}']),
        ]
    )
    use_client(monkeypatch, client)

    with pytest.raises(RuntimeError, match="video unavailable"):
        gemini.global_refine_subtitles(
            source,
            output,
            "key",
            None,
            "refiner",
            "high",
            context_urls=["https://youtu.be/VIDEO_ID"],
        )

    assert len(client.requests) == 2
    assert output.read_text(encoding="utf-8") == "previous"
    assert read_captions(source)[0][2] == "First"


def test_youtube_analysis_server_error_propagates_without_host_retry(
    tmp_path, monkeypatch
):
    source = write_vtt(
        tmp_path / "source.vtt", [("00:00:00.000", "00:00:01.000", "First")]
    )
    output = tmp_path / "output.vtt"
    output.write_text("previous", encoding="utf-8")
    client = ScriptedGeminiClient(
        [
            research_call(),
            youtube_call(
                error=errors.ServerError(
                    503,
                    {
                        "error": {
                            "code": 503,
                            "message": "Deadline expired before operation could complete.",
                            "status": "UNAVAILABLE",
                        }
                    },
                )
            ),
            refinement_call(['{"changes": [{"id": 0, "text": "Changed"}]}']),
        ]
    )
    use_client(monkeypatch, client)

    with pytest.raises(errors.ServerError, match="503"):
        gemini.global_refine_subtitles(
            source,
            output,
            "key",
            None,
            "refiner",
            "high",
            context_urls=["https://youtu.be/VIDEO_ID"],
        )

    assert len(client.requests) == 2
    assert output.read_text(encoding="utf-8") == "previous"
    assert read_captions(source)[0][2] == "First"


def test_research_sdk_failure_preserves_previous_output(tmp_path, monkeypatch):
    source = write_vtt(
        tmp_path / "source.vtt", [("00:00:00.000", "00:00:01.000", "First")]
    )
    output = tmp_path / "output.vtt"
    output.write_text("previous", encoding="utf-8")
    client = ScriptedGeminiClient(
        [
            research_call(error=RuntimeError("search unavailable")),
            refinement_call(['{"changes": []}']),
        ]
    )
    use_client(monkeypatch, client)

    with pytest.raises(RuntimeError, match="search unavailable"):
        gemini.global_refine_subtitles(source, output, "key", None, "refiner", "high")

    assert len(client.requests) == 1
    assert output.read_text(encoding="utf-8") == "previous"


def test_invalid_refinement_json_preserves_source_and_output(tmp_path, monkeypatch):
    source = write_vtt(
        tmp_path / "source.vtt",
        [
            ("00:00:00.000", "00:00:01.000", "First"),
            ("00:00:02.000", "00:00:03.000", "Second"),
        ],
    )
    output = tmp_path / "output.vtt"
    output.write_text("previous", encoding="utf-8")
    client = ScriptedGeminiClient(
        [
            research_call(),
            refinement_call(["not json"]),
        ]
    )
    use_client(monkeypatch, client)

    with pytest.raises(RuntimeError, match=r"Raw response:\nnot json"):
        gemini.global_refine_subtitles(source, output, "key", None, "refiner", "high")

    assert read_captions(source)[0][2] == "First"
    assert output.read_text(encoding="utf-8") == "previous"


@pytest.mark.parametrize(
    "changes",
    [
        json.dumps({"changes": [{"id": 2, "text": "Out of range"}]}),
        json.dumps({"changes": [{"id": 0, "text": "One"}, {"id": 0, "text": "Two"}]}),
        json.dumps({"changes": [{"id": 0, "text": "   "}]}),
    ],
)
def test_invalid_refinement_changes_rejected_without_mutation(
    tmp_path, monkeypatch, changes
):
    source = write_vtt(
        tmp_path / "source.vtt",
        [
            ("00:00:00.000", "00:00:01.000", "First"),
            ("00:00:02.000", "00:00:03.000", "Second"),
        ],
    )
    output = tmp_path / "output.vtt"
    output.write_text("previous", encoding="utf-8")
    client = ScriptedGeminiClient([research_call(), refinement_call([changes])])
    use_client(monkeypatch, client)

    with pytest.raises(
        RuntimeError, match="Parsing or validating the model refinement response failed"
    ):
        gemini.global_refine_subtitles(source, output, "key", None, "refiner", "high")

    assert read_captions(source) == [
        ("00:00:00.000", "00:00:01.000", "First"),
        ("00:00:02.000", "00:00:03.000", "Second"),
    ]
    assert output.read_text(encoding="utf-8") == "previous"


def test_refinement_applies_text_changes_without_deleting_cues(tmp_path, monkeypatch):
    source = write_vtt(
        tmp_path / "source.vtt",
        [
            ("00:00:00.000", "00:00:04.000", "Host: Shared opening line"),
            ("00:00:03.000", "00:00:05.000", "Guest: Different line"),
        ],
    )
    output = tmp_path / "output.vtt"
    client = ScriptedGeminiClient(
        [
            research_call(),
            refinement_call(
                [
                    json.dumps(
                        {"changes": [{"id": 1, "text": "Host: Shared opening line"}]}
                    )
                ]
            ),
        ]
    )
    use_client(monkeypatch, client)

    gemini.global_refine_subtitles(source, output, "key", None, "refiner", "high")

    assert read_captions(output) == [
        ("00:00:00.000", "00:00:04.000", "Host: Shared opening line"),
        ("00:00:03.000", "00:00:05.000", "Host: Shared opening line"),
    ]
