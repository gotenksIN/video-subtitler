"""Global refinement through the Gemini adapter."""

import json

import pytest
from google.genai import types

from modules import gemini
from tests.support.gemini_fakes import (
    ScriptedGeminiClient,
    grounding_candidate,
    read_captions,
    refinement_call,
    research_call,
    use_client,
    write_vtt,
    youtube_call,
)


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
            research_call(queries=["who is the host"]),
            refinement_call(
                [json.dumps({"changes": [{"id": 1, "text": "Rewritten"}]})]
            ),
        ]
    )
    use_client(monkeypatch, client)

    gemini.global_refine_subtitles(source, output, "key", None, "refiner", "medium")

    refinement = client.requests[1]
    assert (
        refinement.config.thinking_config.thinking_level == types.ThinkingLevel.MEDIUM
    )
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
            research_call(queries=["who is the host"]),
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
            research_call(queries=["who is the host"]),
            refinement_call(['{"changes": [{"id": 0, "text":', '"Assembled"}]}']),
        ]
    )
    use_client(monkeypatch, client)

    gemini.global_refine_subtitles(source, output, "key", None, "refiner", "high")

    assert read_captions(output)[0][2] == "Assembled"


def test_identity_research_requests_grounded_plain_text(tmp_path, monkeypatch, capsys):
    source = write_vtt(
        tmp_path / "source.vtt", [("00:00:00.000", "00:00:01.000", "Only")]
    )
    output = tmp_path / "output.vtt"
    client = ScriptedGeminiClient(
        [
            research_call(
                pieces=("Researched identities",),
                queries=["participants of the show"],
                sources=[("Show Wiki", "https://wiki.example/show")],
            ),
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
    assert research.config.thinking_config.thinking_level == types.ThinkingLevel.MEDIUM
    assert any(tool.google_search is not None for tool in research.config.tools)

    printed = capsys.readouterr().out
    assert "participants of the show" in printed
    assert "https://wiki.example/show" in printed
    assert read_captions(output) == [("00:00:00.000", "00:00:01.000", "Only")]


def test_identity_research_collects_grounding_from_later_stream_chunks(
    tmp_path, monkeypatch, capsys
):
    source = write_vtt(
        tmp_path / "source.vtt", [("00:00:00.000", "00:00:01.000", "Only")]
    )
    output = tmp_path / "output.vtt"
    research = research_call(pieces=("Identity ", "research"))
    research.candidate_chunks = [
        [],
        grounding_candidate(sources=[("Official page", "https://example.com/show")]),
    ]
    client = ScriptedGeminiClient([research, refinement_call(['{"changes": []}'])])
    use_client(monkeypatch, client)

    gemini.global_refine_subtitles(source, output, "key", None, "refiner", "medium")

    assert "Official page: https://example.com/show" in capsys.readouterr().out
    assert read_captions(output) == [("00:00:00.000", "00:00:01.000", "Only")]


def test_missing_search_grounding_fails_before_publication(tmp_path, monkeypatch):
    source = write_vtt(
        tmp_path / "source.vtt", [("00:00:00.000", "00:00:01.000", "First")]
    )
    output = tmp_path / "output.vtt"
    output.write_text("previous", encoding="utf-8")
    client = ScriptedGeminiClient(
        [
            research_call(),
            refinement_call(['{"changes": [{"id": 0, "text": "Changed"}]}']),
        ]
    )
    use_client(monkeypatch, client)

    with pytest.raises(RuntimeError, match="no Google Search grounding"):
        gemini.global_refine_subtitles(source, output, "key", None, "refiner", "high")

    assert len(client.requests) == 1
    assert output.read_text(encoding="utf-8") == "previous"
    assert read_captions(source)[0][2] == "First"


def test_missing_context_url_retrieval_fails_before_publication(tmp_path, monkeypatch):
    source = write_vtt(
        tmp_path / "source.vtt", [("00:00:00.000", "00:00:01.000", "First")]
    )
    output = tmp_path / "output.vtt"
    output.write_text("previous", encoding="utf-8")
    client = ScriptedGeminiClient(
        [research_call(queries=["who is the host"]), refinement_call([])]
    )
    use_client(monkeypatch, client)

    with pytest.raises(RuntimeError, match="was not retrieved"):
        gemini.global_refine_subtitles(
            source,
            output,
            "key",
            None,
            "refiner",
            "high",
            context_urls=["https://example.com/notes"],
        )

    assert len(client.requests) == 1
    assert output.read_text(encoding="utf-8") == "previous"


def test_failed_context_url_retrieval_fails_before_publication(tmp_path, monkeypatch):
    source = write_vtt(
        tmp_path / "source.vtt", [("00:00:00.000", "00:00:01.000", "First")]
    )
    output = tmp_path / "output.vtt"
    output.write_text("previous", encoding="utf-8")
    client = ScriptedGeminiClient(
        [
            research_call(
                queries=["who is the host"],
                retrieved=[
                    ("https://example.com/notes", "URL_RETRIEVAL_STATUS_PAYWALL")
                ],
            ),
            refinement_call([]),
        ]
    )
    use_client(monkeypatch, client)

    with pytest.raises(
        RuntimeError, match="retrieval failed with URL_RETRIEVAL_STATUS_PAYWALL"
    ):
        gemini.global_refine_subtitles(
            source,
            output,
            "key",
            None,
            "refiner",
            "high",
            context_urls=["https://example.com/notes"],
        )

    assert len(client.requests) == 1
    assert output.read_text(encoding="utf-8") == "previous"


def test_equivalent_retrieved_url_identity_is_accepted(tmp_path, monkeypatch):
    source = write_vtt(
        tmp_path / "source.vtt", [("00:00:00.000", "00:00:01.000", "Only")]
    )
    output = tmp_path / "output.vtt"
    client = ScriptedGeminiClient(
        [
            research_call(
                queries=["who is the host"],
                retrieved=[
                    (
                        "https://EXAMPLE.com/notes/",
                        "URL_RETRIEVAL_STATUS_SUCCESS",
                    )
                ],
            ),
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
        context_urls=["https://example.com/notes"],
    )

    assert read_captions(output)[0][2] == "Only"


def test_ordinary_context_urls_enable_url_context_tool(tmp_path, monkeypatch):
    source = write_vtt(
        tmp_path / "source.vtt", [("00:00:00.000", "00:00:01.000", "Only")]
    )
    output = tmp_path / "output.vtt"
    client = ScriptedGeminiClient(
        [
            research_call(
                queries=["who is the host"],
                retrieved=[
                    ("https://example.com/notes", "URL_RETRIEVAL_STATUS_SUCCESS")
                ],
            ),
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
        context_urls=["https://example.com/notes"],
    )

    tools = client.requests[0].config.tools
    assert any(tool.url_context is not None for tool in tools)
    assert read_captions(output)[0][2] == "Only"


def test_youtube_context_urls_become_direct_video_analysis(tmp_path, monkeypatch):
    source = write_vtt(
        tmp_path / "source.vtt", [("00:00:00.000", "00:00:01.000", "Only")]
    )
    output = tmp_path / "output.vtt"
    youtube_url = "https://www.youtube.com/watch?v=VIDEO_ID&t=30"
    client = ScriptedGeminiClient(
        [
            research_call(queries=["who is in the video"]),
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
            research_call(
                queries=["who is in the video"],
                retrieved=[(ordinary_url, "URL_RETRIEVAL_STATUS_SUCCESS")],
            ),
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
            research_call(queries=["who is the host"]),
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
            research_call(queries=["who is the host"]),
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
            research_call(queries=["who is the host"]),
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
    client = ScriptedGeminiClient(
        [research_call(queries=["who is the host"]), refinement_call([changes])]
    )
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


def test_refinement_removes_boundary_duplicate_created_by_a_text_change(
    tmp_path, monkeypatch
):
    source = write_vtt(
        tmp_path / "staging.vtt",
        [
            ("00:00:00.000", "00:00:04.000", "Host: Shared opening line"),
            ("00:00:03.000", "00:00:05.000", "Guest: Different line"),
        ],
    )
    output = tmp_path / "output.vtt"
    client = ScriptedGeminiClient(
        [
            research_call(queries=["who is the host"]),
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

    gemini.global_refine_subtitles(
        source,
        output,
        "key",
        None,
        "refiner",
        "high",
        boundary_provenance=[0, 1],
    )

    assert read_captions(output) == [
        ("00:00:00.000", "00:00:04.000", "Host: Shared opening line")
    ]


def test_mismatched_provenance_is_rejected_before_any_request(tmp_path, monkeypatch):
    source = write_vtt(
        tmp_path / "source.vtt", [("00:00:00.000", "00:00:01.000", "Only")]
    )
    output = tmp_path / "output.vtt"
    output.write_text("previous", encoding="utf-8")
    client = ScriptedGeminiClient(
        [research_call(queries=["who is the host"]), refinement_call([])]
    )
    use_client(monkeypatch, client)

    with pytest.raises(ValueError, match="one chunk index per caption"):
        gemini.global_refine_subtitles(
            source,
            output,
            "key",
            None,
            "refiner",
            "high",
            boundary_provenance=[],
        )

    assert client.requests == []
    assert output.read_text(encoding="utf-8") == "previous"
