"""Gemini adapter behaviors and request safety."""

import pytest

from modules import core, gemini
from tests.support.gemini_fakes import (
    ScriptedGeminiClient,
    StreamCall,
    grounded_chunk,
    use_client,
)


def test_research_requires_and_preserves_grounding(monkeypatch):
    client = ScriptedGeminiClient([StreamCall([grounded_chunk()])])
    use_client(monkeypatch, client)

    context = gemini.research_song_context("key", None, "pro", "Song", "Artist")

    config = client.requests[0].config.model_dump(exclude_none=True)
    assert context.search_queries == ["song artist"]
    assert [source.url for source in context.sources] == ["https://example.com/song"]
    assert config["automatic_function_calling"]["disable"] is True


def test_research_rejects_plain_ungrounded_text(monkeypatch):
    client = ScriptedGeminiClient([StreamCall(["unsupported context"])])
    use_client(monkeypatch, client)
    with pytest.raises(RuntimeError, match="no Google Search grounding"):
        gemini.research_song_context("key", None, "pro", "Song", None)


def test_transcription_returns_timed_lyrics_and_disables_afc(tmp_path, monkeypatch):
    audio = tmp_path / "audio.ogg"
    audio.write_bytes(b"whole audio")
    response = (
        '{"lines":[{"id":"L1","start":"0.1","end":"0.9","text":"annyeong hello"}]}'
    )
    client = ScriptedGeminiClient([StreamCall([response])])
    use_client(monkeypatch, client)
    context = core.SongContext(summary="verified", search_queries=["song"], sources=[])

    lines = gemini.transcribe_song("key", None, "flash", audio, 1.0, context, tmp_path)

    config = client.requests[0].config.model_dump(exclude_none=True)
    assert lines[0].text == "annyeong hello"
    assert config["automatic_function_calling"]["disable"] is True


def test_refinement_applies_corrections_and_preserves_other_lines(
    tmp_path, monkeypatch
):
    audio = tmp_path / "audio.ogg"
    audio.write_bytes(b"whole audio")
    response = '{"changes":[{"id":"L1","end":"0.4","text":"corrected"}]}'
    client = ScriptedGeminiClient([StreamCall([response])])
    use_client(monkeypatch, client)
    context = core.SongContext(summary="verified", search_queries=["song"])
    draft = core.validate_lyrics(
        [
            core.LyricLine(id="L1", start="0.1", end="0.3", text="draft"),
            core.LyricLine(id="L2", start="0.5", end="0.9", text="unchanged"),
        ],
        1.0,
    )

    result = gemini.refine_song(
        "key", None, "pro", audio, 1.0, context, draft, tmp_path
    )

    config = client.requests[0].config.model_dump(exclude_none=True)
    assert result[0].text == "corrected"
    assert result[0].start == draft[0].start
    assert result[0].end == "00:00:00.400"
    assert result[1] == draft[1]
    assert draft[0].text == "draft"
    assert config["automatic_function_calling"]["disable"] is True


def test_translation_preserves_source_id_pairing_and_disables_afc(
    tmp_path, monkeypatch
):
    response = (
        '{"translations":[{"id":"L1","text":"Hello"},{"id":"L2","text":"Again"}]}'
    )
    client = ScriptedGeminiClient([StreamCall([response])])
    use_client(monkeypatch, client)
    context = core.SongContext(summary="verified", search_queries=["song"])
    source = [
        core.LyricLine(id="L1", start="0", end="1", text="Hola"),
        core.LyricLine(id="L2", start="1", end="2", text="Otra vez"),
    ]

    result = gemini.translate_song("key", None, "pro", "en", context, source, tmp_path)

    config = client.requests[0].config.model_dump(exclude_none=True)
    assert [item.id for item in result] == ["L1", "L2"]
    assert config["automatic_function_calling"]["disable"] is True
