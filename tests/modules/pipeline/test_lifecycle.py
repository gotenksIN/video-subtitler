"""Whole-song generation and safe output behavior."""

from dataclasses import replace

import pytest

from modules import core, gemini, media, pipeline


def install_offline_stages(tmp_path, monkeypatch):
    calls = {"research": 0, "transcribe": 0, "refine": 0, "translate": 0}

    monkeypatch.setattr(media, "probe_duration", lambda _path: 5.0)

    def extract(_source, work_dir):
        path = work_dir / media.EXTRACTED_AUDIO_NAME
        path.write_bytes(b"normalized")
        return path, 5.0

    monkeypatch.setattr(media, "extract_complete_audio", extract)

    def research(*_args):
        calls["research"] += 1
        return core.SongContext(summary="verified", search_queries=["song"])

    def transcribe(*_args):
        calls["transcribe"] += 1
        return [core.LyricLine(id="L1", start="1", end="2", text="draft")]

    def refine(*_args):
        calls["refine"] += 1
        return [core.LyricLine(id="L1", start="1", end="2", text="source")]

    def translate(*_args):
        calls["translate"] += 1
        return [core.TranslationLine(id="L1", text="translation")]

    monkeypatch.setattr(gemini, "research_song_context", research)
    monkeypatch.setattr(gemini, "transcribe_song", transcribe)
    monkeypatch.setattr(gemini, "refine_song", refine)
    monkeypatch.setattr(gemini, "translate_song", translate)
    return calls


def test_each_run_generates_source_and_optional_translation_from_scratch(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(pipeline, "WORK_ROOT", str(tmp_path / "work"))
    audio = tmp_path / "song.flac"
    audio.write_bytes(b"audio")
    calls = install_offline_stages(tmp_path, monkeypatch)
    first = pipeline.GenerationConfig(
        audio_path=audio, output_dir=tmp_path / "source", api_key="key"
    )
    pipeline.run_generation(first)

    second = replace(first, output_dir=tmp_path / "bilingual", target_language="en")
    pipeline.run_generation(second)

    assert calls == {"research": 2, "transcribe": 2, "refine": 2, "translate": 1}
    assert (
        (tmp_path / "source/song.lrc")
        .read_text(encoding="utf-8")
        .startswith("[00:01.00]source")
    )
    ttml = (tmp_path / "bilingual/song.ttml").read_text(encoding="utf-8")
    assert "translation" in ttml
    assert core.ITUNES_INTERNAL_NS in ttml
    assert not list((tmp_path / "work").iterdir())


def test_existing_default_sidecar_refuses_before_media_or_api_costs(
    tmp_path, monkeypatch
):
    audio = tmp_path / "song.flac"
    audio.write_bytes(b"audio")
    reference = tmp_path / "song.lrc"
    reference.write_text("benchmark\n", encoding="utf-8")
    monkeypatch.setattr(
        media,
        "extract_complete_audio",
        lambda *_args: (_ for _ in ()).throw(AssertionError("must not extract")),
    )
    config = pipeline.GenerationConfig(audio_path=audio, api_key="key")

    with pytest.raises(FileExistsError, match="song.lrc"):
        pipeline.run_generation(config)

    assert reference.read_text(encoding="utf-8") == "benchmark\n"
    assert not (tmp_path / pipeline.WORK_ROOT).exists()
