"""Runtime configuration and context URL behavior of the pipeline."""

from pathlib import Path

import pytest

import gemini_subs


def make_config(tmp_path, **overrides):
    video = tmp_path / "source.webm"
    video.write_bytes(b"video")
    values = {
        "video_path": video,
        "output_path": tmp_path / "output.vtt",
        "model": "model",
        "api_key": "key",
    }
    values.update(overrides)
    return gemini_subs.GenerationConfig(**values)


def test_chunk_thinking_level_defaults_to_high():
    config = gemini_subs.GenerationConfig(
        video_path=Path("source.webm"),
        output_path=Path("output.vtt"),
        model="gemini-pro",
    )

    assert config.chunk_thinking_level == "high"


def test_chunk_thinking_level_uses_the_explicit_value():
    config = gemini_subs.GenerationConfig(
        video_path=Path("source.webm"),
        output_path=Path("output.vtt"),
        model="gemini-pro",
        thinking_level="low",
    )

    assert config.chunk_thinking_level == "low"


def test_validation_accepts_configured_generation_inputs(tmp_path):
    gemini_subs.validate_generation_config(make_config(tmp_path))


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"chunk_dur": 0}, "chunk-dur must be greater than 0"),
        ({"workers": 0}, "workers must be greater than 0"),
        ({"overlap": -1}, "overlap must be greater than or equal to 0"),
        (
            {"chunk_dur": 5, "overlap": 5},
            "overlap must be smaller than --chunk-dur",
        ),
        (
            {"model": "gemini-pro", "thinking_level": "minimal"},
            "only supported by Flash",
        ),
    ],
)
def test_validation_rejects_invalid_generation_inputs(tmp_path, overrides, message):
    with pytest.raises(ValueError, match=message):
        gemini_subs.validate_generation_config(make_config(tmp_path, **overrides))


def test_validation_requires_an_existing_video_file(tmp_path):
    config = make_config(tmp_path)
    config.video_path.unlink()

    with pytest.raises(RuntimeError, match="Video file not found"):
        gemini_subs.validate_generation_config(config)


def test_validation_requires_an_api_key(tmp_path):
    with pytest.raises(RuntimeError, match="API key not configured"):
        gemini_subs.validate_generation_config(make_config(tmp_path, api_key=None))


def test_validation_rejects_output_resolving_to_the_source_video(tmp_path):
    source = tmp_path / "source.webm"
    source.write_bytes(b"video")
    alias = tmp_path / "alias.webm"
    alias.symlink_to(source)
    config = gemini_subs.GenerationConfig(
        video_path=source,
        output_path=alias,
        model="model",
        api_key="key",
    )

    with pytest.raises(RuntimeError, match="must not resolve to the source video"):
        gemini_subs.validate_generation_config(config)


def test_generation_rejects_malformed_context_url_before_probing(tmp_path, monkeypatch):
    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("media probing must not run for malformed URLs")

    monkeypatch.setattr(gemini_subs, "probe_video_format", fail_if_called)
    config = make_config(tmp_path, context_urls=("not-a-url",))

    with pytest.raises(ValueError, match="context-url"):
        gemini_subs.run_generation(config)


@pytest.mark.parametrize(
    "url",
    [
        "not-a-url",
        "ftp://example.com/file",
        "http://",
        "example.com/path",
        "https:///missing-host",
        "https://example.com:bad/path",
        "https://example .com/path",
    ],
)
def test_context_url_validation_rejects_malformed_values(url):
    with pytest.raises(ValueError, match="context-url"):
        gemini_subs.validate_context_urls([url])


def test_context_url_validation_deduplicates_preserving_first_occurrence():
    result = gemini_subs.validate_context_urls(
        ["https://example.com/a", "https://example.com/a", "http://example.com/b"]
    )

    assert result == ["https://example.com/a", "http://example.com/b"]


@pytest.mark.parametrize(
    "url",
    [
        "https://www.youtube.com/watch?v=abc",
        "https://youtube.com/watch?v=abc&t=30",
        "https://m.youtube.com/watch?v=abc",
        "https://youtu.be/abc",
        "https://youtu.be/abc?t=30",
    ],
)
def test_youtube_video_url_detection_accepts_watch_and_share_forms(url):
    assert gemini_subs.is_youtube_video_url(url)


@pytest.mark.parametrize(
    "url",
    [
        "https://youtube.com/channel/UC123",
        "https://www.youtube.com/playlist?list=x",
        "https://example.com/watch?v=abc",
        "https://youtu.be/",
        "https://notyoutube.com/watch?v=abc",
    ],
)
def test_youtube_video_url_detection_rejects_other_pages(url):
    assert not gemini_subs.is_youtube_video_url(url)


def test_context_url_classification_splits_youtube_from_ordinary():
    youtube_urls, ordinary_urls = gemini_subs.classify_context_urls(
        [
            "https://youtu.be/abc?t=5",
            "https://example.com/notes?id=1",
            "https://www.youtube.com/watch?v=abc",
        ]
    )

    assert youtube_urls == [
        "https://youtu.be/abc?t=5",
        "https://www.youtube.com/watch?v=abc",
    ]
    assert ordinary_urls == ["https://example.com/notes?id=1"]
