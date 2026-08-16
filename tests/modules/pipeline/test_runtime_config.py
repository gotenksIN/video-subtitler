"""Runtime configuration policy of the generation pipeline."""

from pathlib import Path

import pytest

from modules import media, pipeline


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
    return pipeline.GenerationConfig(**values)


def test_chunk_thinking_level_defaults_to_high():
    config = pipeline.GenerationConfig(
        video_path=Path("source.webm"),
        output_path=Path("output.vtt"),
        model="gemini-pro",
    )

    assert config.chunk_thinking_level == "high"


def test_chunk_thinking_level_uses_the_explicit_value():
    config = pipeline.GenerationConfig(
        video_path=Path("source.webm"),
        output_path=Path("output.vtt"),
        model="gemini-pro",
        thinking_level="low",
    )

    assert config.chunk_thinking_level == "low"


def test_validation_accepts_configured_generation_inputs(tmp_path):
    pipeline.validate_generation_config(make_config(tmp_path))


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
        pipeline.validate_generation_config(make_config(tmp_path, **overrides))


def test_validation_requires_an_existing_video_file(tmp_path):
    config = make_config(tmp_path)
    config.video_path.unlink()

    with pytest.raises(RuntimeError, match="Video file not found"):
        pipeline.validate_generation_config(config)


def test_validation_requires_an_api_key(tmp_path):
    with pytest.raises(RuntimeError, match="API key not configured"):
        pipeline.validate_generation_config(make_config(tmp_path, api_key=None))


def test_validation_rejects_output_resolving_to_the_source_video(tmp_path):
    source = tmp_path / "source.webm"
    source.write_bytes(b"video")
    alias = tmp_path / "alias.webm"
    alias.symlink_to(source)
    config = pipeline.GenerationConfig(
        video_path=source,
        output_path=alias,
        model="model",
        api_key="key",
    )

    with pytest.raises(RuntimeError, match="must not resolve to the source video"):
        pipeline.validate_generation_config(config)


def test_generation_rejects_malformed_context_url_before_probing(tmp_path, monkeypatch):
    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("media probing must not run for malformed URLs")

    monkeypatch.setattr(media, "probe_video_format", fail_if_called)
    config = make_config(tmp_path, context_urls=("not-a-url",))

    with pytest.raises(ValueError, match="context-url"):
        pipeline.run_generation(config)
