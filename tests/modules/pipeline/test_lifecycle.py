"""Complete generation lifecycle and recovery outcomes."""

import pytest
import webvtt

from modules import gemini, media, pipeline
from tests.support.workdir import FakeMediaTools, write_chunk_subtitles


def prepare_generation(
    tmp_path, monkeypatch, audio_refine=False, refine_text=False, output_path=None
):
    video = tmp_path / "source.mp4"
    video.write_bytes(b"fake video")
    monkeypatch.setattr(
        media,
        "probe_video_format",
        lambda _path: (".mp4", "video/mp4", "h264"),
    )
    monkeypatch.setattr(pipeline, "CHUNK_ROOT", str(tmp_path / "work_root"))
    tools = FakeMediaTools()
    monkeypatch.setattr(media.subprocess, "run", tools.run)
    config = pipeline.GenerationConfig(
        video_path=video,
        output_path=output_path or tmp_path / "output.vtt",
        model="model",
        api_key="key",
        chunk_dur=60,
        workers=2,
        thinking_level="high",
        audio_refine=audio_refine,
        refine_text=refine_text,
    )
    return config, tools


def write_two_chunk_subtitles(monkeypatch):
    def process(_key, _base, chunk, chunk_dir, *_args):
        index = chunk["idx"]
        write_chunk_subtitles(
            chunk_dir,
            index,
            [{"id": 0, "start": "0.5", "end": "1.5", "text": f"Cue {index}"}],
        )
        return True

    monkeypatch.setattr(gemini, "process_chunk", process)


def install_refiners(monkeypatch):
    def audio_refine(**kwargs):
        value = webvtt.read(kwargs["stitched_vtt"])
        for caption in value.captions:
            caption.text = f"Audio: {caption.text}"
        value.save(str(kwargs["output_vtt"]))

    def text_refine(input_vtt, output_vtt, *_args, **_kwargs):
        value = webvtt.read(input_vtt)
        for caption in value.captions:
            caption.text = f"Text: {caption.text}"
        value.save(str(output_vtt))

    monkeypatch.setattr(gemini, "boundary_audio_refine_subtitles", audio_refine)
    monkeypatch.setattr(gemini, "global_refine_subtitles", text_refine)


def active_work_directory(tmp_path):
    return next(iter((tmp_path / "work_root").iterdir()))


@pytest.mark.parametrize(
    ("audio_refine", "refine_text", "expected"),
    [
        (True, True, ["Text: Audio: Cue 0", "Text: Audio: Cue 1"]),
        (True, False, ["Audio: Cue 0", "Audio: Cue 1"]),
        (False, True, ["Text: Cue 0", "Text: Cue 1"]),
        (False, False, ["Cue 0", "Cue 1"]),
    ],
    ids=["audio-and-text", "audio-only", "text-only", "neither"],
)
def test_toggle_combinations_publish_the_selected_artifact_and_clean_work(
    tmp_path, monkeypatch, audio_refine, refine_text, expected
):
    config, _tools = prepare_generation(
        tmp_path, monkeypatch, audio_refine=audio_refine, refine_text=refine_text
    )
    write_two_chunk_subtitles(monkeypatch)
    install_refiners(monkeypatch)

    pipeline.run_generation(config)

    result = webvtt.read(config.output_path)
    assert [caption.text for caption in result] == expected
    assert [(caption.start, caption.end) for caption in result] == [
        ("00:00:00.500", "00:00:01.500"),
        ("00:00:02.500", "00:00:03.500"),
    ]
    work = active_work_directory(tmp_path)
    assert sorted(path.name for path in work.iterdir()) == [pipeline.LOCK_NAME]


def test_missing_audio_stream_fails_before_splitting(tmp_path, monkeypatch):
    config, tools = prepare_generation(tmp_path, monkeypatch, audio_refine=True)
    tools.audio_streams = ()

    def process(*_args):
        raise AssertionError("chunk generation must not run without audio")

    monkeypatch.setattr(gemini, "process_chunk", process)

    with pytest.raises(RuntimeError, match="Failed to extract complete audio"):
        pipeline.run_generation(config)

    assert tools.split_calls() == []
    assert not config.output_path.exists()


def test_retry_resumes_failed_work_without_resplitting_and_publishes(
    tmp_path, monkeypatch
):
    config, tools = prepare_generation(tmp_path, monkeypatch, audio_refine=True)
    failed_once = False

    def process(_key, _base, chunk, chunk_dir, *_args):
        nonlocal failed_once
        index = chunk["idx"]
        if index == 1 and not failed_once:
            failed_once = True
            return False
        write_chunk_subtitles(
            chunk_dir,
            index,
            [{"id": 0, "start": "0.5", "end": "1.5", "text": f"Cue {index}"}],
        )
        return True

    monkeypatch.setattr(gemini, "process_chunk", process)
    install_refiners(monkeypatch)

    with pytest.raises(RuntimeError, match="Failed to process 1 chunk"):
        pipeline.run_generation(config)
    pipeline.run_generation(config)

    assert len(tools.split_calls()) == 1
    assert len(tools.audio_extraction_calls()) == 1
    result = webvtt.read(config.output_path)
    assert [caption.text for caption in result] == ["Audio: Cue 0", "Audio: Cue 1"]


@pytest.mark.parametrize(
    "stage",
    ["chunk", "audio_refine", "text_refine"],
    ids=["chunk API failure", "audio refinement failure", "text refinement failure"],
)
def test_stage_failures_preserve_previous_output_and_resume_state(
    tmp_path, monkeypatch, stage
):
    output = tmp_path / "output.vtt"
    output.write_text("previous", encoding="utf-8")
    config, _tools = prepare_generation(
        tmp_path,
        monkeypatch,
        audio_refine=(stage == "audio_refine"),
        refine_text=(stage == "text_refine"),
        output_path=output,
    )
    if stage == "chunk":
        monkeypatch.setattr(gemini, "process_chunk", lambda *_args: False)
    else:
        write_two_chunk_subtitles(monkeypatch)
        if stage == "audio_refine":
            monkeypatch.setattr(
                gemini,
                "boundary_audio_refine_subtitles",
                lambda **_kw: (_ for _ in ()).throw(RuntimeError("fail")),
            )
        elif stage == "text_refine":
            monkeypatch.setattr(
                gemini,
                "global_refine_subtitles",
                lambda *_args, **_kw: (_ for _ in ()).throw(RuntimeError("fail")),
            )

    with pytest.raises(RuntimeError):
        pipeline.run_generation(config)

    assert output.read_text(encoding="utf-8") == "previous"
    work = active_work_directory(tmp_path)
    assert media.SPLIT_COMPLETE_MARKER in [p.name for p in work.iterdir()]
    assert not list(tmp_path.glob("*.staging.vtt"))
