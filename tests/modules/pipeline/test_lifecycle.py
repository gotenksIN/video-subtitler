"""Complete generation lifecycle and recovery outcomes."""

import json
import shutil
import subprocess
from dataclasses import replace
from threading import Event

import pytest
import webvtt

from modules import core, gemini, media, pipeline
from tests.support.gemini_fakes import (
    ScriptedGeminiClient,
    audio_call,
    chunk_call,
    read_captions,
    use_client,
)
from tests.support.workdir import FakeMediaTools, write_chunk_subtitles


def prepare_generation(
    tmp_path, monkeypatch, audio_refine=False, refine_text=False, output_path=None
):
    video = tmp_path / "source.mp4"
    video.write_bytes(b"fake video")
    monkeypatch.setattr(pipeline, "CHUNK_ROOT", str(tmp_path / "work_root"))
    install_preflight(monkeypatch)
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


def install_preflight(monkeypatch, grounded_names=()):
    monkeypatch.setattr(
        gemini,
        "run_preflight_context",
        lambda *_args, **_kwargs: core.PreflightContext(
            grounded_names=list(grounded_names)
        ),
    )


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


def test_real_media_trailing_visual_survives_audio_refinement(tmp_path, monkeypatch):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg or not shutil.which("ffprobe"):
        pytest.fail("this test requires ffmpeg and ffprobe in PATH")
    video = tmp_path / "source.mp4"
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=64x64:rate=25:duration=2.4",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=1.4",
            "-c:v",
            "libx264",
            "-force_key_frames",
            "1",
            "-c:a",
            "aac",
            str(video),
        ],
        check=True,
        capture_output=True,
    )
    monkeypatch.setattr(pipeline, "CHUNK_ROOT", str(tmp_path / "work_root"))
    install_preflight(monkeypatch)
    client = ScriptedGeminiClient(
        [
            chunk_call([json.dumps({"captions": []})]),
            chunk_call(
                [
                    json.dumps(
                        {
                            "captions": [
                                {
                                    "id": 0,
                                    "start": "0.8",
                                    "end": "1.3",
                                    "text": "[End card]",
                                }
                            ]
                        }
                    )
                ]
            ),
            audio_call(
                [json.dumps({"contractVersion": "sparse-patch-v1", "cues": []})]
            ),
        ]
    )
    use_client(monkeypatch, client)
    monkeypatch.setattr(pipeline, "clean_completed_work", lambda _directory: None)
    config = pipeline.GenerationConfig(
        video_path=video,
        output_path=tmp_path / "output.vtt",
        model="model",
        api_key="key",
        chunk_dur=1,
        workers=1,
        thinking_level="high",
        audio_refine=True,
        refine_text=False,
    )

    pipeline.run_generation(config)

    chunks = media.list_chunks(active_work_directory(tmp_path))
    assert len(chunks) == 2
    offset = chunks[1]["start"]
    assert read_captions(config.output_path) == [
        (core.format_time(offset + 0.8), core.format_time(offset + 1.3), "[End card]")
    ]
    assert len(client.requests) == 3


def test_single_chunk_skips_audio_extraction_and_boundary_refinement(
    tmp_path, monkeypatch
):
    config, tools = prepare_generation(tmp_path, monkeypatch, audio_refine=True)
    tools.chunk_rows = ((0, 4),)
    write_two_chunk_subtitles(monkeypatch)

    def unexpected_refine(**_kwargs):
        raise AssertionError("Pass 2 must not run without boundaries")

    monkeypatch.setattr(gemini, "boundary_audio_refine_subtitles", unexpected_refine)

    pipeline.run_generation(config)

    assert read_captions(config.output_path) == [
        ("00:00:00.500", "00:00:01.500", "Cue 0")
    ]
    assert tools.audio_extraction_calls() == []


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


def test_preflight_grounded_names_reach_chunk_workers_and_text_refinement(
    tmp_path, monkeypatch
):
    config, _tools = prepare_generation(tmp_path, monkeypatch, refine_text=True)
    install_preflight(monkeypatch, grounded_names=["Jane Doe"])
    received = []

    def process(_key, _base, chunk, chunk_dir, _model, _mime, _level, _title, names):
        received.append(names)
        write_chunk_subtitles(
            chunk_dir,
            chunk["idx"],
            [{"id": 0, "start": "0.5", "end": "1.5", "text": f"Cue {chunk['idx']}"}],
        )
        return True

    monkeypatch.setattr(gemini, "process_chunk", process)

    seen = {}

    def text_refine(input_vtt, output_vtt, *_args, **kwargs):
        seen["preflight"] = kwargs["preflight_context"]
        value = webvtt.read(input_vtt)
        value.save(str(output_vtt))

    monkeypatch.setattr(gemini, "global_refine_subtitles", text_refine)

    pipeline.run_generation(config)

    assert received == [["Jane Doe"], ["Jane Doe"]]
    assert seen["preflight"].grounded_names == ["Jane Doe"]


@pytest.mark.parametrize(
    "refine_model",
    [None, "refine-model"],
    ids=["default-refinement", "different-refinement-model"],
)
def test_preflight_uses_the_generation_model_and_run_context(
    tmp_path, monkeypatch, refine_model
):
    config, _tools = prepare_generation(tmp_path, monkeypatch)
    config = replace(
        config,
        refine_model=refine_model,
        context_urls=("https://example.com/notes",),
    )
    received = {}

    def preflight(*_args, **kwargs):
        received.update(kwargs)
        return core.PreflightContext()

    monkeypatch.setattr(gemini, "run_preflight_context", preflight)
    write_two_chunk_subtitles(monkeypatch)

    pipeline.run_generation(config)

    assert received["model_name"] == config.model
    assert received["thinking_level"] == gemini.REFINEMENT_THINKING_LEVEL
    assert received["source_title"] == "source"
    assert received["context_urls"] == ["https://example.com/notes"]


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


def test_preflight_overlaps_splitting_and_audio_extraction(tmp_path, monkeypatch):
    config, _tools = prepare_generation(tmp_path, monkeypatch, audio_refine=True)
    preflight_started = Event()
    audio_started = Event()
    split = media.split_video
    extract = media.extract_complete_audio

    def preflight(**_kwargs):
        preflight_started.set()
        assert audio_started.wait(5), "audio extraction must not wait for preflight"
        return core.PreflightContext()

    def splitting(*args):
        assert preflight_started.wait(5), "splitting must overlap preflight"
        return split(*args)

    def extraction(*args, **kwargs):
        audio_started.set()
        return extract(*args, **kwargs)

    monkeypatch.setattr(gemini, "run_preflight_context", preflight)
    monkeypatch.setattr(media, "split_video", splitting)
    monkeypatch.setattr(media, "extract_complete_audio", extraction)
    write_two_chunk_subtitles(monkeypatch)
    install_refiners(monkeypatch)
    pipeline.run_generation(config)
    assert config.output_path.exists()


@pytest.mark.parametrize("stage", ["split_video", "extract_complete_audio"])
def test_media_failure_settles_preflight_before_releasing_lock(
    tmp_path, monkeypatch, stage
):
    config, _tools = prepare_generation(tmp_path, monkeypatch, audio_refine=True)
    media_failed = Event()
    settled = []

    def preflight(**_kwargs):
        assert media_failed.wait(5)
        work = active_work_directory(tmp_path)
        with pytest.raises(RuntimeError, match="already using"):
            pipeline.acquire_lock(work)
        settled.append(True)
        return core.PreflightContext()

    def failure(*_args, **_kwargs):
        media_failed.set()
        raise RuntimeError("media failed")

    monkeypatch.setattr(gemini, "run_preflight_context", preflight)
    monkeypatch.setattr(media, stage, failure)
    with pytest.raises(RuntimeError, match="media failed"):
        pipeline.run_generation(config)
    assert settled == [True]
    lock = pipeline.acquire_lock(active_work_directory(tmp_path))
    pipeline.release_lock(lock)


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
