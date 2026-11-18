"""Complete generation capabilities with real media and one external Gemini fake."""

import base64
import json
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from threading import Barrier, Event

import httpx2
import pytest
from google.genai import errors, types

from modules import core, io, media, pipeline
from scripts import benchmark
from tests.support.gemini_fakes import (
    ScriptedGeminiClient,
    StreamCall,
    audio_call,
    chunk_call,
    research_call,
    stream_chunk,
    text_call,
    use_client,
)

pytestmark = pytest.mark.usefixtures("fast_retries")


@pytest.fixture
def config(tmp_path, monkeypatch, media_files):
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "source.mp4"
    shutil.copyfile(media_files["h264"], source)
    return pipeline.GenerationConfig(
        video_path=source,
        output_path=tmp_path / "output.vtt",
        model="generation-model",
        api_key="test-key",
        base_url="https://gemini.invalid/proxy",
        chunk_dur=1,
        workers=2,
    )


def calls(*, chunks=2, audio=True, text=True):
    return [
        research_call(),
        *(chunk_call() for _ in range(chunks)),
        *([audio_call()] if audio and chunks > 1 else []),
        *([text_call()] if text else []),
    ]


def run(monkeypatch, config, scenario):
    client = ScriptedGeminiClient(scenario)
    use_client(monkeypatch, client)
    pipeline.run_generation(config)
    assert not client.pending
    return client


def workdirs():
    return sorted(Path(pipeline.CHUNK_ROOT).glob("*"))


def assert_wire_config(request):
    assert request.config.automatic_function_calling.disable is True
    config = request.body["generationConfig"]
    if "responseSchema" in config:
        schema = json.dumps(config["responseSchema"])
        for keyword in (
            "additionalProperties",
            "additional_properties",
            "$defs",
            "$ref",
        ):
            assert keyword not in schema
        assert config["responseMimeType"] == "application/json"


def test_stitch_clamps_rounded_boundaries_before_editorial_merging(
    tmp_path, media_files, monkeypatch
):
    rows = []
    for index, (start, end) in enumerate(
        ((0, 1.0006), (1.0006, 2.0012), (2.0012, 3.0012))
    ):
        name = f"chunk_{index:03d}.mp4"
        shutil.copyfile(media_files["h264"], tmp_path / name)
        rows.append(f"{name},{start},{end}\n")
        captions = [
            {"id": 0, "start": "0", "end": "1.001", "text": "[Banner]"},
            {"id": 1, "start": "0", "end": "1.001", "text": f"Host: Chunk {index}"},
        ]
        if index == 1:
            captions.append(
                {"id": 2, "start": "0.999", "end": "1.001", "text": "Host: Brief"}
            )
        io.atomic_write_json(tmp_path / f"subtitle_chunk_{index:03d}.json", captions)
    (tmp_path / "segments.csv").write_text("".join(rows), encoding="ascii")
    output = tmp_path / "stitched.vtt"
    pipeline.stitch(tmp_path, output)
    cues = io.read_plain_vtt(output)
    assert [(cue.start, cue.end) for cue in cues] == [
        ("00:00:00.000", "00:00:03.001"),
        ("00:00:00.000", "00:00:01.001"),
        ("00:00:01.001", "00:00:02.001"),
        ("00:00:02.000", "00:00:02.001"),
        ("00:00:02.001", "00:00:03.001"),
    ]
    assert cues[0].raw_text == "[Banner]"
    assert all(core.parse_time(cue.end) > core.parse_time(cue.start) for cue in cues)

    previous = output.read_bytes()
    io.atomic_write_json(
        tmp_path / "subtitle_chunk_001.json",
        [{"id": 0, "start": "1.000", "end": "1.001", "text": "Too late"}],
    )
    with pytest.raises(ValueError, match="non-positive stitched interval"):
        pipeline.stitch(tmp_path, output)
    assert output.read_bytes() == previous

    client = ScriptedGeminiClient(
        [
            StreamCall(
                "chunk",
                ['{"captions":[{"id":0,"start":"1.000","end":"1.001","text":"Late"}]}'],
            ),
            chunk_call("Host: Regenerated."),
        ]
    )
    use_client(monkeypatch, client)
    assert not pipeline.process_chunks(
        "test-key",
        "https://gemini.invalid/proxy",
        tmp_path,
        media.list_chunks(tmp_path),
        1,
        "generation-model",
        "video/mp4",
        "high",
    )
    assert [request.kind for request in client.requests] == ["chunk", "chunk"]
    pipeline.stitch(tmp_path, output)
    assert "Host: Regenerated." in [cue.raw_text for cue in io.read_plain_vtt(output)]


def test_four_pass_pipeline_overlaps_preflight_media_and_chunks_and_publishes(
    config, media_files, monkeypatch
):
    config = replace(
        config,
        video_path=media_files["long"],
        chunk_dur=16,
        thinking_level="low",
        context_urls=("https://example.com/context", "https://youtu.be/example"),
    )
    media_ready, split_started, research_started, youtube_started = (
        Event(),
        Event(),
        Event(),
        Event(),
    )
    chunk_barrier = Barrier(2)
    real_run = subprocess.run

    def record_media(command, **kwargs):
        if command[0] == "ffmpeg" and "-segment_time" in command:
            split_started.set()
            assert research_started.wait(5)
        result = real_run(command, **kwargs)
        if command[0] == "ffmpeg" and str(command[-1]).endswith(".ogg.tmp"):
            assert research_started.is_set()
            media_ready.set()
        return result

    def research(_request):
        research_started.set()
        assert youtube_started.wait(5)
        assert split_started.wait(5)
        assert media_ready.wait(5)

    def youtube(_request):
        youtube_started.set()
        assert research_started.wait(5)

    monkeypatch.setattr(subprocess, "run", record_media)
    client = run(
        monkeypatch,
        config,
        [
            research_call(
                urls=[("https://example.com/context", "URL_RETRIEVAL_STATUS_SUCCESS")],
                on_request=research,
            ),
            StreamCall("youtube", ["Direct identity context"], on_request=youtube),
            *(
                chunk_call(banner_end=15.9, on_request=lambda _r: chunk_barrier.wait(5))
                for _ in range(2)
            ),
            audio_call(
                [
                    {
                        "sourceIds": [1],
                        "start": "00:00:00.100",
                        "end": "00:00:00.500",
                        "text": "HOST: Unauthorized.",
                    },
                    {
                        "sourceIds": [2],
                        "start": "00:00:16.200",
                        "end": "00:00:16.600",
                        "text": "HOST: Repaired.",
                    },
                    {
                        "sourceIds": [],
                        "start": "00:00:30.000",
                        "end": "00:00:31.000",
                        "text": "Outside repair",
                    },
                ]
            ),
            text_call([{"id": 2, "text": "HOST: Corrected."}]),
        ],
    )
    captions = io.read_plain_vtt(config.output_path)
    assert [cue.raw_text for cue in captions] == [
        "[Title]",
        "Host: Hello.",
        "Host: Corrected.",
    ]
    assert core.parse_time(captions[0].end) > 31
    assert captions[2].start == "00:00:16.200"
    assert len(client.requests) == 6
    assert {r.kind for r in client.requests} == {
        "research",
        "youtube",
        "chunk",
        "audio",
        "text",
    }
    for request in client.requests:
        cfg = request.config
        assert_wire_config(request)
        assert request.url.startswith(config.base_url + "/")
        assert request.headers["x-goog-api-key"] == "test-key"
        assert request.http_options.base_url == config.base_url
        assert request.http_options.retry_options == types.HttpRetryOptions()
        assert cfg.max_output_tokens == 65536
        assert cfg.thinking_config.include_thoughts is True
        assert cfg.thinking_config.thinking_level == (
            types.ThinkingLevel.LOW
            if request.kind == "chunk"
            else types.ThinkingLevel.HIGH
        )
        assert bool(cfg.tools) is (request.kind == "research")
    research_request = next(r for r in client.requests if r.kind == "research")
    assert any(tool.google_search is not None for tool in research_request.config.tools)
    assert any(tool.url_context is not None for tool in research_request.config.tools)
    youtube_request = next(r for r in client.requests if r.kind == "youtube")
    assert youtube_request.contents[0].file_data.file_uri == "https://youtu.be/example"
    assert [path.name for path in workdirs()[0].iterdir()] == [pipeline.LOCK_NAME]


def test_gemini_4_long_decode_continues_text_and_publishes(config, monkeypatch):
    config = replace(
        config,
        model="MoDeLs/GeMiNi-4-carbon",
        audio_refine_model="GoOgLe/GeMiNi-4-argon",
        refine_model="gemini-4-hydrogen",
        context_urls=("https://youtu.be/example",),
    )
    client = run(
        monkeypatch,
        config,
        [
            *calls(text=False),
            StreamCall("youtube", ["Direct identity context"]),
            StreamCall(
                "text",
                [
                    stream_chunk("not JSON", thought=True),
                    stream_chunk(
                        '{"changes":[{"id":0,',
                        finish_reason="CONTINUATION",
                        continuation_token=b"hop-2-token",
                    ),
                ],
            ),
            StreamCall(
                "text",
                [stream_chunk('"text":"HOST: Corrected."}]}', finish_reason="STOP")],
            ),
        ],
    )
    assert {r.kind for r in client.requests} == {
        "research",
        "youtube",
        "chunk",
        "audio",
        "text",
    }
    for request in client.requests:
        assert request.body["generationConfig"]["maxOutputTokens"] == 1_048_576
        assert_wire_config(request)
    text_requests = [r for r in client.requests if r.kind == "text"]
    assert len(text_requests) == 2
    assert "continuationToken" not in text_requests[0].body
    assert (
        base64.b64decode(text_requests[1].body["continuationToken"]) == b"hop-2-token"
    )
    assert [cue.raw_text for cue in io.read_plain_vtt(config.output_path)] == [
        "Host: Corrected.",
        "Host: Hello.",
    ]
    assert [path.name for path in workdirs()[0].iterdir()] == [pipeline.LOCK_NAME]


def test_preflight_grounding_failures_abort_before_chunks_and_preserve_output(
    config, monkeypatch
):
    config.output_path.write_text("previous", encoding="ascii")
    for scenario, urls, message in (
        (
            [
                StreamCall(
                    "research", error=errors.ServerError(503, {"message": "exhausted"})
                )
                for _ in range(5)
            ],
            (),
            "exhausted",
        ),
        (
            [
                StreamCall(
                    "research",
                    [
                        types.GenerateContentResponse(
                            prompt_feedback=types.GenerateContentResponsePromptFeedback(
                                block_reason="SAFETY"
                            )
                        )
                    ],
                )
            ],
            (),
            "Gemini request blocked: SAFETY",
        ),
        (
            [StreamCall("research", [types.GenerateContentResponse()])],
            (),
            "Gemini stream returned no candidates",
        ),
        ([research_call(grounding=False)], (), "no Google Search grounding"),
        ([research_call()], ("https://example.com/context",), "was not retrieved"),
        (
            [
                research_call(),
                StreamCall(
                    "youtube",
                    error=errors.ClientError(400, {"message": "video failed"}),
                ),
            ],
            ("https://youtu.be/example",),
            "video failed",
        ),
    ):
        client = ScriptedGeminiClient(scenario)
        use_client(monkeypatch, client)
        with pytest.raises((RuntimeError, errors.APIError), match=message):
            pipeline.run_generation(replace(config, context_urls=urls))
        assert not client.pending
        assert not any(r.kind == "chunk" for r in client.requests)
        assert config.output_path.read_text() == "previous"
        assert list(workdirs()[0].glob("chunk_*.mp4"))


def test_refinement_toggle_matrix_and_programmatic_model_fallback(config, monkeypatch):
    for audio, text in ((False, False), (False, True), (True, False), (True, True)):
        selected = replace(config, audio_refine=audio, refine_text=text)
        scenario = [
            research_call(),
            chunk_call(),
            chunk_call(),
            *(
                [
                    audio_call(
                        [
                            {
                                "sourceIds": [0],
                                "start": "00:00:00.100",
                                "end": "00:00:00.500",
                                "text": "HOST: Repaired.",
                            }
                        ]
                    )
                ]
                if audio
                else []
            ),
            *([text_call()] if text else []),
        ]
        client = run(monkeypatch, selected, scenario)
        assert {r.kind for r in client.requests} == {
            "research",
            "chunk",
            *({"audio"} if audio else set()),
            *({"text"} if text else set()),
        }
        assert {r.model for r in client.requests} == {"generation-model"}
        for request in client.requests:
            if request.kind == "text":
                parts = request.body["contents"][0]["parts"]
                assert any("inlineData" in part for part in parts) is audio
                assert_wire_config(request)
        assert [cue.raw_text for cue in io.read_plain_vtt(config.output_path)] == [
            ("Host: " if text else "HOST: ") + ("Repaired." if audio else "Hello."),
            "Host: Hello." if text else "HOST: Hello.",
        ]


def test_single_chunk_skips_audio_request_but_runs_text_refinement(config, monkeypatch):
    client = run(
        monkeypatch,
        replace(config, chunk_dur=60),
        [
            StreamCall(
                "research",
                [
                    stream_chunk(
                        "**PARTICIPANTS AND SPEAKERS:**\nHost: Presenter\n"
                        "Car, the Garden: Singer\n"
                        "## TOPIC TERMINOLOGY AND PROPER NOUNS:\nOpus: Audio codec",
                        grounding=True,
                    )
                ],
            ),
            chunk_call("CAR, THE GARDEN: Hello.\nOPUS: Codec."),
            text_call(),
        ],
    )
    assert [r.kind for r in client.requests] == ["research", "chunk", "text"]
    assert (
        client.requests[-1].contents[0].inline_data.mime_type == media.AUDIO_MIME_TYPE
    )
    assert [cue.raw_text for cue in io.read_plain_vtt(config.output_path)] == [
        "Car, the Garden: Hello.\nOPUS: Codec."
    ]


@pytest.mark.parametrize(
    "retry_conflict", [False, True], ids=["parse", "delete-conflict"]
)
def test_audio_text_refinement_retains_deletes_and_edits_across_surgical_retries(
    config, monkeypatch, retry_conflict
):
    texts = [
        "HOST: First.",
        "HOST: First.",
        "HOST: Secnd.",
        "HOST: Third.",
        "HOST: Third.",
        "[?????]",
        "HOST: Fourth.",
    ]
    client = run(
        monkeypatch,
        replace(config, chunk_dur=60),
        [
            research_call(),
            StreamCall(
                "chunk",
                [
                    json.dumps(
                        {
                            "captions": [
                                {
                                    "id": i,
                                    "start": str(i / 10),
                                    "end": str((i + 1) / 10),
                                    "text": text,
                                }
                                for i, text in enumerate(texts)
                            ]
                        }
                    )
                ],
            ),
            StreamCall(
                "text",
                [
                    json.dumps(
                        {
                            "changes": [
                                {"id": 2, "text": "HOST: Second."},
                                {"id": 4, "text": "Missing label"},
                                {"id": 6, "text": "Missing label"},
                            ],
                            "deletedIds": [0, 3, 5],
                        }
                    )
                ],
            ),
            StreamCall(
                "text",
                [
                    json.dumps(
                        {
                            "changes": [{"id": 6, "text": "HOST: Fourth!"}],
                            "deletedIds": [4],
                        }
                    )
                    if retry_conflict
                    else "not JSON"
                ],
            ),
            StreamCall(
                "text",
                [
                    json.dumps(
                        {
                            "changes": [
                                {"id": 2, "text": "HOST: Stray overwrite."},
                                {"id": 6, "text": "HOST: Fourth!"},
                            ],
                            "deletedIds": [],
                        }
                    )
                ],
            ),
        ],
    )
    captions = io.read_plain_vtt(config.output_path)
    assert [cue.raw_text for cue in captions] == [
        "Host: First.",
        "Host: Second.",
        "Host: Third.",
        "[?????]",
        "Host: Fourth!",
    ]
    assert [cue.start for cue in captions] == [
        core.format_time(i / 10) for i in (1, 2, 4, 5, 6)
    ]
    requests = [request for request in client.requests if request.kind == "text"]
    assert len(requests) == 3
    for request in requests:
        assert_wire_config(request)
        assert request.contents[0] == requests[0].contents[0]
        assert request.contents[0].inline_data.mime_type == media.AUDIO_MIME_TYPE
        assert request.contents[1].startswith(requests[0].contents[1])
        assert (
            "deletedIds"
            in request.body["generationConfig"]["responseSchema"]["properties"]
        )


def test_text_only_refinement_retries_deletes_without_attaching_audio(
    config, monkeypatch
):
    source = config.output_path.with_name("input.vtt")
    source.write_text(
        "WEBVTT\n\n00:00:00.000 --> 00:00:00.500\nHost: Hello.\n\n"
        "00:00:00.500 --> 00:00:01.000\nHost: Hello.\n",
        encoding="ascii",
    )
    client = ScriptedGeminiClient(
        [
            StreamCall("text", ['{"changes": [], "deletedIds": [0]}']),
            text_call(),
        ]
    )
    use_client(monkeypatch, client)
    benchmark.run_text_refinement(
        config,
        "text-model",
        source,
        config.output_path,
        "Source",
        [],
        core.PreflightContext(),
    )
    assert not client.pending
    assert [cue.raw_text for cue in io.read_plain_vtt(config.output_path)] == [
        "Host: Hello.",
        "Host: Hello.",
    ]
    for request in client.requests:
        assert_wire_config(request)
        assert all(
            "inlineData" not in part
            for content in request.body["contents"]
            for part in content["parts"]
        )


def test_silent_video_fails_before_api_when_audio_refinement_is_enabled(
    config, media_files, monkeypatch
):
    client = ScriptedGeminiClient()
    use_client(monkeypatch, client)
    with pytest.raises(RuntimeError, match="audio stream"):
        pipeline.run_generation(replace(config, video_path=media_files["silent"]))
    assert client.requests == []
    run(
        monkeypatch,
        replace(
            config,
            video_path=media_files["silent"],
            audio_refine=False,
            refine_text=False,
        ),
        calls(audio=False, text=False),
    )


def test_partial_chunk_failure_keeps_valid_results_for_resume(config, monkeypatch):
    client = ScriptedGeminiClient(
        [
            research_call(),
            chunk_call(),
            StreamCall(
                "chunk", error=errors.ClientError(403, {"message": "forbidden"})
            ),
        ]
    )
    use_client(monkeypatch, client)
    config.output_path.write_text("previous", encoding="ascii")
    with pytest.raises(errors.ClientError):
        pipeline.run_generation(config)
    work = workdirs()[0]
    assert len(list(work.glob("subtitle_chunk_*.json"))) == 1
    assert config.output_path.read_text() == "previous"
    before = {path.name: path.stat().st_mtime_ns for path in work.glob("chunk_*.mp4")}

    def check_reused_media(_request):
        assert {
            path.name: path.stat().st_mtime_ns for path in work.glob("chunk_*.mp4")
        } == before

    resumed = run(
        monkeypatch,
        config,
        [
            chunk_call(on_request=check_reused_media),
            audio_call(),
            text_call(),
        ],
    )
    assert [r.kind for r in resumed.requests].count("chunk") == 1
    assert len(io.read_plain_vtt(config.output_path)) == 2


def test_text_failure_retains_preflight_chunk_and_audio_caches_for_resume(
    config, monkeypatch
):
    config.output_path.write_text("previous", encoding="ascii")
    scenario = calls(text=False) + [
        StreamCall(
            "text", [stream_chunk('{"changes":[]}', finish_reason="MAX_TOKENS")]
        ),
    ]
    client = ScriptedGeminiClient(scenario)
    use_client(monkeypatch, client)
    with pytest.raises(RuntimeError, match="MAX_TOKENS"):
        pipeline.run_generation(config)
    work = workdirs()[0]
    assert (work / "audio_refinement.json").is_file()
    assert (work / "preflight_context.json").is_file()
    assert config.output_path.read_text() == "previous"
    (work / "segments.csv").write_text("damaged split index\n", encoding="ascii")
    first_cleanup, second_cleanup = Event(), Event()
    real_unlink = Path.unlink

    def overlapping_cleanup(path, *args, **kwargs):
        if path == work / "subtitle_chunk_000.json":
            if not first_cleanup.is_set():
                first_cleanup.set()
                assert second_cleanup.wait(5)
            else:
                result = real_unlink(path, *args, **kwargs)
                second_cleanup.set()
                return result
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", overlapping_cleanup)
    url = "https://example.com/updated"
    resumed = run(
        monkeypatch,
        replace(config, context_urls=(url,)),
        [
            research_call(urls=[(url, "URL_RETRIEVAL_STATUS_SUCCESS")]),
            chunk_call("Host: Updated."),
            chunk_call("Host: Updated."),
            audio_call(),
            text_call(),
        ],
    )
    assert [r.kind for r in resumed.requests] == [
        "research",
        "chunk",
        "chunk",
        "audio",
        "text",
    ]
    assert [cue.raw_text for cue in io.read_plain_vtt(config.output_path)] == [
        "Host: Updated.",
        "Host: Updated.",
    ]
    assert second_cleanup.is_set()
    assert [p.name for p in work.iterdir()] == [pipeline.LOCK_NAME]


def test_work_isolation_tracks_source_and_chunk_options_not_refinement_options(
    config, monkeypatch
):
    def retain(selected, scenario):
        client = ScriptedGeminiClient(
            [
                *scenario,
                StreamCall(
                    "text", error=errors.ClientError(400, {"message": "retain"})
                ),
            ]
        )
        use_client(monkeypatch, client)
        with pytest.raises(errors.ClientError, match="retain"):
            pipeline.run_generation(selected)
        return client

    retain(config, calls(text=False))
    original = workdirs()
    client = retain(
        replace(
            config,
            workers=1,
            refine_model="other-text",
            output_path=config.output_path.with_name("other.vtt"),
        ),
        [],
    )
    assert [r.kind for r in client.requests] == ["text"]
    assert workdirs() == original
    for selected in (
        replace(config, model="other-generation"),
        replace(config, thinking_level="medium"),
        replace(config, chunk_dur=2),
    ):
        before = len(workdirs())
        retain(selected, calls(chunks=1 if selected.chunk_dur == 2 else 2, text=False))
        assert len(workdirs()) == before + 1
    with config.video_path.open("ab") as source:
        source.write(b"changed-content")
    before = len(workdirs())
    retain(config, calls(text=False))
    assert len(workdirs()) == before + 1


@pytest.mark.parametrize(
    "cache",
    ["preflight_context.json", "subtitle_chunk_000.json", "audio_refinement.json"],
)
def test_corrupt_persisted_cache_regenerates_only_its_capability(
    config, monkeypatch, cache
):
    client = ScriptedGeminiClient(
        [
            *calls(text=False),
            StreamCall("text", error=errors.ClientError(400, {"message": "retain"})),
        ]
    )
    use_client(monkeypatch, client)
    with pytest.raises(errors.ClientError, match="retain"):
        pipeline.run_generation(config)
    (workdirs()[0] / cache).write_text("corrupt", encoding="ascii")
    replacement = {
        "preflight_context.json": research_call,
        "subtitle_chunk_000.json": chunk_call,
        "audio_refinement.json": audio_call,
    }[cache]()
    client = run(monkeypatch, config, [replacement, text_call()])
    assert [r.kind for r in client.requests] == [replacement.kind, "text"]


def test_lock_stays_held_until_preflight_settles_after_media_failure(
    config, monkeypatch
):
    started, release = Event(), Event()

    def blocked(_request):
        started.set()
        assert release.wait(5)

    client = ScriptedGeminiClient([research_call(on_request=blocked)])
    use_client(monkeypatch, client)
    real_run = subprocess.run
    media_failed = Event()

    def fail_split(command, **kwargs):
        if command[0] == "ffmpeg":
            assert started.wait(5)
            config.video_path.write_bytes(b"damaged after inspection")
            try:
                return real_run(command, **kwargs)
            finally:
                media_failed.set()
        return real_run(command, **kwargs)

    monkeypatch.setattr(subprocess, "run", fail_split)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(pipeline.run_generation, config)
        try:
            assert media_failed.wait(5)
            work = workdirs()[0]
            with pytest.raises(RuntimeError, match="already using"):
                pipeline.acquire_lock(work)
            assert not future.done()
        finally:
            release.set()
        with pytest.raises(subprocess.CalledProcessError):
            future.result(timeout=5)
    lock = pipeline.acquire_lock(work)
    pipeline.release_lock(lock)
    assert (work / "preflight_context.json").is_file()


def test_stream_and_validation_retries_discard_partial_answers(
    config, monkeypatch, capsys
):
    selected = replace(config, workers=1, audio_refine=False, refine_text=False)

    def check_no_partial_cache(_request):
        assert not (workdirs()[0] / "preflight_context.json").exists()

    def check_completed_research(_request):
        saved = (workdirs()[0] / "preflight_context.json").read_text(encoding="utf-8")
        assert "Partial Speaker" not in saved
        assert "Host: Presenter" in saved

    client = run(
        monkeypatch,
        selected,
        [
            StreamCall(
                "research",
                [
                    stream_chunk(
                        "Partial Speaker: Incomplete", grounding="incomplete query"
                    ),
                    stream_chunk("still thinking", thought=True),
                ],
                complete=False,
            ),
            research_call(on_request=check_no_partial_cache),
            StreamCall(
                "chunk",
                [errors.ServerError(503, {"message": "SSE error"})],
                on_request=check_completed_research,
            ),
            StreamCall("chunk", ['{"captions":', httpx2.TransportError("interrupted")]),
            StreamCall(
                "chunk", ['{"captions":[{"id":0,"start":"1","end":"0","text":"Bad"}]}']
            ),
            chunk_call("Host: Too\nmany\nlines"),
            chunk_call(),
            chunk_call(),
        ],
    )
    assert [r.kind for r in client.requests[:2]] == ["research", "research"]
    assert [r.kind for r in client.requests].count("chunk") == 6
    out = capsys.readouterr().out
    assert "incomplete query" not in out
    assert "speaker identities" in out
    assert [cue.raw_text for cue in io.read_plain_vtt(config.output_path)] == [
        "HOST: Hello.",
        "HOST: Hello.",
    ]


@pytest.mark.parametrize("exhaust", [False, True], ids=["corrected", "exhausted"])
@pytest.mark.parametrize(
    "parse_first", [True, False], ids=["parse-first", "parse-retry"]
)
def test_text_validation_retries_report_all_invalid_cues(
    config, monkeypatch, capsys, exhaust, parse_first
):
    selected = replace(config, chunk_dur=60, audio_refine=False)
    sources = [
        "Host: This is",
        "Host: a test.",
        "and a continuation.",
        "without a label.",
        "Host: The end.",
    ]
    invalid = [
        {"id": 1, "text": "the test."},
        {"id": 2, "text": "Host: and a continuation."},
        {"id": 3, "text": "Host: without a label."},
        {"id": 4, "text": "The ending."},
    ]
    corrected = [
        {"id": 0, "text": "Host: This is indeed"},
        {"id": 1, "text": "Host: the test."},
        {"id": 4, "text": "Host: The ending."},
    ]
    client = ScriptedGeminiClient(
        [
            research_call(),
            StreamCall(
                "chunk",
                [
                    json.dumps(
                        {
                            "captions": [
                                {
                                    "id": index,
                                    "start": str(index * 0.3),
                                    "end": str((index + 1) * 0.3),
                                    "text": text,
                                }
                                for index, text in enumerate(sources)
                            ]
                        }
                    )
                ],
            ),
            StreamCall("text", ['{"changes":'])
            if parse_first
            else text_call([corrected[0], *invalid]),
            text_call([corrected[0], *invalid])
            if parse_first
            else StreamCall("text", ['{"changes":']),
            text_call(
                invalid
                if exhaust
                else [
                    *corrected[1:],
                    {"id": 0, "text": "Host: Stray 1."},
                    {"id": 0, "text": "Host: Stray 2."},
                ]
            ),
        ]
    )
    use_client(monkeypatch, client)
    config.output_path.write_text("previous", encoding="ascii")
    if exhaust:
        with pytest.raises(
            RuntimeError,
            match="Parsing or validating the model refinement response failed:",
        ) as failure:
            pipeline.run_generation(selected)
        for change in invalid:
            assert f"subtitle ID {change['id']}" in str(failure.value)
        assert config.output_path.read_text() == "previous"
        assert (workdirs()[0] / "subtitle_chunk_000.json").is_file()
        assert (workdirs()[0] / "preflight_context.json").is_file()
        run_log = (workdirs()[0] / io.LOG_NAME).read_text(encoding="utf-8")
        for expected in (
            "Warning: Global refinement attempt 1 failed",
            "Warning: Global refinement attempt 2 failed",
            "ERROR:",
            "Traceback (most recent call last):",
            "Raw response:\n",
            '{"changes":',
            json.dumps({"changes": [corrected[0], *invalid]}),
        ):
            assert expected in run_log
        assert "Global refinement attempt 3 failed:" not in run_log
    else:
        pipeline.run_generation(selected)
        captions = io.read_plain_vtt(config.output_path)
        assert [cue.raw_text for cue in captions] == [
            corrected[0]["text"],
            corrected[1]["text"],
            sources[2],
            sources[3],
            corrected[2]["text"],
        ]
        assert [(cue.start, cue.end) for cue in captions] == [
            (core.format_time(index * 0.3), core.format_time((index + 1) * 0.3))
            for index in range(len(sources))
        ]
        assert {path.name for path in workdirs()[0].iterdir()} == {pipeline.LOCK_NAME}
    assert not client.pending
    requests = [request for request in client.requests if request.kind == "text"]
    assert len(requests) == 3
    assert "json_invalid" in requests[1 if parse_first else 2].contents
    assert all(
        request.contents.startswith(requests[0].contents) for request in requests
    )
    feedback = requests[2 if parse_first else 1].contents
    for change in invalid:
        cue_id = change["id"]
        assert f"subtitle ID {cue_id} changes dialogue turns or labels" in feedback
        assert f"Source text: {sources[cue_id]!r}" in feedback
        assert f"Proposed text: {change['text']!r}" in feedback
    for request in requests:
        assert_wire_config(request)
    console = capsys.readouterr().out
    assert "retrying in 1s" in console
    if not exhaust:
        assert "Raw response:" not in console
        assert "Model proposed changes to 3 lines out of 5." in console


def test_audio_validation_failure_preserves_existing_output_and_no_valid_cache(
    config, monkeypatch
):
    config.output_path.write_text("previous", encoding="ascii")
    client = ScriptedGeminiClient(
        [
            research_call(),
            chunk_call(),
            chunk_call(),
            *[
                audio_call(
                    [
                        {
                            "sourceIds": [99],
                            "start": "00:00:00.000",
                            "end": "00:00:01.000",
                            "text": "Unknown",
                        }
                    ]
                )
            ]
            * 3,
        ]
    )
    use_client(monkeypatch, client)
    with pytest.raises(RuntimeError, match="unknown source IDs"):
        pipeline.run_generation(config)
    assert config.output_path.read_text() == "previous"
    assert not (workdirs()[0] / "audio_refinement.json").exists()
    run_log = (workdirs()[0] / io.LOG_NAME).read_text(encoding="utf-8")
    assert "unknown source IDs" in run_log
    assert "Raw response:" in run_log
    assert not any(r.kind == "text" for r in client.requests)


def test_audio_context_cache_and_validation_retries(config, monkeypatch):
    work = config.output_path.parent
    audio_path, duration, source_duration, _ = media.extract_complete_audio(
        config.video_path, work
    )
    stitched = work / "stitched.vtt"
    stitched.write_text(
        "WEBVTT\n\n00:00:00.100 --> 00:00:00.500\nHost: Opus.\n",
        encoding="ascii",
    )
    client = ScriptedGeminiClient(
        [
            StreamCall("audio", ["not JSON"]),
            audio_call(deleted=[99]),
            audio_call(),
            audio_call(),
            audio_call(),
        ]
    )
    use_client(monkeypatch, client)
    for names, terms, expected_cached in (
        (["Host"], "Opus", False),
        (["Host"], "Opus", True),
        (["Presenter"], "Opus", False),
        (["Presenter"], "Opus codec", False),
    ):
        _, cached = benchmark.run_audio_refinement(
            config,
            "audio-model",
            stitched,
            config.output_path,
            audio_path,
            duration,
            source_duration,
            [1],
            work,
            "Source",
            grounded_names=names,
            terminology_context=terms,
        )
        assert cached is expected_cached
    assert not client.pending
    assert [cue.raw_text for cue in io.read_plain_vtt(config.output_path)] == [
        "Host: Opus."
    ]
    for request in client.requests:
        assert_wire_config(request)
        assert request.contents[0].inline_data.mime_type == media.AUDIO_MIME_TYPE
    assert client.requests[0].contents[:2] == client.requests[1].contents[:2]
    assert client.requests[1].contents[:2] == client.requests[2].contents[:2]
    assert len(client.requests[1].contents) == 3


def test_benchmark_reuses_per_model_preflight_and_reports_stage_metrics(
    config, monkeypatch
):
    reference = config.output_path.with_name("reference.vtt")
    reference.write_text(
        "WEBVTT\n\n00:00:00.100 --> 00:00:00.500\nHello.\n\n"
        "00:00:01.100 --> 00:00:01.500\nHello.\n",
        encoding="ascii",
    )
    output = config.output_path.parent / "benchmark"
    long_prefix = "projects/" + "long-project-" * 8 + "/locations/global/models/"
    gen_model, audio_model, text_model = (
        long_prefix + suffix for suffix in ("gen-a", "audio", "text")
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark.py",
            str(config.video_path),
            "--api-key",
            "test",
            "--chunk-dur",
            "1",
            "--case",
            f"{gen_model}:{audio_model}:{text_model}",
            "--model",
            "gen-b",
            "--output-dir",
            str(output),
            "--reference-vtt",
            str(reference),
        ],
    )
    args = benchmark.parse_args()
    invalid_output = output.with_name("invalid-benchmark")
    for overrides, message in (
        ({"video_file": "absent.mp4"}, "Source video not found"),
        ({"reference_vtt": Path("absent.vtt")}, "Reference VTT not found"),
        ({"api_key": None}, "API key not configured"),
        ({"model": [""]}, "model names must not be empty"),
        ({"case": [("gen", " ", "text")]}, "model names must not be empty"),
        ({"case": [("gen", "audio", " ")]}, "model names must not be empty"),
        ({"workers": 0}, "--workers"),
        ({"chunk_dur": 0}, "--chunk-dur"),
        ({"thinking_level": "invalid"}, "--thinking-level"),
        ({"context_url": ["invalid"]}, "--context-url"),
    ):
        invalid_args = type(args)(
            **{**vars(args), "output_dir": invalid_output, **overrides}
        )
        with pytest.raises((ValueError, RuntimeError), match=message):
            benchmark.run_full_matrix(invalid_args)
        assert not invalid_output.exists()
    client = ScriptedGeminiClient(
        [
            research_call(),
            research_call(),
            *(chunk_call() for _ in range(4)),
            audio_call(),
            text_call(),
        ]
    )
    use_client(monkeypatch, client)
    benchmark.run_full_matrix(benchmark.parse_args())
    assert not client.pending
    assert {r.model for r in client.requests if r.kind == "research"} == {
        gen_model,
        "gen-b",
    }
    assert len(list(output.glob("*.vtt"))) == 4
    text_request = next(
        request for request in client.requests if request.kind == "text"
    )
    assert text_request.contents[0].inline_data.mime_type == media.AUDIO_MIME_TYPE
    assert_wire_config(text_request)
    results = json.loads((output / "benchmark-results.json").read_text())
    assert len(results) == 2
    for result in results:
        assert result["preflight_seconds"] >= 0
        assert result["preflight_cached"] is False
        assert result["generation_cached"] is False
        assert result["audio_refine_cached"] is (result["audio_refine_model"] is None)
        assert result["comparison"]["text_similarity"] == 1
        assert result["comparison"]["temporal_iou"] > 0.8
    resumed = ScriptedGeminiClient([text_call()])
    use_client(monkeypatch, resumed)
    benchmark.run_full_matrix(benchmark.parse_args())
    assert [r.kind for r in resumed.requests] == ["text"]
    results = json.loads((output / "benchmark-results.json").read_text())
    assert all(result["generation_cached"] for result in results)
    assert all(result["preflight_cached"] for result in results)
    assert all(result["audio_refine_cached"] for result in results)
    url = "https://example.com/updated"
    monkeypatch.setattr(sys, "argv", [*sys.argv, "--context-url", url])
    updated = ScriptedGeminiClient(
        [
            *(
                research_call(urls=[(url, "URL_RETRIEVAL_STATUS_SUCCESS")])
                for _ in range(2)
            ),
            *(chunk_call("HOST: Updated.") for _ in range(4)),
            audio_call(),
            text_call(),
        ]
    )
    use_client(monkeypatch, updated)
    benchmark.run_full_matrix(benchmark.parse_args())
    assert not updated.pending
    assert [r.kind for r in updated.requests].count("chunk") == 4
    for path in output.glob("*.generated.vtt"):
        assert {cue.raw_text for cue in io.read_plain_vtt(path)} == {"HOST: Updated."}
    results = json.loads((output / "benchmark-results.json").read_text())
    assert all(not result["generation_cached"] for result in results)
    assert all(not result["preflight_cached"] for result in results)
    assert results[0]["audio_refine_cached"] is False

    single = type(args)(
        **{
            **vars(args),
            "chunk_dur": 60,
            "output_dir": output.with_name("single-benchmark"),
        }
    )
    single_client = ScriptedGeminiClient(
        [research_call(), research_call(), chunk_call(), chunk_call(), text_call()]
    )
    use_client(monkeypatch, single_client)
    benchmark.run_full_matrix(single)
    assert not single_client.pending
    assert not any(request.kind == "audio" for request in single_client.requests)
    assert (
        single_client.requests[-1].contents[0].inline_data.mime_type
        == media.AUDIO_MIME_TYPE
    )
    generation_cache = next(
        path
        for path in (output / "work").rglob("subtitle_chunk_000.json")
        if "Updated." in path.read_text()
    )
    generation_cache.write_text("corrupt", encoding="ascii")
    for path in (output / "work").rglob("audio_refinement.json"):
        path.write_text("corrupt", encoding="ascii")
    repaired = ScriptedGeminiClient(
        [chunk_call("HOST: Updated."), audio_call(), text_call()]
    )
    use_client(monkeypatch, repaired)
    benchmark.run_full_matrix(benchmark.parse_args())
    assert not repaired.pending
    results = json.loads((output / "benchmark-results.json").read_text())
    assert sum(result["generation_cached"] for result in results) == 1
    assert results[0]["audio_refine_cached"] is False
