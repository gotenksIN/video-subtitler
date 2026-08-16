import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import webvtt
from google.genai import types
from pydantic import ValidationError

import gemini_subs


def make_caption(caption_id, start, end, text="Text"):
    return gemini_subs.Caption(id=caption_id, start=start, end=end, text=text)


def make_manifest(overlap=0.0, codec="h264"):
    ext = ".webm" if codec == "vp9" else ".mp4"
    mime = "video/webm" if ext == ".webm" else "video/mp4"
    return {
        "mode": "generate",
        "overlap": overlap,
        "chunk_ext": ext,
        "chunk_mime": mime,
        "process_ext": ext,
        "process_mime": mime,
        "video_codec": codec,
    }


def write_layout(directory, rows, overlap=0.0):
    Path(directory, gemini_subs.MANIFEST_NAME).write_text(
        json.dumps(make_manifest(overlap)), encoding="utf-8"
    )
    Path(directory, "segments.csv").write_text(
        "".join(f"{name},{start},{end}\n" for name, start, end in rows),
        encoding="utf-8",
    )


def write_subtitles(directory, index, captions):
    Path(directory, f"subtitle_chunk_{index:03d}.json").write_text(
        json.dumps(captions), encoding="utf-8"
    )


def write_vtt(path, captions):
    value = webvtt.WebVTT()
    value.captions.extend(
        webvtt.Caption(start, end, text) for start, end, text in captions
    )
    value.save(path)


class ImmediateFuture:
    def __init__(self, function, args):
        try:
            self.value = function(*args)
            self.error = None
        except Exception as error:  # noqa: BLE001 - Emulate Future exception capture.
            self.value = None
            self.error = error

    def result(self):
        if self.error is not None:
            raise self.error
        return self.value


class ImmediateExecutor:
    def __init__(self, max_workers):
        # ThreadPoolExecutor takes max_workers. Immediate execution does not need it.
        del max_workers

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def submit(self, function, *args):
        return ImmediateFuture(function, args)


class FakeGeminiClient:
    """Fake SDK boundary that consumes each request and enforces the documented contract."""

    def __init__(
        self,
        model_name,
        response_pieces,
        verify_contents,
        verify_config=None,
        candidate_metadata=None,
    ):
        self.model_name = model_name
        self.response_pieces = response_pieces
        self.verify_contents = verify_contents
        self.verify_config = verify_config
        self.candidate_metadata = candidate_metadata or []

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    @property
    def models(self):
        return self

    def generate_content_stream(self, **kwargs):
        if kwargs["model"] != self.model_name:
            raise AssertionError(
                f"request must use model {self.model_name!r}, got {kwargs['model']!r}"
            )
        config = kwargs["config"]
        if config.response_mime_type != "application/json":
            raise AssertionError("request config must require application/json")
        # Consume the response schema like the SDK structured-output boundary does.
        config.response_schema.model_validate_json("".join(self.response_pieces))
        self.verify_contents(kwargs["contents"])
        if self.verify_config:
            self.verify_config(config)
        return iter(
            SimpleNamespace(
                text=piece,
                candidates=(
                    self.candidate_metadata[index]
                    if index < len(self.candidate_metadata)
                    else None
                ),
            )
            for index, piece in enumerate(self.response_pieces)
        )


def search_candidate(queries=(), sources=(), urls=()):
    """Build one stream candidate carrying grounding and URL retrieval metadata."""
    grounding = None
    if queries or sources:
        grounding = types.GroundingMetadata(
            web_search_queries=list(queries),
            grounding_chunks=[
                types.GroundingChunk(web=types.GroundingChunkWeb(title=title, uri=uri))
                for title, uri in sources
            ],
        )
    url_context = None
    if urls:
        url_context = types.UrlContextMetadata(
            url_metadata=[
                types.UrlMetadata(retrieved_url=url, url_retrieval_status=status)
                for url, status in urls
            ]
        )
    return [
        types.Candidate(grounding_metadata=grounding, url_context_metadata=url_context)
    ]


def research_call(
    pieces=("Research text",),
    queries=(),
    sources=(),
    urls=(),
    verify_contents=None,
    verify_config=None,
):
    """One expected plain-text grounded identity research request."""
    return {
        "structured": False,
        "pieces": list(pieces),
        "candidates": [search_candidate(queries, sources, urls)],
        "verify_contents": verify_contents,
        "verify_config": verify_config,
    }


def youtube_call(pieces=("YouTube analysis",), verify_contents=None, error=None):
    """One expected plain-text direct YouTube analysis request."""
    return {
        "youtube": True,
        "pieces": list(pieces),
        "candidates": [],
        "verify_contents": verify_contents,
        "error": error,
    }


def refinement_call(pieces, verify_contents=None, verify_config=None):
    """One expected structured refinement request."""
    return {
        "structured": True,
        "pieces": list(pieces),
        "candidates": [],
        "verify_contents": verify_contents,
        "verify_config": verify_config,
    }


class FakeRefinementClient:
    """Fake SDK boundary for the three-request refinement flow.

    The first request must be plain text with the Google Search tool and no
    video Parts. An optional second request analyzes YouTube videos with
    plain text, video Parts, and no tools. The final request must be
    structured JSON without any tool.
    """

    def __init__(self, model_name, calls):
        self.model_name = model_name
        self.calls = list(calls)

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    @property
    def models(self):
        return self

    @property
    def pending_calls(self):
        return len(self.calls)

    def generate_content_stream(self, **kwargs):
        if kwargs["model"] != self.model_name:
            raise AssertionError(
                f"request must use model {self.model_name!r}, got {kwargs['model']!r}"
            )
        if not self.calls:
            raise AssertionError(
                "the refinement pipeline issued an unexpected API request"
            )
        call = self.calls.pop(0)
        if call.get("error"):
            raise call["error"]
        config = kwargs["config"]
        tools = config.tools or []
        if call.get("structured"):
            if config.response_mime_type != "application/json":
                raise AssertionError("structured request must require application/json")
            if config.response_schema is not gemini_subs.RefinementResponse:
                raise AssertionError(
                    "structured request must use the refinement schema"
                )
            if tools:
                raise AssertionError("structured refinement must not enable tools")
            # Consume the response schema like the SDK structured-output boundary does.
            config.response_schema.model_validate_json("".join(call["pieces"]))
        elif call.get("youtube"):
            if config.response_mime_type is not None:
                raise AssertionError("YouTube request must use plain text output")
            if config.response_schema is not None:
                raise AssertionError("YouTube request must not use a response schema")
            if tools:
                raise AssertionError("YouTube request must not enable tools")
        else:
            if config.response_mime_type is not None:
                raise AssertionError("research request must use plain text output")
            if config.response_schema is not None:
                raise AssertionError("research request must not use a response schema")
            if not any(tool.google_search is not None for tool in tools):
                raise AssertionError("research request must enable Google Search")
        if call.get("verify_config"):
            call["verify_config"](config)
        if call.get("verify_contents"):
            call["verify_contents"](kwargs["contents"])
        candidates = call["candidates"]
        return iter(
            SimpleNamespace(
                text=piece,
                candidates=(candidates[index] if index < len(candidates) else None),
            )
            for index, piece in enumerate(call["pieces"])
        )


@pytest.fixture
def immediate_execution(monkeypatch):
    monkeypatch.setattr(
        gemini_subs.concurrent.futures, "ThreadPoolExecutor", ImmediateExecutor
    )
    monkeypatch.setattr(
        gemini_subs.concurrent.futures, "as_completed", lambda futures: list(futures)
    )


def test_atomic_json_uses_atomic_replacement_and_leaves_no_temporary_files(
    tmp_path, monkeypatch
):
    target = tmp_path / "captions.json"
    target.write_text("old", encoding="utf-8")
    replacements = []
    real_replace = os.replace

    def replace(source, destination):
        replacements.append((Path(source), Path(destination)))
        real_replace(source, destination)

    monkeypatch.setattr(gemini_subs.os, "replace", replace)

    gemini_subs.atomic_write_json(target, {"text": "plain ascii"})

    assert json.loads(target.read_text(encoding="utf-8")) == {"text": "plain ascii"}
    assert replacements == [(Path(f"{target}.tmp"), target)]
    assert sorted(path.name for path in tmp_path.iterdir()) == ["captions.json"]


def test_atomic_vtt_uses_unique_atomic_temporary_files_in_output_directory(
    tmp_path, monkeypatch
):
    output = tmp_path / "output.vtt"
    sources = []
    real_replace = os.replace

    def replace(source, destination):
        sources.append(Path(source))
        real_replace(source, destination)

    monkeypatch.setattr(gemini_subs.os, "replace", replace)
    value = webvtt.WebVTT()
    value.captions.extend(
        [
            webvtt.Caption("00:00:00.000", "00:00:01.000", "One"),
            webvtt.Caption("00:00:01.000", "00:00:02.000", "Two"),
        ]
    )

    gemini_subs.atomic_save_vtt(value, output)
    gemini_subs.atomic_save_vtt(value, output)

    result = webvtt.read(output)
    assert [caption.text for caption in result] == ["One", "Two"]
    assert sorted(path.name for path in tmp_path.iterdir()) == ["output.vtt"]
    assert len(set(sources)) == 2
    assert all(path.parent == tmp_path for path in sources)
    assert all(path.name.endswith(".tmp.vtt") for path in sources)


@pytest.mark.parametrize("failure", ["save", "replace"])
def test_atomic_vtt_removes_temporary_file_after_failure(
    tmp_path, monkeypatch, failure
):
    output = tmp_path / "output.vtt"
    value = MagicMock()
    if failure == "save":
        value.save.side_effect = OSError("save failed")
    else:
        monkeypatch.setattr(
            gemini_subs.os, "replace", MagicMock(side_effect=OSError("replace failed"))
        )

    with pytest.raises(OSError, match=f"{failure} failed"):
        gemini_subs.atomic_save_vtt(value, output)

    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    ("value", "seconds"),
    [("1.25", 1.25), ("02:03,5", 123.5), ("01:02:03.004", 3723.004)],
)
def test_timestamp_parser_accepts_documented_shapes(value, seconds):
    assert gemini_subs.parse_time(value) == seconds


@pytest.mark.parametrize("value", ["-0.1", "-00:00:00.1", "1:2:3:4"])
def test_timestamp_parser_rejects_negative_or_malformed_values(value):
    with pytest.raises(ValueError):
        gemini_subs.parse_time(value)


@pytest.mark.parametrize(
    ("seconds", "formatted"),
    [(0, "00:00:00.000"), (3661.2346, "01:01:01.235"), (59.9996, "00:01:00.000")],
)
def test_timestamp_formatter_rounds_to_milliseconds(seconds, formatted):
    assert gemini_subs.format_time(seconds) == formatted


def test_timestamp_formatter_rejects_negative_values():
    with pytest.raises(ValueError, match="Negative timestamp"):
        gemini_subs.format_time(-0.001)


def test_caption_validation_sorts_canonicalizes_and_preserves_overlap():
    result = gemini_subs.validate_captions(
        [
            make_caption(2, "1", "3", "Later"),
            make_caption(1, "00:00:00,250", "2", "Earlier"),
        ],
        5,
    )

    assert result[0] == {
        "id": 1,
        "start": "00:00:00.250",
        "end": "00:00:02.000",
        "text": "Earlier",
    }
    assert result[1] == {
        "id": 2,
        "start": "00:00:01.000",
        "end": "00:00:03.000",
        "text": "Later",
    }


def test_caption_validation_preserves_cues_with_the_same_start():
    result = gemini_subs.validate_captions(
        [make_caption(0, "1", "2"), make_caption(1, "1", "3")], 5
    )

    assert result[0] == {
        "id": 0,
        "start": "00:00:01.000",
        "end": "00:00:02.000",
        "text": "Text",
    }
    assert result[1] == {
        "id": 1,
        "start": "00:00:01.000",
        "end": "00:00:03.000",
        "text": "Text",
    }


@pytest.mark.parametrize(
    "captions",
    [
        [make_caption(0, "0", "1"), make_caption(0, "2", "3")],
        [make_caption(0, "1", "1")],
        [make_caption(0, "9", "10.6")],
        [make_caption(0, "10.2", "10.4")],
        [make_caption(0, "9.9996", "10.4")],
    ],
    ids=[
        "duplicate IDs",
        "non-positive interval",
        "overrun beyond tolerance",
        "clamp makes interval invalid",
        "rounding collapses interval",
    ],
)
def test_caption_validation_rejects_invalid_responses(captions):
    with pytest.raises(ValueError):
        gemini_subs.validate_captions(captions, 10)


def test_caption_validation_clamps_allowed_end_overrun_to_clip_duration():
    result = gemini_subs.validate_captions([make_caption(0, "9", "10.4")], 10)
    assert result[0]["end"] == "00:00:10.000"


@pytest.mark.parametrize(
    ("probe_output", "expected"),
    [
        ("vp9\n", (".webm", "video/webm", "vp9")),
        ("h264\n", (".mp4", "video/mp4", "h264")),
        ("h265\n", (".mp4", "video/mp4", "hevc")),
    ],
)
def test_primary_video_probe_maps_supported_codecs(monkeypatch, probe_output, expected):
    run = MagicMock(
        return_value=subprocess.CompletedProcess([], 0, stdout=probe_output, stderr="")
    )
    monkeypatch.setattr(gemini_subs.subprocess, "run", run)

    assert gemini_subs.probe_video_format("source") == expected
    command = run.call_args.args[0]
    assert command[command.index("-select_streams") + 1] == "v:0"


def test_primary_video_probe_rejects_unsupported_codec(monkeypatch):
    monkeypatch.setattr(
        gemini_subs.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            [], 0, stdout="av1\n", stderr=""
        ),
    )
    with pytest.raises(RuntimeError, match="Video format not supported"):
        gemini_subs.probe_video_format("source")


def test_primary_video_probe_wraps_tool_failure(monkeypatch):
    monkeypatch.setattr(
        gemini_subs.subprocess,
        "run",
        MagicMock(side_effect=FileNotFoundError("ffprobe")),
    )
    with pytest.raises(RuntimeError, match="Failed to probe video format"):
        gemini_subs.probe_video_format("source")


def test_manifest_contains_documented_identity_and_uses_sorted_hash(
    tmp_path, monkeypatch
):
    video = tmp_path / "source.mp4"
    video.write_bytes(b"video")
    config = gemini_subs.GenerationConfig(
        video_path=video,
        output_path=tmp_path / "out.vtt",
        model="chunk-model",
        thinking_level="high",
        chunk_dur=60,
        overlap=5.0,
    )
    monkeypatch.setattr(
        gemini_subs,
        "probe_video_format",
        lambda _path: (".mp4", "video/mp4", "h264"),
    )
    monkeypatch.setattr(gemini_subs, "CHUNK_ROOT", str(tmp_path / "work"))

    manifest, work_dir = gemini_subs.build_manifest(config)
    expected_fields = {
        "video",
        "chunk_dur",
        "format",
        "mode",
        "model",
        "chunk_thinking_level",
        "overlap",
        "chunk_ext",
        "chunk_mime",
        "process_ext",
        "process_mime",
        "video_codec",
    }
    digest = hashlib.sha256(
        json.dumps(manifest, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]

    assert set(manifest) == expected_fields
    assert manifest["video"] == {
        "path": str(video.resolve()),
        "size": 5,
        "mtime_ns": video.stat().st_mtime_ns,
    }
    assert manifest["format"] == "stream-copy-v1"
    assert manifest["mode"] == "generate"
    assert Path(work_dir) == tmp_path / "work" / digest


def test_manifest_identity_changes_with_runtime_input(tmp_path, monkeypatch):
    video = tmp_path / "source.webm"
    video.write_bytes(b"video")

    def make_config(model):
        return gemini_subs.GenerationConfig(
            video_path=video,
            output_path=tmp_path / "out.vtt",
            model=model,
            thinking_level="high",
            chunk_dur=60,
            overlap=0,
        )

    monkeypatch.setattr(
        gemini_subs,
        "probe_video_format",
        lambda _path: (".webm", "video/webm", "vp9"),
    )

    _, first = gemini_subs.build_manifest(make_config("model-a"))
    _, second = gemini_subs.build_manifest(make_config("model-b"))

    assert first != second


def test_segment_index_is_source_of_chunk_timing_and_physical_index(tmp_path):
    (tmp_path / "segments.csv").write_text(
        "ignored\nchunk_001.mp4,1.5,4.25,extra\n", encoding="utf-8"
    )

    assert gemini_subs.list_chunks(tmp_path) == [
        {
            "idx": 1,
            "name": "chunk_001.mp4",
            "start": 1.5,
            "end": 4.25,
            "duration": 2.75,
        }
    ]


def test_completed_split_with_nonempty_chunks_is_reused(tmp_path, monkeypatch):
    write_layout(tmp_path, [("chunk_000.mp4", 0, 2)])
    chunk = tmp_path / "chunk_000.mp4"
    chunk.write_bytes(b"cached chunk")
    marker = tmp_path / gemini_subs.SPLIT_COMPLETE_MARKER
    marker.write_text("ok\n", encoding="utf-8")

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("a valid completed split must not run ffmpeg")

    monkeypatch.setattr(gemini_subs.subprocess, "run", fail_if_called)

    gemini_subs.split_video("source.mp4", tmp_path, 60, make_manifest())

    assert chunk.read_bytes() == b"cached chunk"
    assert marker.read_text(encoding="utf-8") == "ok\n"


def test_invalid_completed_split_is_cleaned_then_recreated(tmp_path, monkeypatch):
    write_layout(
        tmp_path,
        [("chunk_000.mp4", 0, 2), ("chunk_001.mp4", 2, 4)],
    )
    (tmp_path / "chunk_000.mp4").write_bytes(b"only one chunk")
    marker = tmp_path / gemini_subs.SPLIT_COMPLETE_MARKER
    marker.write_text("ok\n", encoding="utf-8")

    def run(command, **_kwargs):
        assert not marker.exists()
        assert not (tmp_path / "chunk_000.mp4").exists()
        assert command[command.index("-map") + 1] == "0:v:0"
        assert "0:a?" in command
        assert "-sn" in command
        for name in ("chunk_000.mp4", "chunk_001.mp4"):
            (tmp_path / name).write_bytes(b"fresh chunk")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(gemini_subs.subprocess, "run", run)
    gemini_subs.split_video("source.mp4", tmp_path, 60, make_manifest())

    assert marker.read_text(encoding="utf-8") == "ok\n"
    assert (tmp_path / "chunk_000.mp4").read_bytes() == b"fresh chunk"
    assert (tmp_path / "chunk_001.mp4").read_bytes() == b"fresh chunk"


def test_failed_split_leaves_no_completion_marker_for_resume(tmp_path, monkeypatch):
    marker = tmp_path / gemini_subs.SPLIT_COMPLETE_MARKER
    marker.write_text("ok\n", encoding="utf-8")
    (tmp_path / "segments.csv").write_text("chunk_000.mp4,0,2\n", encoding="utf-8")
    monkeypatch.setattr(
        gemini_subs.subprocess,
        "run",
        MagicMock(side_effect=subprocess.CalledProcessError(1, "ffmpeg")),
    )

    with pytest.raises(subprocess.CalledProcessError):
        gemini_subs.split_video("source.mp4", tmp_path, 60, make_manifest())

    assert not marker.exists()


def test_processing_windows_clip_context_at_video_edges():
    chunks = [
        {"idx": 0, "name": "chunk_000.mp4", "start": 0.0, "end": 10.0},
        {"idx": 1, "name": "chunk_001.mp4", "start": 10.0, "end": 18.0},
    ]

    windows = gemini_subs.get_processing_windows(chunks, 2.0)

    assert windows[0]["clip_start"] == 0
    assert windows[0]["clip_end"] == 12
    assert windows[1]["clip_start"] == 8
    assert windows[1]["clip_end"] == 18
    assert windows[1]["clip_duration"] == 10
    assert windows[1]["owner_start_rel"] == 2
    assert windows[1]["owner_end_rel"] == 10


@pytest.mark.parametrize(
    ("ext", "codec", "encoder", "audio"),
    [
        (".webm", "vp9", "libvpx-vp9", "libopus"),
        (".mp4", "h264", "libx264", "aac"),
        (".mp4", "hevc", "libx265", "aac"),
    ],
)
def test_overlap_codec_configuration_matches_primary_codec(ext, codec, encoder, audio):
    arguments = gemini_subs.overlap_codec_args(ext, codec)
    assert encoder in arguments
    assert audio in arguments
    assert arguments[arguments.index("-crf") + 1] == "32"


@pytest.mark.parametrize(
    ("ext", "codec", "expected"),
    [(".webm", "vp9", "3"), (".mp4", "h264", "3"), (".mp4", "hevc", "8")],
)
def test_overlap_codec_threads_keep_hevc_fixed(ext, codec, expected):
    arguments = gemini_subs.overlap_codec_args(ext, codec, threads=3)

    assert arguments[arguments.index("-threads") + 1] == expected


def test_overlap_codec_configuration_rejects_container_mismatch():
    with pytest.raises(ValueError, match="H.264 input requires MP4"):
        gemini_subs.overlap_codec_args(".webm", "h264")


def test_overlap_clip_command_seeks_before_input(tmp_path, monkeypatch):
    (tmp_path / gemini_subs.MANIFEST_NAME).write_text(
        json.dumps(make_manifest(codec="h264")), encoding="utf-8"
    )
    calls = []

    def run(command, **_kwargs):
        calls.append(command)
        Path(command[-1]).write_bytes(b"new clip")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(gemini_subs.subprocess, "run", run)
    result = gemini_subs.create_overlap_clip(
        "source.mp4", tmp_path, 0, 1.25, 4.75, ".mp4"
    )

    assert result == "context_chunk_000.mp4"
    command = calls[0]
    assert command.index("-ss") < command.index("-i")
    assert command[command.index("-ss") + 1] == "00:00:01.250"


@pytest.mark.parametrize(
    ("workers", "cpu_count", "expected"),
    [(1, 24, 0), (3, 24, 8), (8, 24, 3)],
)
def test_ffmpeg_threads_scale_with_clip_workers(
    monkeypatch, workers, cpu_count, expected
):
    monkeypatch.setattr(gemini_subs.os, "process_cpu_count", lambda: cpu_count)

    assert gemini_subs.ffmpeg_threads_for_workers(workers) == expected


@pytest.mark.parametrize(
    ("api_workers", "cpu_count", "expected"),
    [(4, 24, 4), (16, 24, 16), (32, 24, 24)],
)
def test_suggested_clip_workers_follows_api_workers(
    monkeypatch, api_workers, cpu_count, expected
):
    monkeypatch.setattr(gemini_subs.os, "process_cpu_count", lambda: cpu_count)

    assert gemini_subs.suggested_clip_workers(api_workers) == expected


def test_valid_overlap_clip_cache_is_reused_without_reencoding(tmp_path, monkeypatch):
    clip = tmp_path / "context_chunk_000.mp4"
    clip.write_bytes(b"cached")
    run = MagicMock(
        return_value=subprocess.CompletedProcess([], 0, stdout="2.5\n", stderr="")
    )
    monkeypatch.setattr(gemini_subs.subprocess, "run", run)

    result = gemini_subs.create_overlap_clip("source.mp4", tmp_path, 0, 0, 2.5, ".mp4")

    assert result == clip.name
    assert clip.read_bytes() == b"cached"


def test_invalid_overlap_cache_is_regenerated_and_leaves_no_temporary_file(
    tmp_path, monkeypatch
):
    clip = tmp_path / "context_chunk_002.mp4"
    clip.write_bytes(b"invalid")
    (tmp_path / gemini_subs.MANIFEST_NAME).write_text(
        json.dumps(make_manifest(codec="h264")), encoding="utf-8"
    )
    calls = []

    def run(command, **_kwargs):
        calls.append(command)
        if command[0] == "ffprobe":
            return subprocess.CompletedProcess(command, 1, stdout="", stderr="bad")
        Path(command[-1]).write_bytes(b"new clip")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(gemini_subs.subprocess, "run", run)
    result = gemini_subs.create_overlap_clip(
        "source.mp4", tmp_path, 2, 1.25, 4.75, ".mp4"
    )

    assert result == clip.name
    assert clip.read_bytes() == b"new clip"
    assert not Path(f"{clip}.tmp").exists()
    command = calls[1]
    assert command[command.index("-ss") + 1] == "00:00:01.250"
    assert command[command.index("-t") + 1] == "3.500"
    assert "libx264" in command
    assert command[command.index("-crf") + 1] == "32"


def test_subtitle_response_schema_accepts_documented_caption_shape():
    response = gemini_subs.SubtitleResponse.model_validate(
        {"captions": [{"id": 0, "start": "0", "end": "1", "text": "Hi"}]}
    )

    assert response.captions[0].id == 0
    assert response.captions[0].text == "Hi"


def test_subtitle_response_schema_rejects_malformed_captions():
    with pytest.raises(ValidationError):
        gemini_subs.SubtitleResponse.model_validate(
            {"captions": [{"id": "not-an-int"}]}
        )


def test_generation_config_meets_documented_request_contract(monkeypatch):
    contract = {}

    def make_config(**kwargs):
        contract.update(kwargs)
        return object()

    def make_thinking_config(**kwargs):
        return kwargs

    def make_afc_config(**kwargs):
        return SimpleNamespace(**kwargs)

    monkeypatch.setattr(
        gemini_subs,
        "types",
        SimpleNamespace(
            GenerateContentConfig=make_config,
            ThinkingConfig=make_thinking_config,
            AutomaticFunctionCallingConfig=make_afc_config,
        ),
    )

    gemini_subs.generate_content_config("high")

    assert contract["response_mime_type"] == "application/json"
    assert contract["automatic_function_calling"].disable is True
    assert "tools" not in contract
    schema = contract["response_schema"]
    parsed = schema.model_validate(
        {"captions": [{"id": 0, "start": "0", "end": "1", "text": "Hi"}]}
    )
    assert parsed.captions[0].text == "Hi"
    assert contract["thinking_config"] == {"thinking_level": "HIGH"}


def test_minimal_thinking_is_only_valid_for_flash_models():
    gemini_subs.validate_thinking_level_for_model("Gemini-FLASH", "minimal")
    gemini_subs.validate_thinking_level_for_model("gemini-pro", "high")
    with pytest.raises(ValueError, match="only supported by Flash"):
        gemini_subs.validate_thinking_level_for_model("gemini-pro", "minimal")


def test_generation_prompt_includes_clip_owner_timing_and_source_title():
    prompt = gemini_subs.build_generation_prompt(12.0, 2.0, 10.0, "Show Title")

    assert "12.000-second" in prompt
    assert "00:00:02.000 to 00:00:10.000" in prompt
    assert "Source title: Show Title" in prompt


def test_generation_prompt_omits_source_block_without_title():
    prompt = gemini_subs.build_generation_prompt(12.0, 2.0, 10.0)

    assert "SOURCE CONTEXT" not in prompt


def test_valid_chunk_cache_skips_media_read_and_api(monkeypatch):
    chunk = {
        "idx": 0,
        "clip_name": "missing.mp4",
        "clip_duration": 2,
        "owner_start_rel": 0,
        "owner_end_rel": 2,
    }
    monkeypatch.setattr(gemini_subs, "load_cached_captions", lambda *_args: [])

    def fail_if_called(*_args):
        raise AssertionError("no API client may be created for cached captions")

    monkeypatch.setattr(gemini_subs, "create_client", fail_if_called)

    assert gemini_subs.process_chunk(
        "key", None, chunk, "/missing", "model", "video/mp4", "high"
    )


def test_invalid_chunk_cache_is_removed_for_regeneration(tmp_path):
    cache = tmp_path / "subtitle_chunk_000.json"
    cache.write_text("invalid", encoding="utf-8")

    assert gemini_subs.load_cached_captions(cache, 2) is None
    assert not cache.exists()


def test_chunk_request_saves_canonical_caption_array(tmp_path, monkeypatch):
    (tmp_path / "clip.mp4").write_bytes(b"video bytes")
    chunk = {
        "idx": 3,
        "clip_name": "clip.mp4",
        "clip_duration": 2,
        "owner_start_rel": 0,
        "owner_end_rel": 2,
    }
    pieces = ['{"captions": [{"id": 0,', '"start": "0", "end": "1", "text": "Hi"}]}']

    def verify_contents(contents):
        video_part, prompt = contents
        if video_part.inline_data.mime_type != "video/mp4":
            raise AssertionError(
                "video part must use video/mp4, "
                f"got {video_part.inline_data.mime_type!r}"
            )
        if video_part.inline_data.data != b"video bytes":
            raise AssertionError("video part must contain the clip bytes")
        if not isinstance(prompt, str) or not prompt.strip():
            raise AssertionError("request must include the generation prompt")
        if "Source title: Show Title" not in prompt:
            raise AssertionError("generation prompt must include the source title")

    client = FakeGeminiClient("model", pieces, verify_contents)
    monkeypatch.setattr(gemini_subs, "create_client", lambda *_args: client)

    assert gemini_subs.process_chunk(
        "key",
        "base",
        chunk,
        tmp_path,
        "model",
        "video/mp4",
        "high",
        "Show Title",
    )

    saved = json.loads(
        (tmp_path / "subtitle_chunk_003.json").read_text(encoding="utf-8")
    )
    assert saved == [
        {
            "id": 0,
            "start": "00:00:00.000",
            "end": "00:00:01.000",
            "text": "Hi",
        }
    ]


def test_chunk_request_failure_does_not_publish_result(tmp_path, monkeypatch):
    (tmp_path / "clip.mp4").write_bytes(b"video")
    chunk = {
        "idx": 0,
        "clip_name": "clip.mp4",
        "clip_duration": 2,
        "owner_start_rel": 0,
        "owner_end_rel": 2,
    }
    client = FakeGeminiClient("model", ["bad"], lambda _contents: None)
    monkeypatch.setattr(gemini_subs, "create_client", lambda *_args: client)

    assert not gemini_subs.process_chunk(
        "key", None, chunk, tmp_path, "model", "video/mp4", "high"
    )
    assert not (tmp_path / "subtitle_chunk_000.json").exists()


def test_process_chunks_without_overlap_processes_every_chunk_and_reports_failures(
    tmp_path, monkeypatch, immediate_execution
):
    chunks = [
        {"idx": 0, "name": "chunk_000.mp4", "start": 0, "end": 2},
        {"idx": 1, "name": "chunk_001.mp4", "start": 2, "end": 4},
    ]

    def process(_key, _base, chunk, chunk_dir, *_args):
        Path(chunk_dir, f"processed_{chunk['idx']}").write_text(
            "done", encoding="utf-8"
        )
        return chunk["idx"] == 0

    monkeypatch.setattr(gemini_subs, "process_chunk", process)
    failed = gemini_subs.process_chunks(
        "key",
        None,
        "source",
        str(tmp_path),
        chunks,
        0,
        ".mp4",
        7,
        3,
        "model",
        "video/mp4",
        "high",
    )

    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "processed_0",
        "processed_1",
    ]
    assert failed == ["chunk_001.mp4"]


def test_process_chunks_overlap_keeps_processing_clips_after_a_clip_failure(
    tmp_path, monkeypatch, immediate_execution
):
    chunks = [
        {"idx": 0, "name": "chunk_000.mp4", "start": 0, "end": 2},
        {"idx": 1, "name": "chunk_001.mp4", "start": 2, "end": 4},
    ]

    def attach(_video, _directory, chunk, _overlap, _ext, *_args):
        if chunk["idx"] == 1:
            raise RuntimeError("encode failed")
        return {**chunk, "clip_name": "context_chunk_000.mp4"}

    def process(_key, _base, chunk, chunk_dir, *_args):
        Path(chunk_dir, f"processed_{chunk['idx']}").write_text(
            "done", encoding="utf-8"
        )
        return False

    monkeypatch.setattr(gemini_subs, "attach_overlap_clip", attach)
    monkeypatch.setattr(gemini_subs, "process_chunk", process)
    failed = gemini_subs.process_chunks(
        "key",
        None,
        "source",
        str(tmp_path),
        chunks,
        1,
        ".mp4",
        2,
        4,
        "model",
        "video/mp4",
        "high",
    )

    assert (tmp_path / "processed_0").read_text(encoding="utf-8") == "done"
    assert not (tmp_path / "processed_1").exists()
    assert failed == ["context_chunk_001.mp4", "context_chunk_000.mp4"]


def test_stitch_applies_clip_offset_and_half_open_midpoint_ownership(tmp_path):
    write_layout(
        tmp_path,
        [("chunk_000.mp4", 0, 10), ("chunk_001.mp4", 10, 20)],
        overlap=2,
    )
    write_subtitles(
        tmp_path,
        0,
        [
            {"start": "9", "end": "11", "text": "Right edge excluded"},
            {"start": "8", "end": "10", "text": "First owner"},
        ],
    )
    write_subtitles(
        tmp_path,
        1,
        [
            {"start": "1", "end": "3", "text": "Left edge included"},
            {"start": "4", "end": "6", "text": "Offset caption"},
        ],
    )
    output = tmp_path / "output.vtt"

    provenance = gemini_subs.stitch(tmp_path, output)

    result = webvtt.read(output)
    assert [caption.text for caption in result] == [
        "First owner",
        "Left edge included",
        "Offset caption",
    ]
    assert [caption.start for caption in result] == [
        "00:00:08.000",
        "00:00:09.000",
        "00:00:12.000",
    ]
    assert provenance == [0, 1, 1]


def test_stitch_rejects_missing_and_unexpected_chunk_results(tmp_path):
    write_layout(tmp_path, [("chunk_000.mp4", 0, 5)])
    write_subtitles(tmp_path, 2, [])

    with pytest.raises(
        ValueError,
        match=r"missing chunk indices: \[0\].*unexpected chunk indices: \[2\]",
    ):
        gemini_subs.stitch(tmp_path, tmp_path / "output.vtt")


def test_stitch_preserves_repeated_text_and_cross_chunk_overlap(tmp_path):
    write_layout(
        tmp_path,
        [("chunk_000.mp4", 0, 5), ("chunk_001.mp4", 5, 10)],
    )
    write_subtitles(tmp_path, 0, [{"start": "1", "end": "5.5", "text": "Again"}])
    write_subtitles(tmp_path, 1, [{"start": "0", "end": "1", "text": "Again"}])
    output = tmp_path / "output.vtt"

    provenance = gemini_subs.stitch(tmp_path, output)

    result = webvtt.read(output)
    assert [caption.text for caption in result] == ["Again", "Again"]
    assert result[0].start == "00:00:01.000"
    assert result[0].end == "00:00:05.500"
    assert result[1].start == "00:00:05.000"
    assert result[1].end == "00:00:06.000"
    assert provenance is None


def test_stitch_preserves_text_for_player_layout(tmp_path):
    text = (
        "Host: This generic turn is deliberately longer than forty-two characters\n"
        "[On-screen banner remains unchanged]\n"
        "Plain unlabeled line remains unchanged\n"
        "Guest: This separate turn remains unchanged"
    )
    write_layout(tmp_path, [("chunk_000.mp4", 0, 5)])
    write_subtitles(tmp_path, 0, [{"start": "1", "end": "3", "text": text}])
    output = tmp_path / "output.vtt"

    gemini_subs.stitch(tmp_path, output)

    caption = webvtt.read(output)[0]
    assert (caption.start, caption.end) == ("00:00:01.000", "00:00:03.000")
    assert caption.text == text


def test_stitch_removes_exact_boundary_echo_and_keeps_new_text(tmp_path):
    write_layout(
        tmp_path,
        [("chunk_000.mp4", 0, 5), ("chunk_001.mp4", 5, 10)],
        overlap=1,
    )
    write_subtitles(
        tmp_path,
        0,
        [
            {
                "start": "1",
                "end": "5.5",
                "text": "Host: Intro repeated phrase.",
            }
        ],
    )
    write_subtitles(
        tmp_path,
        1,
        [
            {
                "start": "1",
                "end": "2",
                "text": "Host: Repeated phrase.\nHost: Keep this detail.",
            }
        ],
    )
    output = tmp_path / "output.vtt"

    provenance = gemini_subs.stitch(tmp_path, output)

    assert [caption.text for caption in webvtt.read(output)] == [
        "Host: Intro repeated phrase.",
        "Host: Keep this detail.",
    ]
    assert provenance == [0, 1]


@pytest.mark.parametrize(
    ("earlier_text", "later_text"),
    [
        ("Host: Repeat this phrase.", "Guest: Repeat this phrase."),
        ("Repeat this phrase.", "Repeat this phrase."),
        ("Host: Again.", "Host: Again."),
        ("Host: Keep this phrase.", "Host: Keep that phrase."),
        (
            "Host: Repeat this phrase.\nGuest: Different reply.",
            "Host: Repeat this phrase.",
        ),
    ],
)
def test_stitch_preserves_ambiguous_boundary_repetition(
    tmp_path, earlier_text, later_text
):
    write_layout(
        tmp_path,
        [("chunk_000.mp4", 0, 5), ("chunk_001.mp4", 5, 10)],
        overlap=1,
    )
    write_subtitles(
        tmp_path,
        0,
        [{"start": "1", "end": "5.5", "text": earlier_text}],
    )
    write_subtitles(
        tmp_path,
        1,
        [{"start": "1", "end": "2", "text": later_text}],
    )
    output = tmp_path / "output.vtt"

    provenance = gemini_subs.stitch(tmp_path, output)

    assert [caption.text for caption in webvtt.read(output)] == [
        earlier_text,
        later_text,
    ]
    assert provenance == [0, 1]


def test_refinement_prompt_contains_script_title_and_identity_context():
    script = "[37] 12:34:56.789 --> 12:34:58.012: Unique caption text"
    prompt = gemini_subs.build_refinement_prompt(
        script, "Show Title", "Jane Doe: Host of the show.", "Video observations"
    )

    assert script in prompt
    assert "Source title: Show Title" in prompt
    assert "GROUNDED IDENTITY CONTEXT" in prompt
    assert "Jane Doe: Host of the show." in prompt
    assert "DIRECT VIDEO IDENTITY ANALYSIS" in prompt
    assert "Video observations" in prompt


def test_refinement_prompt_omits_optional_source_blocks():
    prompt = gemini_subs.build_refinement_prompt("script")

    assert "Source title:" not in prompt
    assert "GROUNDED IDENTITY CONTEXT" not in prompt
    assert "DIRECT VIDEO IDENTITY ANALYSIS" not in prompt


def test_research_prompt_lists_urls_and_defers_youtube_analysis():
    prompt = gemini_subs.build_identity_research_prompt(
        "Show Title",
        ["https://example.com/notes"],
        ["https://www.youtube.com/watch?v=VIDEO_ID"],
    )

    assert "Show Title" in prompt
    assert "- https://example.com/notes" in prompt
    assert "https://www.youtube.com/watch?v=VIDEO_ID" in prompt
    assert "separate pass" in prompt


def test_research_prompt_omits_optional_sections():
    prompt = gemini_subs.build_identity_research_prompt()

    assert "SOURCE TITLE" not in prompt
    assert "CONTEXT URLS" not in prompt
    assert "YouTube" not in prompt


def test_youtube_analysis_prompt_includes_title_and_observations():
    prompt = gemini_subs.build_youtube_analysis_prompt("Show Title")

    assert "Show Title" in prompt
    assert "speaker-identification observations" in prompt


@pytest.mark.parametrize(
    "changes",
    [
        [
            gemini_subs.RefinedCaption(id=0, text="One"),
            gemini_subs.RefinedCaption(id=0, text="Two"),
        ],
        [gemini_subs.RefinedCaption(id=2, text="Out")],
        [gemini_subs.RefinedCaption(id=0, text="  ")],
    ],
    ids=["duplicate", "out of range", "empty"],
)
def test_refinement_change_validation_is_all_or_nothing(changes):
    with pytest.raises(ValueError):
        gemini_subs.validate_refinement_changes(changes, 2)


def test_global_refinement_changes_only_text_and_preserves_timestamps(
    tmp_path, monkeypatch
):
    source = tmp_path / "source.vtt"
    output = tmp_path / "output.vtt"
    write_vtt(
        source,
        [
            ("00:00:00.000", "00:00:01.000", "Old\nline"),
            ("00:00:02.000", "00:00:03.000", "Keep"),
        ],
    )
    order = []

    def verify_research_contents(contents):
        order.append("research")
        assert isinstance(contents, str), (
            "research request must be a single text prompt"
        )
        if "Show Title" not in contents:
            raise AssertionError("research prompt must include the source title")

    def verify_refinement_contents(contents):
        order.append("refine")
        assert isinstance(contents, str), (
            "refinement request must be a single text prompt"
        )
        if "[0] 00:00:00.000 --> 00:00:01.000: Old\nline" not in contents:
            raise AssertionError("refinement prompt must contain the indexed script")
        if "Source title: Show Title" not in contents:
            raise AssertionError("refinement prompt must include the source title")
        if "Research text" not in contents:
            raise AssertionError("refinement prompt must include the research text")

    client = FakeRefinementClient(
        "refiner",
        [
            research_call(
                pieces=("Research text",),
                queries=["who is in this video"],
                sources=[("Site", "https://example.com/site")],
                verify_contents=verify_research_contents,
            ),
            refinement_call(
                ['{"changes": [{"id": 0, "text": "New"}]}'],
                verify_contents=verify_refinement_contents,
            ),
        ],
    )
    monkeypatch.setattr(gemini_subs, "create_client", lambda *_args: client)

    gemini_subs.global_refine_subtitles(
        source, output, "key", None, "refiner", "high", source_title="Show Title"
    )

    assert order == ["research", "refine"]
    assert client.pending_calls == 0
    result = webvtt.read(output)
    assert len(result) == 2
    assert [caption.text for caption in result] == ["New", "Keep"]
    assert [(caption.start, caption.end) for caption in result] == [
        ("00:00:00.000", "00:00:01.000"),
        ("00:00:02.000", "00:00:03.000"),
    ]


@pytest.mark.parametrize(
    ("earlier_text", "refined_text", "provenance", "expected_texts"),
    [
        (
            "Host: First unique sentence",
            "Host: First unique sentence",
            [0, 1],
            ["Host: First unique sentence"],
        ),
        (
            "Host: Intro before the repeated boundary\nphrase",
            "Host: Repeated boundary\nphrase",
            [0, 1],
            ["Host: Intro before the repeated boundary\nphrase"],
        ),
        (
            "Host: Shared opening.\nGuest: Shared response.",
            "Host: Shared opening.\nGuest: Shared response.",
            [0, 1],
            ["Host: Shared opening.\nGuest: Shared response."],
        ),
        (
            "Host: Shared opening.\nGuest: Shared response.",
            ("Host: Shared opening.\nGuest: Shared response.\nNarrator: Fresh detail."),
            [0, 1],
            [
                "Host: Shared opening.\nGuest: Shared response.",
                "Narrator: Fresh detail.",
            ],
        ),
        (
            "Host: Shared opening.\nGuest: Yes.",
            "Host: Shared opening.\nGuest: Yes.",
            [0, 1],
            [
                "Host: Shared opening.\nGuest: Yes.",
                "Host: Shared opening.\nGuest: Yes.",
            ],
        ),
        (
            "Host: Shared opening.\n[On-screen card]",
            "Host: Shared opening.",
            [0, 1],
            ["Host: Shared opening.\n[On-screen card]", "Host: Shared opening."],
        ),
        (
            "Host: Shared opening.\n[On-screen card]\nGuest: Shared response.",
            "Host: Shared opening.\n[On-screen card]\nGuest: Shared response.",
            [0, 1],
            [
                "Host: Shared opening.\n[On-screen card]\nGuest: Shared response.",
                "Host: Shared opening.\n[On-screen card]\nGuest: Shared response.",
            ],
        ),
        (
            "Host: First unique sentence",
            "Host: First unique sentence\nGuest: New follow-up line",
            [0, 1],
            ["Host: First unique sentence", "Guest: New follow-up line"],
        ),
        (
            "Host: Repeated words",
            "Host: Repeated words\nHost: New detail",
            [0, 1],
            ["Host: Repeated words", "Host: New detail"],
        ),
        (
            "Host: First unique sentence",
            "Host: First unique sentence",
            None,
            ["Host: First unique sentence", "Host: First unique sentence"],
        ),
    ],
)
def test_refinement_applies_boundary_cleanup(
    tmp_path, monkeypatch, earlier_text, refined_text, provenance, expected_texts
):
    source = tmp_path / "staging.vtt"
    output = tmp_path / "output.vtt"
    write_vtt(
        source,
        [
            ("00:00:00.000", "00:00:04.000", earlier_text),
            ("00:00:03.000", "00:00:05.000", "Guest: Different text"),
        ],
    )
    client = FakeRefinementClient(
        "refiner",
        [
            research_call(queries=["who is the host"]),
            refinement_call(
                [json.dumps({"changes": [{"id": 1, "text": refined_text}]})]
            ),
        ],
    )
    monkeypatch.setattr(gemini_subs, "create_client", lambda *_args: client)

    gemini_subs.global_refine_subtitles(
        source,
        output,
        "key",
        None,
        "refiner",
        "high",
        boundary_provenance=provenance,
    )

    result = webvtt.read(output)
    assert [caption.text for caption in result] == expected_texts
    assert [(caption.start, caption.end) for caption in result] == [
        ("00:00:00.000", "00:00:04.000"),
        ("00:00:03.000", "00:00:05.000"),
    ][: len(expected_texts)]


def test_refinement_rejects_mismatched_provenance_before_publication(tmp_path):
    source = tmp_path / "staging.vtt"
    output = tmp_path / "output.vtt"
    write_vtt(source, [("00:00:00.000", "00:00:01.000", "Only cue")])
    output.write_text("previous output", encoding="utf-8")

    with pytest.raises(ValueError, match="one chunk index per caption"):
        gemini_subs.global_refine_subtitles(
            source,
            output,
            "key",
            None,
            "refiner",
            "high",
            boundary_provenance=[],
        )

    assert output.read_text(encoding="utf-8") == "previous output"


def test_invalid_refinement_does_not_mutate_or_publish(tmp_path, monkeypatch):
    source = tmp_path / "source.vtt"
    output = tmp_path / "output.vtt"
    output.write_text("previous", encoding="utf-8")
    write_vtt(
        source,
        [
            ("00:00:00.000", "00:00:01.000", "First"),
            ("00:00:02.000", "00:00:03.000", "Second"),
        ],
    )
    client = FakeRefinementClient(
        "model",
        [
            research_call(queries=["who is the host"]),
            refinement_call(
                [
                    json.dumps(
                        {
                            "changes": [
                                {"id": 0, "text": "Changed"},
                                {"id": 2, "text": "Bad"},
                            ]
                        }
                    )
                ]
            ),
        ],
    )
    monkeypatch.setattr(gemini_subs, "create_client", lambda *_args: client)

    with pytest.raises(SystemExit, match="1"):
        gemini_subs.global_refine_subtitles(
            source,
            output,
            "key",
            None,
            "model",
            "high",
        )

    result = webvtt.read(source)
    assert [caption.text for caption in result] == ["First", "Second"]
    assert output.read_text(encoding="utf-8") == "previous"


def test_research_enables_url_context_and_lists_urls_when_supplied(
    tmp_path, monkeypatch
):
    source = tmp_path / "source.vtt"
    output = tmp_path / "output.vtt"
    write_vtt(source, [("00:00:00.000", "00:00:01.000", "Only")])

    def verify_contents(contents):
        if "- https://example.com/notes" not in contents:
            raise AssertionError("research prompt must list supplied context URLs")

    def verify_config(config):
        tools = config.tools or []
        if not any(tool.url_context is not None for tool in tools):
            raise AssertionError("ordinary context URLs must enable URL Context")

    client = FakeRefinementClient(
        "refiner",
        [
            research_call(
                queries=["who is the host"],
                urls=[("https://example.com/notes", "URL_RETRIEVAL_STATUS_SUCCESS")],
                verify_contents=verify_contents,
                verify_config=verify_config,
            ),
            refinement_call(['{"changes": []}']),
        ],
    )
    monkeypatch.setattr(gemini_subs, "create_client", lambda *_args: client)

    gemini_subs.global_refine_subtitles(
        source,
        output,
        "key",
        None,
        "refiner",
        "high",
        source_title="Show Title",
        context_urls=["https://example.com/notes"],
    )

    assert webvtt.read(output)[0].text == "Only"


def test_refinement_without_search_grounding_fails_without_publication(
    tmp_path, monkeypatch, capsys
):
    source = tmp_path / "source.vtt"
    output = tmp_path / "output.vtt"
    output.write_text("previous", encoding="utf-8")
    write_vtt(source, [("00:00:00.000", "00:00:01.000", "First")])

    client = FakeRefinementClient(
        "refiner",
        [
            research_call(),
            youtube_call(),
            refinement_call(['{"changes": [{"id": 0, "text": "Changed"}]}']),
        ],
    )
    monkeypatch.setattr(gemini_subs, "create_client", lambda *_args: client)

    with pytest.raises(SystemExit, match="1"):
        gemini_subs.global_refine_subtitles(
            source,
            output,
            "key",
            None,
            "refiner",
            "high",
            context_urls=["https://www.youtube.com/watch?v=VIDEO_ID"],
        )

    assert "no Google Search grounding" in capsys.readouterr().out
    assert client.pending_calls == 2
    assert output.read_text(encoding="utf-8") == "previous"
    assert [caption.text for caption in webvtt.read(source)] == ["First"]


def test_refinement_fails_when_context_url_retrieval_fails(
    tmp_path, monkeypatch, capsys
):
    source = tmp_path / "source.vtt"
    output = tmp_path / "output.vtt"
    output.write_text("previous", encoding="utf-8")
    write_vtt(source, [("00:00:00.000", "00:00:01.000", "First")])
    youtube_url = "https://www.youtube.com/watch?v=VIDEO_ID"

    client = FakeRefinementClient(
        "refiner",
        [
            research_call(
                queries=["who is the host"],
                urls=[("https://example.com/notes", "URL_RETRIEVAL_STATUS_PAYWALL")],
            ),
            youtube_call(),
            refinement_call(['{"changes": []}']),
        ],
    )
    monkeypatch.setattr(gemini_subs, "create_client", lambda *_args: client)

    with pytest.raises(SystemExit, match="1"):
        gemini_subs.global_refine_subtitles(
            source,
            output,
            "key",
            None,
            "refiner",
            "high",
            context_urls=["https://example.com/notes", youtube_url],
        )

    assert "retrieval failed with URL_RETRIEVAL_STATUS_PAYWALL" in (
        capsys.readouterr().out
    )
    assert client.pending_calls == 2
    assert output.read_text(encoding="utf-8") == "previous"


def test_refinement_fails_when_context_url_retrieval_is_missing(
    tmp_path, monkeypatch, capsys
):
    source = tmp_path / "source.vtt"
    output = tmp_path / "output.vtt"
    output.write_text("previous", encoding="utf-8")
    write_vtt(source, [("00:00:00.000", "00:00:01.000", "First")])

    client = FakeRefinementClient(
        "refiner",
        [
            research_call(queries=["who is the host"]),
            refinement_call(['{"changes": []}']),
        ],
    )
    monkeypatch.setattr(gemini_subs, "create_client", lambda *_args: client)

    with pytest.raises(SystemExit, match="1"):
        gemini_subs.global_refine_subtitles(
            source,
            output,
            "key",
            None,
            "refiner",
            "high",
            context_urls=["https://example.com/notes"],
        )

    assert "was not retrieved" in capsys.readouterr().out
    assert client.pending_calls == 1
    assert output.read_text(encoding="utf-8") == "previous"


def test_refinement_accepts_equivalent_retrieved_url_identity(tmp_path, monkeypatch):
    source = tmp_path / "source.vtt"
    output = tmp_path / "output.vtt"
    write_vtt(source, [("00:00:00.000", "00:00:01.000", "Only")])

    client = FakeRefinementClient(
        "refiner",
        [
            research_call(
                queries=["who is the host"],
                urls=[
                    (
                        "https://example.com/notes/",
                        "URL_RETRIEVAL_STATUS_SUCCESS",
                    )
                ],
            ),
            refinement_call(['{"changes": []}']),
        ],
    )
    monkeypatch.setattr(gemini_subs, "create_client", lambda *_args: client)

    gemini_subs.global_refine_subtitles(
        source,
        output,
        "key",
        None,
        "refiner",
        "high",
        context_urls=["https://example.com/notes"],
    )

    assert webvtt.read(output)[0].text == "Only"


def test_youtube_context_url_becomes_separate_direct_video_analysis(
    tmp_path, monkeypatch, capsys
):
    source = tmp_path / "source.vtt"
    output = tmp_path / "output.vtt"
    write_vtt(source, [("00:00:00.000", "00:00:01.000", "Only")])
    youtube_url = "https://www.youtube.com/watch?v=VIDEO_ID&t=30"
    order = []

    def verify_research_contents(contents):
        order.append("research")
        assert isinstance(contents, str), "research request must not attach video Parts"
        if youtube_url not in contents:
            raise AssertionError(
                "research prompt must list the YouTube URL as an identifier"
            )

    def verify_youtube_contents(contents):
        order.append("youtube")
        video_parts = [part for part in contents if not isinstance(part, str)]
        if [part.file_data.file_uri for part in video_parts] != [youtube_url]:
            raise AssertionError("video Part must carry the YouTube URL")
        if not isinstance(contents[-1], str) or not contents[-1].strip():
            raise AssertionError("YouTube request must end with the text prompt")

    def verify_refinement_contents(contents):
        order.append("refine")
        if "YouTube analysis" not in contents:
            raise AssertionError(
                "refinement prompt must include the YouTube analysis text"
            )

    client = FakeRefinementClient(
        "refiner",
        [
            research_call(
                queries=["who is in the video"],
                sources=[("Site", "https://example.com/site")],
                verify_contents=verify_research_contents,
            ),
            youtube_call(
                pieces=("YouTube analysis",),
                verify_contents=verify_youtube_contents,
            ),
            refinement_call(
                ['{"changes": [{"id": 0, "text": "Refined"}]}'],
                verify_contents=verify_refinement_contents,
            ),
        ],
    )
    monkeypatch.setattr(gemini_subs, "create_client", lambda *_args: client)

    gemini_subs.global_refine_subtitles(
        source,
        output,
        "key",
        None,
        "refiner",
        "high",
        context_urls=[youtube_url],
    )

    assert order == ["research", "youtube", "refine"]
    assert client.pending_calls == 0
    assert webvtt.read(output)[0].text == "Refined"
    assert youtube_url in capsys.readouterr().out


def test_youtube_analysis_sdk_failure_preserves_output(tmp_path, monkeypatch, capsys):
    source = tmp_path / "source.vtt"
    output = tmp_path / "output.vtt"
    output.write_text("previous", encoding="utf-8")
    write_vtt(source, [("00:00:00.000", "00:00:01.000", "First")])
    youtube_url = "https://www.youtube.com/watch?v=VIDEO_ID"

    client = FakeRefinementClient(
        "refiner",
        [
            research_call(queries=["who is the host"]),
            youtube_call(error=RuntimeError("video unavailable")),
            refinement_call(['{"changes": [{"id": 0, "text": "Changed"}]}']),
        ],
    )
    monkeypatch.setattr(gemini_subs, "create_client", lambda *_args: client)

    with pytest.raises(RuntimeError, match="video unavailable"):
        gemini_subs.global_refine_subtitles(
            source,
            output,
            "key",
            None,
            "refiner",
            "high",
            context_urls=[youtube_url],
        )

    assert youtube_url in capsys.readouterr().out
    assert client.pending_calls == 1
    assert output.read_text(encoding="utf-8") == "previous"
    assert [caption.text for caption in webvtt.read(source)] == ["First"]


def test_mixed_context_urls_split_youtube_from_ordinary(tmp_path, monkeypatch):
    source = tmp_path / "source.vtt"
    output = tmp_path / "output.vtt"
    write_vtt(source, [("00:00:00.000", "00:00:01.000", "Only")])
    youtube_url = "https://youtu.be/VIDEO_ID"
    ordinary_url = "https://example.com/notes?id=7"

    def verify_research_contents(contents):
        assert isinstance(contents, str), "research request must not attach video Parts"
        if f"- {ordinary_url}" not in contents:
            raise AssertionError("ordinary URL must stay in the research prompt")

    def verify_research_config(config):
        tools = config.tools or []
        if not any(tool.url_context is not None for tool in tools):
            raise AssertionError("ordinary URL must enable URL Context")

    def verify_youtube_contents(contents):
        video_parts = [part for part in contents if not isinstance(part, str)]
        if [part.file_data.file_uri for part in video_parts] != [youtube_url]:
            raise AssertionError("only the YouTube URL may be a video Part")

    client = FakeRefinementClient(
        "refiner",
        [
            research_call(
                queries=["who is the host"],
                urls=[(ordinary_url, "URL_RETRIEVAL_STATUS_SUCCESS")],
                verify_contents=verify_research_contents,
                verify_config=verify_research_config,
            ),
            youtube_call(
                pieces=("YouTube analysis",),
                verify_contents=verify_youtube_contents,
            ),
            refinement_call(['{"changes": []}']),
        ],
    )
    monkeypatch.setattr(gemini_subs, "create_client", lambda *_args: client)

    gemini_subs.global_refine_subtitles(
        source,
        output,
        "key",
        None,
        "refiner",
        "high",
        context_urls=[youtube_url, ordinary_url],
    )

    assert webvtt.read(output)[0].text == "Only"


def test_refinement_requires_each_distinct_url_query():
    with pytest.raises(SystemExit, match="1"):
        gemini_subs.verify_refinement_grounding(
            ["who is the host"],
            [],
            {
                "https://example.com/notes?id=1": "URL_RETRIEVAL_STATUS_SUCCESS",
            },
            [
                "https://example.com/notes?id=1",
                "https://example.com/notes?id=2",
            ],
        )


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


def test_context_url_validation_accepts_and_deduplicates_http_urls():
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


def test_context_url_classification_preserves_queries():
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


def test_refinement_rejects_malformed_context_url_before_api(tmp_path, monkeypatch):
    source = tmp_path / "source.vtt"
    write_vtt(source, [("00:00:00.000", "00:00:01.000", "First")])

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("no API client may be created for malformed URLs")

    monkeypatch.setattr(gemini_subs, "create_client", fail_if_called)

    with pytest.raises(ValueError, match="context-url"):
        gemini_subs.global_refine_subtitles(
            source,
            tmp_path / "output.vtt",
            "key",
            None,
            "refiner",
            "high",
            context_urls=["not-a-url"],
        )


def test_generation_rejects_malformed_context_url_before_media(tmp_path, monkeypatch):
    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("media probing must not run for malformed context URLs")

    monkeypatch.setattr(gemini_subs, "probe_video_format", fail_if_called)
    config = gemini_subs.GenerationConfig(
        video_path=tmp_path / "missing.webm",
        output_path=tmp_path / "output.vtt",
        model="gemini-flash",
        api_key="key",
        context_urls=("not-a-url",),
    )

    with pytest.raises(ValueError, match="context-url"):
        gemini_subs.run_generation(config)


def test_cli_rejects_malformed_context_url_before_refinement(
    tmp_path, monkeypatch, capsys
):
    source = tmp_path / "source.vtt"
    source.write_text("WEBVTT\n", encoding="utf-8")

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("refinement must not run for malformed context URLs")

    monkeypatch.setattr(gemini_subs, "global_refine_subtitles", fail_if_called)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gemini_subs.py",
            str(source),
            "--refine-only",
            "--api-key",
            "key",
            "--context-url",
            "not-a-url",
        ],
    )

    with pytest.raises(SystemExit, match="1"):
        gemini_subs.main()

    assert "Invalid --context-url" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("QWER boxing match [EP.38].webm", "QWER boxing match [EP.38]"),
        ("[EP.7] guest story | inn.webm", "[EP.7] guest story | inn"),
        ("QWER boxing match.webm.vtt", "QWER boxing match"),
        ("show.ko.vtt", "show"),
        ("show.en-US.vtt", "show"),
        ("movie.webm.en.vtt", "movie"),
        ("episode.BTS.webm", "episode.BTS"),
        ("episode.bts.webm", "episode.bts"),
        ("plain.vtt", "plain"),
        ("plain.mp4", "plain"),
    ],
)
def test_source_title_derivation_strips_media_subtitle_and_language_suffixes(
    filename, expected
):
    assert gemini_subs.derive_source_title(Path(filename)) == expected


def test_process_lock_blocks_second_owner_and_survives_release(tmp_path):
    first = gemini_subs.acquire_lock(tmp_path)
    lock_path = tmp_path / gemini_subs.LOCK_NAME
    try:
        assert lock_path.read_text(encoding="utf-8") == str(os.getpid())
        with pytest.raises(RuntimeError, match=f"PID {os.getpid()}"):
            gemini_subs.acquire_lock(tmp_path)
    finally:
        gemini_subs.release_lock(first)

    second = gemini_subs.acquire_lock(tmp_path)
    gemini_subs.release_lock(second)
    assert lock_path.exists()


def test_process_lock_treats_unlocked_pid_text_as_stale(tmp_path):
    lock_path = tmp_path / gemini_subs.LOCK_NAME
    lock_path.write_text("999999", encoding="utf-8")

    lock = gemini_subs.acquire_lock(tmp_path)
    try:
        assert lock_path.read_text(encoding="utf-8") == str(os.getpid())
    finally:
        gemini_subs.release_lock(lock)


@pytest.mark.parametrize(
    ("arguments", "existing_input", "message"),
    [
        (["missing.mp4", "--api-key", "key"], False, "Video file not found"),
        (["input.mp4"], True, "API key not configured"),
        (["input.mp4", "--api-key", "key", "--chunk-dur", "0"], True, "chunk-dur"),
        (["input.mp4", "--api-key", "key", "--workers", "0"], True, "workers"),
        (["input.mp4", "--api-key", "key", "--overlap", "-1"], True, "overlap"),
        (
            ["input.mp4", "--api-key", "key", "--chunk-dur", "5", "--overlap", "5"],
            True,
            "smaller",
        ),
        (
            [
                "input.mp4",
                "--api-key",
                "key",
                "--model",
                "pro",
                "--thinking-level",
                "minimal",
            ],
            True,
            "Flash",
        ),
    ],
)
def test_cli_rejects_invalid_generation_inputs_before_pipeline(
    tmp_path, monkeypatch, capsys, arguments, existing_input, message
):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    input_path = tmp_path / arguments[0]
    if existing_input:
        input_path.write_bytes(b"video")
    arguments = [str(input_path), *arguments[1:]]

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("pipeline must not start for invalid CLI inputs")

    monkeypatch.setattr(gemini_subs, "build_manifest", fail_if_called)
    monkeypatch.setattr(sys, "argv", ["gemini_subs.py", *arguments])

    with pytest.raises(SystemExit, match="1"):
        gemini_subs.main()

    assert message in capsys.readouterr().out


def test_cli_rejects_generation_output_resolving_to_source(tmp_path, monkeypatch):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"video")
    alias = tmp_path / "alias.mp4"
    alias.symlink_to(source)
    monkeypatch.setattr(
        sys,
        "argv",
        ["gemini_subs.py", str(source), "--api-key", "key", "--output", str(alias)],
    )

    with pytest.raises(SystemExit, match="1"):
        gemini_subs.main()


@pytest.mark.parametrize(
    "input_exists", [False, True], ids=["missing input", "missing key"]
)
def test_refine_only_validates_input_and_api_key(tmp_path, monkeypatch, input_exists):
    source = tmp_path / "source.vtt"
    if input_exists:
        source.write_text("WEBVTT\n", encoding="utf-8")

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("refinement must not run when CLI validation fails")

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(gemini_subs, "global_refine_subtitles", fail_if_called)
    arguments = ["gemini_subs.py", str(source), "--refine-only"]
    if not input_exists:
        arguments.extend(["--api-key", "key"])
    monkeypatch.setattr(sys, "argv", arguments)

    with pytest.raises(SystemExit, match="1"):
        gemini_subs.main()


def test_refine_only_refines_in_place_without_running_video_pipeline(
    tmp_path, monkeypatch
):
    source = tmp_path / "source.vtt"
    source.write_text("WEBVTT\n", encoding="utf-8")

    def fail_if_called(*_args):
        raise AssertionError("video pipeline must not run in refine-only mode")

    def refine(_input_path, output_path, *_args, **_kwargs):
        Path(output_path).write_text("refined\n", encoding="utf-8")

    monkeypatch.setattr(gemini_subs, "global_refine_subtitles", refine)
    monkeypatch.setattr(gemini_subs, "build_manifest", fail_if_called)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gemini_subs.py",
            str(source),
            "--refine-only",
            "--api-key",
            "key",
            "--output",
            str(source),
        ],
    )

    with pytest.raises(SystemExit, match="0"):
        gemini_subs.main()

    assert source.read_text(encoding="utf-8") == "refined\n"


def prepare_generation_config(
    tmp_path, monkeypatch, process_result=None, refine_text=True, output_path=None
):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"video")
    work = tmp_path / "work"
    work.mkdir()
    chunks = [{"idx": 0, "name": "chunk_000.mp4", "start": 0, "end": 1}]
    monkeypatch.setattr(
        gemini_subs, "build_manifest", lambda _config: (make_manifest(), str(work))
    )
    monkeypatch.setattr(gemini_subs, "split_video", MagicMock())
    monkeypatch.setattr(gemini_subs, "list_chunks", lambda _path: chunks)
    monkeypatch.setattr(
        gemini_subs,
        "process_chunks",
        MagicMock(return_value=[] if process_result is None else process_result),
    )
    return gemini_subs.GenerationConfig(
        video_path=source,
        output_path=output_path or tmp_path / "output_subtitles.vtt",
        model="model",
        api_key="key",
        chunk_dur=60,
        overlap=5.0,
        workers=7,
        thinking_level="high",
        refine_text=refine_text,
    ), work


def test_generation_passes_source_title_to_chunks_and_refinement(tmp_path, monkeypatch):
    config, _work = prepare_generation_config(tmp_path, monkeypatch)
    received = {}

    def stitch(_directory, path):
        Path(path).write_text("stitched", encoding="utf-8")

    def refine(_input_path, output_path, *_args, **kwargs):
        received["refinement_title"] = kwargs["source_title"]
        Path(output_path).write_text("refined", encoding="utf-8")

    monkeypatch.setattr(gemini_subs, "stitch", stitch)
    monkeypatch.setattr(gemini_subs, "global_refine_subtitles", refine)

    gemini_subs.run_generation(config)

    assert gemini_subs.process_chunks.call_args.args[-1] == "source"
    assert received["refinement_title"] == "source"


def test_generation_removes_boundary_echo_created_by_refinement(tmp_path, monkeypatch):
    config, work = prepare_generation_config(tmp_path, monkeypatch)
    chunks = [
        {"idx": 0, "name": "chunk_000.mp4", "start": 0, "end": 5},
        {"idx": 1, "name": "chunk_001.mp4", "start": 5, "end": 10},
    ]
    write_layout(
        work,
        [(chunk["name"], chunk["start"], chunk["end"]) for chunk in chunks],
        overlap=1,
    )
    write_subtitles(
        work,
        0,
        [{"start": "1", "end": "5.5", "text": "Host: Shared phrase."}],
    )
    write_subtitles(
        work,
        1,
        [{"start": "1", "end": "2", "text": "Guest: Different phrase."}],
    )
    manifest = make_manifest(overlap=1)
    monkeypatch.setattr(
        gemini_subs, "build_manifest", lambda _config: (manifest, str(work))
    )
    monkeypatch.setattr(gemini_subs, "list_chunks", lambda _path: chunks)
    client = FakeRefinementClient(
        "model",
        [
            research_call(queries=["who is the host"]),
            refinement_call(
                ['{"changes": [{"id": 1, "text": "Host: Shared phrase."}]}']
            ),
        ],
    )
    monkeypatch.setattr(gemini_subs, "create_client", lambda *_args: client)

    gemini_subs.run_generation(config)

    result = webvtt.read(config.output_path)
    assert [(caption.start, caption.end, caption.text) for caption in result] == [
        ("00:00:01.000", "00:00:05.500", "Host: Shared phrase.")
    ]


def test_successful_generation_without_refinement_cleans_work_before_unlock(
    tmp_path, monkeypatch
):
    config, work = prepare_generation_config(tmp_path, monkeypatch, refine_text=False)
    artifact = work / "manifest.json"
    artifact.write_text("state", encoding="utf-8")
    real_release = gemini_subs.release_lock

    def release_after_cleanup(lock_file):
        # The documented contract releases the lock only after work cleanup.
        assert sorted(path.name for path in work.iterdir()) == [gemini_subs.LOCK_NAME]
        real_release(lock_file)

    def stitch(_directory, path):
        Path(path).write_text("stitched\n", encoding="utf-8")

    monkeypatch.setattr(gemini_subs, "release_lock", release_after_cleanup)
    monkeypatch.setattr(gemini_subs, "stitch", stitch)

    gemini_subs.run_generation(config)

    assert (tmp_path / "output_subtitles.vtt").read_text(encoding="utf-8") == (
        "stitched\n"
    )
    assert sorted(path.name for path in work.iterdir()) == [gemini_subs.LOCK_NAME]
    lock = gemini_subs.acquire_lock(work)
    gemini_subs.release_lock(lock)


def test_failed_chunk_processing_keeps_resume_state_and_releases_lock(
    tmp_path, monkeypatch
):
    config, work = prepare_generation_config(
        tmp_path, monkeypatch, process_result=["chunk_000.mp4"]
    )
    artifact = work / "subtitle_chunk_000.json"
    artifact.write_text("[]", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Failed to process 1 chunk"):
        gemini_subs.run_generation(config)

    assert artifact.exists()
    lock = gemini_subs.acquire_lock(work)
    gemini_subs.release_lock(lock)


def test_refinement_failure_preserves_output_removes_staging_and_resume_state(
    tmp_path, monkeypatch
):
    output = tmp_path / "output.vtt"
    output.write_text("previous", encoding="utf-8")
    config, work = prepare_generation_config(tmp_path, monkeypatch, output_path=output)
    state = work / "manifest.json"
    state.write_text("state", encoding="utf-8")
    received = {}

    def stitch(_directory, path):
        Path(path).write_text("stitched", encoding="utf-8")

    def refine(input_path, *_args, **_kwargs):
        received["path"] = Path(input_path)
        received["content"] = Path(input_path).read_text(encoding="utf-8")
        raise RuntimeError("refinement failed")

    monkeypatch.setattr(gemini_subs, "stitch", stitch)
    monkeypatch.setattr(gemini_subs, "global_refine_subtitles", refine)

    with pytest.raises(RuntimeError, match="refinement failed"):
        gemini_subs.run_generation(config)

    assert received["content"] == "stitched"
    assert received["path"].name.endswith(".staging.vtt")
    assert received["path"].parent == output.parent
    assert output.read_text(encoding="utf-8") == "previous"
    assert state.exists()
    assert not list(tmp_path.glob("*.staging.vtt"))
