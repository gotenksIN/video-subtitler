"""Media pipeline behavior against real FFmpeg fixtures.

Tests exercise probing, splitting, and segment listing on tiny generated
videos. They assert observable artifact properties, not command lines or
internal representation.
"""

import itertools
import shutil
import subprocess

import pytest

from modules import io, media

VIDEO_WIDTH = 160
VIDEO_HEIGHT = 90
VIDEO_RATE = 25
VIDEO_SECONDS = 6.0
KEYFRAME_INTERVAL = 25  # One keyframe per second at 25 fps.

CODEC_SPECS = {
    "vp9": {
        "ext": ".webm",
        "mime": "video/webm",
        "codec": "vp9",
        "encode_args": [
            "-c:v",
            "libvpx-vp9",
            "-b:v",
            "150k",
            "-deadline",
            "good",
            "-cpu-used",
            "4",
            "-g",
            str(KEYFRAME_INTERVAL),
            "-c:a",
            "libopus",
            "-b:a",
            "24k",
        ],
    },
    "h264": {
        "ext": ".mp4",
        "mime": "video/mp4",
        "codec": "h264",
        "encode_args": [
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "32",
            "-g",
            str(KEYFRAME_INTERVAL),
            "-c:a",
            "aac",
            "-b:a",
            "24k",
            "-movflags",
            "+faststart",
        ],
    },
    "hevc": {
        "ext": ".mp4",
        "mime": "video/mp4",
        "codec": "hevc",
        "encode_args": [
            "-c:v",
            "libx265",
            "-preset",
            "veryfast",
            "-crf",
            "32",
            "-x265-params",
            f"keyint={KEYFRAME_INTERVAL}:min-keyint={KEYFRAME_INTERVAL}",
            "-c:a",
            "aac",
            "-b:a",
            "24k",
            "-movflags",
            "+faststart",
            "-tag:v",
            "hvc1",
        ],
    },
}


@pytest.fixture(scope="session")
def media_tools():
    """Resolve ffmpeg and ffprobe or fail with a clear message."""
    tools = {name: shutil.which(name) for name in ("ffmpeg", "ffprobe")}
    missing = [name for name, path in tools.items() if not path]
    if missing:
        pytest.fail(
            "media tests require " + " and ".join(missing) + " in PATH; "
            "install them with ./scripts/ffmpeg.sh",
            pytrace=False,
        )
    return tools


def _generate_video(ffmpeg, directory, name, encode_args):
    """Encode one tiny test video and fail clearly when FFmpeg cannot."""
    path = directory / name
    command = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        (
            f"testsrc2=size={VIDEO_WIDTH}x{VIDEO_HEIGHT}:"
            f"rate={VIDEO_RATE}:duration={VIDEO_SECONDS}"
        ),
        "-f",
        "lavfi",
        "-i",
        f"sine=frequency=440:sample_rate=44100:duration={VIDEO_SECONDS}",
        "-shortest",
        *encode_args,
        str(path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        pytest.fail(
            f"could not generate the {name} fixture with {ffmpeg}:\n"
            f"{result.stderr.strip()}",
            pytrace=False,
        )
    return path


@pytest.fixture(scope="session", params=list(CODEC_SPECS))
def video_fixture(request, media_tools, tmp_path_factory):
    """One tiny session-scoped VP9, H.264, and HEVC source video."""
    spec = CODEC_SPECS[request.param]
    directory = tmp_path_factory.mktemp(f"fixture-{request.param}")
    path = _generate_video(
        media_tools["ffmpeg"],
        directory,
        f"source{spec['ext']}",
        spec["encode_args"],
    )
    return {
        "codec": spec["codec"],
        "ext": spec["ext"],
        "mime": spec["mime"],
        "path": path,
    }


@pytest.fixture(scope="session")
def mpeg4_video(media_tools, tmp_path_factory):
    """A decodable MPEG-4 video that the pipeline does not support."""
    directory = tmp_path_factory.mktemp("fixture-unsupported")
    return _generate_video(
        media_tools["ffmpeg"],
        directory,
        "unsupported.avi",
        ["-c:v", "mpeg4", "-q:v", "8", "-an", "-f", "avi"],
    )


@pytest.fixture(scope="session")
def h264_video_with_subtitles(media_tools, tmp_path_factory):
    directory = tmp_path_factory.mktemp("fixture-subtitles")
    source = _generate_video(
        media_tools["ffmpeg"],
        directory,
        "source.mp4",
        CODEC_SPECS["h264"]["encode_args"],
    )
    subtitles = directory / "captions.srt"
    subtitles.write_text(
        "1\n00:00:00,000 --> 00:00:05,000\nInternal subtitle\n",
        encoding="utf-8",
    )
    output = directory / "source-with-subtitles.mp4"
    command = [
        media_tools["ffmpeg"],
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-i",
        str(subtitles),
        "-map",
        "0",
        "-map",
        "1:0",
        "-c",
        "copy",
        "-c:s",
        "mov_text",
        str(output),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        pytest.fail(
            f"could not generate a subtitle-stream fixture:\n{result.stderr.strip()}",
            pytrace=False,
        )
    return output


def _ffprobe_lines(ffprobe, path, entries, select_streams=None):
    """Return nonempty ffprobe output lines for one show_entries query."""
    command = [ffprobe, "-v", "error"]
    if select_streams:
        command += ["-select_streams", select_streams]
    command += [
        "-show_entries",
        entries,
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        pytest.fail(
            f"ffprobe failed for {path}:\n{result.stderr.strip()}", pytrace=False
        )
    return [line for line in result.stdout.splitlines() if line]


def _primary_video_codec(ffprobe, path):
    lines = _ffprobe_lines(ffprobe, path, "stream=codec_name", select_streams="v:0")
    if len(lines) != 1:
        pytest.fail(
            f"expected one video stream in {path}, got {lines!r}", pytrace=False
        )
    return lines[0]


def _container_duration(ffprobe, path):
    lines = _ffprobe_lines(ffprobe, path, "format=duration")
    if len(lines) != 1:
        pytest.fail(f"expected one duration for {path}, got {lines!r}", pytrace=False)
    return float(lines[0])


def _stream_types(ffprobe, path):
    return _ffprobe_lines(ffprobe, path, "stream=codec_type")


def test_probe_video_format_identifies_supported_primary_codec(video_fixture):
    assert media.probe_video_format(str(video_fixture["path"])) == (
        video_fixture["ext"],
        video_fixture["mime"],
        video_fixture["codec"],
    )


def test_probe_video_format_rejects_unsupported_codec(mpeg4_video):
    with pytest.raises(RuntimeError, match="Video format not supported"):
        media.probe_video_format(str(mpeg4_video))


def test_probe_video_format_reports_missing_input(media_tools, tmp_path):
    with pytest.raises(RuntimeError, match="Failed to probe video format"):
        media.probe_video_format(str(tmp_path / "missing.webm"))


def test_split_creates_decodable_chunks_and_completion_marker(
    video_fixture, media_tools, tmp_path
):
    chunk_dir = tmp_path / "work"
    media.split_video(
        str(video_fixture["path"]),
        str(chunk_dir),
        2,
        {"chunk_ext": video_fixture["ext"]},
    )

    marker = chunk_dir / media.SPLIT_COMPLETE_MARKER
    assert marker.read_text(encoding="utf-8") == "ok\n"
    chunks = media.list_chunks(str(chunk_dir))
    assert len(chunks) >= 2
    source_duration = _container_duration(media_tools["ffprobe"], video_fixture["path"])
    for position, chunk in enumerate(chunks):
        path = chunk_dir / chunk["name"]
        assert path.is_file() and path.stat().st_size > 0
        assert chunk["idx"] == position
        assert chunk["duration"] > 0
        assert (
            _primary_video_codec(media_tools["ffprobe"], path) == video_fixture["codec"]
        )
        assert _stream_types(media_tools["ffprobe"], path) == ["video", "audio"]
        assert _container_duration(media_tools["ffprobe"], path) > 0

    assert chunks[0]["start"] == 0.0
    for previous, current in itertools.pairwise(chunks):
        assert current["start"] >= previous["end"] - 0.01
    assert chunks[-1]["end"] == pytest.approx(source_duration, abs=0.5)


def _split_artifacts(chunk_dir):
    """Snapshot split artifacts; the manifest is rewritten on every call."""
    return {
        path.name: (path.stat().st_size, path.stat().st_mtime_ns, path.read_bytes())
        for path in chunk_dir.iterdir()
        if path.is_file() and path.name != io.MANIFEST_NAME
    }


def test_completed_split_is_reused_unchanged(video_fixture, tmp_path):
    chunk_dir = tmp_path / "work"
    manifest = {"chunk_ext": video_fixture["ext"]}
    media.split_video(str(video_fixture["path"]), str(chunk_dir), 2, manifest)
    before = _split_artifacts(chunk_dir)

    media.split_video(str(video_fixture["path"]), str(chunk_dir), 2, manifest)

    assert _split_artifacts(chunk_dir) == before


def test_corrupted_chunk_triggers_split_regeneration(
    video_fixture, media_tools, tmp_path
):
    chunk_dir = tmp_path / "work"
    manifest = {"chunk_ext": video_fixture["ext"]}
    media.split_video(str(video_fixture["path"]), str(chunk_dir), 2, manifest)
    original_names = [chunk["name"] for chunk in media.list_chunks(str(chunk_dir))]
    (chunk_dir / original_names[0]).write_bytes(b"")

    media.split_video(str(video_fixture["path"]), str(chunk_dir), 2, manifest)

    marker = chunk_dir / media.SPLIT_COMPLETE_MARKER
    assert marker.read_text(encoding="utf-8") == "ok\n"
    regenerated = media.list_chunks(str(chunk_dir))
    assert [chunk["name"] for chunk in regenerated] == original_names
    for chunk in regenerated:
        path = chunk_dir / chunk["name"]
        assert path.stat().st_size > 0
        assert _container_duration(media_tools["ffprobe"], path) > 0


@pytest.mark.parametrize("corruption", ["malformed", "truncated"])
def test_invalid_segment_index_triggers_split_regeneration(
    video_fixture, media_tools, tmp_path, corruption
):
    chunk_dir = tmp_path / "work"
    manifest = {"chunk_ext": video_fixture["ext"]}
    media.split_video(str(video_fixture["path"]), str(chunk_dir), 2, manifest)
    original_names = [chunk["name"] for chunk in media.list_chunks(str(chunk_dir))]
    if corruption == "malformed":
        index = f"{original_names[0]},0,not-a-number\n"
        (chunk_dir / "segments.csv").write_text(index, encoding="utf-8")
    else:
        index = f"{original_names[0]},0,1\n"
        (chunk_dir / "segments.csv").write_text(index, encoding="utf-8")

    media.split_video(str(video_fixture["path"]), str(chunk_dir), 2, manifest)

    marker = chunk_dir / media.SPLIT_COMPLETE_MARKER
    assert marker.read_text(encoding="utf-8") == "ok\n"
    regenerated = media.list_chunks(str(chunk_dir))
    assert [chunk["name"] for chunk in regenerated] == original_names
    for chunk in regenerated:
        path = chunk_dir / chunk["name"]
        assert path.stat().st_size > 0
        assert _container_duration(media_tools["ffprobe"], path) > 0


def test_failed_split_leaves_no_completion_marker(video_fixture, tmp_path):
    chunk_dir = tmp_path / "work"
    chunk_dir.mkdir()
    (chunk_dir / media.SPLIT_COMPLETE_MARKER).write_text("ok\n", encoding="utf-8")
    (chunk_dir / "segments.csv").write_text("chunk_000.mp4,0,1\n", encoding="utf-8")

    with pytest.raises(subprocess.CalledProcessError):
        media.split_video(
            str(tmp_path / "missing-source.mp4"),
            str(chunk_dir),
            2,
            {"chunk_ext": video_fixture["ext"]},
        )

    assert not (chunk_dir / media.SPLIT_COMPLETE_MARKER).exists()


def test_generated_media_keeps_audio_and_excludes_source_subtitles(
    h264_video_with_subtitles, media_tools, tmp_path
):
    chunk_dir = tmp_path / "work"
    media.split_video(
        str(h264_video_with_subtitles),
        str(chunk_dir),
        2,
        {"chunk_ext": ".mp4"},
    )

    for chunk in media.list_chunks(str(chunk_dir)):
        assert _stream_types(media_tools["ffprobe"], chunk_dir / chunk["name"]) == [
            "video",
            "audio",
        ]


def test_segment_index_is_the_source_of_chunk_timing(tmp_path):
    (tmp_path / "segments.csv").write_text(
        "chunk_000.mp4,0.0,2.0\nchunk_001.mp4,2.0,4.5,ignored,fields\n",
        encoding="utf-8",
    )

    assert media.list_chunks(str(tmp_path)) == [
        {
            "idx": 0,
            "name": "chunk_000.mp4",
            "start": 0.0,
            "end": 2.0,
            "duration": 2.0,
        },
        {
            "idx": 1,
            "name": "chunk_001.mp4",
            "start": 2.0,
            "end": 4.5,
            "duration": 2.5,
        },
    ]


def test_segment_index_missing_means_no_chunks(tmp_path):
    assert media.list_chunks(str(tmp_path)) == []


@pytest.mark.parametrize(
    "index",
    [
        "unexpected.mp4,0,1\n",
        "chunk_000.mp4,0,1\nchunk_000.mp4,1,2\n",
    ],
)
def test_segment_index_rejects_invalid_or_duplicate_chunk_names(tmp_path, index):
    (tmp_path / "segments.csv").write_text(index, encoding="utf-8")

    assert media.list_chunks(str(tmp_path)) == []


def test_extract_complete_audio_produces_mono_opus_at_48khz(
    video_fixture, media_tools, tmp_path
):
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    audio_path, duration, source_dur, reused = media.extract_complete_audio(
        str(video_fixture["path"]), str(work_dir)
    )
    assert reused is False
    assert (work_dir / media.EXTRACTED_AUDIO_NAME).is_file()
    assert duration > 0
    assert abs(duration - source_dur) <= media.AUDIO_DURATION_TOLERANCE_SECONDS
    assert media.extracted_audio_is_valid(audio_path) is True


def test_extract_complete_audio_fails_when_source_has_no_audio(mpeg4_video, tmp_path):
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    assert media.has_audio_stream(str(mpeg4_video)) is False
    with pytest.raises(RuntimeError, match="may not contain an audio stream"):
        media.extract_complete_audio(str(mpeg4_video), str(work_dir))


def test_extract_complete_audio_reuses_valid_cache_and_regenerates_corrupt(
    video_fixture, tmp_path
):
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    _audio_path, dur1, _, reused1 = media.extract_complete_audio(
        str(video_fixture["path"]), str(work_dir)
    )
    assert reused1 is False

    # Second run reuses valid cache
    _, dur2, _, reused2 = media.extract_complete_audio(
        str(video_fixture["path"]), str(work_dir)
    )
    assert reused2 is True
    assert dur2 == dur1

    # Corrupted cache is removed and regenerated
    (work_dir / media.EXTRACTED_AUDIO_NAME).write_bytes(b"corrupted")
    _, dur3, _, reused3 = media.extract_complete_audio(
        str(video_fixture["path"]), str(work_dir)
    )
    assert reused3 is False
    assert dur3 > 0
