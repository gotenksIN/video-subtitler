"""Real media fixtures shared by capability tests."""

import shutil
import subprocess

import pytest


@pytest.fixture(scope="session")
def media_files(tmp_path_factory):
    for binary in ("ffmpeg", "ffprobe"):
        if not shutil.which(binary):
            pytest.fail(f"Required media tool is absent: {binary}")
    root = tmp_path_factory.mktemp("real-media")

    def create(name, encoder, *, duration=2, audio=True, audio_duration=None, extra=()):
        path = root / name
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"color=c=blue:s=64x64:r=4:d={duration}",
        ]
        if audio:
            command += [
                "-f",
                "lavfi",
                "-i",
                f"sine=frequency=440:duration={audio_duration or duration}",
            ]
        command += ["-c:v", encoder, "-g", "4", "-pix_fmt", "yuv420p", "-threads", "1"]
        if encoder == "libx264":
            command += ["-preset", "ultrafast", "-bf", "0"]
        elif encoder == "libx265":
            command += [
                "-preset",
                "ultrafast",
                "-x265-params",
                "pools=none:frame-threads=1:log-level=error:bframes=0",
            ]
        elif encoder == "libvpx-vp9":
            command += ["-deadline", "realtime", "-cpu-used", "8"]
        if audio:
            command += ["-c:a", "libopus" if name.endswith(".webm") else "aac"]
        result = subprocess.run(
            [*command, *extra, str(path)], capture_output=True, text=True, check=False
        )
        if result.returncode:
            pytest.fail(f"FFmpeg fixture requires encoder {encoder}: {result.stderr}")
        return path

    files = {
        "h264": create("h264.mp4", "libx264"),
        "vp9": create("vp9.webm", "libvpx-vp9"),
        "hevc": create("hevc.mp4", "libx265"),
        "silent": create("silent.mp4", "libx264", audio=False),
        "unsupported": create("mpeg4.mp4", "mpeg4", audio=False),
        "long": create("long.mp4", "libx264", duration=32, audio_duration=30.5),
    }
    subtitle = root / "embedded.srt"
    subtitle.write_text(
        "1\n00:00:00,000 --> 00:00:01,000\nEmbedded\n", encoding="ascii"
    )
    files["subtitled"] = root / "subtitled.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(files["h264"]),
            "-i",
            str(subtitle),
            "-map",
            "0",
            "-map",
            "1",
            "-c",
            "copy",
            "-c:s",
            "mov_text",
            str(files["subtitled"]),
        ],
        check=True,
        capture_output=True,
    )
    return files


@pytest.fixture
def fast_retries(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _delay: None)
