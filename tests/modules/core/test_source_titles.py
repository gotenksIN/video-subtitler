"""Source title derivation from media and subtitle filenames."""

import pytest

from modules import core


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("movie.webm.en.vtt", "movie"),
        ("show.ko.vtt", "show"),
        ("movie.en-US.vtt", "movie"),
        ("episode.BTS.webm", "episode.BTS"),
        ("episode.1080p.vtt", "episode.1080p"),
        ("movie.mp4.vtt", "movie"),
        ("plain.vtt", "plain"),
        ("plain.srt", "plain"),
        ("plain.sub", "plain"),
        ("plain.sbv", "plain"),
        ("Talk Show.mp4", "Talk Show"),
        ("movie.WEBM", "movie"),
        ("movie.mkv", "movie"),
        ("movie.mov", "movie"),
        ("movie.avi", "movie"),
        ("movie.m4v", "movie"),
        ("no-extension", "no-extension"),
        ("media.webm.ko", "media.webm.ko"),
        ("full/path/to/Show - Episode 1.mp4", "Show - Episode 1"),
    ],
)
def test_derive_source_title(filename, expected):
    assert core.derive_source_title(filename) == expected
