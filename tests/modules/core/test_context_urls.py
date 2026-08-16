"""Context URL validation, YouTube detection, and classification."""

import pytest

from modules import core


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
        core.validate_context_urls([url])


def test_context_url_validation_deduplicates_preserving_first_occurrence():
    result = core.validate_context_urls(
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
    assert core.is_youtube_video_url(url)


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
    assert not core.is_youtube_video_url(url)


def test_context_url_classification_splits_youtube_from_ordinary():
    youtube_urls, ordinary_urls = core.classify_context_urls(
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
