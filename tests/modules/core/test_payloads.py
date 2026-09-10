"""Public lyric output format behavior."""

from xml.etree import ElementTree as ET

import pytest

from modules import core


def source_lines():
    return [
        core.LyricLine(
            id="L1", start="00:00:01.000", end="00:00:02.000", text="Annyeong"
        ),
        core.LyricLine(id="L2", start="00:00:03.000", end="00:00:04.000", text="Hello"),
    ]


def test_lrc_keeps_source_text_and_marks_real_gaps():
    assert core.serialize_lrc(source_lines(), 5.0) == (
        "[00:01.00]Annyeong\n[00:02.00]\n[00:03.00]Hello\n[00:04.00]\n"
    )


def test_ttml_links_translation_metadata_to_internal_itunes_line_keys():
    translations = [
        core.TranslationLine(id="L1", text="Hello"),
        core.TranslationLine(id="L2", text="Bonjour"),
    ]

    payload = core.serialize_ttml(source_lines(), translations, 5.0, "en", "Song")
    root = ET.fromstring(payload)
    ns = {"tt": core.TTML_NS, "itunes": core.ITUNES_INTERNAL_NS}
    paragraphs = root.findall(".//tt:p", ns)
    linked = root.findall(".//itunes:translation/itunes:text", ns)

    assert root.attrib[f"{{{core.ITUNES_INTERNAL_NS}}}timing"] == "Line"
    assert [p.attrib[f"{{{core.ITUNES_INTERNAL_NS}}}key"] for p in paragraphs] == [
        "L1",
        "L2",
    ]
    assert [p.text for p in paragraphs] == ["Annyeong", "Hello"]
    assert [(item.attrib["for"], item.text) for item in linked] == [
        ("L1", "Hello"),
        ("L2", "Bonjour"),
    ]


def test_translation_requires_exact_source_id_coverage():
    response = core.TranslationResponse(
        translations=[core.TranslationLine(id="L2", text="Only one")]
    )
    with pytest.raises(ValueError, match="every source line"):
        core.validate_translation(response, source_lines())
