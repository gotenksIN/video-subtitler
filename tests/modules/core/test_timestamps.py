"""Timestamp parsing and formatting behavior."""

import pytest

from modules import core


@pytest.mark.parametrize(
    ("value", "seconds"),
    [
        ("0", 0.0),
        ("1.25", 1.25),
        ("0.125", 0.125),
        ("12:34", 754.0),
        ("02:03,5", 123.5),
        ("01:02:03.004", 3723.004),
        ("00:00:00.001", 0.001),
        ("1:2:03.5", 3723.5),
    ],
)
def test_parse_time_accepts_documented_shapes(value, seconds):
    assert core.parse_time(value) == seconds


def test_parse_time_treats_decimal_commas_as_points():
    assert core.parse_time("00:00:01,500") == 1.5


@pytest.mark.parametrize(
    "value",
    ["-0.1", "-00:00:00.1", "1:2:3:4", "abc", "", "1..5", "1.2.3"],
)
def test_parse_time_rejects_negative_or_malformed_values(value):
    with pytest.raises(ValueError):
        core.parse_time(value)


@pytest.mark.parametrize(
    ("seconds", "formatted"),
    [
        (0, "00:00:00.000"),
        (0.001, "00:00:00.001"),
        (0.0004, "00:00:00.000"),
        (60, "00:01:00.000"),
        (59.9996, "00:01:00.000"),
        (3600, "01:00:00.000"),
        (3599.9999, "01:00:00.000"),
        (3661.2346, "01:01:01.235"),
    ],
)
def test_format_time_rounds_to_milliseconds(seconds, formatted):
    assert core.format_time(seconds) == formatted


def test_format_time_rejects_negative_values():
    with pytest.raises(ValueError, match="Negative timestamp"):
        core.format_time(-0.001)


@pytest.mark.parametrize("seconds", [0, 0.001, 1.234, 61.5, 3599.9999, 12345.678])
def test_format_time_round_trips_within_half_a_millisecond(seconds):
    reparsed = core.parse_time(core.format_time(seconds))

    assert abs(reparsed - seconds) <= 0.0005
