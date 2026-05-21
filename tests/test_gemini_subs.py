import unittest

from gemini_subs import Caption, format_time, parse_time, validate_captions


class TimeHelpersTest(unittest.TestCase):
    def test_parse_time_accepts_common_timestamp_shapes(self):
        self.assertAlmostEqual(parse_time("01:02:03.456"), 3723.456)
        self.assertAlmostEqual(parse_time("02:03.456"), 123.456)
        self.assertAlmostEqual(parse_time("3,5"), 3.5)

    def test_format_time_rounds_to_milliseconds(self):
        self.assertEqual(format_time(3723.4564), "01:02:03.456")
        self.assertEqual(format_time(3723.4566), "01:02:03.457")

    def test_format_time_rejects_negative_values(self):
        with self.assertRaisesRegex(ValueError, "Negative timestamp"):
            format_time(-0.001)


class CaptionValidationTest(unittest.TestCase):
    def test_generation_rejects_duplicate_ids(self):
        captions = [
            Caption(id=1, start="00:00:00.000", end="00:00:01.000", text="First"),
            Caption(id=1, start="00:00:01.500", end="00:00:02.000", text="Second"),
        ]

        with self.assertRaisesRegex(ValueError, "Duplicate caption IDs"):
            validate_captions(captions, chunk_duration=10.0)

    def test_generation_clamps_long_end_and_heals_overlap(self):
        captions = [
            Caption(id=0, start="00:00:01.000", end="00:00:03.000", text="First"),
            Caption(id=1, start="00:00:02.000", end="00:00:04.000", text="Second"),
        ]

        validated = validate_captions(captions, chunk_duration=2.0)

        self.assertEqual(validated[0]["start"], "00:00:01.000")
        self.assertEqual(validated[0]["end"], "00:00:02.000")
        self.assertEqual(validated[1]["start"], "00:00:02.000")
        self.assertEqual(validated[1]["end"], "00:00:02.500")

    def test_alignment_restores_missing_cues_and_preserves_text(self):
        original_cues = [
            {
                "id": 0,
                "start": "00:00:00.000",
                "end": "00:00:01.000",
                "text": "Original text",
            },
            {
                "id": 1,
                "start": "00:00:01.500",
                "end": "00:00:02.000",
                "text": "Missing cue",
            },
        ]
        captions = [
            Caption(
                id=0,
                start="00:00:00.100",
                end="00:00:01.100",
                text="Model tried to edit this",
            )
        ]

        validated = validate_captions(captions, chunk_duration=3.0, original_cues=original_cues)

        self.assertEqual(len(validated), 2)
        self.assertEqual(validated[0]["start"], "00:00:00.100")
        self.assertEqual(validated[0]["text"], "Original text")
        self.assertEqual(validated[1], original_cues[1])


if __name__ == "__main__":
    unittest.main()
