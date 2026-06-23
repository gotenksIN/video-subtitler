import unittest
import subprocess
from types import SimpleNamespace
from unittest.mock import patch

from gemini_subs import (
    build_manifest,
    Caption,
    default_chunk_thinking_level,
    format_time,
    generate_content_config,
    overlap_codec_args,
    parse_time,
    probe_video_format,
    validate_captions,
    validate_thinking_level_for_model,
)


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


class ThinkingConfigTest(unittest.TestCase):
    def test_default_chunk_thinking_level_uses_minimal_for_flash(self):
        self.assertEqual(default_chunk_thinking_level("gemini-3.5-flash"), "minimal")
        self.assertEqual(default_chunk_thinking_level("gemini-3.1-pro-preview"), "low")

    def test_generate_content_config_sets_thinking_level(self):
        config = generate_content_config("low")

        self.assertIsNotNone(config.thinking_config)
        self.assertEqual(config.thinking_config.thinking_level.value, "LOW")
        self.assertIsNone(config.thinking_config.thinking_budget)

    def test_minimal_thinking_level_is_flash_only(self):
        validate_thinking_level_for_model("gemini-3.5-flash", "minimal")

        with self.assertRaisesRegex(ValueError, "Flash models"):
            validate_thinking_level_for_model("gemini-3.1-pro-preview", "minimal")


class VideoFormatTest(unittest.TestCase):
    def test_vp9_webm_uses_webm_chunks(self):
        probe_output = "vp9\nopus\nmatroska,webm\n"

        with patch("gemini_subs.subprocess.run") as run:
            run.return_value.stdout = probe_output

            ext, mime, codec = probe_video_format("input.webm")

        self.assertEqual(ext, ".webm")
        self.assertEqual(mime, "video/webm")
        self.assertEqual(codec, "vp9")

    def test_h264_mkv_uses_mp4_chunks(self):
        probe_output = "h264\naac\nsubrip\nmatroska,webm\n"

        with patch("gemini_subs.subprocess.run") as run:
            run.return_value.stdout = probe_output

            ext, mime, codec = probe_video_format("input.mkv")

        self.assertEqual(ext, ".mp4")
        self.assertEqual(mime, "video/mp4")
        self.assertEqual(codec, "h264")

    def test_hevc_mkv_uses_mp4_chunks_and_hevc_codec(self):
        probe_output = "hevc\naac\nsubrip\nsubrip\nmatroska,webm\n"

        with patch("gemini_subs.subprocess.run") as run:
            run.return_value.stdout = probe_output

            ext, mime, codec = probe_video_format("input.mkv")

        self.assertEqual(ext, ".mp4")
        self.assertEqual(mime, "video/mp4")
        self.assertEqual(codec, "hevc")

    def test_manifest_stores_video_codec(self):
        args = SimpleNamespace(
            video_file="input.mkv",
            vtt_file=None,
            chunk_dur=60,
            model="gemini-3.1-pro-preview",
            chunk_thinking_level="low",
            overlap=5,
        )

        with (
            patch("gemini_subs.probe_video_format", return_value=(".mp4", "video/mp4", "hevc")),
            patch("gemini_subs.file_fingerprint", return_value={"path": "input.mkv", "size": 1, "mtime_ns": 2}),
        ):
            manifest, _chunk_dir = build_manifest(args)

        self.assertEqual(manifest["video_codec"], "hevc")
        self.assertEqual(manifest["process_ext"], ".mp4")
        self.assertEqual(manifest["process_mime"], "video/mp4")

    def test_vp9_manifest_uses_webm_overlap_clips(self):
        args = SimpleNamespace(
            video_file="input.webm",
            vtt_file=None,
            chunk_dur=60,
            model="gemini-3.1-pro-preview",
            chunk_thinking_level="low",
            overlap=5,
        )

        with (
            patch("gemini_subs.probe_video_format", return_value=(".webm", "video/webm", "vp9")),
            patch("gemini_subs.file_fingerprint", return_value={"path": "input.webm", "size": 1, "mtime_ns": 2}),
        ):
            manifest, _chunk_dir = build_manifest(args)

        self.assertEqual(manifest["video_codec"], "vp9")
        self.assertEqual(manifest["process_ext"], ".webm")
        self.assertEqual(manifest["process_mime"], "video/webm")
        self.assertNotIn("overlap_format", manifest)

    def test_unsupported_video_codec_fails_early(self):
        probe_output = "av1\nopus\nmatroska,webm\n"

        with patch("gemini_subs.subprocess.run") as run:
            run.return_value.stdout = probe_output

            with self.assertRaisesRegex(RuntimeError, "Video format not supported"):
                probe_video_format("input.mkv")

    def test_ffprobe_failure_fails_early(self):
        with patch("gemini_subs.subprocess.run") as run:
            run.side_effect = subprocess.CalledProcessError(1, ["ffprobe"])

            with self.assertRaisesRegex(RuntimeError, "Failed to probe video format"):
                probe_video_format("input.mkv")

    def test_mp4_overlap_uses_x264_flags_matching_vp9_quality(self):
        args = overlap_codec_args(".mp4", "h264")

        self.assertIn("libx264", args)
        self.assertIn("-crf", args)
        self.assertIn("32", args)
        self.assertIn("-b:v", args)
        self.assertIn("0", args)
        self.assertIn("-threads", args)
        self.assertIn("8", args)
        self.assertIn("+faststart", args)

    def test_mp4_overlap_uses_x265_for_hevc(self):
        args = overlap_codec_args(".mp4", "hevc")

        self.assertIn("libx265", args)
        self.assertIn("-crf", args)
        self.assertIn("32", args)
        self.assertIn("-threads", args)
        self.assertIn("8", args)
        self.assertIn("+faststart", args)

    def test_unsupported_overlap_format_fails_early(self):
        with self.assertRaisesRegex(ValueError, "requires MP4 overlap clips"):
            overlap_codec_args(".mkv", "h264")

    def test_incompatible_overlap_format_fails_early(self):
        with self.assertRaisesRegex(ValueError, "requires MP4 overlap clips"):
            overlap_codec_args(".webm", "hevc")

    def test_vp9_requires_webm_overlap_clips(self):
        with self.assertRaisesRegex(ValueError, "requires WebM overlap clips"):
            overlap_codec_args(".mp4", "vp9")


if __name__ == "__main__":
    unittest.main()
