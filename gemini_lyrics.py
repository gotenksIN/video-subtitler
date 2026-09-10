"""Command-line entry point for whole-song lyrics generation."""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from modules import gemini, pipeline

load_dotenv()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate source-language timed lyrics from complete song audio."
    )
    parser.add_argument("audio_file", help="Path to a song audio file.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory. The audio directory is used by default.",
    )
    parser.add_argument("--title", help="Song title used for grounded research.")
    parser.add_argument("--artist", help="Artist used for grounded research.")
    parser.add_argument(
        "--target-language",
        help="BCP 47 target language tag for optional bilingual TTML, such as en.",
    )
    parser.add_argument(
        "--flash-model",
        default=os.environ.get("GEMINI_FLASH_MODEL", gemini.DEFAULT_FLASH_MODEL),
        help="Gemini Flash model for the draft transcription.",
    )
    parser.add_argument(
        "--pro-model",
        default=os.environ.get("GEMINI_PRO_MODEL", gemini.DEFAULT_PRO_MODEL),
        help="Gemini Pro model for research, refinement, and translation.",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("GEMINI_API_KEY"),
        help="Gemini API key override.",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("GEMINI_API_BASE"),
        help="Optional Gemini-compatible proxy base URL.",
    )
    arguments = parser.parse_args()
    config = pipeline.GenerationConfig(
        audio_path=Path(arguments.audio_file),
        output_dir=arguments.output_dir,
        api_key=arguments.api_key,
        base_url=arguments.base_url,
        title=arguments.title,
        artist=arguments.artist,
        target_language=arguments.target_language,
        flash_model=arguments.flash_model,
        pro_model=arguments.pro_model,
    )
    try:
        pipeline.run_generation(config)
    except Exception as error:  # noqa: BLE001 - Convert failures to CLI diagnostics.
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
