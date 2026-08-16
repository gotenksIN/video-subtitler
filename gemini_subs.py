"""Command-line entry point for VTT subtitle generation and refinement."""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from modules import core, gemini, pipeline

# Load environment variables from .env file
load_dotenv()


def main():
    parser = argparse.ArgumentParser(
        description="Generate VTT subtitles for a video using Gemini API."
    )
    parser.add_argument(
        "video_file_or_vtt",
        help="Path to the original video file (OR path to input VTT if --refine-only is used)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="output_subtitles.vtt",
        help="Output path for the generated VTT file",
    )
    parser.add_argument(
        "--api-key", default=os.environ.get("GEMINI_API_KEY"), help="Gemini API Key"
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("GEMINI_API_BASE"),
        help="Base URL for Gemini API (optional)",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("GEMINI_MODEL", gemini.DEFAULT_CHUNK_MODEL),
        help="Gemini model to use for chunk video generation",
    )
    parser.add_argument(
        "--refine-model",
        default=os.environ.get("GEMINI_REFINE_MODEL", gemini.DEFAULT_REFINE_MODEL),
        help="Gemini model to use for the global refinement pass",
    )
    parser.add_argument(
        "--disable-text-refine",
        action="store_true",
        help="Disable the global text refinement pass after generation",
    )
    parser.add_argument(
        "--refine-only",
        action="store_true",
        help="Skip video processing entirely; only run global text refinement on the input VTT file",
    )
    parser.add_argument(
        "--chunk-dur",
        type=int,
        default=60,
        help="Chunk duration in seconds (default: 60)",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=5.0,
        help="Seconds of context to add before and after each chunk (default: 5)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=pipeline.DEFAULT_API_WORKERS,
        help="Max concurrent API workers",
    )
    parser.add_argument(
        "--thinking-level",
        choices=gemini.THINKING_LEVELS,
        default=None,
        help=(
            "Chunk Gemini thinking level. Default: high. "
            "Lowest supported: minimal for Flash models, low otherwise."
        ),
    )
    parser.add_argument(
        "--context-url",
        action="append",
        default=None,
        help=(
            "Absolute HTTP(S) URL used as grounding context for global "
            "refinement. Repeatable. Public YouTube watch or share URLs are "
            "analyzed in a separate direct-video pass. Other URLs use the "
            "URL Context tool and refinement fails if one is not retrieved "
            "successfully."
        ),
    )

    args = parser.parse_args()

    try:
        context_urls = core.validate_context_urls(args.context_url)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    if args.refine_only:
        if not os.path.exists(args.video_file_or_vtt):
            print(f"Error: Input VTT file not found: {args.video_file_or_vtt}")
            sys.exit(1)
        if not args.api_key:
            print(
                "Error: Gemini API key not configured. Set GEMINI_API_KEY in .env or the environment, or pass --api-key."
            )
            sys.exit(1)
        try:
            gemini.global_refine_subtitles(
                args.video_file_or_vtt,
                args.output,
                args.api_key,
                args.base_url,
                args.refine_model or args.model,
                gemini.REFINEMENT_THINKING_LEVEL,
                source_title=core.derive_source_title(Path(args.video_file_or_vtt)),
                context_urls=context_urls,
            )
        except Exception as e:  # noqa: BLE001 - Convert refinement failures to CLI errors.
            print(f"Error: {e}")
            sys.exit(1)
        sys.exit(0)

    config = pipeline.GenerationConfig(
        video_path=Path(args.video_file_or_vtt),
        output_path=Path(args.output),
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
        refine_model=args.refine_model,
        chunk_dur=args.chunk_dur,
        overlap=args.overlap,
        workers=args.workers,
        thinking_level=args.thinking_level,
        refine_text=not args.disable_text_refine,
        context_urls=tuple(context_urls),
    )
    try:
        pipeline.run_generation(config)
    except Exception as e:  # noqa: BLE001 - Convert pipeline failures to CLI errors.
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
