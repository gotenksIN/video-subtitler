"""Command-line entry point for VTT subtitle generation and refinement."""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from modules import core, gemini, pipeline

load_dotenv()


def main():
    parser = argparse.ArgumentParser(
        description="Generate English WebVTT subtitles from video using the Gemini API."
    )
    parser.add_argument(
        "video_file_or_vtt",
        help="Path to the source video file, or input WebVTT file when using --refine-only.",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="output_subtitles.vtt",
        help="Output path for the generated WebVTT file.",
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
    parser.add_argument(
        "--model",
        default=os.environ.get("GEMINI_MODEL", gemini.DEFAULT_CHUNK_MODEL),
        help="Gemini model for chunk video subtitle generation.",
    )
    parser.add_argument(
        "--refine-model",
        default=os.environ.get("GEMINI_REFINE_MODEL", gemini.DEFAULT_REFINE_MODEL),
        help="Gemini model for the global text refinement pass.",
    )
    parser.add_argument(
        "--audio-refine-model",
        default=os.environ.get(
            "GEMINI_AUDIO_REFINE_MODEL", gemini.DEFAULT_AUDIO_REFINE_MODEL
        ),
        help="Gemini model for the boundary audio refinement pass.",
    )
    parser.add_argument(
        "--disable-audio-refine",
        action="store_true",
        help="Disable the boundary audio refinement pass after generation.",
    )
    parser.add_argument(
        "--disable-text-refine",
        action="store_true",
        help="Disable the global text refinement pass after generation.",
    )
    parser.add_argument(
        "--refine-only",
        action="store_true",
        help="Skip video processing and run global text refinement on an input WebVTT file.",
    )
    parser.add_argument(
        "--chunk-dur",
        type=int,
        default=60,
        help="Video chunk duration in seconds.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=pipeline.DEFAULT_API_WORKERS,
        help="Maximum concurrent API workers.",
    )
    parser.add_argument(
        "--thinking-level",
        choices=gemini.THINKING_LEVELS,
        default=None,
        help=(
            "Gemini thinking level for chunk video requests (minimal, low, medium, high). "
            "minimal requires a Flash model."
        ),
    )
    parser.add_argument(
        "--context-url",
        action="append",
        default=None,
        help=(
            "Absolute HTTP(S) URL used as grounding context for global refinement. "
            "Repeat the option to supply several URLs."
        ),
    )

    args = parser.parse_args()

    if args.refine_only:
        try:
            context_urls = core.validate_context_urls(args.context_url)
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)
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
        audio_refine_model=args.audio_refine_model,
        chunk_dur=args.chunk_dur,
        workers=args.workers,
        thinking_level=args.thinking_level,
        audio_refine=not args.disable_audio_refine,
        refine_text=not args.disable_text_refine,
        context_urls=tuple(args.context_url or ()),
    )
    try:
        pipeline.run_generation(config)
    except Exception as e:  # noqa: BLE001 - Convert pipeline failures to CLI errors.
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
