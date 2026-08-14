# Gemini subtitle generator

A Python CLI that uses the Google Gemini API to generate English WebVTT subtitles from video.
It splits video into chunks and creates overlapping context clips by default.
It processes clips in parallel, stitches validated JSON results into a WebVTT file, and refines the complete script.

## Features

- **Generation mode:** Provide a video file to create English subtitles with accurate timestamps.
- **Concurrent processing:** Process video chunks in parallel using multiple Gemini API workers.
- **Structured outputs:** Validate model responses with Pydantic schemas to catch malformed timestamps, duplicate IDs, and invalid chunk output.
- **Resumable failures:** Keep temporary work directories on failure so retries reuse valid completed chunks.
  Clean up temporary chunk files on success.
- **Safe outputs:** Write chunk JSON and final WebVTT files atomically to prevent corrupted output.

## Prerequisites

- [uv](https://github.com/astral-sh/uv) - Python package installer and resolver.
- [FFmpeg](https://ffmpeg.org/) - Install `ffmpeg` and `ffprobe` in your system `PATH`.
  - **Headless and X11-free installation (WSL/Ubuntu Server):** Install a precompiled GPL static build to avoid X11 and GUI dependencies.
    Use the BtbN static build installer for your user binary directory:
    ```bash
    ./scripts/ffmpeg.sh
    ```
    Add `~/.local/bin` to your `PATH` by adding `export PATH="$HOME/.local/bin:$PATH"` to your shell profile.
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) - Optional tool executed automatically with `uvx`.

## Installation

1. Clone this repository.
2. Install dependencies with `uv`:
   ```bash
   uv sync
   ```
3. Create a `.env` file in the root directory and add your Gemini API credentials:
   ```env
   GEMINI_API_KEY=your_api_key_here

   # Optional: set a custom base URL or change default models
   GEMINI_API_BASE=https://main.your-proxy-domain.com/google/v1beta
   GEMINI_MODEL=gemini-3.7-flash
   GEMINI_REFINE_MODEL=gemini-3.1-pro-preview
   ```

## Usage

Run the CLI using `uv run`.

### Helper scripts

Install or upgrade the BtbN GPL static FFmpeg build in `~/.local/bin`:
```bash
./scripts/ffmpeg.sh
```

Download a YouTube video as VP9 video with audio, falling back to WebM when VP9 is unavailable:
```bash
./scripts/yt-dl.sh "https://youtube.com/watch?v=..."
```

Generate subtitles for a local video:
```bash
./scripts/subtitle.sh "your_video.webm"
```

The subtitle helper writes output to `your_video.webm.vtt`.

Benchmark subtitle generation and refinement models across full video runs:
```bash
./scripts/benchmark.py "your_video.webm" --case gemini-3.7-flash:gemini-3.1-pro-preview
```

Compare generated subtitles against a reference WebVTT file:
```bash
./scripts/benchmark.py "your_video.webm" --model gemini-3.7-flash --reference-vtt "reference.vtt"
```

The benchmark saves generated subtitles and `benchmark-results.json` to the output directory.

### Generation mode

Generate English subtitles from a video:
```bash
uv run python gemini_subs.py "your_video.webm" --output "generated_subtitles.vtt"
```

### Two-stage processing and text refinement

By default, the pipeline uses `gemini-3.7-flash` for chunk video generation and `gemini-3.1-pro-preview` for global text refinement.
Chunk processing limits context to 60 seconds per chunk.
A final global pass provides the complete subtitle script to the refinement model.
It corrects inconsistent character names, terminology, and continuity errors without changing timestamps.

To skip the global refinement pass:
```bash
uv run python gemini_subs.py "your_video.webm" --disable-text-refine
```

To run global text refinement on an existing WebVTT file without video processing:
```bash
uv run python gemini_subs.py "generated_subtitles.vtt" --refine-only -o "polished_subtitles.vtt"
```

### Additional options

- `--disable-text-refine`: Disable the global text refinement pass after generation.
- `--refine-only`: Skip video processing and run global text refinement on an input WebVTT file.
- `--chunk-dur`: Video chunk duration in seconds (default: `60`).
- `--overlap`: Seconds of context to add before and after each chunk (default: `5.0`).
  This creates temporary re-encoded overlap clips for accurate boundary timing.
  The input codec determines the clip container and video encoder.
- `--workers`: Maximum concurrent API workers (default: `7`).
- `--thinking-level`: Gemini thinking level for chunk video requests (default: `high`).
  Supported levels are `minimal`, `low`, `medium`, and `high`.
  `minimal` requires a Flash model.
  The global refinement pass always uses `medium`.
- `--api-key`: Override `GEMINI_API_KEY` from `.env` or the environment.
- `--base-url`: Override `GEMINI_API_BASE` for a custom Gemini-compatible proxy.
- `--model`: Override `GEMINI_MODEL` for chunk video generation (default: `gemini-3.7-flash`).
- `--refine-model`: Override `GEMINI_REFINE_MODEL` for global text refinement (default: `gemini-3.1-pro-preview`).

## Notes

- The initial split uses stream copy (`-c copy`).
  Supported input codecs are VP9, H.264, and HEVC/H.265.
  VP9 chunks use WebM format, while H.264 and HEVC chunks use MP4 format.
- AV1 input is rejected during probing because the processing pipeline supports VP9, H.264, and HEVC/H.265 only.
- With the default `--overlap 5`, temporary overlap clips are re-encoded with the matching video codec family so chunk boundaries align with subtitle timing.
- Set `--overlap 0` to disable overlap re-encoding and process stream-copy chunks directly.
- Keep inline video requests below 20 MiB; reduce `--chunk-dur` if chunk uploads fail.
- When a chunk fails validation or API processing, stitching stops and the work directory is preserved for retry.
  Successful runs clean up the temporary work directory.
- Output WebVTT files are ignored by Git by default.
  Move or rename files to track specific subtitle outputs.

## Development checks

```bash
shellcheck scripts/subtitle.sh scripts/yt-dl.sh scripts/ffmpeg.sh
uv run ruff check .
uv run ruff format --check .
uv run python -m compileall -q .
uv run pytest
uv run gemini_subs.py --help
./scripts/benchmark.py --help
```
