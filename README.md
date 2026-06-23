# Gemini Subtitle Generator

A Python CLI that uses the Google Gemini API to generate new English subtitles from scratch. It splits video into chunks, adds overlapping context clips by default, processes them concurrently, stitches validated JSON results into a final VTT file, and optionally runs a full-script text refinement pass.

## Features

- **Generation Mode:** Pass a video file, and Gemini generates English subtitles with accurate timestamps from scratch.
- **Concurrent Processing:** Processes chunks in parallel using multiple Gemini API workers.
- **Structured Outputs:** Uses Pydantic JSON schemas and local validation to catch malformed timestamps, duplicate IDs, and invalid chunk output.
- **Resumable Failures:** Failed runs keep their work directory so a retry can skip valid completed chunks. Successful runs clean up temporary chunks.
- **Safe Outputs:** Writes chunk JSON and the final VTT atomically to avoid corrupting previous results.

## Prerequisites

- [uv](https://github.com/astral-sh/uv) - Fast Python package installer and resolver.
- [FFmpeg](https://ffmpeg.org/) - `ffmpeg` and `ffprobe` must be installed and available in your system's PATH.
  - **Headless / X11-Free Installation (WSL/Ubuntu Server):** To avoid installing heavy X11/GUI dependencies, install or upgrade a pre-compiled GPL static build from the official [BtbN/FFmpeg-Builds](https://github.com/BtbN/FFmpeg-Builds) repository directly into your local user binary directory:
    ```bash
    ./scripts/ffmpeg.sh
    ```
    Ensure `~/.local/bin` is in your `PATH` (by adding `export PATH="$HOME/.local/bin:$PATH"` to your `~/.bashrc` or `~/.zshrc`).
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) - Optional, run automatically on-demand via `uvx` (no manual installation needed).

## Installation

1. Clone this repository.
2. Install the required dependencies using `uv`:
   ```bash
   uv sync
   ```
3. Create a `.env` file in the root directory and add your Gemini API credentials:
   ```env
   GEMINI_API_KEY=your_api_key_here

   # Optional: Set a custom base URL or change the model
   GEMINI_API_BASE=https://main.your-proxy-domain.com/google/v1beta
   GEMINI_MODEL=gemini-3.1-pro-preview
   ```

## Usage

You can run the script using `uv run`.

### Helper Scripts

Install or upgrade the latest BtbN GPL static FFmpeg build into `~/.local/bin`:
```bash
./scripts/ffmpeg.sh
```

Download a YouTube video as best available VP9 video plus best audio, falling back to best WebM when VP9 is unavailable:
```bash
./scripts/yt-dl.sh "https://youtube.com/watch?v=..."
```

Generate subtitles for a local video using the repository's preferred worker settings:
```bash
./scripts/subtitle.sh "your_video.webm"
```

The subtitle helper writes to `your_video.webm.vtt`.

Benchmark local overlap clip generation and Gemini processing for a sample clip:
```bash
./scripts/benchmark.py "your_video.webm"
```

The benchmark makes a real Gemini API request and reports FFmpeg time, Gemini time, and a suggested `--workers` value for `scripts/subtitle.sh`.

### Generation Mode (Creating new subtitles from scratch)
To generate completely new English subtitles from a video with no existing VTT:
```bash
uv run python gemini_subs.py "your_video.webm" --output "generated_subtitles.vtt"
```

### Two-Stage Processing & Text Refinement
By default, Gemini runs a **global text refinement pass** on the final VTT after chunk processing. Because chunk-based processing limits context to 60 seconds, a final global pass allows the model to see the *entire* subtitle script at once, fixing inconsistent character names, over-localized memes, and continuity errors without altering the generated timestamps.

To skip the global refinement pass:
```bash
uv run python gemini_subs.py "your_video.webm" --disable-text-refine
```

If you already generated a VTT and only want to run the global text refinement pass (skipping video processing entirely):
```bash
uv run python gemini_subs.py "generated_subtitles.vtt" --refine-only -o "polished_subtitles.vtt"
```

### Additional Options
- `--disable-text-refine`: Disable the global text refinement pass after generation.
- `--refine-only`: Skip video processing entirely; only run global text refinement on the input VTT file.
- `--chunk-dur`: Video chunk duration in seconds (default: `60`)
- `--overlap`: Seconds of extra context to include before and after each chunk. This creates temporary re-encoded overlap clips for better boundary timing. The clip container and video encoder are derived from the input codec. Default: `5`.
- `--workers`: Max concurrent API workers (default: `4`)
- `--thinking-level`: Gemini thinking level for chunk video calls. Default: `minimal` for Flash models, `low` otherwise. The global refinement pass always uses `high`.
- `--api-key`: Override `GEMINI_API_KEY` from `.env` or the environment.
- `--base-url`: Override `GEMINI_API_BASE` for a custom Gemini-compatible proxy.
- `--model`: Override `GEMINI_MODEL`.

## Notes

- The initial split uses `-c copy`. Supported input codecs are VP9, H.264, and HEVC/H.265. VP9 chunks use WebM, while H.264 and HEVC chunks use MP4.
- AV1 input is rejected early because the processing pipeline currently supports VP9, H.264, and HEVC/H.265 only.
- With the default `--overlap 5`, temporary overlap clips are re-encoded with the same video codec family as the input so chunk boundaries can land exactly where subtitle timing needs them.
- Set `--overlap 0` to disable overlap re-encoding and process stream-copy chunks directly.
- Gemini's inline video guidance recommends keeping requests below 20 MB; reduce `--chunk-dur` if chunk uploads fail.
- If any chunk fails validation or API processing, stitching is aborted and the work directory is kept for retry. Successful runs remove their temporary work directory.
- Output VTT files are ignored by Git by default; move or rename files if you want to track specific subtitle outputs.

## Development Checks

```bash
shellcheck scripts/subtitle.sh scripts/yt-dl.sh scripts/ffmpeg.sh
uv run ruff check .
uv run ruff format --check .
uv run python -m compileall -q .
uv run python -m unittest discover -s tests
uv run gemini_subs.py --help
./scripts/benchmark.py --help
```
