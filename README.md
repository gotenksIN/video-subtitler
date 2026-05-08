# Gemini Subtitle Aligner & Generator

A Python tool that uses the Google Gemini API to align existing VTT subtitles to a video or generate new English subtitles from scratch. It splits the video into chunks, adds overlapping context clips by default, processes them concurrently, and stitches the validated JSON results back into a final VTT file.

## Features

- **Alignment Mode:** Pass an existing VTT file, and Gemini will fix the timestamps to perfectly match when dialogue is spoken or when editor text appears on screen.
- **Generation Mode:** Pass just a video file, and Gemini will generate translated English subtitles with accurate timestamps from scratch.
- **Concurrent Processing:** Processes chunks in parallel using multiple Gemini API workers.
- **Structured Outputs:** Uses Pydantic JSON schemas and local validation to catch malformed timestamps, dropped captions, or edited alignment text.
- **Resumable:** Failed runs keep their work directory so a retry can skip valid completed chunks.
- **Safe Outputs:** Writes chunk JSON and the final VTT atomically to avoid corrupting previous results.

## Prerequisites

- [uv](https://github.com/astral-sh/uv) - Fast Python package installer and resolver.
- [FFmpeg](https://ffmpeg.org/) - `ffmpeg` and `ffprobe` must be installed and available in your system's PATH.

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

### Alignment Mode (Fixing existing VTT timestamps)
To fix broken timestamps in an existing VTT file while preserving the original text:
```bash
uv run python gemini_subs.py "your_video.webm" "your_subtitles.vtt" --output "fixed_output.vtt"
```

To change the default 5-second boundary context window:
```bash
uv run python gemini_subs.py "your_video.webm" "your_subtitles.vtt" --overlap 3 --output "fixed_output.vtt"
```

### Generation Mode (Creating new subtitles from scratch)
To generate completely new English subtitles from a video with no existing VTT:
```bash
uv run python gemini_subs.py "your_video.webm" --output "generated_subtitles.vtt"
```

### Two-Stage Processing & Text Refinement
By default, Gemini runs a **global text refinement pass** on the final VTT after chunk processing. Because chunk-based processing limits context to 60 seconds, a final global pass allows the model to see the *entire* subtitle script at once, fixing inconsistent character names, over-localized memes, and continuity errors without altering the video-aligned timestamps.

To skip the global refinement pass:
```bash
uv run python gemini_subs.py "your_video.webm" --disable-text-refine
```

If you already generated a VTT and only want to run the global text refinement pass (skipping video processing entirely):
```bash
uv run python gemini_subs.py "generated_subtitles.vtt" --refine-only -o "polished_subtitles.vtt"
```

### Additional Options
- `--disable-text-refine`: Disable the global text refinement pass after alignment/generation.
- `--refine-only`: Skip video processing entirely; only run global text refinement on the input VTT file.
- `--chunk-dur`: Video chunk duration in seconds (default: `60`)
- `--overlap`: Seconds of extra context to include before and after each chunk. This creates temporary re-encoded overlap clips for better boundary timing. Default: `5`.
- `--overlap-format`: Container for overlap clips. Default: `mp4`.
- `--clip-workers`: Number of overlap clip encodes to run in parallel. `0` uses an automatic value.
- `--workers`: Max concurrent API workers (default: `4`)
- `--thinking-budget`: Gemini thinking token budget for chunk video calls. Default: `0`. The global refinement pass omits thinking config unless you pass a positive budget.
- `--api-key`: Override `GEMINI_API_KEY` from `.env` or the environment.
- `--base-url`: Override `GEMINI_API_BASE` for a custom Gemini-compatible proxy.
- `--model`: Override `GEMINI_MODEL`.
- `--keep-chunks`: Keep the per-input work directory under `temp_video_chunks/` after successful processing.

## Notes

- The initial split uses `-c copy`. With the default `--overlap 5`, temporary overlap clips are re-encoded so chunk boundaries can land exactly where subtitle timing needs them.
- Alignment mode assigns captions to chunks by cue midpoint instead of raw start time, which reduces drift for lines that straddle chunk boundaries.
- Set `--overlap 0` to disable overlap re-encoding and process stream-copy chunks directly.
- Gemini's inline video guidance recommends keeping requests below 20 MB; reduce `--chunk-dur` if chunk uploads fail.
- If any chunk fails validation or API processing, stitching is aborted and the work directory is kept for retry.
- Output VTT files are ignored by Git by default; move or rename files if you want to track specific subtitle outputs.
