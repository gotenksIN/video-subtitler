# Gemini Subtitle Aligner & Generator

A Python tool that uses the Google Gemini API to align existing VTT subtitles to a video or generate new English subtitles from scratch. It splits the video into chunks using FFmpeg stream-copying, processes them concurrently to avoid long API/proxy requests, and stitches the validated JSON results back into a final VTT file.

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
   GEMINI_MODEL=gemini-3-flash-preview
   ```

## Usage

You can run the script using `uv run`.

### Alignment Mode (Fixing existing VTT timestamps)
To fix broken timestamps in an existing VTT file while preserving the original text:
```bash
uv run python gemini_subs.py "your_video.webm" "your_subtitles.vtt" --output "fixed_output.vtt"
```

If the original subtitles are mistranslated and you want Gemini to correct the English while keeping the same cue structure:
```bash
uv run python gemini_subs.py "your_video.webm" "your_subtitles.vtt" --text-mode fix --output "fixed_output.vtt"
```

If captions near chunk boundaries still drift, add overlap context around each chunk:
```bash
uv run python gemini_subs.py "your_video.webm" "your_subtitles.vtt" --text-mode fix --overlap 5 --output "fixed_output.vtt"
```

### Generation Mode (Creating new subtitles from scratch)
To generate completely new English subtitles from a video with no existing VTT:
```bash
uv run python gemini_subs.py "your_video.webm" --output "generated_subtitles.vtt"
```
`--text-mode` is ignored in generation mode because there is no existing VTT text to preserve or fix.

### Additional Options
- `--chunk-dur`: Video chunk duration in seconds (default: `60`)
- `--overlap`: Seconds of extra context to include before and after each chunk. This creates temporary re-encoded overlap clips for better boundary timing.
- `--overlap-format`: Container for overlap clips. Default: `webm`.
- `--clip-workers`: Number of overlap clip encodes to run in parallel. `0` uses an automatic value.
- `--workers`: Max concurrent API workers (default: `4`)
- `--api-key`: Override `GEMINI_API_KEY` from `.env` or the environment.
- `--base-url`: Override `GEMINI_API_BASE` for a custom Gemini-compatible proxy.
- `--model`: Override `GEMINI_MODEL`.
- `--text-mode`: In alignment mode, either preserve the original subtitle text or let the model fix awkward translation. Choices: `preserve`, `fix`.
- `--keep-chunks`: Keep the per-input work directory under `temp_video_chunks/` after successful processing.

## Notes

- The script uses `-c copy` to avoid re-encoding. It parses the FFmpeg segment manifest to account for dynamic keyframe chunk durations, guaranteeing precise subtitle timestamps without the CPU overhead of re-encoding.
- Alignment mode assigns captions to chunks by cue midpoint instead of raw start time, which reduces drift for lines that straddle chunk boundaries.
- `--overlap` is the quality-first option: it re-encodes temporary context clips so chunk boundaries can land exactly where subtitle timing needs them.
- Gemini's inline video guidance recommends keeping requests below 20 MB; reduce `--chunk-dur` if chunk uploads fail.
- If any chunk fails validation or API processing, stitching is aborted and the work directory is kept for retry.
- Output VTT files are ignored by Git by default; move or rename files if you want to track specific subtitle outputs.
