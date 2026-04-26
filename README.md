# Gemini Subtitle Aligner & Generator

A Python tool that uses the Google Gemini API to align existing VTT subtitles to a video or generate new English subtitles from scratch. It splits the video into 60-second chunks and processes them concurrently to bypass API context limits and connection timeouts, then stitches them back together.

## Features

- **Alignment Mode:** Pass an existing VTT file, and Gemini will fix the timestamps to perfectly match when dialogue is spoken or when editor text appears on screen.
- **Generation Mode:** Pass just a video file, and Gemini will generate translated English subtitles with accurate timestamps from scratch.
- **Concurrent Processing:** Splits the video into manageable chunks and processes them in parallel using multithreading for speed.
- **Structured Outputs:** Uses Pydantic JSON schemas to enforce valid millisecond timestamps and prevent data loss.
- **Resumable:** If an API error or timeout occurs, you can safely re-run the script and it will resume by skipping any already-processed chunks.

## Prerequisites

- [uv](https://github.com/astral-sh/uv) - Fast Python package installer and resolver.
- [FFmpeg](https://ffmpeg.org/) - Must be installed and available in your system's PATH.

## Installation

1. Clone this repository.
2. Install the required dependencies using `uv`:
   ```bash
   uv init
   uv add google-genai pydub webvtt-py python-dotenv pydantic
   ```
3. Create a `.env` file in the root directory and add your Gemini API credentials:
   ```env
   GEMINI_API_KEY=your_api_key_here
   
   # Optional: Set a custom base URL or change the model
   GEMINI_API_BASE=https://main.your-proxy-domain.com/google/vertex
   GEMINI_MODEL=gemini-3.1-flash-lite-preview
   ```

## Usage

You can run the script using `uv run`.

### Alignment Mode (Fixing existing VTT timestamps)
To fix broken timestamps in an existing VTT file (maintaining the original editor text):
```bash
uv run python gemini_subs.py "your_video.webm" "your_subtitles.vtt" --output "fixed_output.vtt"
```

### Generation Mode (Creating new subtitles from scratch)
To generate completely new English subtitles from a video with no existing VTT:
```bash
uv run python gemini_subs.py "your_video.webm" --output "generated_subtitles.vtt"
```

### Additional Options
- `--chunk-dur`: Video chunk duration in seconds (default: `60`)
- `--workers`: Max concurrent API workers (default: `4`)
- `--keep-chunks`: Keep the temporary `temp_video_chunks` directory after processing instead of deleting it (useful for debugging).
