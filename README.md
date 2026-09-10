# Lyrics generator

This command-line tool generates line-timed lyrics from complete song audio files.
It recovers lyrics directly from audio without requiring published text.
It supports unsegmented audio inputs, including FLAC and MP3 formats.
Gemini Pro grounds song and artist context with Google Search.
Gemini Flash transcribes the source language across the complete audio.
Gemini Pro checks the draft against the complete audio and returns text and timing corrections for existing lines.
Unchanged lines and fields retain their Flash values.
The audio remains authoritative throughout generation.

The tool outputs a same-stem `.lrc` file containing source lyrics.
An optional translation target produces a same-stem `.ttml` file formatted for Gramophone.
The TTML body preserves source lyrics and timing.
Translation metadata links to source lines with internal `itunes:key` attributes.

## Non-goals

The project explicitly excludes these features:
- Video chunking or video stream processing.
- Word-level karaoke timing spans.
- Performer assignment or duet tags.
- Musical part divisions such as verse or chorus tags.
- Transliteration or romanization tracks.

## Requirements

- Python 3.14 or later
- [uv](https://docs.astral.sh/uv/)
- `ffmpeg` and `ffprobe` in `PATH`
- A Gemini API key

On Linux, you can install static FFmpeg binaries in `~/.local/bin`:

```bash
./scripts/ffmpeg.sh
```

Install dependencies:

```bash
uv sync
```

Copy `.env.example` to `.env` and set `GEMINI_API_KEY`:

```bash
cp .env.example .env
```

## Usage

Generate source lyrics next to an audio file:

```bash
uv run python gemini_lyrics.py "song.mp3"
```

Generate source lyrics and bilingual TTML with explicit metadata:

```bash
uv run python gemini_lyrics.py "song.flac" \
  --artist "Artist Name" \
  --title "Song Title" \
  --target-language en \
  --output-dir output
```

### Local benchmark files and output protection

To evaluate output against a reference, provide a local song and same-stem lyric file, for example:
- `3RACHA (Bang Chan, Changbin, HAN).flac`
- `3RACHA (Bang Chan, Changbin, HAN).lrc`

The benchmark LRC file serves only as an offline evaluation target.
The pipeline never reads existing lyric files as prompt context or generation input.

The CLI protects existing files from accidental replacement.
When an output file already exists, validation halts before media processing or API calls.
Running the tool on this local FLAC file without `--output-dir` fails because the same-stem benchmark LRC already exists.
Specify `--output-dir` when testing against benchmark audio:

```bash
uv run python gemini_lyrics.py "3RACHA (Bang Chan, Changbin, HAN).flac" \
  --artist "3RACHA" \
  --title "3RACHA" \
  --target-language en \
  --output-dir lyrics-output
```

## Options

| Option | Environment variable | Default | Description |
| --- | --- | --- | --- |
| `audio_file` | None | Required | Path to input audio file (such as FLAC or MP3). |
| `--output-dir` | None | Audio file directory | Directory for generated `.lrc` and `.ttml` sidecars. |
| `--title` | None | Audio file stem | Song title for grounded search. |
| `--artist` | None | None | Artist name for grounded search. |
| `--target-language` | None | None | BCP 47 language tag for optional bilingual TTML (such as `en`). |
| `--flash-model` | `GEMINI_FLASH_MODEL` | `gemini-3.8-flash` | Gemini model for draft transcription. |
| `--pro-model` | `GEMINI_PRO_MODEL` | `gemini-3.1-pro-preview` | Gemini model for research, refinement, and translation. |
| `--api-key` | `GEMINI_API_KEY` | None | Gemini API key override. |
| `--base-url` | `GEMINI_API_BASE` | None | Optional Gemini-compatible proxy base URL. |

## Player compatibility

The tool produces two output sidecars:

1. **Source-only LRC (`.lrc`)**:
   Emits standard `[mm:ss.xx]` line timestamps with half-up centisecond rounding.
   Empty timestamps mark audible gaps between lyric lines.
   Compatible with standard music players, including Jellyfin, Feishin, MusicBee, and Gramophone.

2. **Bilingual TTML (`.ttml`)**:
   Emits Apple-style TTML with line timing.
   Uses the internal Apple namespace `http://music.apple.com/lyric-ttml-internal` for timing and translation metadata.
   Designed specifically for Gramophone on Android.
   Gramophone pairs the subtitle translation directly with each active source line during playback.

## Pipeline

The production CLI executes a whole-audio pipeline:

1. **Audio normalization**: Extracts the first audio stream to mono Ogg Opus at 48000 Hz and 64 kbps.
2. **Research**: Identifies the song and artist using Google Search grounding with Gemini Pro.
3. **Transcription**: Creates a complete source lyric draft from normalized audio with Gemini Flash.
4. **Refinement**: Applies audio-supported text and timing corrections with Gemini Pro, preserving draft line structure.
5. **Translation**: Translates refined lines by source ID with Gemini Pro when you set `--target-language`.

## Working files

Every invocation runs its selected stages from scratch without caching stage results across runs.
Each run uses a unique directory under `.lyrics-generator-work/` for normalized audio and failure diagnostics.
Successful runs remove their work directory.
Failed runs retain their working files and print the directory for inspection.
Retries start a fresh run.

The CLI reports each request's stage and model, elapsed time, received stream chunks, and separate answer and thought character counts.
It flushes an update on the first chunk, then at most once every five seconds as chunks arrive, and when the stream closes.
These counts reflect SDK response chunks, including thought-only chunks, rather than raw SSE keepalive messages.
If no chunks arrive, the counters stop updating.
The CLI does not print lyrics or reasoning text during streaming.
When the API provides usage metadata, the closing summary includes answer and thought token counts.
Failed structured streams retain their partial answer and count/timing diagnostics in the run directory.

## Development

Run code quality checks and tests:

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
uv run python gemini_lyrics.py --help
```
