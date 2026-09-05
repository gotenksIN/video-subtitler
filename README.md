# Gemini subtitle generator

This Linux amd64 command-line tool creates English WebVTT subtitles from video.
It uses an audio-first pipeline with four Gemini passes:

1. Research participant identities and topic terminology before chunk generation.
2. Generate subtitles from stream-copy video chunks in parallel with Gemini Flash.
3. Repair chunk-boundary faults against the complete extracted audio track.
4. Proofread the complete script with grounded preflight context.

FFmpeg performs local media processing.
The executable does not require Python or `uv`.

## Requirements

- Linux amd64 host.
- `ffmpeg` and `ffprobe` in `PATH`.
- A Gemini API key.
- Go 1.27 or later when you compile from source.

The optional helper `scripts/yt-dl.sh` requires `uvx` to run `yt-dlp`.
The Go subtitle executable and `scripts/subtitle.sh` do not use Python, `uv`, or `uvx`.

To install a headless static FFmpeg release on Linux hosts, run:

```bash
./scripts/ffmpeg.sh
```

## Install the repository binary

Keep the executable at `bin/video-subtitler` inside this repository.
Run the installation commands from the repository root.

### Build with Go

If Go is available, compile the executable:

```bash
mkdir -p bin
CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -trimpath -o bin/video-subtitler ./cmd/video-subtitler
```

The build works on `go-rewrite` without changing the default branch.
See [Development](#development) for the maintained local toolchain paths.

### Download without Go

If Go is unavailable, open [GitHub Releases](https://github.com/gotenksIN/video-subtitler/releases) and choose a release that matches your checkout.
Download the Linux amd64 assets `video-subtitler` and `video-subtitler.sha256` into the repository's `bin/` directory.
Create that directory first if needed:

```bash
mkdir -p bin
```

Verify the download, then make the binary executable:

```bash
(cd bin && sha256sum --check video-subtitler.sha256) && chmod +x bin/video-subtitler
```

If no matching release exists, build with Go or publish one through the [manual release workflow](#ci-and-manual-release).
Both installation methods provide the same repository-local executable for `scripts/subtitle.sh`.

## Configuration and precedence

Copy `.env.example` to `.env` in the repository root:

```bash
cp .env.example .env
```

Set your Gemini API key in `.env`:

```env
GEMINI_API_KEY=your_api_key_here
```

The executable resolves `.env` relative to its repository `bin` directory.
It finds the repository `.env` when you run the binary from any working directory.

Configuration values follow this strict precedence:

1. Command-line options take precedence over environment variables and `.env`.
2. Process environment variables take precedence over `.env` values.
3. Repository `.env` values provide defaults when environment variables are unset.
4. Built-in defaults apply when no option, environment variable, or `.env` entry exists.

| Variable | CLI option | Default | Description |
| --- | --- | --- | --- |
| `GEMINI_API_KEY` | `--api-key` | None | Required Gemini API credential. |
| `GEMINI_API_BASE` | `--base-url` | SDK default | Optional Gemini-compatible proxy base URL. |
| `GEMINI_MODEL` | `--model` | `gemini-3.8-flash` | Video chunk subtitle generation model. |
| `GEMINI_AUDIO_REFINE_MODEL` | `--audio-refine-model` | `gemini-3.8-flash` | Chunk-boundary audio repair model. |
| `GEMINI_REFINE_MODEL` | `--refine-model` | `gemini-3.1-pro-preview` | Full-script grounded text refinement model. |

## Generate subtitles

Run the executable directly:

```bash
bin/video-subtitler "video.webm" --output "video.vtt"
```

You can place command-line options before or after the input file path.

You can also run the wrapper script from any working directory:

```bash
/path/to/video-subtitler/scripts/subtitle.sh "video.webm"
```

The wrapper invokes the repository binary at `bin/video-subtitler` and writes output to `<video>.vtt`.
When standard input is a terminal, the script prompts for one optional grounding URL.

Supported primary video codecs are VP9, H.264, and HEVC/H.265.
VP9 chunks use WebM containers.
H.264 and HEVC chunks use MP4 containers.
Unsupported codecs return an error during probing.

### Options

- `-o`, `--output PATH`: Subtitle output path.
  Default: `output_subtitles.vtt`.
- `--api-key KEY`: Override `GEMINI_API_KEY`.
- `--base-url URL`: Override `GEMINI_API_BASE` for a compatible proxy endpoint.
- `--model MODEL`: Chunk generation model.
  Default: `gemini-3.8-flash`.
- `--audio-refine-model MODEL`: Boundary audio refinement model.
  Default: `gemini-3.8-flash`.
- `--refine-model MODEL`: Global text refinement model.
  Default: `gemini-3.1-pro-preview`.
- `--chunk-dur SECONDS`: Requested chunk duration in seconds.
  Default: `60`.
- `--workers COUNT`: Maximum concurrent chunk generation workers.
  Default: `7`.
- `--thinking-level LEVEL`: Thinking level for chunk generation.
  Accepted choices: `minimal`, `low`, `medium`, `high`.
  Default: `high`.
  `minimal` requires a Flash model.
- `--context-url URL`: Grounding URL for preflight research and text refinement.
  Repeat this option to supply multiple URLs.
  Public YouTube watch and share URLs use direct video analysis.
  Other HTTP and HTTPS URLs use Google URL Context.
- `--disable-audio-refine`: Skip boundary audio repair.
  Use this option for silent videos or when source audio extraction is unnecessary.
- `--disable-text-refine`: Skip global text refinement.
- `--refine-only`: Refine an existing WebVTT file without video processing.
- `-h`, `--help`: Show command help.

### Refine existing subtitles

Run global text refinement on an existing subtitle file without video processing:

```bash
bin/video-subtitler subtitles.vtt --refine-only -o polished.vtt --context-url "https://example.com/topic"
```

## Work state, resume, and output safety

The tool creates a deterministic work directory under `temp_video_chunks/<manifest-hash>/`.
The 16-character hash prefix identifies the source video fingerprint, chunk duration, model, codec, and format.

Manifest hashes and cache identities use Go JSON serialization.
The pipeline reuses valid artifacts from previous Go runs:
- `segments.csv` and matching chunk video files skip stream-copy splitting.
- Valid `subtitle_chunk_NNN.json` files skip chunk model requests.
- Valid `extracted_audio.ogg` files skip complete audio extraction.
- Valid `preflight_context.json` files skip preflight web research.
- Valid `audio_refinement.json` matching the audio cache identity skips boundary audio repair.

No automatic cache revision invalidates stored work.
Failed runs retain valid artifacts to support immediate resume.
Successful runs clean intermediate work files while retaining directory and lock inodes.

The process acquires two exclusive non-blocking POSIX file locks before processing:
1. `temp_video_chunks/<manifest-hash>/.lock` serializes access to the work directory.
2. `.<output>.video-subtitler.lock` serializes access to the target output file.

Both lock files store the running process ID.
Lock inodes remain in place to avoid deletion races between processes.
Locks release automatically when the process exits.

Intermediate JSON caches, staging files, and final VTT outputs write to temporary files before atomic replacement with `os.Rename`.
An interrupted run or error never leaves a corrupted final output file.

## Development

Packages follow an acyclic architecture:
- `internal/vtt`: WebVTT parsing, formatting, and atomic file publication.
- `internal/storage`: Atomic JSON publication, file fingerprints, and cache hashing.
- `internal/core`: Schemas, timestamps, title derivation, URL policies, cue classification, speaker casing, and repair envelopes.
- `internal/media`: FFmpeg and FFprobe media operations.
- `internal/gemini`: Gemini client management, prompts, response parsing, and cache validation.
- `internal/pipeline`: Process locking, worker scheduling, segment stitching, and run lifecycle.
- `cmd/video-subtitler`: CLI argument parsing, environment loading, and dispatch.

Run offline verification checks with portable defaults:

```bash
gofmt -w cmd internal
go test ./...
go vet ./...
CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -trimpath -o bin/video-subtitler ./cmd/video-subtitler
shellcheck scripts/*.sh
```

In the maintained development environment:
- Go installation (`GOROOT`): `/home/gotenks/Projects/go`.
- Go tools: `/home/gotenks/Projects/go/bin` (`go`, `gofmt`).
- Workspace (`GOPATH`): `/home/gotenks/Projects/go-workspace`.
- Module cache (`GOMODCACHE`): `/home/gotenks/Projects/go-workspace/pkg/mod`.
- Installed tools binary directory: `/home/gotenks/Projects/go-workspace/bin`.
- Build cache: `~/.cache/go-build`.
- LLVM tools: `/home/gotenks/Projects/llvm/bin`.
- C compiler (`CC`): `/home/gotenks/Projects/llvm/bin/clang`.

Run full local verification including the race detector:

```bash
PATH=/home/gotenks/Projects/go/bin:/home/gotenks/Projects/llvm/bin:$PATH gofmt -w cmd internal
PATH=/home/gotenks/Projects/go/bin:/home/gotenks/Projects/llvm/bin:$PATH CC=/home/gotenks/Projects/llvm/bin/clang CGO_ENABLED=1 /home/gotenks/Projects/go/bin/go test -race ./...
PATH=/home/gotenks/Projects/go/bin:$PATH /home/gotenks/Projects/go/bin/go vet ./...
CGO_ENABLED=0 GOOS=linux GOARCH=amd64 /home/gotenks/Projects/go/bin/go build -trimpath -o bin/video-subtitler ./cmd/video-subtitler
shellcheck scripts/*.sh
```

Offline tests use local HTTP test servers and synthetic FFmpeg media fixtures.
They do not make live or paid API calls.
Continuous integration also executes race-enabled tests in Ubuntu runner environments.

## CI and manual release

The CI workflow `.github/workflows/ci.yml` runs on pull requests and pushes to the `go-rewrite` branch.
It validates code formatting, runs the test suite with the race detector, runs `go vet`, checks shell scripts with `shellcheck`, and builds a static Linux amd64 executable.

The release workflow `.github/workflows/release.yml` uses only manual `workflow_dispatch` triggers.
GitHub registers a `workflow_dispatch` action only after the workflow file exists on the default repository branch.
To register the action:
1. Merge or copy `.github/workflows/release.yml` to the default branch without adding automatic push or tag triggers.
2. Open the GitHub Actions tab and select **Manual release**.
3. Enter `go-rewrite` or an exact commit SHA in the `ref` input field.
4. Enter the desired release tag in the `tag` input field.

The workflow resolves the exact commit SHA of the selected ref, checks out that commit, runs full offline verification, compiles a static Linux amd64 binary with stripped symbols, computes its SHA256 checksum, tags that exact commit, and publishes the GitHub release.
Releases never trigger automatically from branch pushes or tag pushes.
