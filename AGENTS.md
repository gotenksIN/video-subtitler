# Agent specification

## Agent skills

### Issue tracker

Track issues in GitHub Issues for `gotenksIN/video-subtitler`.
See `docs/agents/issue-tracker.md`.

### Triage labels

Use the five default triage labels.
See `docs/agents/triage-labels.md`.

### Domain docs

Use the single-context domain layout.
See `docs/agents/domain.md`.

`CONTEXT.md` is the authoritative technical and architectural specification.
Read it before changing code, tests, scripts, or documentation.
Update it when architecture, schemas, pipeline flow, or domain concepts change.

## Development rules

Use ASCII for documentation, code, and comments unless existing content or a Unicode behavior fixture requires another character set.
Keep comments rare and explain non-obvious behavior.
Write readable idiomatic Go with one statement per line, normal multiline control flow, descriptive names, explicit error handling, and manageable functions.
Run `gofmt` as you work.
Do not use compressed semicolon-delimited Go.

Maintain the authoritative local development environment:
- `GOROOT=/home/gotenks/Projects/go`
- Go tools live at `/home/gotenks/Projects/go/bin` (`go`, `gofmt`).
- `GOPATH=/home/gotenks/Projects/go-workspace`
- `GOMODCACHE=/home/gotenks/Projects/go-workspace/pkg/mod`
- Tool binaries installed under GOPATH live at `/home/gotenks/Projects/go-workspace/bin`.
- Build cache lives at `~/.cache/go-build`.
- LLVM tools live at `/home/gotenks/Projects/llvm/bin`.
- C compiler is `CC=/home/gotenks/Projects/llvm/bin/clang` (Clang 23.1.0) with libc/GCC dev packages for Cgo linking.
- Do not use `~/go` for cache or source lookups.
- Propagate these paths to every child agent.

Add `/home/gotenks/Projects/go/bin` and `/home/gotenks/Projects/llvm/bin` to `PATH`.
The supported target is Linux amd64.
Build final binaries with `CGO_ENABLED=0`.
The optional `scripts/yt-dl.sh` helper may retain its `uvx` and `yt-dlp` prerequisite.
Do not add Python or `uv` requirements to `bin/video-subtitler` or `scripts/subtitle.sh`.

Use atomic publication for final files and caches.
Do not add automatic cache revisions without approval.
Keep cache reuse and invalidation correct for artifacts produced by the Go pipeline.
Do not add compatibility code or parity tests for the retired Python implementation.
Do not change timing semantics casually.
Retain failed-run artifacts that support resume.
Keep `.env` values and API keys out of output and tests.

The Go Gemini SDK has no client-side automatic function calling loop.
Direct model requests must not add function declarations or host function execution.
Keep Google Search and URL Context enabled where the preflight contract requires them.
Set `HTTPOptions.RetryOptions` explicitly on every Gemini client.
Wrap the client HTTP transport with the project-owned SSE normalizing transport to handle leading and repeated empty stream lines.
Keep thought streaming enabled for configured thinking levels and exclude thought parts from assembled response text.

Keep project packages acyclic:

- `internal/vtt` and `internal/storage` are foundations.
- `internal/core` owns schemas, timing, classification, and repair authority.
- `internal/media` owns FFmpeg and FFprobe operations.
- `internal/gemini` owns prompts, request translation, response parsing, and Gemini caches.
- `internal/pipeline` owns locking, scheduling, stitching, publication, and run lifecycle.
- `cmd/video-subtitler` owns dotenv loading, argument parsing, validation routing, and dispatch.

## Testing principles

Test contracts through public or executable interfaces.
Assert outputs, side effects, errors, and externally visible state.
Every test must protect a behavioral contract.
Let the compiler enforce static type relationships.

Test adapters at the project-owned boundary.
Use local HTTP servers for Gemini request and response translation.
Do not call live or paid providers in tests.
Use real FFmpeg media fixtures for media behavior.
Do not test private helper names, prompt prose, exact FFmpeg command arrays, or worker formulas.

## Simplicity

Implement only explicit requirements.
Do not add speculative abstractions, options, dependencies, or tests.
Search the repository and standard library before adding a helper or dependency.
Ask before adding a direct dependency.

## Validation matrix

Run only checks relevant to changed files unless a cross-package migration requires the complete offline suite.

| Changed files | Required checks |
| --- | --- |
| Documentation or instructions only | Check Markdown semantics manually. |
| `scripts/*.sh` | `shellcheck <changed-script>`. |
| Go production or test files | `/home/gotenks/Projects/go/bin/gofmt -w <changed-files>` and `/home/gotenks/Projects/go/bin/go test <affected-packages>`. |
| Cross-package Go changes | `PATH=/home/gotenks/Projects/go/bin:/home/gotenks/Projects/llvm/bin:$PATH CC=/home/gotenks/Projects/llvm/bin/clang CGO_ENABLED=1 /home/gotenks/Projects/go/bin/go test -race ./...` and `/home/gotenks/Projects/go/bin/go vet ./...`. |
| CLI changes | Build `bin/video-subtitler` and run `bin/video-subtitler --help`. |
| Release or build changes | `CGO_ENABLED=0 GOOS=linux GOARCH=amd64 /home/gotenks/Projects/go/bin/go build -trimpath -o bin/video-subtitler ./cmd/video-subtitler` and inspect the binary format. |

## Git workflow

Never create commits unless the user explicitly asks.
Never push, rewrite history, or change the default branch unless the user explicitly asks.
Preserve unrelated work.

Before every authorized commit, run the exact commands `git status`, `git diff`, and `git log -10`.
Stage only files for the task.
Use the subject format `<scope>: <Capitalized imperative summary>`.
Keep the subject at or below 72 characters and omit a trailing period.

GitHub exposes a manual `workflow_dispatch` workflow only when its workflow file exists on the default branch.
Do not claim a release workflow present only on `go-rewrite` is dispatchable.
Keep releases manual-only and build the selected ref and tag the exact selected commit.
