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

`CONTEXT.md` is the authoritative technical and architectural specification for this repository.
It describes the pipeline, schemas, authority rules, caching, and domain model from first principles.
Read `CONTEXT.md` before changing code, tests, scripts, or documentation.
Always keep `CONTEXT.md` up-to-date whenever architecture, schemas, pipeline flow, or domain concepts change.

## Development rules

Use ASCII for documentation, code, and comments unless existing content requires another character set.
Keep comments rare and explain non-obvious behavior.
PEP 758 parenthesis-free `except X, Y:` syntax is valid in this repository.
Use atomic output publication for final files.
Do not use legacy `google-generativeai`.

When a Gemini request does not require automatic function calling, construct its config with `_build_content_config()` so AFC is disabled explicitly instead of relying on SDK defaults.
This requirement applies to structured, plain-text, tool-enabled, and tool-free requests that do not require automatic function calling.
Assert required request configuration, including disabled AFC and sanitized wire schemas, on requests recorded through the real SDK client in end-to-end pipeline tests.
Do not add standalone unit tests for config builder helpers.

Do not add automatic cache revisions without approval.
Do not change timing semantics casually.
Do not remove failed-run artifacts that support resume.

Keep the five modules in `modules/` on an acyclic dependency graph:
- `modules/core.py` and `modules/io.py` are foundations with no project-internal imports.
- `modules/media.py` depends only on `io`.
- `modules/gemini.py` depends on `core`, `io`, and `media` (for `AUDIO_MIME_TYPE`).
- `modules/pipeline.py` orchestrates media and Gemini and owns the run lifecycle.
- `gemini_subs.py` stays CLI-only: dotenv loading, argument parsing, validation, and dispatch.

Use semantic line breaks in Markdown prose: put each complete sentence on its own source line.
Use active voice, present tense, ASD-STE100 short sentences, and sentence-case headings.
Do not use exclamation points in documentation.

## Testing principles

Keep the test suite small, fast, and pipeline-oriented (target 45 to 70 collected items across `tests/`).
Organize tests by capability rather than mirroring `modules/` one-for-one:
- `tests/test_e2e_pipeline.py` covers `pipeline.run_generation` and `scripts/benchmark.py` end-to-end across Passes 0 through 3.
- `tests/test_cli.py` covers `gemini_subs.py` subprocess dispatch, environment precedence, diagnostics, and `--refine-only`.
- `tests/test_media_io.py` covers FFmpeg probing, splitting, audio extraction, cache recovery, and atomic JSON/VTT I/O.
- `tests/test_audio_patch.py` covers combinatorial `sparse-patch-v1` authority, lineage, visual preservation, and duration clamping rules.
- `tests/test_text_contracts.py` covers Pass 3 structural-edit guards, speaker-label casing, caption validation, timestamps, source titles, and context URLs.

Default to end-to-end pipeline scenarios using tiny session-scoped real FFmpeg fixtures and `ScriptedGeminiClient` wrapping real `google.genai.Client` instances with `httpx2.MockTransport`.
Never replace FFmpeg with fake command runners, never bypass `google.genai.Client` with a hand-rolled fake client, and never monkeypatch internal project functions (`media.*`, `gemini.*` stages, or `pipeline.*` helpers) to isolate pipeline stages.
Do not add source-text, AST, prompt-prose, exact FFmpeg argument-array, manifest JSON, or private-helper assertions.

Test each behavioral contract once at its highest practical boundary.
Do not duplicate validation, caching, or formatting checks across modules, adapters, and the CLI.
Add direct contract tests only for combinatorial validation rules, multi-codec media behavior, or filesystem failure paths that are impractical to trigger end-to-end.
Extend an existing scenario when new behavior fits an established flow instead of adding single-assertion micro-tests.
Group pure validation inputs by equivalence class instead of parametrizing every file extension, URL host, `argparse` error, or manifest key as separate pytest items.
Parametrize across all supported codecs (`vp9`, `h264`, `hevc`) only when testing codec-specific probing and stream-copy splitting; use one shared H.264 fixture for codec-independent scenarios.

Assert outputs, files, parsed WebVTT cues, persisted resume state, exit status, diagnostics, SDK config fields, and serialized wire schemas.
Remove tests that only prove a feature, command, or symbol exists.
Let the typechecker enforce static type relationships.

## Simplicity (YAGNI)

Implement only current, explicit requirements.
Do not add speculative features, abstractions, configuration, dependencies, or extensibility for hypothetical future use.
Prefer the smallest clear change that reuses existing code and standard facilities.
Delete obsolete code when safe.

Before writing a utility or adding a dependency, search the repository for an existing implementation and its callers.
Then check the standard library and already-declared dependencies.
Reuse an established option when it fits.
Ask before adding a new dependency.

## Validation matrix

Run only checks strictly relevant to the changed files.
Never chain unrelated tool checks for small, focused edits.
For documentation or instruction edits, do not run pytest, ruff, compileall, shellcheck, or CLI help commands.

| Changed files | Required checks |
| --- | --- |
| Documentation or instructions only | No code validation. Check Markdown semantics manually. |
| `scripts/*.sh` | `shellcheck <changed-script>` on the changed script only. |
| `modules/core.py` | `uv run pytest tests/test_audio_patch.py tests/test_text_contracts.py`. |
| `modules/io.py` or `modules/media.py` | `uv run pytest tests/test_media_io.py`. |
| `modules/gemini.py` or `modules/pipeline.py` | `uv run pytest tests/test_e2e_pipeline.py`. |
| `gemini_subs.py` | `uv run pytest tests/test_cli.py` and `uv run python gemini_subs.py --help`. |
| `scripts/benchmark.py` | `uv run pytest tests/test_e2e_pipeline.py -k benchmark` and `./scripts/benchmark.py --help`. |
| `tests/` | Run `uv run pytest <changed-test-file>` (or `uv run pytest` when editing `tests/conftest.py` or `tests/support/`). |
| Python production or test files | `uv run ruff check <changed-file>` and `uv run ruff format --check <changed-file>`. |

Run the full `uv run pytest` suite when the user requests it or when a change alters cross-module contracts.

## Git workflow

Never create commits unless the user explicitly asks for them.
When the user requests per-task commits, commit each discrete task before starting the next one.
Before every commit, run the exact full commands `git status`, `git diff`, and `git log -10`.
Do not replace these required inspections with abbreviated variants such as `git status --short`, `git diff --stat`, or `git log --oneline`.
Read the full commit messages from `git log -10`, including their bodies and trailers.
Stage only files that belong to the current task.

Format commit messages per repository conventions:
- Use the subject format `<scope>: <Capitalized summary>`. Derive the lowercase scope from the component or directory you changed.
- Write the summary in the imperative mood and do not end it with a period.
- Keep the subject near 50 characters and never longer than 72 characters.
- Add a concise, technical body when the subject does not provide enough context. Explain what changed and why.
- Separate the subject from the body with a blank line and wrap body text at 72 characters.

Check commit signing once per session with `git config commit.gpgsign` and `git config user.signingkey`.
Remember the result for the rest of the session.
If both are set, sign every commit with the configured method and use `git commit --signoff`.
Do not amend commits, push, or rewrite history unless the user explicitly asks.
