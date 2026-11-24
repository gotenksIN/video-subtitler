# Agent specification

## Domain documentation

`CONTEXT.md` documents the project's architectural choices, domain rules, and design decisions.
Read `CONTEXT.md` before changing code, tests, scripts, or documentation.
Keep `CONTEXT.md` up-to-date whenever architectural choices, domain rules, or design decisions change.
Use terms defined in `CONTEXT.md` rather than introducing synonyms.

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

## Testing principles

`tests/README.md` defines the test file layout, automated contract matrix, and anti-bloat boundaries.
Keep the test suite small, fast, and pipeline-oriented (target 45 to 70 collected items across `tests/`).

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
