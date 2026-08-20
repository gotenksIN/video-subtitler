# Issue tracker: GitHub

Issues and specs live in GitHub Issues.
Use the `gh` CLI for all operations.

## Conventions

- Create issues with `gh issue create`.
- Read issues and comments with `gh issue view <number> --comments`.
- List issues with `gh issue list`.
- Comment with `gh issue comment <number>`.
- Change labels with `gh issue edit <number>`.
- Close issues with `gh issue close <number>`.

Infer the repository from the configured Git remote.

## Pull requests as a triage surface

**PRs as a request surface: no.**

## Skill operations

When a skill says to publish to the issue tracker, create a GitHub issue.
When a skill says to fetch a ticket, use `gh issue view <number> --comments`.
Use GitHub sub-issues and native issue dependencies for wayfinding when available.
Fall back to task lists and `Blocked by: #<number>` lines when they are unavailable.
