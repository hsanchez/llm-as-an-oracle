# LLM-as-an-Oracle — Agent Instructions

> Shared context for Claude Code, Codex, Gemini, and other AI coding agents.
> Tool-specific files (CLAUDE.md, CODEX.md, GEMINI.md) layer on top of this with overrides only.

## Mission

LLM-as-an-Oracle is an evaluation orchestrator. It sits above the Judge and the
Verifier and decides which one is the better fit for a given task. It does
not evaluate trajectories itself — it routes to the evaluator that is most
likely to produce a reliable result, then returns that evaluator's output
alongside the routing decision that led to it.

The term "Oracle" here means adaptive evaluation layer, not an all-knowing
model. It is a system design pattern.

## Memory Discipline

- `MEMORY.md` at the repo root holds the current compressed project state —
  rewritten to stay concise, never append-only.
- `MEMORY_LOG.md` at the repo root holds raw timestamped entries.
- Both files are committed to the repo — they are shared team context, not
  personal artifacts.
- Never store secrets, credentials, or speculation in either file.
- Log entry format: `YYYY-MM-DD HH:MM — <1-3 line factual update>`
- Never write timestamps manually — use the `memlog` command to append
  entries, which provides the current local time automatically.
- At the start of each task, read `MEMORY.md`. Consult `MEMORY_LOG.md` only
  when debugging or recovering context.
- When asked to "update memory", append a new entry to `MEMORY_LOG.md` only.
- When asked to "compress memory", rewrite `MEMORY.md` to reflect current
  project state and remove the compressed entries from `MEMORY_LOG.md`.
  Compress when `MEMORY_LOG.md` exceeds ~100 entries, or when explicitly asked.
- Never read or write `MEMORY.local.md` or `MEMORY_LOG.local.md` — those are
  private human scratch files.

## Project Management

- This project uses GitHub Issues for task tracking — do not create or maintain PLAN.md
- When starting a task, ask the user for the relevant issue number if not provided
- Reference issue numbers in commit messages: `fix #42` or `closes #42`
- Never create, close, or comment on issues without explicit user instruction

## Required Development Workflow

**CRITICAL**: Always run these commands before committing:

```bash
uv sync                       # Install dependencies
uv run prek run --all-files   # Lint (ruff) + type check (ty)
uv run pytest                 # Run tests
```

All must pass — enforced by CI.

## Code Standards

- Python ≥ 3.14 with full type annotations
- Use `from __future__ import annotations` only where needed: files with forward references, TYPE_CHECKING blocks, or complex type annotations
- 2-space indentation, never tabs
- Imports at module level — never inside functions or test bodies
- Never use bare `except` — always specify the exception type
- Minimize `# type: ignore` — fix types properly instead
- No abbreviations in variable names (`queue` not `q`)
- No comments that summarize what code does — only explain non-obvious *why*
- Prioritize clarity over cleverness
- Follow existing patterns; maintain consistency across the codebase

## Docstring Standards

Follow lightweight Google style — type annotations already document types, so docstrings should only add what the signature cannot express.

- **Module**: one or two sentences max; purpose only, no RST/NumPy section headers.
- **Class**: one-line summary. Add a short paragraph only when behavior is non-obvious from the name and type annotations.
- **Dataclass `Attributes:`**: never — field names and type annotations are self-documenting.
- **Method `Args:`**: omit when parameters are obvious from names and types. Add only when behavior differs from what the signature implies (e.g., a flag that is silently ignored, a surprising default).
- **Method `Returns:`**: include only for non-obvious return shapes (tuples, composite objects). Skip when the return type annotation is sufficient.
- **`Raises:`**: always include — callers need it.

## Module & Package Standards

- Prefer implementing functionality in existing files unless it is a genuinely new logical component
- Avoid creating many small files
- `__init__.py` exports (`__all__`) are a DX surface for framework *users*, not internal convenience — curate ruthlessly; the framework itself imports directly from modules

## Testing Standards

- Each test: atomic, self-contained, covers a single piece of functionality
- Use parameterization for multiple examples of the same behavior
- Use separate tests for distinct functionality
- `asyncio_mode = "auto"` is set globally — never add `@pytest.mark.asyncio` to individual tests
- Run pytest after any significant change
- Every new feature needs corresponding tests

## Documentation

### User-Facing Docs (`docs/`)

Update for any user-facing change: new features, configuration options, CLI changes, template syntax.

When updating:
- Read related docs first to understand structure and voice
- Place content where users would naturally look for it
- Update related sections that reference affected functionality
- Remove or update content that becomes incorrect
- Do not append — integrate holistically; restructure if needed
- Every word must serve the user; pro forma documentation is unacceptable

### `dev-docs/ARCHITECTURE.md`

Live architectural overview. Update whenever the system design changes.

### `dev-docs/core/`

Frozen reference documentation — generated or otherwise locked. **Read only; never edit or add files here.**

### `dev-docs/decisions/`

Architecture Decision Records (ADRs). Records *why* core decisions were made.
Format: `NNN-short-name.md` (e.g., `001-mvp-scope.md`)

Create a new ADR when:
- Choosing between multiple valid approaches
- Making scope decisions (what's in/out)
- Selecting technologies
- Establishing patterns that affect multiple components

Not every small choice — only significant architectural decisions.


## Git & Commits

- Never force-push on collaborative repos
- Never amend commits
- Commit messages: brief headlines only
- Run all checks before committing

## Writing Style (for docs and comments)

- Be brief and direct
- State what something *is* — never use "This isn't..." or "not just..." constructions

## Response Compression

Use terse responses by default when technical accuracy would remain intact.

- Always ON unless the user asks for more detail
- Lead with the answer, fix, or finding
- Remove filler, pleasantries, and hedging unless they materially improve trust or accuracy
- Keep all technical substance exact: constraints, caveats, commands, paths, identifiers, versions, and next steps
- Fragments are acceptable when clear
- Prefer short familiar words and standard abbreviations when unambiguous
- Avoid restating the prompt, repeating context, or explaining the obvious
- Prefer this pattern when it fits: `<thing> <action> <reason>. <next step>`
- For reviews and findings, prefer: `<location>: <issue>. <fix>`
- Keep code, commands, paths, URLs, version numbers, and other literals unchanged
- Do not drift back toward verbose prose over time
- If brevity would hide uncertainty, risk, safety constraints, or important tradeoffs, keep the extra words
