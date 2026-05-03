# Claude Code — LLM-as-an-Oracle

See [AGENTS.md](AGENTS.md) for all shared instructions.

## Memory Discipline (Claude-specific)

AGENTS.md is the authoritative source for memory rules. Read it before any memory operation.
Key rules reproduced here to prevent mistakes:
- **Never write to** `~/.claude/projects/*/memory/MEMORY.md` when a memory discipline is specified in this file or in `AGENTS.md`.

## Claude-Specific Notes

- Prefer `bash` tool for running `uv` commands over explaining them
- When uncertain about architecture, read `dev-docs/architecture.md` before proceeding
- Use `TodoWrite` to track multi-step tasks; check off items as completed — persists within the session only, not to disk
