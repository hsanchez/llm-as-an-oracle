#!/usr/bin/env python3
"""LLM Oracle — CLI entry-point.

Usage
-----
    # Run the full end-to-end demo (no API keys needed — uses StubProvider)
    uv run python main.py demo

    # Route a single prompt/task and print the routing decision
    uv run python main.py route --task "Implement binary search in Python" --difficulty medium

    # Compare verifier vs judge on a task (offline stub)
    uv run python main.py compare --task "Sort a list of integers" --trajectories 3

    # Run the test suite
    uv run python main.py test

    # Print package info
    uv run python main.py info
"""

from __future__ import annotations

import os
import sys

if __name__ == "__main__":
  # Ensure src/ is on the path so the package is importable when running
  # directly without installation (e.g., `python main.py demo`).
  src_path = os.path.join(os.path.dirname(__file__), "src")
  if src_path not in sys.path:
    sys.path.insert(0, src_path)

from llm_oracle._cli import main  # noqa: E402

if __name__ == "__main__":
  main()
