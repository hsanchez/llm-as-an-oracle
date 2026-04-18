"""Evaluation package for LLM Oracle.

Exports the evaluation harness and hardness comparison tools for side-by-side
benchmarking of the Verifier and Judge strategies.
"""

from llm_oracle.evaluation.harness import (
  EvaluationHarness,
  HarnessReport,
  TaskHardnessRecord,
)

__all__ = [
  "EvaluationHarness",
  "HarnessReport",
  "TaskHardnessRecord",
]
