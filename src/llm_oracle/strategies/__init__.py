"""Strategies package for LLM Oracle.

Exports the two evaluation strategy implementations:
  - VerifierStrategy: LLM-as-a-Verifier with logprob-based fine-grained scoring
  - JudgeStrategy: LLM-as-a-Judge with rubric-based chain-of-thought evaluation
"""

from llm_oracle.strategies.judge import JudgeStrategy
from llm_oracle.strategies.verifier import VerifierStrategy

__all__ = [
  "VerifierStrategy",
  "JudgeStrategy",
]
