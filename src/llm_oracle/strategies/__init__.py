"""Strategies package for LLM Oracle.

Exports the evaluation strategy implementations:
  - AdversarialVerifierStrategy: claim verifier with confirmation/challenge passes
  - VerifierStrategy: LLM-as-a-Verifier with logprob-based fine-grained scoring
  - JudgeStrategy: LLM-as-a-Judge with rubric-based chain-of-thought evaluation
"""

from llm_oracle.strategies.adversarial import AdversarialDecision, AdversarialVerifierStrategy
from llm_oracle.strategies.judge import JudgeStrategy
from llm_oracle.strategies.verifier import VerifierStrategy

__all__ = [
  "AdversarialDecision",
  "AdversarialVerifierStrategy",
  "VerifierStrategy",
  "JudgeStrategy",
]
