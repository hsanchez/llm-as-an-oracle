"""LLM Oracle — LLM-as-a-Verifier · LLM-as-a-Judge · Intelligent Router.

This package provides a modern, elegant Python implementation of two
complementary LLM evaluation strategies and an intelligent routing layer
that automatically selects the right strategy for each task.

Quick-start
-----------
>>> from llm_oracle import (
...     OracleRouter, VerifierStrategy, JudgeStrategy,
...     EvaluationHarness, EvaluationCriterion,
...     Task, Trajectory, ScoringConfig,
...     StubProvider, TaskDifficulty,
... )
>>>
>>> # 1. Build a shared stub provider (swap for OpenAIProvider / GeminiProvider)
>>> model   = StubProvider(default_score="C", default_score_a="C", default_score_b="E")
>>> config  = ScoringConfig(granularity=20, num_verifications=2, num_criteria=2)
>>> criteria = [
...     EvaluationCriterion("correctness", "Correctness", "Is the solution correct?"),
...     EvaluationCriterion("clarity",     "Clarity",     "Is the solution clear?"),
... ]
>>>
>>> # 2. Instantiate both strategies
>>> verifier = VerifierStrategy(model, config, criteria)
>>> judge    = JudgeStrategy(model, config, criteria)
>>>
>>> # 3. Create the router with the default five-policy chain
>>> router = OracleRouter.default(verifier, judge)
>>>
>>> # 4. Define a task and some candidate trajectories
>>> task = Task(
...     id="task-1",
...     description="Sort a list of integers",
...     problem_statement="Write a function that sorts a list of integers in ascending order.",
...     difficulty=TaskDifficulty.EASY,
... )
>>> trajectories = [
...     Trajectory("t1", "task-1", "def sort_list(lst): return sorted(lst)"),
...     Trajectory("t2", "task-1", "def sort_list(lst): lst.sort(); return lst"),
... ]
>>>
>>> # 5. Route and evaluate in one call
>>> result, decision = router.evaluate(task, trajectories)
>>> print(decision.selected_strategy, decision.confidence)
>>> print(result.best_trajectory_id)

Package layout
--------------
llm_oracle/
├── core/
│   ├── models.py       — Data-model dataclasses (Task, Trajectory, …)
│   ├── strategy.py     — Abstract BaseStrategy + LanguageModel protocol
│   └── providers.py    — OpenAI, Anthropic, Gemini, and Stub providers
├── strategies/
│   ├── verifier.py     — LLM-as-a-Verifier (logprob + tournament selection)
│   └── judge.py        — LLM-as-a-Judge  (rubric + chain-of-thought)
├── evaluation/
│   └── harness.py      — Side-by-side hardness comparison harness
└── routing/
    └── router.py       — Signal-based policy-chain router
"""

from __future__ import annotations

# ── Core models ───────────────────────────────────────────────────────────────
from llm_oracle.core.models import (
  CriterionList,
  EvaluationCriterion,
  EvaluationResult,
  ModelConfig,
  PairwiseComparison,
  RoutingDecision,
  ScoreDict,
  ScoreResult,
  ScoringConfig,
  StrategyType,
  Task,
  TaskDifficulty,
  Trajectory,
  TrajectoryList,
)

# ── Providers ─────────────────────────────────────────────────────────────────
from llm_oracle.core.providers import (
  AnthropicProvider,
  BaseProvider,
  GeminiProvider,
  OpenAIProvider,
  StubProvider,
  StubResponse,
  create_provider,
  get_provider,
  register_provider,
)

# ── Strategy base ─────────────────────────────────────────────────────────────
from llm_oracle.core.strategy import BaseStrategy, LanguageModel

# ── Evaluation harness ────────────────────────────────────────────────────────
from llm_oracle.evaluation.harness import (
  EvaluationHarness,
  HarnessReport,
  TaskHardnessRecord,
)

# ── Routing ───────────────────────────────────────────────────────────────────
from llm_oracle.routing.router import (
  DetailedRoutingDecision,
  DifficultyPolicy,
  GroundTruthPolicy,
  KeywordDomainPolicy,
  OracleRouter,
  OutputAvailabilityPolicy,
  PolicyChain,
  PolicyVote,
  PriorHardnessPolicy,
  RoutingPolicy,
  RoutingSignals,
  SignalExtractor,
  TrajectoryCountPolicy,
)

# ── Evaluation strategies ─────────────────────────────────────────────────────
from llm_oracle.strategies.judge import JudgeStrategy
from llm_oracle.strategies.verifier import VerifierStrategy

__version__: str = "0.1.0"
__author__: str = "LLM Oracle"

__all__ = [
  # ── Version ──────────────────────────────────────────────────────────────
  "__version__",
  "__author__",
  # ── Data models ───────────────────────────────────────────────────────────
  "CriterionList",
  "EvaluationCriterion",
  "EvaluationResult",
  "ModelConfig",
  "PairwiseComparison",
  "RoutingDecision",
  "ScoreDict",
  "ScoreResult",
  "ScoringConfig",
  "StrategyType",
  "Task",
  "TaskDifficulty",
  "Trajectory",
  "TrajectoryList",
  # ── Strategy base ─────────────────────────────────────────────────────────
  "BaseStrategy",
  "LanguageModel",
  # ── Providers ─────────────────────────────────────────────────────────────
  "BaseProvider",
  "OpenAIProvider",
  "AnthropicProvider",
  "GeminiProvider",
  "StubProvider",
  "StubResponse",
  "create_provider",
  "get_provider",
  "register_provider",
  # ── Strategies ────────────────────────────────────────────────────────────
  "VerifierStrategy",
  "JudgeStrategy",
  # ── Harness ───────────────────────────────────────────────────────────────
  "EvaluationHarness",
  "HarnessReport",
  "TaskHardnessRecord",
  # ── Router ────────────────────────────────────────────────────────────────
  "OracleRouter",
  "PolicyChain",
  "RoutingPolicy",
  "RoutingSignals",
  "PolicyVote",
  "DetailedRoutingDecision",
  "SignalExtractor",
  # Built-in policies
  "GroundTruthPolicy",
  "KeywordDomainPolicy",
  "DifficultyPolicy",
  "TrajectoryCountPolicy",
  "PriorHardnessPolicy",
  "OutputAvailabilityPolicy",
]
