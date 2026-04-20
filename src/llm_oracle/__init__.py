"""Top-level public API for llm_oracle.

This package exposes the main workflow objects for building Judge, Verifier,
Harness, and Oracle-based evaluation pipelines. Advanced routing, provider,
and base-abstraction names are also available here for convenience.
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
