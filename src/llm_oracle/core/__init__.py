"""Core package for LLM Oracle.

Exports the fundamental data models, strategy base class, and provider
implementations used throughout the system.
"""

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
from llm_oracle.core.strategy import BaseStrategy, LanguageModel

__all__ = [
  # Models
  "EvaluationCriterion",
  "EvaluationResult",
  "ModelConfig",
  "PairwiseComparison",
  "RoutingDecision",
  "ScoreResult",
  "ScoringConfig",
  "StrategyType",
  "Task",
  "TaskDifficulty",
  "Trajectory",
  "TrajectoryList",
  "CriterionList",
  "ScoreDict",
  # Strategy
  "BaseStrategy",
  "LanguageModel",
  # Providers
  "BaseProvider",
  "OpenAIProvider",
  "AnthropicProvider",
  "GeminiProvider",
  "StubProvider",
  "StubResponse",
  "create_provider",
  "get_provider",
  "register_provider",
]
