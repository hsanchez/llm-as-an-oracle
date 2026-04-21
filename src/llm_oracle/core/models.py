"""Core data models and types for the LLM Oracle system."""

from __future__ import annotations

import enum
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any


class TaskDifficulty(enum.Enum):
  """Task difficulty levels for routing decisions."""

  EASY = "easy"
  MEDIUM = "medium"
  HARD = "hard"
  UNKNOWN = "unknown"


class StrategyType(enum.Enum):
  """Types of evaluation strategies."""

  VERIFIER = "verifier"
  JUDGE = "judge"


@dataclass(frozen=True)
class Task:
  """Immutable task descriptor."""

  id: str
  description: str
  problem_statement: str
  test_cases: list[dict[str, Any]] | None = None
  ground_truth: str | None = None
  metadata: dict[str, Any] = field(default_factory=dict)
  difficulty: TaskDifficulty = TaskDifficulty.UNKNOWN


@dataclass
class Trajectory:
  """Execution trajectory or solution attempt."""

  id: str
  task_id: str
  content: str
  output: str | None = None
  metadata: dict[str, Any] = field(default_factory=dict)
  reward: float | None = None


@dataclass
class EvaluationCriterion:
  """A single evaluation criterion with a scoring weight."""

  id: str
  name: str
  description: str
  weight: float = 1.0


@dataclass
class ScoreResult:
  """Scoring result for one trajectory."""

  trajectory_id: str
  score: float
  raw_score: float | None = None
  confidence: float = 1.0
  criterion_scores: dict[str, float] = field(default_factory=dict)
  reasoning: str = ""
  metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PairwiseComparison:
  """Pairwise comparison result for two trajectories."""

  trajectory_a_id: str
  trajectory_b_id: str
  score_a: float
  score_b: float
  winner: str | None = None
  criterion_id: str | None = None
  confidence: float = 1.0
  reasoning: str = ""


@dataclass
class EvaluationResult:
  """Complete evaluation result for a set of trajectories."""

  task_id: str
  strategy_type: StrategyType
  best_trajectory_id: str
  trajectory_scores: dict[str, ScoreResult]
  pairwise_comparisons: list[PairwiseComparison] = field(default_factory=list)
  metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RoutingDecision:
  """Routing decision indicating which strategy to use for a task."""

  task_id: str
  selected_strategy: StrategyType
  confidence: float
  reasoning: str
  metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelConfig:
  """Language model configuration."""

  model_id: str
  provider: str
  api_key: str | None = None
  temperature: float = 1.0
  max_tokens: int = 4096
  top_p: float = 1.0
  additional_params: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScoringConfig:
  """Scoring configuration shared by both strategies.

  Strategy-specific fields: ``use_logprobs`` and ``num_verifications`` apply
  to the Verifier only; ``num_criteria`` applies to the Judge only.
  ``fuzzy_threshold`` must be in ``[0.0, 1.0]``; ``enable_fuzzy_alignment``
  is reserved and currently has no effect.
  """

  granularity: int = 20
  num_verifications: int = 4
  num_criteria: int = 3
  enable_fuzzy_alignment: bool = True
  fuzzy_threshold: float = 0.75
  use_logprobs: bool = True
  temperature: float | None = None
  max_tokens: int | None = None
  additional_params: dict[str, Any] = field(default_factory=dict)


# Type aliases for convenience
TrajectoryList = Sequence[Trajectory]
ScoreDict = dict[str, float]
CriterionList = Sequence[EvaluationCriterion]
