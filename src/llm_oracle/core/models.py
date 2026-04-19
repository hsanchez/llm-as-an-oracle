"""Core data models and types for LLM Oracle system.

This module defines the fundamental data structures used throughout the system,
including tasks, trajectories, evaluations, and scoring results.
"""

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
  """Represents a task to be evaluated.

  Attributes:
    id: Unique task identifier
    description: Natural language description of the task
    problem_statement: Formal problem statement
    test_cases: Optional test cases for verification
    ground_truth: Optional ground truth solution or output
    metadata: Additional task metadata
    difficulty: Estimated task difficulty
  """

  id: str
  description: str
  problem_statement: str
  test_cases: list[dict[str, Any]] | None = None
  ground_truth: str | None = None
  metadata: dict[str, Any] = field(default_factory=dict)
  difficulty: TaskDifficulty = TaskDifficulty.UNKNOWN


@dataclass
class Trajectory:
  """Represents an execution trajectory or solution attempt.

  Attributes:
    id: Unique trajectory identifier
    task_id: Associated task ID
    content: The actual trajectory content (code, actions, reasoning)
    output: Execution output or result
    metadata: Additional trajectory metadata
    reward: Optional reward/success indicator (0.0-1.0)
  """

  id: str
  task_id: str
  content: str
  output: str | None = None
  metadata: dict[str, Any] = field(default_factory=dict)
  reward: float | None = None


@dataclass
class EvaluationCriterion:
  """A single evaluation criterion.

  Attributes:
    id: Unique criterion identifier
    name: Human-readable criterion name
    description: Detailed description of what to evaluate
    weight: Relative weight in overall evaluation (0.0-1.0)
  """

  id: str
  name: str
  description: str
  weight: float = 1.0


@dataclass
class ScoreResult:
  """Result of scoring a single trajectory.

  Attributes:
    trajectory_id: ID of the scored trajectory
    score: Normalized score (0.0-1.0)
    raw_score: Raw score before normalization
    confidence: Confidence in the score (0.0-1.0)
    criterion_scores: Scores per criterion
    reasoning: Textual explanation of the score
    metadata: Additional scoring metadata
  """

  trajectory_id: str
  score: float
  raw_score: float | None = None
  confidence: float = 1.0
  criterion_scores: dict[str, float] = field(default_factory=dict)
  reasoning: str = ""
  metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PairwiseComparison:
  """Result of comparing two trajectories.

  Attributes:
    trajectory_a_id: ID of first trajectory
    trajectory_b_id: ID of second trajectory
    score_a: Score for trajectory A
    score_b: Score for trajectory B
    winner: ID of winning trajectory (or None for tie)
    criterion_id: Criterion used for comparison
    confidence: Confidence in the comparison
    reasoning: Textual explanation
  """

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
  """Complete evaluation result for a set of trajectories.

  Attributes:
    task_id: ID of the evaluated task
    strategy_type: Strategy used for evaluation
    best_trajectory_id: ID of the best trajectory
    trajectory_scores: Scores for all trajectories
    pairwise_comparisons: All pairwise comparisons performed
    metadata: Additional evaluation metadata
  """

  task_id: str
  strategy_type: StrategyType
  best_trajectory_id: str
  trajectory_scores: dict[str, ScoreResult]
  pairwise_comparisons: list[PairwiseComparison] = field(default_factory=list)
  metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RoutingDecision:
  """Decision about which strategy to use for a task.

  Attributes:
    task_id: ID of the task
    selected_strategy: Strategy type selected
    confidence: Confidence in the routing decision
    reasoning: Explanation for the decision
    metadata: Additional routing metadata
  """

  task_id: str
  selected_strategy: StrategyType
  confidence: float
  reasoning: str
  metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelConfig:
  """Configuration for language model.

  Attributes:
    model_id: Model identifier (e.g., "gpt-4", "claude-3")
    provider: Provider name (e.g., "openai", "anthropic")
    api_key: API key for the provider
    temperature: Sampling temperature
    max_tokens: Maximum tokens in response
    top_p: Nucleus sampling parameter
    additional_params: Provider-specific parameters
  """

  model_id: str
  provider: str
  api_key: str | None = None
  temperature: float = 1.0
  max_tokens: int = 4096
  top_p: float = 1.0
  additional_params: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScoringConfig:
  """Configuration for scoring strategies.

  **Strategy compatibility** — not all fields apply to every strategy:

  * ``use_logprobs`` — Verifier only; ignored by Judge.
  * ``num_verifications`` — Verifier only; ignored by Judge.
  * ``num_criteria`` — Judge only; ignored by Verifier.
  * ``fuzzy_threshold`` — must be in ``[0.0, 1.0]`` (validated at construction).
  * ``enable_fuzzy_alignment`` — reserved; currently has no effect.

  All other fields apply to every strategy.

  Attributes:
    granularity: Number of score levels (e.g., 20 for A–T scale).
    num_verifications: Repeated verifications per trajectory (Verifier only).
    num_criteria: Number of evaluation criteria (Judge only).
    enable_fuzzy_alignment: Reserved; currently unused.
    fuzzy_threshold: Fuzzy-match threshold in ``[0.0, 1.0]``.
    use_logprobs: Use log-probability scoring (Verifier only).
    temperature: Sampling temperature; ``None`` defers to the strategy default.
    max_tokens: Max response tokens; ``None`` defers to the strategy default.
    additional_params: Strategy-specific parameters not covered by named fields.
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
