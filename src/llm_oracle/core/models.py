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


class AlignmentStatus(enum.Enum):
  """Status of extraction alignment with source text."""

  MATCH_EXACT = "match_exact"
  MATCH_FUZZY = "match_fuzzy"
  MATCH_LESSER = "match_lesser"
  NO_MATCH = "no_match"


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
    features: Features extracted for routing decision
    metadata: Additional routing metadata
  """

  task_id: str
  selected_strategy: StrategyType
  confidence: float
  reasoning: str
  features: dict[str, Any] = field(default_factory=dict)
  metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TokenInterval:
  """Token-based interval in text.

  Attributes:
    start_index: Starting token index (inclusive)
    end_index: Ending token index (exclusive)
  """

  start_index: int
  end_index: int


@dataclass
class CharInterval:
  """Character-based interval in text.

  Attributes:
    start_pos: Starting character position (inclusive)
    end_pos: Ending character position (exclusive)
  """

  start_pos: int
  end_pos: int


@dataclass
class Extraction:
  """Represents an extracted entity from text.

  Attributes:
    extraction_class: Type/class of the extraction
    extraction_text: The extracted text content
    extraction_index: Position index in sequence
    group_index: Group index for batched extractions
    token_interval: Token-level position in source
    char_interval: Character-level position in source
    alignment_status: How well aligned with source text
    attributes: Additional extraction attributes
  """

  extraction_class: str
  extraction_text: str
  extraction_index: int
  group_index: int = 0
  token_interval: TokenInterval | None = None
  char_interval: CharInterval | None = None
  alignment_status: AlignmentStatus | None = None
  attributes: dict[str, Any] | None = None


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

  Attributes:
    granularity: Number of score levels (e.g., 20 for A-T scale)
    num_verifications: Number of repeated verifications
    num_criteria: Number of evaluation criteria
    enable_fuzzy_alignment: Enable fuzzy text alignment
    fuzzy_threshold: Threshold for fuzzy matching
    use_logprobs: Use log probabilities for scoring
    additional_params: Strategy-specific parameters (e.g., temperature, max_tokens)
  """

  granularity: int = 20
  num_verifications: int = 4
  num_criteria: int = 3
  enable_fuzzy_alignment: bool = True
  fuzzy_threshold: float = 0.75
  use_logprobs: bool = True
  additional_params: dict[str, Any] = field(default_factory=dict)


# Type aliases for convenience
TrajectoryList = Sequence[Trajectory]
ScoreDict = dict[str, float]
CriterionList = Sequence[EvaluationCriterion]
