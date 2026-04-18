"""Base strategy abstract class for LLM evaluation strategies.

This module defines the abstract base class for all evaluation strategies,
providing a common interface for verifier and judge implementations.
"""

from __future__ import annotations

import abc
from typing import Any, Protocol

from llm_oracle.core.models import (
  EvaluationCriterion,
  EvaluationResult,
  PairwiseComparison,
  ScoreResult,
  ScoringConfig,
  StrategyType,
  Task,
  Trajectory,
  TrajectoryList,
)


class LanguageModel(Protocol):
  """Protocol for language model implementations."""

  def generate(
    self,
    prompt: str,
    temperature: float = 1.0,
    max_tokens: int = 4096,
    return_logprobs: bool = False,
    **kwargs: Any,
  ) -> tuple[str, list[Any] | None, list[Any] | None]:
    """Generate text from prompt.

    Args:
      prompt: Input prompt
      temperature: Sampling temperature
      max_tokens: Maximum tokens to generate
      return_logprobs: Whether to return log probabilities
      **kwargs: Additional model-specific parameters

    Returns:
      Tuple of (generated_text, tokens, logprobs)
    """
    ...


class BaseStrategy(abc.ABC):
  """Abstract base class for evaluation strategies.

  This class defines the interface that all evaluation strategies must implement,
  including verifiers and judges. Strategies are responsible for scoring
  trajectories and selecting the best one for a given task.
  """

  def __init__(
    self,
    model: LanguageModel,
    config: ScoringConfig,
    criteria: list[EvaluationCriterion],
  ):
    """Initialize the strategy.

    Args:
      model: Language model to use for evaluation
      config: Configuration for scoring behavior
      criteria: List of evaluation criteria to use
    """
    self.model = model
    self.config = config
    self.criteria = criteria
    self._validate_config()

  def _validate_config(self) -> None:
    """Validate configuration parameters.

    Raises:
      ValueError: If configuration is invalid
    """
    if self.config.granularity < 2:
      raise ValueError(f"Granularity must be at least 2, got {self.config.granularity}")

    if not 0.0 <= self.config.fuzzy_threshold <= 1.0:
      raise ValueError(f"Fuzzy threshold must be in [0, 1], got {self.config.fuzzy_threshold}")

    if self.config.num_verifications < 1:
      raise ValueError(
        f"Number of verifications must be at least 1, got {self.config.num_verifications}"
      )

    if not self.criteria:
      raise ValueError("At least one evaluation criterion is required")

  @abc.abstractmethod
  def evaluate(
    self,
    task: Task,
    trajectories: TrajectoryList,
    **kwargs: Any,
  ) -> EvaluationResult:
    """Evaluate trajectories for a given task.

    Args:
      task: The task being evaluated
      trajectories: List of candidate trajectories
      **kwargs: Additional strategy-specific parameters

    Returns:
      Complete evaluation result with scores and best trajectory

    Raises:
      ValueError: If trajectories list is empty or invalid
    """
    pass

  @abc.abstractmethod
  def score_trajectory(
    self,
    task: Task,
    trajectory: Trajectory,
    criterion: EvaluationCriterion,
    **kwargs: Any,
  ) -> ScoreResult:
    """Score a single trajectory against a criterion.

    Args:
      task: The task being evaluated
      trajectory: The trajectory to score
      criterion: Evaluation criterion to use
      **kwargs: Additional strategy-specific parameters

    Returns:
      Score result with normalized score and metadata
    """
    pass

  @abc.abstractmethod
  def compare_trajectories(
    self,
    task: Task,
    trajectory_a: Trajectory,
    trajectory_b: Trajectory,
    criterion: EvaluationCriterion,
    **kwargs: Any,
  ) -> PairwiseComparison:
    """Compare two trajectories pairwise.

    Args:
      task: The task being evaluated
      trajectory_a: First trajectory
      trajectory_b: Second trajectory
      criterion: Criterion for comparison
      **kwargs: Additional strategy-specific parameters

    Returns:
      Pairwise comparison result with scores for both
    """
    pass

  @abc.abstractmethod
  def get_strategy_type(self) -> StrategyType:
    """Return the type of this strategy.

    Returns:
      StrategyType enum value
    """
    pass

  def select_best_trajectory(
    self,
    task: Task,
    trajectories: TrajectoryList,
    scores: dict[str, ScoreResult],
  ) -> str:
    """Select the best trajectory based on scores.

    Default implementation selects trajectory with highest average score.
    Subclasses can override for custom selection logic.

    Args:
      task: The task being evaluated
      trajectories: List of candidate trajectories
      scores: Scores for each trajectory

    Returns:
      ID of the best trajectory
    """
    if not trajectories:
      raise ValueError("Cannot select from empty trajectory list")

    if not scores:
      # If no scores available, return first trajectory
      return trajectories[0].id

    best_id = max(scores.keys(), key=lambda tid: scores[tid].score)
    return best_id

  def aggregate_criterion_scores(
    self,
    criterion_scores: dict[str, float],
  ) -> float:
    """Aggregate scores across criteria into a single score.

    Default implementation computes weighted average based on criterion weights.

    Args:
      criterion_scores: Scores per criterion ID

    Returns:
      Aggregated score (0.0-1.0)
    """
    if not criterion_scores:
      return 0.5

    # Map criterion IDs to their weights
    criterion_weights = {c.id: c.weight for c in self.criteria}

    total_weight = 0.0
    weighted_sum = 0.0

    for crit_id, score in criterion_scores.items():
      weight = criterion_weights.get(crit_id, 1.0)
      weighted_sum += score * weight
      total_weight += weight

    return weighted_sum / total_weight if total_weight > 0 else 0.5

  def get_scale_description(self) -> dict[str, Any]:
    """Get description of the scoring scale.

    Returns:
      Dictionary describing the scale levels and valid tokens
    """
    g = self.config.granularity

    # Generate letter-based scale (A-Z for up to 26 levels)
    if g <= 26:
      valid_tokens = {}
      for i in range(g):
        letter = chr(65 + i)  # A, B, C, ...
        score_value = float(g - i)
        valid_tokens[letter] = score_value
        valid_tokens[letter.lower()] = score_value

      scale_desc = self._generate_scale_description(g)

      return {
        "scale_description": scale_desc,
        "score_format": f"LETTER_A_TO_{chr(65 + g - 1)}",
        "valid_tokens": valid_tokens,
        "granularity": g,
      }
    else:
      # For granularity > 26, use numeric scale
      return {
        "scale_description": f"Rate on a scale from 1 to {g}",
        "score_format": f"NUMBER_1_TO_{g}",
        "valid_tokens": {str(i): float(i) for i in range(1, g + 1)},
        "granularity": g,
      }

  def _generate_scale_description(self, granularity: int) -> str:
    """Generate a human-readable scale description.

    Args:
      granularity: Number of scale levels

    Returns:
      Scale description text
    """
    if granularity == 20:
      return (
        "Rate how likely the agent correctly solved the task on a "
        "20-point scale using letters A through T:\n"
        "  A = clearly and completely succeeded with verified output (best)\n"
        "  B-D = succeeded with only minor issues\n"
        "  E-G = above average, mostly correct with some issues\n"
        "  H-J = uncertain, leans toward success\n"
        "  K-M = uncertain, leans toward failure\n"
        "  N-P = below average, significant issues remain\n"
        "  Q-S = failed with some partial progress\n"
        "  T = clearly and completely failed (worst)"
      )
    else:
      end_letter = chr(65 + granularity - 1)
      return (
        f"Rate how likely the agent correctly solved the task on a "
        f"{granularity}-point scale using letters A through {end_letter}:\n"
        f"  A = clearly and completely succeeded (best)\n"
        f"  {end_letter} = clearly and completely failed (worst)\n"
        f"  Use intermediate letters for partial success."
      )

  def normalize_score(
    self,
    raw_score: float,
    min_val: float | None = None,
    max_val: float | None = None,
  ) -> float:
    """Normalize a raw score to [0, 1] range.

    Args:
      raw_score: Raw score value
      min_val: Minimum possible value (defaults to 1)
      max_val: Maximum possible value (defaults to granularity)

    Returns:
      Normalized score in [0, 1]
    """
    if min_val is None:
      min_val = 1.0
    if max_val is None:
      max_val = float(self.config.granularity)

    if max_val <= min_val:
      return 0.5

    normalized = (raw_score - min_val) / (max_val - min_val)
    return max(0.0, min(1.0, normalized))
