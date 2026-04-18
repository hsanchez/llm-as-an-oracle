"""LLM as Verifier strategy implementation.

This module implements the LLM-as-a-Verifier approach, which uses fine-grained
scoring with log probabilities, repeated verification, and criteria decomposition
to evaluate trajectories. It achieves higher accuracy than traditional judge
approaches by scaling scoring granularity and using pairwise tournament selection.
"""

from __future__ import annotations

import math
import re
from itertools import combinations
from typing import Any

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
from llm_oracle.core.strategy import BaseStrategy, LanguageModel


class VerifierStrategy(BaseStrategy):
  """LLM-as-a-Verifier evaluation strategy.

  This strategy approximates the reward of a trajectory using:
    R(t, τ) = (1/CK) Σ_c Σ_k Σ_g p_θ(v_g | t, c, τ) · φ(v_g)

  Where:
    - C = number of evaluation criteria
    - K = number of repeated verifications
    - G = number of score tokens (granularity level)
    - p_θ(v_g | t, c, τ) = probability assigned to score token v_g
    - φ(v_g) = mapping of score token to scalar value

  Selection uses round-robin tournament: for every pair (i, j), the trajectory
  with higher reward receives a win. The trajectory with most wins is selected.
  """

  def __init__(
    self,
    model: LanguageModel,
    config: ScoringConfig,
    criteria: list[EvaluationCriterion],
  ):
    """Initialize the verifier strategy.

    Args:
      model: Language model supporting log probability extraction
      config: Scoring configuration with granularity and verification settings
      criteria: List of evaluation criteria for decomposition
    """
    super().__init__(model, config, criteria)
    self._scale_info = self.get_scale_description()

  def get_strategy_type(self) -> StrategyType:
    """Return the strategy type."""
    return StrategyType.VERIFIER

  def evaluate(
    self,
    task: Task,
    trajectories: TrajectoryList,
    **kwargs: Any,
  ) -> EvaluationResult:
    """Evaluate trajectories using verifier approach with tournament selection.

    Args:
      task: The task being evaluated
      trajectories: List of candidate trajectories (n >= 1)
      **kwargs: Additional parameters

    Returns:
      Complete evaluation result with best trajectory selected via tournament

    Raises:
      ValueError: If trajectories list is empty
    """
    if not trajectories:
      raise ValueError("Cannot evaluate empty trajectory list")

    if len(trajectories) == 1:
      # Single trajectory - just score it
      trajectory = trajectories[0]
      score_result = self._score_trajectory_multi_criterion(task, trajectory)
      return EvaluationResult(
        task_id=task.id,
        strategy_type=self.get_strategy_type(),
        best_trajectory_id=trajectory.id,
        trajectory_scores={trajectory.id: score_result},
        pairwise_comparisons=[],
      )

    # Multiple trajectories - use pairwise tournament
    pairwise_comparisons = []
    trajectory_scores = {}

    # Perform all pairwise comparisons
    for traj_a, traj_b in combinations(trajectories, 2):
      for criterion in self.criteria:
        # Perform K repeated verifications per criterion
        for rep in range(self.config.num_verifications):
          comparison = self.compare_trajectories(task, traj_a, traj_b, criterion, repetition=rep)
          pairwise_comparisons.append(comparison)

    # Aggregate scores for each trajectory
    for trajectory in trajectories:
      criterion_scores = self._aggregate_trajectory_scores(trajectory.id, pairwise_comparisons)
      overall_score = self.aggregate_criterion_scores(criterion_scores)
      trajectory_scores[trajectory.id] = ScoreResult(
        trajectory_id=trajectory.id,
        score=overall_score,
        criterion_scores=criterion_scores,
        confidence=self._compute_confidence(criterion_scores),
      )

    # Select best via tournament
    best_id = self._tournament_selection(trajectories, pairwise_comparisons)

    return EvaluationResult(
      task_id=task.id,
      strategy_type=self.get_strategy_type(),
      best_trajectory_id=best_id,
      trajectory_scores=trajectory_scores,
      pairwise_comparisons=pairwise_comparisons,
      metadata={
        "num_comparisons": len(pairwise_comparisons),
        "granularity": self.config.granularity,
        "num_verifications": self.config.num_verifications,
      },
    )

  def score_trajectory(
    self,
    task: Task,
    trajectory: Trajectory,
    criterion: EvaluationCriterion,
    **kwargs: Any,
  ) -> ScoreResult:
    """Score a single trajectory against a criterion using log probabilities.

    Args:
      task: The task being evaluated
      trajectory: The trajectory to score
      criterion: Evaluation criterion to use
      **kwargs: Additional parameters

    Returns:
      Score result with normalized score from log probabilities
    """
    prompt = self._create_scoring_prompt(task, trajectory, criterion)

    # Generate with log probabilities
    text, tokens, position_logprobs = self.model.generate(
      prompt,
      temperature=self.config.additional_params.get("temperature", 1.0),
      max_tokens=self.config.additional_params.get("max_tokens", 4096),
      return_logprobs=True,
    )

    # Extract score from log probabilities
    tag = "<score>"
    raw_score, confidence = self._extract_score_from_logprobs(text, tokens, position_logprobs, tag)

    # Normalize to [0, 1]
    normalized_score = self.normalize_score(raw_score)

    return ScoreResult(
      trajectory_id=trajectory.id,
      score=normalized_score,
      raw_score=raw_score,
      confidence=confidence,
      criterion_scores={criterion.id: normalized_score},
      reasoning=text,
      metadata={
        "criterion_id": criterion.id,
        "extraction_tag": tag,
      },
    )

  def compare_trajectories(
    self,
    task: Task,
    trajectory_a: Trajectory,
    trajectory_b: Trajectory,
    criterion: EvaluationCriterion,
    **kwargs: Any,
  ) -> PairwiseComparison:
    """Compare two trajectories pairwise for a specific criterion.

    Args:
      task: The task being evaluated
      trajectory_a: First trajectory
      trajectory_b: Second trajectory
      criterion: Criterion for comparison
      **kwargs: Additional parameters (e.g., repetition number)

    Returns:
      Pairwise comparison with scores for both trajectories
    """
    prompt = self._create_pairwise_prompt(task, trajectory_a, trajectory_b, criterion)

    # Generate with log probabilities
    text, tokens, position_logprobs = self.model.generate(
      prompt,
      temperature=self.config.additional_params.get("temperature", 1.0),
      max_tokens=self.config.additional_params.get("max_tokens", 4096),
      return_logprobs=True,
    )

    # Extract scores for both trajectories
    score_a, conf_a = self._extract_score_from_logprobs(
      text, tokens, position_logprobs, "<score_A>"
    )
    score_b, conf_b = self._extract_score_from_logprobs(
      text, tokens, position_logprobs, "<score_B>"
    )

    # Normalize scores
    norm_score_a = self.normalize_score(score_a)
    norm_score_b = self.normalize_score(score_b)

    # Determine winner
    winner = None
    if norm_score_a > norm_score_b:
      winner = trajectory_a.id
    elif norm_score_b > norm_score_a:
      winner = trajectory_b.id

    return PairwiseComparison(
      trajectory_a_id=trajectory_a.id,
      trajectory_b_id=trajectory_b.id,
      score_a=norm_score_a,
      score_b=norm_score_b,
      winner=winner,
      criterion_id=criterion.id,
      confidence=min(conf_a, conf_b),
      reasoning=text,
    )

  def _score_trajectory_multi_criterion(
    self,
    task: Task,
    trajectory: Trajectory,
  ) -> ScoreResult:
    """Score trajectory across all criteria with repeated verifications.

    Args:
      task: The task being evaluated
      trajectory: The trajectory to score

    Returns:
      Aggregated score result across all criteria
    """
    criterion_scores: dict[str, list[float]] = {c.id: [] for c in self.criteria}

    for criterion in self.criteria:
      for _ in range(self.config.num_verifications):
        result = self.score_trajectory(task, trajectory, criterion)
        criterion_scores[criterion.id].append(result.score)

    # Average scores across repetitions
    avg_criterion_scores = {
      cid: sum(scores) / len(scores) for cid, scores in criterion_scores.items()
    }

    overall_score = self.aggregate_criterion_scores(avg_criterion_scores)

    return ScoreResult(
      trajectory_id=trajectory.id,
      score=overall_score,
      criterion_scores=avg_criterion_scores,
      confidence=self._compute_confidence(avg_criterion_scores),
    )

  def _create_scoring_prompt(
    self,
    task: Task,
    trajectory: Trajectory,
    criterion: EvaluationCriterion,
  ) -> str:
    """Create prompt for scoring a single trajectory.

    Args:
      task: The task being evaluated
      trajectory: The trajectory to score
      criterion: Evaluation criterion

    Returns:
      Formatted prompt string
    """
    ground_truth_note = ""
    if task.ground_truth:
      ground_truth_note = (
        f"**Ground Truth Reference:**\n{task.ground_truth}\nUse this to verify correctness.\n"
      )

    test_cases_block = ""
    if task.test_cases:
      cases_text = "\n".join(
        f"  Input: {c.get('input', '')!r} → Expected: {c.get('expected', '')!r}"
        for c in task.test_cases
      )
      test_cases_block = f"\n**Test Cases:**\n{cases_text}\n"

    output_block = ""
    if trajectory.output:
      output_block = f"\n**Execution Output:**\n```\n{trajectory.output}\n```"

    scale_desc = self._scale_info["scale_description"]
    score_format = self._scale_info["score_format"]

    return f"""You are an expert evaluator of AI agent trajectories.

{ground_truth_note}{test_cases_block}

**Task Description:**
{task.problem_statement}

**Agent Trajectory:**
{trajectory.content}{output_block}

**Evaluation Criterion — {criterion.name}:**
{criterion.description}

Evaluate how well this trajectory satisfies the criterion "{criterion.name}".
Focus exclusively on this criterion and ignore other aspects.

**Rating Scale:**
{scale_desc}

Provide your analysis and then output your final score in this exact format:
<score>{score_format}</score>

Begin your evaluation now."""

  def _create_pairwise_prompt(
    self,
    task: Task,
    trajectory_a: Trajectory,
    trajectory_b: Trajectory,
    criterion: EvaluationCriterion,
  ) -> str:
    """Create prompt for pairwise trajectory comparison.

    Args:
      task: The task being evaluated
      trajectory_a: First trajectory
      trajectory_b: Second trajectory
      criterion: Evaluation criterion

    Returns:
      Formatted prompt string
    """
    ground_truth_note = ""
    if task.ground_truth:
      ground_truth_note = (
        f"**Ground Truth Reference:**\n{task.ground_truth}\nUse this to verify correctness.\n"
      )

    test_cases_block = ""
    if task.test_cases:
      cases_text = "\n".join(
        f"  Input: {c.get('input', '')!r} → Expected: {c.get('expected', '')!r}"
        for c in task.test_cases
      )
      test_cases_block = f"\n**Test Cases:**\n{cases_text}\n"

    output_a_block = ""
    if trajectory_a.output:
      output_a_block = f"\n**Output A:**\n```\n{trajectory_a.output}\n```"

    output_b_block = ""
    if trajectory_b.output:
      output_b_block = f"\n**Output B:**\n```\n{trajectory_b.output}\n```"

    scale_desc = self._scale_info["scale_description"]
    score_format = self._scale_info["score_format"]

    return f"""You are an expert evaluator of AI agent trajectories.
You will see a task description and two agent trajectories.
Your job is to evaluate them on ONE specific criterion: **{criterion.name}**.

{ground_truth_note}{test_cases_block}

**Task:**
{task.problem_statement}

**Trajectory A:**
{trajectory_a.content}{output_a_block}

**Trajectory B:**
{trajectory_b.content}{output_b_block}

**Evaluation Guideline — {criterion.name}:**
{criterion.description}

Score each trajectory ONLY on this specific criterion. Ignore other aspects
that are not relevant to "{criterion.name}".

**Rating Scale:**
{scale_desc}

Then output your final scores:
<score_A>{score_format}</score_A>
<score_B>{score_format}</score_B>

Begin your analysis now."""

  def _extract_score_from_logprobs(
    self,
    text: str,
    tokens: list[str] | None,
    position_logprobs: list[list[tuple[str, float]]] | None,
    tag: str,
  ) -> tuple[float, float]:
    """Extract score from log probabilities at the specified tag.

    Args:
      text: Generated text
      tokens: List of generated tokens
      position_logprobs: Log probabilities for each position
      tag: XML-style tag to extract score from (e.g., "<score>")

    Returns:
      Tuple of (raw_score, confidence)
    """
    valid_tokens = self._scale_info["valid_tokens"]

    # Try to extract from log probabilities first
    if tokens and position_logprobs and self.config.use_logprobs:
      tag_logprobs = self._find_tag_logprobs(tokens, position_logprobs, tag)

      if tag_logprobs:
        probs: dict[float, float] = {}

        for tok_str, logprob in tag_logprobs:
          tok = tok_str.strip()
          if tok in valid_tokens:
            val = valid_tokens[tok]
            p = math.exp(logprob)
            probs[val] = max(probs.get(val, 0.0), p)

        if probs:
          # Compute expected value from probability distribution
          total_p = sum(probs.values())
          expected = sum(v * p for v, p in probs.items()) / total_p

          # Confidence is the probability mass on the top choice
          confidence = max(probs.values()) / total_p if total_p > 0 else 0.5

          return expected, confidence

    # Fallback: parse from text
    tag_name = tag.strip("<>")
    pattern = rf"<{re.escape(tag_name)}>\s*(.+?)\s*</{re.escape(tag_name)}>"
    match = re.search(pattern, text or "", re.IGNORECASE)

    if match:
      tok = match.group(1).strip()
      raw_val = valid_tokens.get(tok)

      if raw_val is None:
        # Try case-insensitive match
        for vt, val in valid_tokens.items():
          if tok.lower() == vt.lower():
            raw_val = val
            break

      if raw_val is not None:
        return raw_val, 0.8  # Moderate confidence for text extraction

    # Default to middle of scale
    return float(self.config.granularity) / 2.0, 0.5

  def _find_tag_logprobs(
    self,
    tokens: list[str],
    position_logprobs: list[list[tuple[str, float]]],
    tag: str,
  ) -> list[tuple[str, float]] | None:
    """Find log probabilities at the position immediately after a tag.

    Args:
      tokens: List of generated tokens
      position_logprobs: Log probabilities for each position
      tag: Tag to search for

    Returns:
      List of (token, logprob) tuples at the tag position, or None
    """
    if not tokens or not position_logprobs:
      return None

    text_so_far = ""
    for i, tok in enumerate(tokens):
      text_so_far += tok
      if text_so_far.rstrip().endswith(tag):
        # Return logprobs for next position
        if i + 1 < len(position_logprobs):
          return position_logprobs[i + 1]

    return None

  def _aggregate_trajectory_scores(
    self,
    trajectory_id: str,
    comparisons: list[PairwiseComparison],
  ) -> dict[str, float]:
    """Aggregate scores for a trajectory across all comparisons.

    Args:
      trajectory_id: ID of trajectory to aggregate scores for
      comparisons: All pairwise comparisons

    Returns:
      Dictionary mapping criterion IDs to aggregated scores
    """
    criterion_scores: dict[str, list[float]] = {}

    for comp in comparisons:
      if comp.criterion_id is None:
        continue

      if comp.criterion_id not in criterion_scores:
        criterion_scores[comp.criterion_id] = []

      # Add this trajectory's score from the comparison
      if comp.trajectory_a_id == trajectory_id:
        criterion_scores[comp.criterion_id].append(comp.score_a)
      elif comp.trajectory_b_id == trajectory_id:
        criterion_scores[comp.criterion_id].append(comp.score_b)

    # Average scores for each criterion
    return {cid: sum(scores) / len(scores) for cid, scores in criterion_scores.items() if scores}

  def _tournament_selection(
    self,
    trajectories: TrajectoryList,
    comparisons: list[PairwiseComparison],
  ) -> str:
    """Select best trajectory using round-robin tournament.

    Args:
      trajectories: List of all trajectories
      comparisons: All pairwise comparisons

    Returns:
      ID of winning trajectory
    """
    # Count wins for each trajectory
    wins: dict[str, float] = {t.id: 0.0 for t in trajectories}

    # Group comparisons by trajectory pair and criterion
    comparison_map: dict[tuple[str, str, str], list[PairwiseComparison]] = {}

    for comp in comparisons:
      key = (comp.trajectory_a_id, comp.trajectory_b_id, comp.criterion_id or "")
      if key not in comparison_map:
        comparison_map[key] = []
      comparison_map[key].append(comp)

    # For each unique pair+criterion, aggregate scores and award wins
    for key, comps in comparison_map.items():
      tid_a, tid_b, _ = key

      # Average scores across repetitions
      avg_score_a = sum(c.score_a for c in comps) / len(comps)
      avg_score_b = sum(c.score_b for c in comps) / len(comps)

      if avg_score_a > avg_score_b:
        wins[tid_a] += 1.0
      elif avg_score_b > avg_score_a:
        wins[tid_b] += 1.0
      else:
        # Tie - half win to each
        wins[tid_a] += 0.5
        wins[tid_b] += 0.5

    # Return trajectory with most wins
    return max(wins.keys(), key=lambda tid: wins[tid])

  def _compute_confidence(self, criterion_scores: dict[str, float]) -> float:
    """Compute confidence based on score variance across criteria.

    Args:
      criterion_scores: Scores per criterion

    Returns:
      Confidence value (0.0-1.0)
    """
    if not criterion_scores:
      return 0.5

    scores = list(criterion_scores.values())
    if len(scores) == 1:
      return 0.9  # High confidence with single criterion

    # Lower variance = higher confidence
    mean = sum(scores) / len(scores)
    variance = sum((s - mean) ** 2 for s in scores) / len(scores)

    # Convert variance to confidence (inverse relationship)
    # Variance ranges from 0 (all same) to 0.25 (max spread in [0,1])
    confidence = 1.0 - min(variance * 4.0, 1.0)

    return max(0.5, confidence)  # Minimum 50% confidence
