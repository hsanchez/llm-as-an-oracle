"""LLM-as-a-Judge strategy: rubric-based holistic scoring without log-probability access."""

from __future__ import annotations

import re
import statistics
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

# Rubric anchors used when the judge scores a single trajectory.
_RUBRIC_ANCHORS = {
  1: "Completely failed — no meaningful progress toward the goal.",
  2: "Severely lacking — attempted but produced largely incorrect output.",
  3: "Mostly failed — some correct steps, but the solution is wrong overall.",
  4: "Partially correct — achieves some sub-goals but misses the main one.",
  5: "Borderline — shows understanding but contains significant errors.",
  6: "Acceptable — meets the core requirement with noticeable deficiencies.",
  7: "Good — mostly correct with only minor issues.",
  8: "Very good — correct solution with negligible imperfections.",
  9: "Excellent — fully correct, clean, and well-structured.",
  10: "Perfect — flawless solution that exceeds expectations.",
}

# Regex used to pull numeric scores out of free-form model output.
_SCORE_PATTERN = re.compile(
  r"<score>\s*(\d+(?:\.\d+)?)\s*</score>",
  re.IGNORECASE,
)
_PAIRWISE_SCORE_A_PATTERN = re.compile(
  r"<score_A>\s*(\d+(?:\.\d+)?)\s*</score_A>",
  re.IGNORECASE,
)
_PAIRWISE_SCORE_B_PATTERN = re.compile(
  r"<score_B>\s*(\d+(?:\.\d+)?)\s*</score_B>",
  re.IGNORECASE,
)
_VERDICT_PATTERN = re.compile(
  r"<verdict>\s*(A|B|tie)\s*</verdict>",
  re.IGNORECASE,
)


class JudgeStrategy(BaseStrategy):
  """LLM-as-a-Judge evaluation strategy.

  Evaluates trajectories with structured rubric-based reasoning:

    1. **Pointwise scoring** — each trajectory is scored independently per
       criterion on an integer scale (default 1–10).
    2. **Pairwise preference** — pairs of trajectories are compared side by side
       to reduce systematic positional bias; orderings are swapped and averaged.
    3. **Criteria aggregation** — per-criterion scores are combined via weighted
       average using the weights defined on each ``EvaluationCriterion``.

  The strategy deliberately avoids log-probability access so it works with any
  provider that supports plain text generation (OpenAI, Anthropic, local
  models, etc.).
  """

  def __init__(
    self,
    model: LanguageModel,
    config: ScoringConfig,
    criteria: list[EvaluationCriterion],
    *,
    score_min: float = 1.0,
    score_max: float = 10.0,
    swap_pairwise: bool = True,
    reasoning_depth: str = "detailed",
  ):
    """Initialize the judge strategy.

    **Score scale** — rubric anchors are hardcoded to 1–10.  ``score_min`` and
    ``score_max`` clip and normalize model output but do not regenerate the
    rubric text, so values outside ``[1, 10]`` will produce contradictory
    prompts and unreliable scores.  Keep the defaults unless you provide a
    custom rubric via ``additional_params``.

    Args:
      model: Language model used to generate evaluations.
      config: Scoring configuration (granularity, verification repeats, etc.).
      criteria: Evaluation criteria to score against.
      score_min: Lower bound of the numeric scale (default 1); must be < ``score_max``.
      score_max: Upper bound of the numeric scale (default 10).
      swap_pairwise: Run each pairwise comparison twice with swapped order and
        average the scores to cancel positional bias.
      reasoning_depth: One of ``"brief"``, ``"detailed"``, or
        ``"chain_of_thought"``.
    """
    super().__init__(model, config, criteria)

    if score_min >= score_max:
      raise ValueError(
        f"score_min ({score_min}) must be strictly less than score_max ({score_max})"
      )
    valid_depths = {"brief", "detailed", "chain_of_thought"}
    if reasoning_depth not in valid_depths:
      raise ValueError(f"reasoning_depth must be one of {valid_depths}, got {reasoning_depth!r}")

    self.score_min = score_min
    self.score_max = score_max
    self.swap_pairwise = swap_pairwise
    self.reasoning_depth = reasoning_depth

  # ------------------------------------------------------------------ #
  # BaseStrategy interface                                             #
  # ------------------------------------------------------------------ #

  def get_strategy_type(self) -> StrategyType:
    return StrategyType.JUDGE

  def evaluate(
    self,
    task: Task,
    trajectories: TrajectoryList,
    **kwargs: Any,
  ) -> EvaluationResult:
    """Evaluate trajectories with pointwise scoring and pairwise tie-breaking.

    Raises:
      ValueError: If ``trajectories`` is empty.
    """
    if not trajectories:
      raise ValueError("Cannot evaluate an empty trajectory list")

    trajectory_scores: dict[str, ScoreResult] = {}
    pairwise_comparisons: list[PairwiseComparison] = []

    # ── Pointwise scoring ────────────────────────────────────────────
    for trajectory in trajectories:
      score_result = self._score_trajectory_full(task, trajectory, **kwargs)
      trajectory_scores[trajectory.id] = score_result

    # ── Pairwise comparisons (tie-breaking & confidence) ─────────────
    if len(trajectories) > 1:
      for traj_a, traj_b in combinations(trajectories, 2):
        for criterion in self.criteria:
          comparison = self.compare_trajectories(task, traj_a, traj_b, criterion, **kwargs)
          pairwise_comparisons.append(comparison)

    # ── Select best trajectory ───────────────────────────────────────
    best_id = self._select_best(trajectories, trajectory_scores, pairwise_comparisons)

    return EvaluationResult(
      task_id=task.id,
      strategy_type=self.get_strategy_type(),
      best_trajectory_id=best_id,
      trajectory_scores=trajectory_scores,
      pairwise_comparisons=pairwise_comparisons,
      metadata={
        "score_range": [self.score_min, self.score_max],
        "swap_pairwise": self.swap_pairwise,
        "reasoning_depth": self.reasoning_depth,
        "num_criteria": len(self.criteria),
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
    """Score a trajectory against one criterion via pointwise rubric."""
    prompt = self._build_pointwise_prompt(task, trajectory, criterion)

    text, _, _ = self.model.generate(
      prompt,
      temperature=kwargs.get(
        "temperature", self.config.temperature if self.config.temperature is not None else 0.0
      ),
      max_tokens=kwargs.get(
        "max_tokens", self.config.max_tokens if self.config.max_tokens is not None else 2048
      ),
      return_logprobs=False,
    )

    raw_score = self._parse_pointwise_score(text)
    normalized = self._normalize(raw_score)

    return ScoreResult(
      trajectory_id=trajectory.id,
      score=normalized,
      raw_score=raw_score,
      confidence=self._score_to_confidence(raw_score),
      criterion_scores={criterion.id: normalized},
      reasoning=text,
      metadata={
        "criterion_id": criterion.id,
        "criterion_name": criterion.name,
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
    """Compare two trajectories; when ``swap_pairwise`` is set, runs twice and averages to cancel positional bias."""
    gen_kwargs = {
      "temperature": kwargs.get(
        "temperature", self.config.temperature if self.config.temperature is not None else 0.0
      ),
      "max_tokens": kwargs.get(
        "max_tokens", self.config.max_tokens if self.config.max_tokens is not None else 2048
      ),
      "return_logprobs": False,
    }

    score_a, score_b, reasoning = self._single_pairwise(
      task, trajectory_a, trajectory_b, criterion, **gen_kwargs
    )

    if self.swap_pairwise:
      # Swap order to cancel positional bias; note the score roles are reversed.
      swapped_b, swapped_a, _ = self._single_pairwise(
        task, trajectory_b, trajectory_a, criterion, **gen_kwargs
      )
      score_a = (score_a + swapped_a) / 2.0
      score_b = (score_b + swapped_b) / 2.0

    winner: str | None
    if score_a > score_b:
      winner = trajectory_a.id
    elif score_b > score_a:
      winner = trajectory_b.id
    else:
      winner = None  # tie

    return PairwiseComparison(
      trajectory_a_id=trajectory_a.id,
      trajectory_b_id=trajectory_b.id,
      score_a=self._normalize(score_a),
      score_b=self._normalize(score_b),
      winner=winner,
      criterion_id=criterion.id,
      confidence=self._pairwise_confidence(score_a, score_b),
      reasoning=reasoning,
    )

  # ------------------------------------------------------------------ #
  # Prompt construction                                                #
  # ------------------------------------------------------------------ #

  def _build_pointwise_prompt(
    self,
    task: Task,
    trajectory: Trajectory,
    criterion: EvaluationCriterion,
  ) -> str:
    rubric_lines = "\n".join(f"  {score}: {desc}" for score, desc in _RUBRIC_ANCHORS.items())
    reasoning_instruction = self._reasoning_instruction()

    reference_block = ""
    if task.ground_truth:
      reference_block = (
        f"**Reference Solution:**\n```\n{task.ground_truth}\n```\n"
        "Use the reference to verify factual correctness where applicable.\n\n"
      )

    output_block = ""
    if trajectory.output:
      output_block = f"\n**Execution Output:**\n```\n{trajectory.output}\n```"

    return f"""You are an impartial expert judge evaluating an AI agent's work.

## Task
{task.problem_statement}

{reference_block}## Agent Trajectory
```
{trajectory.content}
```{output_block}

## Evaluation Criterion
**{criterion.name}:** {criterion.description}

## Scoring Rubric ({int(self.score_min)}–{int(self.score_max)})
{rubric_lines}

## Instructions
{reasoning_instruction}

After your analysis, output your final score using exactly this format:
<score>NUMBER</score>

The number must be between {int(self.score_min)} and {int(self.score_max)} (decimals allowed).
Do not include any text inside the tags other than the number."""

  def _build_pairwise_prompt(
    self,
    task: Task,
    trajectory_a: Trajectory,
    trajectory_b: Trajectory,
    criterion: EvaluationCriterion,
  ) -> str:
    reasoning_instruction = self._reasoning_instruction()

    reference_block = ""
    if task.ground_truth:
      reference_block = f"**Reference Solution:**\n```\n{task.ground_truth}\n```\n\n"

    output_a_block = ""
    if trajectory_a.output:
      output_a_block = f"\n**Output A:**\n```\n{trajectory_a.output}\n```"

    output_b_block = ""
    if trajectory_b.output:
      output_b_block = f"\n**Output B:**\n```\n{trajectory_b.output}\n```"

    return f"""You are an impartial expert judge comparing two AI agent trajectories.

## Task
{task.problem_statement}

{reference_block}## Trajectory A
```
{trajectory_a.content}
```{output_a_block}

## Trajectory B
```
{trajectory_b.content}
```{output_b_block}

## Evaluation Criterion
**{criterion.name}:** {criterion.description}

## Instructions
{reasoning_instruction}

Score each trajectory on the criterion above using the scale \
{int(self.score_min)}–{int(self.score_max)}, where {int(self.score_min)} is \
the worst and {int(self.score_max)} is the best.

Then output your final scores and an overall verdict using exactly this format \
(all three tags are required):
<score_A>NUMBER</score_A>
<score_B>NUMBER</score_B>
<verdict>A</verdict>  ← use "A", "B", or "tie"

Numbers must be between {int(self.score_min)} and {int(self.score_max)} \
(decimals allowed).  Do not include any text inside the tags other than the \
required value."""

  def _reasoning_instruction(self) -> str:
    """Return the reasoning instruction for the current depth setting."""
    if self.reasoning_depth == "brief":
      return "Write a concise 1–2 sentence justification before providing your score."
    if self.reasoning_depth == "detailed":
      return (
        "Provide a structured analysis covering: (1) what was done correctly, "
        "(2) what was done incorrectly or is missing, and (3) your overall "
        "assessment. Be specific and reference the trajectory content."
      )
    # chain_of_thought
    return (
      "Think step by step:\n"
      "  1. Identify the key requirements for the criterion.\n"
      "  2. Check each requirement against the trajectory.\n"
      "  3. Note any partial credit considerations.\n"
      "  4. Arrive at a justified score.\n"
      "Show all reasoning explicitly before your final score."
    )

  # ------------------------------------------------------------------ #
  # Scoring helpers                                                    #
  # ------------------------------------------------------------------ #

  def _score_trajectory_full(
    self,
    task: Task,
    trajectory: Trajectory,
    **kwargs: Any,
  ) -> ScoreResult:
    """Score a trajectory across all criteria with repeated verifications."""
    criterion_scores: dict[str, list[float]] = {c.id: [] for c in self.criteria}
    all_reasoning: list[str] = []

    for criterion in self.criteria:
      for rep in range(self.config.num_verifications):
        result = self.score_trajectory(task, trajectory, criterion, **kwargs)
        criterion_scores[criterion.id].append(result.score)
        if rep == 0:
          # Keep one reasoning example per criterion
          all_reasoning.append(f"[{criterion.name}]\n{result.reasoning.strip()}")

    # Average repeated verifications per criterion
    avg_criterion_scores: dict[str, float] = {
      cid: statistics.mean(scores) for cid, scores in criterion_scores.items() if scores
    }

    overall = self.aggregate_criterion_scores(avg_criterion_scores)
    confidence = self._multi_criterion_confidence(avg_criterion_scores)

    return ScoreResult(
      trajectory_id=trajectory.id,
      score=overall,
      criterion_scores=avg_criterion_scores,
      confidence=confidence,
      reasoning="\n\n".join(all_reasoning),
      metadata={
        "num_criteria": len(self.criteria),
        "num_verifications": self.config.num_verifications,
      },
    )

  def _single_pairwise(
    self,
    task: Task,
    trajectory_a: Trajectory,
    trajectory_b: Trajectory,
    criterion: EvaluationCriterion,
    **gen_kwargs: Any,
  ) -> tuple[float, float, str]:
    """Run one pairwise call; returns ``(score_a, score_b, reasoning)``."""
    prompt = self._build_pairwise_prompt(task, trajectory_a, trajectory_b, criterion)
    text, _, _ = self.model.generate(prompt, **gen_kwargs)

    score_a = self._parse_tagged_float(text, _PAIRWISE_SCORE_A_PATTERN)
    score_b = self._parse_tagged_float(text, _PAIRWISE_SCORE_B_PATTERN)

    return score_a, score_b, text

  # ------------------------------------------------------------------ #
  # Best-trajectory selection                                            #
  # ------------------------------------------------------------------ #

  def _select_best(
    self,
    trajectories: TrajectoryList,
    scores: dict[str, ScoreResult],
    comparisons: list[PairwiseComparison],
  ) -> str:
    """Return the best trajectory ID using 70% pointwise + 30% pairwise win-rate."""
    if not trajectories:
      raise ValueError("Trajectory list is empty")

    if len(trajectories) == 1:
      return trajectories[0].id

    # Build combined score: 70 % pointwise + 30 % pairwise win-rate
    win_rates = self._compute_win_rates(trajectories, comparisons)

    def combined(tid: str) -> float:
      pointwise = scores[tid].score if tid in scores else 0.5
      pairwise_wr = win_rates.get(tid, 0.5)
      return 0.7 * pointwise + 0.3 * pairwise_wr

    return max((t.id for t in trajectories), key=combined)

  def _compute_win_rates(
    self,
    trajectories: TrajectoryList,
    comparisons: list[PairwiseComparison],
  ) -> dict[str, float]:
    """Return per-trajectory win rate in [0, 1] from pairwise comparisons."""
    wins: dict[str, float] = {t.id: 0.0 for t in trajectories}
    totals: dict[str, int] = {t.id: 0 for t in trajectories}

    for comp in comparisons:
      # Each side participates in this match
      totals[comp.trajectory_a_id] = totals.get(comp.trajectory_a_id, 0) + 1
      totals[comp.trajectory_b_id] = totals.get(comp.trajectory_b_id, 0) + 1

      if comp.winner == comp.trajectory_a_id:
        wins[comp.trajectory_a_id] = wins.get(comp.trajectory_a_id, 0.0) + 1.0
      elif comp.winner == comp.trajectory_b_id:
        wins[comp.trajectory_b_id] = wins.get(comp.trajectory_b_id, 0.0) + 1.0
      else:
        # Tie
        wins[comp.trajectory_a_id] = wins.get(comp.trajectory_a_id, 0.0) + 0.5
        wins[comp.trajectory_b_id] = wins.get(comp.trajectory_b_id, 0.0) + 0.5

    return {tid: wins[tid] / totals[tid] if totals[tid] > 0 else 0.5 for tid in wins}

  # ------------------------------------------------------------------ #
  # Parsing utilities                                                    #
  # ------------------------------------------------------------------ #

  def _parse_pointwise_score(self, text: str) -> float:
    """Parse a pointwise score from model output, clamped to ``[score_min, score_max]``."""
    return self._parse_tagged_float(text, _SCORE_PATTERN)

  def _parse_tagged_float(
    self,
    text: str,
    pattern: re.Pattern,
  ) -> float:
    """Extract a clamped float from text via regex; returns midpoint on failure."""
    match = pattern.search(text or "")
    if match:
      try:
        raw = float(match.group(1))
        return max(self.score_min, min(self.score_max, raw))
      except ValueError:
        pass

    # Graceful fallback: return midpoint
    return (self.score_min + self.score_max) / 2.0

  # ------------------------------------------------------------------ #
  # Normalisation & confidence                                           #
  # ------------------------------------------------------------------ #

  def _normalize(self, raw: float) -> float:
    """Normalize a raw score from ``[score_min, score_max]`` to ``[0, 1]``."""
    span = self.score_max - self.score_min
    if span == 0:
      return 0.5
    return max(0.0, min(1.0, (raw - self.score_min) / span))

  def _score_to_confidence(self, raw: float) -> float:
    """Map a raw score to confidence; extreme scores are more reliable than mid-range."""
    normalized = self._normalize(raw)
    # Distance from 0.5 (center) → higher distance = more confident
    distance = abs(normalized - 0.5)
    return 0.5 + distance

  def _multi_criterion_confidence(
    self,
    criterion_scores: dict[str, float],
  ) -> float:
    """Estimate confidence from criterion agreement (low stdev = high confidence)."""
    if not criterion_scores:
      return 0.5
    scores = list(criterion_scores.values())
    if len(scores) == 1:
      return self._score_to_confidence(
        scores[0] * (self.score_max - self.score_min) + self.score_min
      )

    try:
      stdev = statistics.stdev(scores)
    except statistics.StatisticsError:
      stdev = 0.0

    # Low stdev → high agreement → high confidence
    # stdev is in [0, 0.5] for normalized scores; map linearly.
    confidence = 1.0 - min(stdev * 2.0, 0.5)
    return max(0.5, confidence)

  def _pairwise_confidence(self, score_a: float, score_b: float) -> float:
    """Compute confidence from pairwise margin; larger gap = higher confidence."""
    span = self.score_max - self.score_min
    if span == 0:
      return 0.5
    margin = abs(score_a - score_b) / span
    # margin ∈ [0, 1]; map to confidence ∈ [0.5, 1.0]
    return 0.5 + margin * 0.5
