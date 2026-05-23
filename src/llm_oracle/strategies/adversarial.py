"""Adversarial claim verification built from confirmation and challenge passes."""

from __future__ import annotations

import enum
from typing import Any

from llm_oracle.core.models import (
  EvaluationCriterion,
  EvaluationResult,
  PairwiseComparison,
  ScoreResult,
  StrategyType,
  Task,
  Trajectory,
  TrajectoryList,
)
from llm_oracle.core.strategy import BaseStrategy


class AdversarialDecision(enum.StrEnum):
  """Decision produced by adversarial claim verification."""

  CONFIRMED = "confirmed"
  REJECTED = "rejected"
  UNCERTAIN = "uncertain"


class AdversarialVerifierStrategy(BaseStrategy):
  """Verifies a claim with confirmation and evidence-based challenge passes.

  The confirmation verifier checks whether the claim is supported. The
  challenge verifier checks whether there is a strong evidence-based reason the
  claim is wrong. Disagreement, weak confidence, and ambiguous evidence produce
  an ``uncertain`` decision so host applications can escalate to a human oracle.
  """

  def __init__(
    self,
    confirmation_verifier: BaseStrategy,
    challenge_verifier: BaseStrategy,
    confirmation_criterion: EvaluationCriterion,
    challenge_criterion: EvaluationCriterion,
    *,
    confirmation_threshold: float = 0.65,
    min_confidence: float = 0.70,
  ) -> None:
    if confirmation_verifier.get_strategy_type() != StrategyType.VERIFIER:
      raise TypeError(
        "'confirmation_verifier' must be a VERIFIER strategy, "
        f"got {confirmation_verifier.get_strategy_type()}"
      )
    if challenge_verifier.get_strategy_type() != StrategyType.VERIFIER:
      raise TypeError(
        "'challenge_verifier' must be a VERIFIER strategy, "
        f"got {challenge_verifier.get_strategy_type()}"
      )
    if (
      confirmation_verifier.config.num_verifications != challenge_verifier.config.num_verifications
    ):
      raise ValueError(
        "confirmation_verifier and challenge_verifier must use the same num_verifications value."
      )
    _validate_probability("confirmation_threshold", confirmation_threshold)
    _validate_probability("min_confidence", min_confidence)

    super().__init__(
      confirmation_verifier.model,
      confirmation_verifier.config,
      [confirmation_criterion, challenge_criterion],
    )

    self.confirmation_verifier = confirmation_verifier
    self.challenge_verifier = challenge_verifier
    self.confirmation_criterion = confirmation_criterion
    self.challenge_criterion = challenge_criterion
    self.confirmation_threshold = confirmation_threshold
    self.min_confidence = min_confidence

  def get_strategy_type(self) -> StrategyType:
    return StrategyType.ADVERSARIAL

  def evaluate(
    self,
    task: Task,
    trajectories: TrajectoryList,
    **kwargs: Any,
  ) -> EvaluationResult:
    """Evaluate claim trajectories with adversarial verification.

    Raises:
      ValueError: If trajectories list is empty.
    """
    if not trajectories:
      raise ValueError("Cannot evaluate empty trajectory list")

    trajectory_scores = {
      trajectory.id: self.score_trajectory(task, trajectory, self.confirmation_criterion, **kwargs)
      for trajectory in trajectories
    }
    best_trajectory_id = self.select_best_trajectory(task, trajectories, trajectory_scores)

    return EvaluationResult(
      task_id=task.id,
      strategy_type=self.get_strategy_type(),
      best_trajectory_id=best_trajectory_id,
      trajectory_scores=trajectory_scores,
      metadata={
        "adversarial": True,
        "confirmation_threshold": self.confirmation_threshold,
        "min_confidence": self.min_confidence,
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
    """Score one claim trajectory with repeated confirmation/challenge passes.

    Raises:
      ValueError: If ``criterion`` is not one of this strategy's configured
        confirmation or challenge criteria.
    """
    self._validate_adversarial_criterion(criterion)
    confirmation_results: list[ScoreResult] = []
    challenge_results: list[ScoreResult] = []

    for _ in range(self.config.num_verifications):
      confirmation_results.append(
        self.confirmation_verifier.score_trajectory(
          task,
          trajectory,
          self.confirmation_criterion,
          **kwargs,
        )
      )
      challenge_results.append(
        self.challenge_verifier.score_trajectory(
          task,
          trajectory,
          self.challenge_criterion,
          **kwargs,
        )
      )

    confirmation_result = _aggregate_pass_results(
      trajectory.id,
      self.confirmation_criterion.id,
      confirmation_results,
    )
    challenge_result = _aggregate_pass_results(
      trajectory.id,
      self.challenge_criterion.id,
      challenge_results,
    )
    decision = self._decide(confirmation_result, challenge_result)
    score = self._claim_support_score(confirmation_result, challenge_result, decision)
    confidence = min(confirmation_result.confidence, challenge_result.confidence)

    return ScoreResult(
      trajectory_id=trajectory.id,
      score=score,
      confidence=confidence,
      criterion_scores={
        self.confirmation_criterion.id: confirmation_result.score,
        self.challenge_criterion.id: challenge_result.score,
      },
      reasoning=_combine_reasoning(confirmation_result, challenge_result),
      metadata={
        "decision": decision.value,
        "confirmation_score": confirmation_result.score,
        "challenge_score": challenge_result.score,
        "confirmation_confidence": confirmation_result.confidence,
        "challenge_confidence": challenge_result.confidence,
        "confirmation_threshold": self.confirmation_threshold,
        "min_confidence": self.min_confidence,
        "num_verifications": self.config.num_verifications,
        "decision_reason": self._decision_reason(
          confirmation_result,
          challenge_result,
          decision,
        ),
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
    """Compare claim trajectories by adversarial claim-support score.

    Raises:
      ValueError: If ``criterion`` is not one of this strategy's configured
        confirmation or challenge criteria.
    """
    self._validate_adversarial_criterion(criterion)
    score_a = self.score_trajectory(task, trajectory_a, criterion, **kwargs)
    score_b = self.score_trajectory(task, trajectory_b, criterion, **kwargs)

    winner = None
    if score_a.score > score_b.score:
      winner = trajectory_a.id
    elif score_b.score > score_a.score:
      winner = trajectory_b.id

    return PairwiseComparison(
      trajectory_a_id=trajectory_a.id,
      trajectory_b_id=trajectory_b.id,
      score_a=score_a.score,
      score_b=score_b.score,
      winner=winner,
      criterion_id=criterion.id,
      confidence=min(score_a.confidence, score_b.confidence),
      reasoning=_combine_pairwise_reasoning(score_a, score_b),
    )

  def _decide(
    self,
    confirmation_result: ScoreResult,
    challenge_result: ScoreResult,
  ) -> AdversarialDecision:
    confirmation_high = confirmation_result.score >= self.confirmation_threshold
    # The first implementation uses one threshold for both "support is strong
    # enough" and "challenge is strong enough". A future challenge_threshold can
    # support stricter rejection than confirmation when false rejections are costlier.
    challenge_high = challenge_result.score >= self.confirmation_threshold
    both_confident = (
      confirmation_result.confidence >= self.min_confidence
      and challenge_result.confidence >= self.min_confidence
    )

    if confirmation_high and not challenge_high and both_confident:
      return AdversarialDecision.CONFIRMED
    if not confirmation_high and challenge_high and both_confident:
      return AdversarialDecision.REJECTED
    return AdversarialDecision.UNCERTAIN

  def _claim_support_score(
    self,
    confirmation_result: ScoreResult,
    challenge_result: ScoreResult,
    decision: AdversarialDecision,
  ) -> float:
    if decision == AdversarialDecision.CONFIRMED:
      return confirmation_result.score
    if decision == AdversarialDecision.REJECTED:
      return 1.0 - challenge_result.score
    return (confirmation_result.score + (1.0 - challenge_result.score)) / 2.0

  def _decision_reason(
    self,
    confirmation_result: ScoreResult,
    challenge_result: ScoreResult,
    decision: AdversarialDecision,
  ) -> str:
    # Keep model justification and deterministic policy explanation separate for auditability.
    if decision == AdversarialDecision.CONFIRMED:
      return "Confirmation is high, challenge is low, and both passes are confident."
    if decision == AdversarialDecision.REJECTED:
      return "Confirmation is low, challenge is high, and both passes are confident."
    if (
      confirmation_result.confidence < self.min_confidence
      or challenge_result.confidence < self.min_confidence
    ):
      return "At least one adversarial verification pass has low confidence."
    return "Confirmation and challenge scores do not produce a decisive claim judgment."

  def _validate_adversarial_criterion(self, criterion: EvaluationCriterion) -> None:
    allowed_ids = {self.confirmation_criterion.id, self.challenge_criterion.id}
    if criterion.id not in allowed_ids:
      allowed = ", ".join(sorted(allowed_ids))
      raise ValueError(
        f"criterion must be one of the configured adversarial criteria: {allowed}. "
        f"Got {criterion.id!r}."
      )


def _combine_reasoning(
  confirmation_result: ScoreResult,
  challenge_result: ScoreResult,
) -> str:
  sections: list[str] = []
  if confirmation_result.reasoning.strip():
    sections.append(f"[Confirmation]\n{confirmation_result.reasoning.strip()}")
  if challenge_result.reasoning.strip():
    sections.append(f"[Challenge]\n{challenge_result.reasoning.strip()}")
  return "\n\n".join(sections)


def _combine_pairwise_reasoning(score_a: ScoreResult, score_b: ScoreResult) -> str:
  sections: list[str] = []
  if score_a.reasoning.strip():
    sections.append(f"[Trajectory A]\n{score_a.reasoning.strip()}")
  if score_b.reasoning.strip():
    sections.append(f"[Trajectory B]\n{score_b.reasoning.strip()}")
  return "\n\n".join(sections)


def _aggregate_pass_results(
  trajectory_id: str,
  criterion_id: str,
  results: list[ScoreResult],
) -> ScoreResult:
  if not results:
    raise ValueError("Cannot aggregate empty adversarial verification pass results.")

  score = sum(result.score for result in results) / len(results)
  confidence = sum(result.confidence for result in results) / len(results)
  reasoning = "\n\n".join(
    f"[Pass {index}]\n{result.reasoning.strip()}"
    for index, result in enumerate(results, start=1)
    if result.reasoning.strip()
  )

  return ScoreResult(
    trajectory_id=trajectory_id,
    score=score,
    confidence=confidence,
    criterion_scores={criterion_id: score},
    reasoning=reasoning,
    metadata={
      "num_verifications": len(results),
    },
  )


def _validate_probability(name: str, value: float) -> None:
  if value < 0.0 or value > 1.0:
    raise ValueError(f"{name} must be between 0.0 and 1.0, got {value}.")
