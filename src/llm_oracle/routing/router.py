"""Signal-based routing layer that selects between the Verifier and Judge strategies.

Policies are deterministic and O(1); each casts a soft vote aggregated by a
PolicyChain. Every decision is fully auditable via DetailedRoutingDecision.
"""

from __future__ import annotations

import abc
import dataclasses
import re
import time
from dataclasses import dataclass, field

from llm_oracle.core.models import (
  EvaluationResult,
  HumanOracle,
  HumanRequest,
  HumanResponse,
  HumanResponsePending,
  RoutingDecision,
  StrategyType,
  Task,
  TaskDifficulty,
  TrajectoryList,
)
from llm_oracle.core.strategy import BaseStrategy

_CONFIDENCE_THRESHOLD: float = 0.60
_UNCERTAINTY_THRESHOLD: float = 0.10

_VERIFIABLE_KEYWORDS: frozenset[str] = frozenset(
  {
    "implement",
    "code",
    "function",
    "algorithm",
    "bug",
    "fix",
    "test",
    "assert",
    "output",
    "return",
    "compile",
    "run",
    "execute",
    "script",
    "program",
    "class",
    "method",
    "sql",
    "query",
    "api",
    "endpoint",
    "parse",
    "format",
    "regex",
    "sort",
    "search",
  }
)

_JUDGMENT_KEYWORDS: frozenset[str] = frozenset(
  {
    "explain",
    "summarize",
    "describe",
    "discuss",
    "argue",
    "essay",
    "compare",
    "analyze",
    "review",
    "critique",
    "evaluate",
    "opinion",
    "recommend",
    "suggest",
    "creative",
    "write",
    "story",
    "poem",
    "design",
    "plan",
    "strategy",
    "brainstorm",
  }
)


@dataclass(frozen=True)
class RoutingSignals:
  """Named feature signals extracted from a task for routing decisions.

  All fields are floats in ``[0, 1]`` except ``trajectory_count`` (raw int)
  and ``prior_hardness`` (``None`` when unavailable). When ``prior_hardness``
  is set, ``PriorHardnessPolicy`` (w=1.8) typically dominates.
  """

  has_ground_truth: float = 0.0
  has_test_cases: float = 0.0
  trajectory_count: int = 1
  stated_difficulty: float = 0.5
  verifiable_keyword_density: float = 0.0
  judgment_keyword_density: float = 0.0
  problem_length: float = 0.0
  output_available: float = 0.0
  prior_hardness: float | None = None


@dataclass(frozen=True)
class PolicyVote:
  """A single policy's routing vote.

  PolicyChain aggregates as ``sum(confidence * weight)`` per strategy.
  All votes contribute to the weighted totals; the threshold gates whether the
  winning strategy is accepted or falls back to the default.
  """

  policy_name: str
  preferred: StrategyType
  confidence: float
  weight: float = 1.0
  signals_used: list[str] = field(default_factory=list)
  reasoning: str = ""


class RoutingPolicy(abc.ABC):
  """Abstract base for all routing policies.

  A policy inspects a :class:`~llm_oracle.core.models.Task` (and optionally
  its trajectories) and casts a soft vote for a strategy.
  """

  name: str = "unnamed_policy"
  weight: float = 1.0

  @abc.abstractmethod
  def vote(
    self,
    task: Task,
    trajectories: TrajectoryList,
    signals: RoutingSignals,
  ) -> PolicyVote:
    """Cast a routing vote for the preferred strategy."""

  def __repr__(self) -> str:
    return f"{self.__class__.__name__}(name={self.name!r}, weight={self.weight})"


class GroundTruthPolicy(RoutingPolicy):
  """Prefer the verifier when ground truth or test cases are present.

  Rationale: the verifier's log-probability scoring is most informative when
  it can compare trajectories against a reference answer; without one, the
  judge's rubric is often more reliable.
  """

  name = "ground_truth"
  weight = 2.0

  def vote(
    self,
    task: Task,
    trajectories: TrajectoryList,
    signals: RoutingSignals,
  ) -> PolicyVote:
    verifier_score = signals.has_ground_truth * 0.5 + signals.has_test_cases * 0.5
    if verifier_score >= 0.5:
      return PolicyVote(
        policy_name=self.name,
        preferred=StrategyType.VERIFIER,
        confidence=0.6 + verifier_score * 0.4,
        weight=self.weight,
        signals_used=["has_ground_truth", "has_test_cases"],
        reasoning=(
          "Ground-truth reference or formal test cases are available; "
          "the verifier can leverage them for precise logprob scoring."
        ),
      )
    return PolicyVote(
      policy_name=self.name,
      preferred=StrategyType.JUDGE,
      confidence=0.55,
      weight=self.weight,
      signals_used=["has_ground_truth", "has_test_cases"],
      reasoning=(
        "No ground-truth or test cases present; the judge's rubric-based "
        "holistic reasoning is better suited to this open-ended evaluation."
      ),
    )


class KeywordDomainPolicy(RoutingPolicy):
  """Classify the task domain from problem-statement keywords.

  * High density of *verifiable* keywords (code, algorithm, …) → verifier.
  * High density of *judgment* keywords (explain, analyze, …) → judge.
  * Mixed / ambiguous → low-confidence judge (default).
  """

  name = "keyword_domain"
  weight = 1.5

  def vote(
    self,
    task: Task,
    trajectories: TrajectoryList,
    signals: RoutingSignals,
  ) -> PolicyVote:
    verifiable_density = signals.verifiable_keyword_density
    judgment_density = signals.judgment_keyword_density
    gap = verifiable_density - judgment_density

    if gap > 0.05:
      confidence = min(0.95, 0.6 + gap * 2.0)
      return PolicyVote(
        policy_name=self.name,
        preferred=StrategyType.VERIFIER,
        confidence=confidence,
        weight=self.weight,
        signals_used=["verifiable_keyword_density", "judgment_keyword_density"],
        reasoning=(
          f"Problem statement has stronger verifiable-domain signal "
          f"(Δ={gap:.2f}); routing to verifier."
        ),
      )

    if gap < -0.05:
      confidence = min(0.95, 0.6 + abs(gap) * 2.0)
      return PolicyVote(
        policy_name=self.name,
        preferred=StrategyType.JUDGE,
        confidence=confidence,
        weight=self.weight,
        signals_used=["verifiable_keyword_density", "judgment_keyword_density"],
        reasoning=(
          f"Problem statement has stronger open-ended-domain signal "
          f"(Δ={gap:.2f}); routing to judge."
        ),
      )

    return PolicyVote(
      policy_name=self.name,
      preferred=StrategyType.JUDGE,
      confidence=0.52,
      weight=self.weight,
      signals_used=["verifiable_keyword_density", "judgment_keyword_density"],
      reasoning="Keyword signals are ambiguous; defaulting to judge.",
    )


class DifficultyPolicy(RoutingPolicy):
  """Route based on stated task difficulty.

  * EASY tasks: judge is fast and sufficiently accurate.
  * HARD tasks: verifier's multi-criterion repeated verification adds value.
  * MEDIUM / UNKNOWN: slight verifier preference for safety.
  """

  name = "difficulty"
  weight = 1.0

  def vote(
    self,
    task: Task,
    trajectories: TrajectoryList,
    signals: RoutingSignals,
  ) -> PolicyVote:
    difficulty = task.difficulty

    if difficulty == TaskDifficulty.EASY:
      return PolicyVote(
        policy_name=self.name,
        preferred=StrategyType.JUDGE,
        confidence=0.65,
        weight=self.weight,
        signals_used=["stated_difficulty"],
        reasoning="Task is marked EASY; the judge's lower overhead is preferred.",
      )

    if difficulty == TaskDifficulty.HARD:
      return PolicyVote(
        policy_name=self.name,
        preferred=StrategyType.VERIFIER,
        confidence=0.72,
        weight=self.weight,
        signals_used=["stated_difficulty"],
        reasoning=(
          "Task is marked HARD; the verifier's granular multi-pass scoring "
          "is more robust for difficult evaluations."
        ),
      )

    return PolicyVote(
      policy_name=self.name,
      preferred=StrategyType.VERIFIER,
      confidence=0.55,
      weight=self.weight,
      signals_used=["stated_difficulty"],
      reasoning=(
        f"Task difficulty is {difficulty.value}; applying slight verifier "
        "preference as the more thorough option."
      ),
    )


class TrajectoryCountPolicy(RoutingPolicy):
  """Adjust routing based on the number of candidate trajectories.

  * 1 trajectory : judge is sufficient (no tournament needed).
  * 2–4           : both work; slight verifier preference for its pairwise
                    tournament selection.
  * 5+            : verifier's O(n²) tournament becomes expensive; judge is
                    preferred unless ground truth is present.
  """

  name = "trajectory_count"
  weight = 0.8

  def vote(
    self,
    task: Task,
    trajectories: TrajectoryList,
    signals: RoutingSignals,
  ) -> PolicyVote:
    trajectory_count = signals.trajectory_count

    if trajectory_count == 1:
      return PolicyVote(
        policy_name=self.name,
        preferred=StrategyType.JUDGE,
        confidence=0.60,
        weight=self.weight,
        signals_used=["trajectory_count"],
        reasoning=(
          "Only one trajectory to evaluate; the verifier's pairwise "
          "tournament adds no value — judge is more efficient."
        ),
      )

    if 2 <= trajectory_count <= 4:
      return PolicyVote(
        policy_name=self.name,
        preferred=StrategyType.VERIFIER,
        confidence=0.58,
        weight=self.weight,
        signals_used=["trajectory_count"],
        reasoning=(
          f"{trajectory_count} trajectories available; verifier's pairwise tournament "
          "can effectively discriminate between them."
        ),
      )

    return PolicyVote(
      policy_name=self.name,
      preferred=StrategyType.JUDGE,
      confidence=0.62,
      weight=self.weight,
      signals_used=["trajectory_count"],
      reasoning=(
        f"{trajectory_count} trajectories would require "
        f"{trajectory_count * (trajectory_count - 1) // 2} pairwise calls "
        "in the verifier; routing to judge for efficiency."
      ),
    )


class PriorHardnessPolicy(RoutingPolicy):
  """Use a cached hardness score from the evaluation harness.

  When a prior hardness score is available (e.g., from an earlier harness
  run on similar tasks), use it to route:

  * Low hardness  (< 0.35): judge is sufficient.
  * High hardness (≥ 0.60): verifier's finer granularity is beneficial.
  * Mid hardness           : slight verifier preference.
  """

  name = "prior_hardness"
  weight = 1.8

  def vote(
    self,
    task: Task,
    trajectories: TrajectoryList,
    signals: RoutingSignals,
  ) -> PolicyVote:
    hardness = signals.prior_hardness

    if hardness is None:
      return PolicyVote(
        policy_name=self.name,
        preferred=StrategyType.JUDGE,
        confidence=0.51,
        weight=self.weight,
        signals_used=["prior_hardness"],
        reasoning="No prior hardness data available; abstaining (weak judge default).",
      )

    if hardness < 0.35:
      confidence = 0.65 + (0.35 - hardness) * 0.5
      return PolicyVote(
        policy_name=self.name,
        preferred=StrategyType.JUDGE,
        confidence=min(0.92, confidence),
        weight=self.weight,
        signals_used=["prior_hardness"],
        reasoning=(
          f"Prior hardness {hardness:.3f} is low; the judge is sufficiently accurate and faster."
        ),
      )

    if hardness >= 0.60:
      confidence = 0.65 + (hardness - 0.60) * 0.5
      return PolicyVote(
        policy_name=self.name,
        preferred=StrategyType.VERIFIER,
        confidence=min(0.95, confidence),
        weight=self.weight,
        signals_used=["prior_hardness"],
        reasoning=(
          f"Prior hardness {hardness:.3f} is high; the verifier's repeated "
          "multi-criterion verification is more reliable on hard tasks."
        ),
      )

    return PolicyVote(
      policy_name=self.name,
      preferred=StrategyType.VERIFIER,
      confidence=0.56,
      weight=self.weight,
      signals_used=["prior_hardness"],
      reasoning=(f"Prior hardness {hardness:.3f} is mid-range; slight verifier preference."),
    )


class OutputAvailabilityPolicy(RoutingPolicy):
  """Prefer the verifier when execution outputs are available.

  Concrete execution outputs (stdout, test results, stack traces) give the
  verifier concrete evidence to score.  Without outputs, the judge's reasoning
  over trajectory *intent* is equally or more informative.
  """

  name = "output_availability"
  weight = 0.9

  def vote(
    self,
    task: Task,
    trajectories: TrajectoryList,
    signals: RoutingSignals,
  ) -> PolicyVote:
    if signals.output_available >= 0.5:
      return PolicyVote(
        policy_name=self.name,
        preferred=StrategyType.VERIFIER,
        confidence=0.65,
        weight=self.weight,
        signals_used=["output_available"],
        reasoning=(
          "Execution outputs are present; the verifier can leverage them "
          "as concrete evidence during logprob scoring."
        ),
      )
    return PolicyVote(
      policy_name=self.name,
      preferred=StrategyType.JUDGE,
      confidence=0.57,
      weight=self.weight,
      signals_used=["output_available"],
      reasoning=(
        "No execution outputs available; judge's reasoning over trajectory intent is preferred."
      ),
    )


class PolicyChain:
  """Aggregates votes from multiple RoutingPolicy instances.

  Computes weighted confidence per strategy and picks the winner. Falls back
  to ``fallback_strategy`` when no policy reaches ``confidence_threshold``.
  """

  def __init__(
    self,
    policies: list[RoutingPolicy],
    *,
    confidence_threshold: float = _CONFIDENCE_THRESHOLD,
    fallback_strategy: StrategyType = StrategyType.JUDGE,
  ) -> None:
    if not policies:
      raise ValueError("PolicyChain requires at least one policy.")
    self._policies = list(policies)
    self._threshold = confidence_threshold
    self._fallback = fallback_strategy

  def decide(
    self,
    task: Task,
    trajectories: TrajectoryList,
    signals: RoutingSignals,
  ) -> tuple[StrategyType, float, list[PolicyVote], str]:
    """Run all policies and return ``(strategy, confidence, votes, reasoning)``."""
    votes: list[PolicyVote] = [
      policy.vote(task, trajectories, signals) for policy in self._policies
    ]

    verifier_score, judge_score = self._aggregate(votes)
    total = verifier_score + judge_score

    if total < 1e-9:
      return self._fallback, 0.5, votes, "All policies abstained; using fallback."

    if verifier_score >= judge_score:
      winner = StrategyType.VERIFIER
      confidence = verifier_score / total
    else:
      winner = StrategyType.JUDGE
      confidence = judge_score / total

    if confidence < self._threshold:
      reasoning = (
        f"Max confidence {confidence:.3f} is below threshold {self._threshold}; "
        f"falling back to {self._fallback.value}."
      )
      return self._fallback, confidence, votes, reasoning

    reasoning = self._build_reasoning(winner, votes, confidence)
    return winner, confidence, votes, reasoning

  @property
  def policies(self) -> list[RoutingPolicy]:
    """Read-only view of registered policies."""
    return list(self._policies)

  @property
  def threshold(self) -> float:
    """Minimum aggregate confidence required to commit to a strategy."""
    return self._threshold

  @property
  def fallback(self) -> StrategyType:
    """Strategy returned when confidence is below the threshold."""
    return self._fallback

  @staticmethod
  def _aggregate(votes: list[PolicyVote]) -> tuple[float, float]:
    """Return ``(verifier_total, judge_total)`` weighted confidence scores."""
    verifier_total = 0.0
    judge_total = 0.0
    for vote in votes:
      weighted = vote.confidence * vote.weight
      if vote.preferred == StrategyType.VERIFIER:
        verifier_total += weighted
      else:
        judge_total += weighted
    return verifier_total, judge_total

  @staticmethod
  def _build_reasoning(
    winner: StrategyType,
    votes: list[PolicyVote],
    confidence: float,
  ) -> str:
    lines = [
      f"Selected {winner.value} with confidence {confidence:.3f}.",
      "",
      "Policy votes:",
    ]
    for vote in votes:
      marker = "→" if vote.preferred == winner else "←"
      lines.append(
        f"  {marker} [{vote.policy_name}] "
        f"{vote.preferred.value} @ {vote.confidence:.2f} (w={vote.weight}): "
        f"{vote.reasoning}"
      )
    return "\n".join(lines)


class SignalExtractor:
  """Extracts :class:`RoutingSignals` from a task and its trajectories.

  This is the only place that touches raw task/trajectory fields, keeping all
  policy logic clean and signal-independent.
  """

  def extract(
    self,
    task: Task,
    trajectories: TrajectoryList,
    *,
    prior_hardness: float | None = None,
  ) -> RoutingSignals:
    """Extract routing signals from a task and its trajectories."""
    text = (task.problem_statement + " " + task.description).lower()
    words = set(re.findall(r"\b\w+\b", text))

    verifiable_matches = len(words & _VERIFIABLE_KEYWORDS)
    judgment_matches = len(words & _JUDGMENT_KEYWORDS)
    total_checked = len(_VERIFIABLE_KEYWORDS | _JUDGMENT_KEYWORDS)

    verifiable_density = verifiable_matches / total_checked if total_checked else 0.0
    judgment_density = judgment_matches / total_checked if total_checked else 0.0

    has_output = float(any(t.output for t in trajectories))

    stated_difficulty = {
      TaskDifficulty.EASY: 0.0,
      TaskDifficulty.MEDIUM: 0.5,
      TaskDifficulty.HARD: 1.0,
      TaskDifficulty.UNKNOWN: 0.5,
    }.get(task.difficulty, 0.5)

    return RoutingSignals(
      has_ground_truth=float(task.ground_truth is not None),
      has_test_cases=float(bool(task.test_cases)),
      trajectory_count=len(trajectories),
      stated_difficulty=stated_difficulty,
      verifiable_keyword_density=verifiable_density,
      judgment_keyword_density=judgment_density,
      problem_length=min(len(task.problem_statement), 2000) / 2000.0,
      output_available=has_output,
      prior_hardness=prior_hardness,
    )


@dataclass
class DetailedRoutingDecision(RoutingDecision):
  """RoutingDecision extended with per-policy votes, signals, and latency."""

  policy_votes: list[PolicyVote] = field(default_factory=list)
  signals: RoutingSignals = field(default_factory=RoutingSignals)
  elapsed_ms: float = 0.0


class OracleRouter:
  """Routes evaluation tasks to the verifier or judge strategy via a PolicyChain."""

  def __init__(
    self,
    verifier: BaseStrategy,
    judge: BaseStrategy,
    policy_chain: PolicyChain,
    signal_extractor: SignalExtractor | None = None,
    hardness_cache: dict[str, float] | None = None,
    human_oracle: HumanOracle | None = None,
    uncertainty_threshold: float = _UNCERTAINTY_THRESHOLD,
  ) -> None:
    self._verifier = verifier
    self._judge = judge
    self._chain = policy_chain
    self._extractor = signal_extractor or SignalExtractor()
    self._hardness_cache: dict[str, float] = hardness_cache or {}
    self._human_oracle = human_oracle
    self._uncertainty_threshold = uncertainty_threshold
    self._decision_log: list[DetailedRoutingDecision] = []

  @classmethod
  def default(
    cls,
    verifier: BaseStrategy,
    judge: BaseStrategy,
    *,
    hardness_cache: dict[str, float] | None = None,
    confidence_threshold: float = _CONFIDENCE_THRESHOLD,
    human_oracle: HumanOracle | None = None,
    uncertainty_threshold: float = _UNCERTAINTY_THRESHOLD,
  ) -> OracleRouter:
    """Create a router with the default six-policy chain.

    Policy weights: prior_hardness(1.8), ground_truth(2.0),
    keyword_domain(1.5), difficulty(1.0), output_availability(0.9),
    trajectory_count(0.8).
    """
    policies: list[RoutingPolicy] = [
      PriorHardnessPolicy(),
      GroundTruthPolicy(),
      KeywordDomainPolicy(),
      DifficultyPolicy(),
      OutputAvailabilityPolicy(),
      TrajectoryCountPolicy(),
    ]
    chain = PolicyChain(
      policies,
      confidence_threshold=confidence_threshold,
      fallback_strategy=StrategyType.JUDGE,
    )
    return cls(
      verifier,
      judge,
      chain,
      hardness_cache=hardness_cache,
      human_oracle=human_oracle,
      uncertainty_threshold=uncertainty_threshold,
    )

  @classmethod
  def verifier_only(
    cls,
    verifier: BaseStrategy,
    judge: BaseStrategy,
  ) -> OracleRouter:
    """Create a router that always selects the verifier (ablation mode)."""

    class _AlwaysVerifierPolicy(RoutingPolicy):
      name = "always_verifier"

      def vote(self, task, trajectories, signals):  # type: ignore[override]
        return PolicyVote(
          policy_name=self.name,
          preferred=StrategyType.VERIFIER,
          confidence=1.0,
          reasoning="Forced verifier routing (ablation mode).",
        )

    chain = PolicyChain([_AlwaysVerifierPolicy()], confidence_threshold=0.0)
    return cls(verifier, judge, chain)

  @classmethod
  def judge_only(
    cls,
    verifier: BaseStrategy,
    judge: BaseStrategy,
  ) -> OracleRouter:
    """Create a router that always selects the judge (ablation mode)."""

    class _AlwaysJudgePolicy(RoutingPolicy):
      name = "always_judge"

      def vote(self, task, trajectories, signals):  # type: ignore[override]
        return PolicyVote(
          policy_name=self.name,
          preferred=StrategyType.JUDGE,
          confidence=1.0,
          reasoning="Forced judge routing (ablation mode).",
        )

    chain = PolicyChain([_AlwaysJudgePolicy()], confidence_threshold=0.0)
    return cls(verifier, judge, chain)

  def route(
    self,
    task: Task,
    trajectories: TrajectoryList,
  ) -> DetailedRoutingDecision:
    """Compute a routing decision without running evaluation."""
    start_time = time.perf_counter()
    prior_hardness = self._hardness_cache.get(task.id)
    signals = self._extractor.extract(task, trajectories, prior_hardness=prior_hardness)
    strategy, confidence, votes, reasoning = self._chain.decide(task, trajectories, signals)
    elapsed_ms = (time.perf_counter() - start_time) * 1_000
    decision = DetailedRoutingDecision(
      task_id=task.id,
      selected_strategy=strategy,
      confidence=confidence,
      reasoning=reasoning,
      policy_votes=votes,
      signals=signals,
      elapsed_ms=elapsed_ms,
    )
    self._decision_log.append(decision)
    return decision

  def evaluate(
    self,
    task: Task,
    trajectories: TrajectoryList,
  ) -> tuple[EvaluationResult, DetailedRoutingDecision]:
    """Route and evaluate a task; returns ``(EvaluationResult, DetailedRoutingDecision)``.

    If a ``human_oracle`` is configured and the score spread across trajectories
    falls below ``uncertainty_threshold``, the oracle asks the human for
    clarification and re-evaluates with the clarified task.

    Raises:
      ValueError: If ``trajectories`` is empty.
      ValueError: If a human answer does not match any clarification key.
      RuntimeError: If the human oracle returns a pending (deferred) response.
    """
    if not trajectories:
      raise ValueError(f"Trajectory list for task '{task.id}' must be non-empty.")

    decision = self.route(task, trajectories)
    strategy = self._resolve_strategy(decision.selected_strategy)
    result = strategy.evaluate(task, trajectories)

    if self._human_oracle is not None and self._is_uncertain(result):
      human_request = self._build_human_request(task, result)
      response = self._human_oracle.ask(human_request)

      if isinstance(response, HumanResponsePending):
        raise RuntimeError(
          f"Deferred human responses are not yet supported. "
          f"Request {response.request_id!r} returned HumanResponsePending."
        )

      clarified_task = self._apply_clarification(task, response)
      decision = self.route(clarified_task, trajectories)
      strategy = self._resolve_strategy(decision.selected_strategy)
      result = strategy.evaluate(clarified_task, trajectories)
      decision.metadata.update(
        {
          "human_escalated": True,
          "human_request_id": human_request.id,
          "human_response": response.answer,
          "human_responder_id": response.responder_id,
        }
      )

    return result, decision

  def _is_uncertain(self, result: EvaluationResult) -> bool:
    """Return True when score spread is below the uncertainty threshold."""
    scores = [s.score for s in result.trajectory_scores.values()]
    if len(scores) < 2:
      return False
    return (max(scores) - min(scores)) < self._uncertainty_threshold

  def _build_human_request(self, task: Task, result: EvaluationResult) -> HumanRequest:
    """Build a HumanRequest from an inconclusive evaluation result."""
    sorted_scores = sorted(result.trajectory_scores.values(), key=lambda s: s.score, reverse=True)
    spread = sorted_scores[0].score - sorted_scores[-1].score
    reasoning_lines = [s.reasoning for s in sorted_scores[:2] if s.reasoning]
    context = " ".join(reasoning_lines)
    question = (
      "The evaluation was inconclusive. Can you clarify the task requirements "
      "or expected outcome?" + (f" Context: {context}" if context else "")
    )
    reason = (
      f"Score spread {spread:.3f} below threshold {self._uncertainty_threshold}. "
      f"Oracle cannot distinguish which trajectory best satisfies '{task.description}'."
    )
    return HumanRequest(
      id=f"uncertainty-{task.id}",
      task_id=task.id,
      question=question,
      reason=reason,
    )

  def _apply_clarification(self, task: Task, response: HumanResponse) -> Task:
    """Return a new Task with the human's clarification applied.

    If ``task.metadata`` contains a ``human_clarifications`` map, the response
    answer is used as a lookup key and the matching overrides are applied as
    field updates. Otherwise, the free-form answer is appended to
    ``problem_statement``.

    Raises:
      ValueError: If a clarifications map exists but the answer key is absent.
    """
    _ALLOWED_FIELDS = {
      "description",
      "problem_statement",
      "ground_truth",
      "test_cases",
      "difficulty",
    }
    clarifications: dict = task.metadata.get("human_clarifications", {})

    if clarifications:
      answer_key = response.answer.strip().lower()
      if answer_key not in clarifications:
        raise ValueError(
          f"Human answer {answer_key!r} does not match any key in "
          f"'human_clarifications'. Available: {sorted(clarifications)}"
        )
      overrides: dict = dict(clarifications[answer_key])
      task_updates = {k: v for k, v in overrides.items() if k in _ALLOWED_FIELDS}
      metadata = {**task.metadata, **(overrides.get("metadata") or {})}
      return dataclasses.replace(task, **task_updates, metadata=metadata)

    return dataclasses.replace(
      task,
      problem_statement=f"{task.problem_statement}\n\nHuman clarification: {response.answer}",
    )

  def register_policy(
    self,
    policy: RoutingPolicy,
    *,
    position: int | None = None,
  ) -> OracleRouter:
    """Add a policy to the chain; returns ``self`` for fluent chaining."""
    policies = list(self._chain.policies)
    if position is None:
      policies.append(policy)
    else:
      policies.insert(position, policy)

    self._chain = PolicyChain(
      policies,
      confidence_threshold=self._chain.threshold,
      fallback_strategy=self._chain.fallback,
    )
    return self

  def update_hardness(self, task_id: str, hardness: float) -> None:
    """Update the hardness cache for a task (called automatically by the harness)."""
    if not 0.0 <= hardness <= 1.0:
      raise ValueError(f"Hardness must be in [0, 1], got {hardness}")
    self._hardness_cache[task_id] = hardness

  @property
  def decision_log(self) -> list[DetailedRoutingDecision]:
    """Read-only list of all routing decisions made so far."""
    return list(self._decision_log)

  def routing_summary(self) -> str:
    """Render a compact audit table of all past routing decisions."""
    if not self._decision_log:
      return "No routing decisions recorded yet."

    verifier_count = sum(
      1 for d in self._decision_log if d.selected_strategy == StrategyType.VERIFIER
    )
    judge_count = len(self._decision_log) - verifier_count
    average_confidence = sum(d.confidence for d in self._decision_log) / len(self._decision_log)
    average_ms = sum(d.elapsed_ms for d in self._decision_log) / len(self._decision_log)

    lines = [
      "",
      "═" * 56,
      "  OracleRouter — Routing Audit",
      "═" * 56,
      f"  Total decisions  : {len(self._decision_log)}",
      f"  → Verifier       : {verifier_count}",
      f"  → Judge          : {judge_count}",
      f"  Avg confidence   : {average_confidence:.3f}",
      f"  Avg latency      : {average_ms:.2f} ms",
      "",
      f"  {'Task':<30s}  {'Strategy':>9s}  {'Conf':>6s}",
      "  " + "-" * 50,
    ]
    for decision in self._decision_log:
      lines.append(
        f"  {decision.task_id:<30s}  {decision.selected_strategy.value:>9s}  {decision.confidence:>6.3f}"
      )
    lines += ["═" * 56, ""]
    return "\n".join(lines)

  def _resolve_strategy(self, strategy_type: StrategyType) -> BaseStrategy:
    """Map a StrategyType to the corresponding strategy instance."""
    if strategy_type == StrategyType.VERIFIER:
      return self._verifier
    return self._judge
