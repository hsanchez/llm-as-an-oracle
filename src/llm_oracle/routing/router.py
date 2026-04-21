"""Signal-based routing layer that selects between the Verifier and Judge strategies.

Policies are deterministic and O(1); each casts a soft vote aggregated by a
PolicyChain. Every decision is fully auditable via DetailedRoutingDecision.
"""

from __future__ import annotations

import abc
import re
import time
from dataclasses import dataclass, field

from llm_oracle.core.models import (
  EvaluationResult,
  RoutingDecision,
  StrategyType,
  Task,
  TaskDifficulty,
  TrajectoryList,
)
from llm_oracle.core.strategy import BaseStrategy

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

# Minimum confidence required to commit to a strategy without fallback.
_CONFIDENCE_THRESHOLD: float = 0.60

# Keywords that strongly suggest a verifiable / executable task.
_VERIFIABLE_KEYWORDS: frozenset[str] = frozenset(
  {
    "implement",
    "code",
    "function",
    "algorithm",
    "bug",
    "fix",
    "test",
    "unit test",
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

# Keywords that suggest an open-ended / creative task better suited to a judge.
_JUDGEMENT_KEYWORDS: frozenset[str] = frozenset(
  {
    "explain",
    "summarise",
    "summarize",
    "describe",
    "discuss",
    "argue",
    "essay",
    "compare",
    "analyse",
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


# ──────────────────────────────────────────────────────────────────────────────
# Signal types
# ──────────────────────────────────────────────────────────────────────────────


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
  judgement_keyword_density: float = 0.0
  problem_length: float = 0.0
  output_available: float = 0.0
  prior_hardness: float | None = None


@dataclass(frozen=True)
class PolicyVote:
  """A single policy's routing vote.

  PolicyChain aggregates as ``sum(confidence * weight)`` per strategy.
  Votes below ``confidence_threshold`` are treated as abstentions.
  """

  policy_name: str
  preferred: StrategyType
  confidence: float
  weight: float = 1.0
  signals_used: list[str] = field(default_factory=list)
  reasoning: str = ""


# ──────────────────────────────────────────────────────────────────────────────
# Base policy
# ──────────────────────────────────────────────────────────────────────────────


class RoutingPolicy(abc.ABC):
  """Abstract base for all routing policies.

  A policy inspects a :class:`~llm_oracle.core.models.Task` (and optionally
  its trajectories) and casts a soft vote for a strategy.
  """

  #: Human-readable name used in audit logs and ``PolicyVote``.
  name: str = "unnamed_policy"

  #: Relative weight when combined in a :class:`PolicyChain`.
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


# ──────────────────────────────────────────────────────────────────────────────
# Built-in routing policies
# ──────────────────────────────────────────────────────────────────────────────


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
  * High density of *judgement* keywords (explain, analyse, …) → judge.
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
    v_density = signals.verifiable_keyword_density
    j_density = signals.judgement_keyword_density
    gap = v_density - j_density

    if gap > 0.05:
      confidence = min(0.95, 0.6 + gap * 2.0)
      return PolicyVote(
        policy_name=self.name,
        preferred=StrategyType.VERIFIER,
        confidence=confidence,
        weight=self.weight,
        signals_used=["verifiable_keyword_density", "judgement_keyword_density"],
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
        signals_used=["verifiable_keyword_density", "judgement_keyword_density"],
        reasoning=(
          f"Problem statement has stronger open-ended-domain signal "
          f"(Δ={gap:.2f}); routing to judge."
        ),
      )

    # Ambiguous — weak judge preference as the safer default
    return PolicyVote(
      policy_name=self.name,
      preferred=StrategyType.JUDGE,
      confidence=0.52,
      weight=self.weight,
      signals_used=["verifiable_keyword_density", "judgement_keyword_density"],
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

    # MEDIUM or UNKNOWN
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
    n = signals.trajectory_count

    if n == 1:
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

    if 2 <= n <= 4:
      return PolicyVote(
        policy_name=self.name,
        preferred=StrategyType.VERIFIER,
        confidence=0.58,
        weight=self.weight,
        signals_used=["trajectory_count"],
        reasoning=(
          f"{n} trajectories available; verifier's pairwise tournament "
          "can effectively discriminate between them."
        ),
      )

    # n >= 5: tournament becomes quadratic; prefer judge
    return PolicyVote(
      policy_name=self.name,
      preferred=StrategyType.JUDGE,
      confidence=0.62,
      weight=self.weight,
      signals_used=["trajectory_count"],
      reasoning=(
        f"{n} trajectories would require {n * (n - 1) // 2} pairwise calls "
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
  weight = 1.8  # Second-highest weight after GroundTruthPolicy (2.0).

  def vote(
    self,
    task: Task,
    trajectories: TrajectoryList,
    signals: RoutingSignals,
  ) -> PolicyVote:
    hardness = signals.prior_hardness

    if hardness is None:
      # No prior data — abstain with a very weak judge preference.
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

    # Mid-range hardness
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


# ──────────────────────────────────────────────────────────────────────────────
# Policy chain — aggregate multiple policy votes
# ──────────────────────────────────────────────────────────────────────────────


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

  # ── Public API ───────────────────────────────────────────────────────────────

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

    # Normalised confidence for the winning side.
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

  # ── Private helpers ──────────────────────────────────────────────────────────

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
    for v in votes:
      marker = "→" if v.preferred == winner else "←"
      lines.append(
        f"  {marker} [{v.policy_name}] "
        f"{v.preferred.value} @ {v.confidence:.2f} (w={v.weight}): "
        f"{v.reasoning}"
      )
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Signal extractor
# ──────────────────────────────────────────────────────────────────────────────


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

    v_matches = len(words & _VERIFIABLE_KEYWORDS)
    j_matches = len(words & _JUDGEMENT_KEYWORDS)
    total_checked = len(_VERIFIABLE_KEYWORDS | _JUDGEMENT_KEYWORDS)

    v_density = v_matches / total_checked if total_checked else 0.0
    j_density = j_matches / total_checked if total_checked else 0.0

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
      verifiable_keyword_density=v_density,
      judgement_keyword_density=j_density,
      problem_length=min(len(task.problem_statement), 2000) / 2000.0,
      output_available=has_output,
      prior_hardness=prior_hardness,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Extended routing decision (includes per-policy votes)
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class DetailedRoutingDecision(RoutingDecision):
  """RoutingDecision extended with per-policy votes, signals, and latency."""

  policy_votes: list[PolicyVote] = field(default_factory=list)
  signals: RoutingSignals = field(default_factory=RoutingSignals)
  elapsed_ms: float = 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Main router
# ──────────────────────────────────────────────────────────────────────────────


class OracleRouter:
  """Routes evaluation tasks to the verifier or judge strategy via a PolicyChain."""

  def __init__(
    self,
    verifier: BaseStrategy,
    judge: BaseStrategy,
    policy_chain: PolicyChain,
    signal_extractor: SignalExtractor | None = None,
    hardness_cache: dict[str, float] | None = None,
  ) -> None:
    self._verifier = verifier
    self._judge = judge
    self._chain = policy_chain
    self._extractor = signal_extractor or SignalExtractor()
    self._hardness_cache: dict[str, float] = hardness_cache or {}
    self._decision_log: list[DetailedRoutingDecision] = []

  # ── Factory constructors ─────────────────────────────────────────────────────

  @classmethod
  def default(
    cls,
    verifier: BaseStrategy,
    judge: BaseStrategy,
    *,
    hardness_cache: dict[str, float] | None = None,
    confidence_threshold: float = _CONFIDENCE_THRESHOLD,
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
    return cls(verifier, judge, chain, hardness_cache=hardness_cache)

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

  # ── Core routing ─────────────────────────────────────────────────────────────

  def route(
    self,
    task: Task,
    trajectories: TrajectoryList,
  ) -> DetailedRoutingDecision:
    """Compute a routing decision without running evaluation."""
    t0 = time.perf_counter()

    prior_hardness = self._hardness_cache.get(task.id)
    signals = self._extractor.extract(task, trajectories, prior_hardness=prior_hardness)

    strategy, confidence, votes, reasoning = self._chain.decide(task, trajectories, signals)

    elapsed_ms = (time.perf_counter() - t0) * 1_000

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

    Raises:
      ValueError: If ``trajectories`` is empty.
    """
    if not trajectories:
      raise ValueError(f"Trajectory list for task '{task.id}' must be non-empty.")

    decision = self.route(task, trajectories)
    strategy = self._resolve_strategy(decision.selected_strategy)
    result = strategy.evaluate(task, trajectories)

    return result, decision

  # ── Policy management ────────────────────────────────────────────────────────

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

  # ── Audit & introspection ────────────────────────────────────────────────────

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
    avg_conf = sum(d.confidence for d in self._decision_log) / len(self._decision_log)
    avg_ms = sum(d.elapsed_ms for d in self._decision_log) / len(self._decision_log)

    lines = [
      "",
      "═" * 56,
      "  OracleRouter — Routing Audit",
      "═" * 56,
      f"  Total decisions  : {len(self._decision_log)}",
      f"  → Verifier       : {verifier_count}",
      f"  → Judge          : {judge_count}",
      f"  Avg confidence   : {avg_conf:.3f}",
      f"  Avg latency      : {avg_ms:.2f} ms",
      "",
      f"  {'Task':<30s}  {'Strategy':>9s}  {'Conf':>6s}",
      "  " + "-" * 50,
    ]
    for d in self._decision_log:
      lines.append(f"  {d.task_id:<30s}  {d.selected_strategy.value:>9s}  {d.confidence:>6.3f}")
    lines += ["═" * 56, ""]
    return "\n".join(lines)

  # ── Private helpers ──────────────────────────────────────────────────────────

  def _resolve_strategy(self, strategy_type: StrategyType) -> BaseStrategy:
    """Map a StrategyType to the corresponding strategy instance.

    Raises:
      ValueError: For unknown strategy types.
    """
    if strategy_type == StrategyType.VERIFIER:
      return self._verifier
    if strategy_type == StrategyType.JUDGE:
      return self._judge
    raise ValueError(f"Unknown strategy type: {strategy_type!r}")
