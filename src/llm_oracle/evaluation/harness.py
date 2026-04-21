"""Side-by-side comparison harness for Verifier vs Judge, with composite hardness metrics.

Hardness is measured across four dimensions: score spread, strategy disagreement,
confidence gap, and oracle gap (selection error).
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

from llm_oracle.core.models import (
  EvaluationResult,
  ScoreResult,
  StrategyType,
  Task,
  TrajectoryList,
)
from llm_oracle.core.strategy import BaseStrategy

# ──────────────────────────────────────────────────────────────────────────────
# Per-task hardness record
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class TaskHardnessRecord:
  """Hardness measurements and evaluation results for a single task."""

  task_id: str
  hardness_score: float = 0.0  # [0, 1] composite; 1 = both strategies struggled badly, 0 = both agreed with high confidence
  score_spread: float = 0.0  # mean absolute difference (not variance) between Verifier and Judge scores per trajectory; high = they disagree on how good each candidate is
  strategy_disagreement: float = 0.0  # fraction of (A, B) pairs where Verifier prefers A but Judge prefers B; high = strategies disagree on relative ranking
  avg_confidence: float = 1.0  # mean confidence across all scores from both strategies; low = evaluators were uncertain about their own verdicts
  oracle_gap_verifier: float = 0.0  # reward forfeited by following the Verifier: best_possible_reward − reward_of_verifier_pick; 0 = perfect pick
  oracle_gap_judge: float = 0.0  # same as oracle_gap_verifier but for the Judge; compare both to see which strategy made the costlier mistake
  verifier_result: EvaluationResult | None = None
  judge_result: EvaluationResult | None = None
  elapsed_verifier_s: float = 0.0
  elapsed_judge_s: float = 0.0

  # Convenience ---------------------------------------------------------------

  @property
  def verifier_wins(self) -> bool:
    """True when the verifier has a smaller oracle gap."""
    return self.oracle_gap_verifier < self.oracle_gap_judge

  @property
  def judge_wins(self) -> bool:
    """True when the judge has a smaller oracle gap."""
    return self.oracle_gap_judge < self.oracle_gap_verifier

  @property
  def strategies_agree(self) -> bool:
    """True when both strategies selected the same best trajectory."""
    if self.verifier_result is None or self.judge_result is None:
      return False
    return self.verifier_result.best_trajectory_id == self.judge_result.best_trajectory_id


# ──────────────────────────────────────────────────────────────────────────────
# Aggregate report
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class HarnessReport:
  """Aggregated comparison report for all evaluated tasks."""

  task_records: list[TaskHardnessRecord] = field(default_factory=list)
  verifier_accuracy: float = 0.0
  judge_accuracy: float = 0.0
  avg_hardness: float = 0.0
  hard_task_threshold: float = 0.6
  total_elapsed_s: float = 0.0

  # ── Slicing helpers ─────────────────────────────────────────────────────────

  @property
  def hard_tasks(self) -> list[TaskHardnessRecord]:
    """Tasks whose hardness score exceeds ``hard_task_threshold``."""
    return [r for r in self.task_records if r.hardness_score >= self.hard_task_threshold]

  @property
  def easy_tasks(self) -> list[TaskHardnessRecord]:
    """Tasks whose hardness score is below ``hard_task_threshold``."""
    return [r for r in self.task_records if r.hardness_score < self.hard_task_threshold]

  @property
  def verifier_wins_count(self) -> int:
    """Number of tasks where the verifier outperformed the judge."""
    return sum(1 for r in self.task_records if r.verifier_wins)

  @property
  def judge_wins_count(self) -> int:
    """Number of tasks where the judge outperformed the verifier."""
    return sum(1 for r in self.task_records if r.judge_wins)

  @property
  def tie_count(self) -> int:
    """Number of tasks where both strategies performed equally."""
    return len(self.task_records) - self.verifier_wins_count - self.judge_wins_count

  def verifier_accuracy_on_hard(self) -> float:
    """Verifier accuracy restricted to hard tasks."""
    return _accuracy_on(self.hard_tasks, StrategyType.VERIFIER)

  def judge_accuracy_on_hard(self) -> float:
    """Judge accuracy restricted to hard tasks."""
    return _accuracy_on(self.hard_tasks, StrategyType.JUDGE)

  def verifier_accuracy_on_easy(self) -> float:
    """Verifier accuracy restricted to easy tasks."""
    return _accuracy_on(self.easy_tasks, StrategyType.VERIFIER)

  def judge_accuracy_on_easy(self) -> float:
    """Judge accuracy restricted to easy tasks."""
    return _accuracy_on(self.easy_tasks, StrategyType.JUDGE)

  # ── Reporting ───────────────────────────────────────────────────────────────

  def summary(self) -> str:
    """Render a human-readable comparison summary table."""
    n = len(self.task_records)
    n_hard = len(self.hard_tasks)
    n_easy = len(self.easy_tasks)

    bar = "═" * 64

    lines = [
      "",
      bar,
      "  LLM Oracle — Evaluation Harness Report",
      bar,
      f"  Tasks evaluated : {n}",
      f"  Hard tasks      : {n_hard}  (threshold ≥ {self.hard_task_threshold:.2f})",
      f"  Easy tasks      : {n_easy}",
      f"  Total runtime   : {self.total_elapsed_s:.1f} s",
      "",
      "  ┌──────────────────────────────┬────────────┬────────────┐",
      "  │ Metric                       │  Verifier  │   Judge    │",
      "  ├──────────────────────────────┼────────────┼────────────┤",
      _row("Overall accuracy", self.verifier_accuracy, self.judge_accuracy),
      _row(
        "Accuracy on hard tasks",
        self.verifier_accuracy_on_hard(),
        self.judge_accuracy_on_hard(),
      ),
      _row(
        "Accuracy on easy tasks",
        self.verifier_accuracy_on_easy(),
        self.judge_accuracy_on_easy(),
      ),
      _row(
        "Wins (lower oracle gap)",
        self.verifier_wins_count / max(n, 1),
        self.judge_wins_count / max(n, 1),
        fmt="count",
        wins_v=self.verifier_wins_count,
        wins_j=self.judge_wins_count,
      ),
      "  ├──────────────────────────────┼────────────┼────────────┤",
      _row(
        "Avg elapsed / task (s)",
        self._avg_elapsed(StrategyType.VERIFIER),
        self._avg_elapsed(StrategyType.JUDGE),
        fmt="time",
      ),
      "  └──────────────────────────────┴────────────┴────────────┘",
      "",
      f"  Average hardness score  : {self.avg_hardness:.3f}",
      f"  Strategy agreement rate : {self._agreement_rate():.1%}",
      f"  Ties                    : {self.tie_count}",
      bar,
      "",
    ]
    return "\n".join(lines)

  def per_task_table(self) -> str:
    """Render a per-task breakdown table."""
    header = (
      f"  {'Task':<30s}  {'Hard':>5s}  {'V-Gap':>6s}  {'J-Gap':>6s}  {'Winner':>9s}  {'Agree':>5s}"
    )
    separator = "  " + "-" * (len(header) - 2)
    rows = [header, separator]

    for r in sorted(self.task_records, key=lambda x: -x.hardness_score):
      hard_flag = "✓" if r.hardness_score >= self.hard_task_threshold else " "
      winner = "Verifier" if r.verifier_wins else ("Judge" if r.judge_wins else "Tie")
      agree = "✓" if r.strategies_agree else "✗"
      rows.append(
        f"  {r.task_id:<30s}  {hard_flag:>5s}  {r.oracle_gap_verifier:>6.3f}  "
        f"{r.oracle_gap_judge:>6.3f}  {winner:>9s}  {agree:>5s}"
      )

    return "\n".join(rows)

  # ── Private helpers ─────────────────────────────────────────────────────────

  def _avg_elapsed(self, strategy: StrategyType) -> float:
    if not self.task_records:
      return 0.0
    if strategy == StrategyType.VERIFIER:
      values = [r.elapsed_verifier_s for r in self.task_records]
    else:
      values = [r.elapsed_judge_s for r in self.task_records]
    return statistics.mean(values) if values else 0.0

  def _agreement_rate(self) -> float:
    if not self.task_records:
      return 0.0
    return sum(1 for r in self.task_records if r.strategies_agree) / len(self.task_records)


# ──────────────────────────────────────────────────────────────────────────────
# Harness
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class EvaluationHarness:
  """Runs both strategies in parallel and aggregates results into a HarnessReport."""

  verifier: BaseStrategy
  judge: BaseStrategy
  max_workers: int = 4
  hard_task_threshold: float = 0.6
  hardness_weights: dict[str, float] = field(
    default_factory=lambda: {
      "score_spread": 0.25,
      "strategy_disagreement": 0.35,
      "confidence_gap": 0.20,
      "oracle_gap": 0.20,
    }
  )

  def __post_init__(self) -> None:
    if self.verifier.get_strategy_type() != StrategyType.VERIFIER:
      raise TypeError(
        f"'verifier' must be a VERIFIER strategy, got {self.verifier.get_strategy_type()}"
      )
    if self.judge.get_strategy_type() != StrategyType.JUDGE:
      raise TypeError(f"'judge' must be a JUDGE strategy, got {self.judge.get_strategy_type()}")
    _validate_hardness_weights(self.hardness_weights)

  # ── Public API ───────────────────────────────────────────────────────────────

  def run(
    self,
    task_trajectories: Sequence[tuple[Task, TrajectoryList]],
    *,
    parallel: bool = True,
  ) -> HarnessReport:
    """Run the full harness; uses a thread pool when ``parallel=True``.

    Raises:
      ValueError: If any trajectory list is empty.
    """
    if not task_trajectories:
      return HarnessReport(hard_task_threshold=self.hard_task_threshold)

    t0 = time.perf_counter()

    if parallel and len(task_trajectories) > 1:
      records = self._run_parallel(task_trajectories)
    else:
      records = list(self._run_sequential(task_trajectories))

    total_elapsed = time.perf_counter() - t0
    return self._build_report(records, total_elapsed)

  def run_single(
    self,
    task: Task,
    trajectories: TrajectoryList,
  ) -> TaskHardnessRecord:
    """Evaluate a single task with both strategies.

    Raises:
      ValueError: If ``trajectories`` is empty.
    """
    if not trajectories:
      raise ValueError(f"Trajectory list for task '{task.id}' is empty")

    verifier_result, elapsed_v = _timed(self.verifier.evaluate, task, trajectories)
    judge_result, elapsed_j = _timed(self.judge.evaluate, task, trajectories)

    return self._compute_record(
      task, trajectories, verifier_result, judge_result, elapsed_v, elapsed_j
    )

  def hardness_score(
    self,
    task: Task,
    trajectories: TrajectoryList,
    verifier_result: EvaluationResult,
    judge_result: EvaluationResult,
  ) -> float:
    """Compute composite hardness in [0, 1] from pre-computed strategy results."""
    record = self._compute_record(task, trajectories, verifier_result, judge_result)
    return record.hardness_score

  # ── Internal run helpers ─────────────────────────────────────────────────────

  def _run_sequential(
    self,
    task_trajectories: Sequence[tuple[Task, TrajectoryList]],
  ) -> Iterator[TaskHardnessRecord]:
    for task, trajectories in task_trajectories:
      yield self.run_single(task, trajectories)

  def _run_parallel(
    self,
    task_trajectories: Sequence[tuple[Task, TrajectoryList]],
  ) -> list[TaskHardnessRecord]:
    records: list[TaskHardnessRecord] = []
    with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
      futures = {
        pool.submit(self.run_single, task, trajs): task.id for task, trajs in task_trajectories
      }
      for future in as_completed(futures):
        records.append(future.result())
    return records

  # ── Record computation ───────────────────────────────────────────────────────

  def _compute_record(
    self,
    task: Task,
    trajectories: TrajectoryList,
    verifier_result: EvaluationResult,
    judge_result: EvaluationResult,
    elapsed_v: float = 0.0,
    elapsed_j: float = 0.0,
  ) -> TaskHardnessRecord:
    oracle_id = _oracle_best(trajectories)

    score_var = _inter_strategy_score_spread(verifier_result, judge_result)
    disagreement = _pairwise_disagreement(trajectories, verifier_result, judge_result)
    avg_conf = _avg_confidence(verifier_result, judge_result)
    oracle_gap_v = _oracle_gap(oracle_id, verifier_result, trajectories)
    oracle_gap_j = _oracle_gap(oracle_id, judge_result, trajectories)

    # Oracle gap component: use average of both gaps.
    oracle_gap_component = (oracle_gap_v + oracle_gap_j) / 2.0

    # Confidence gap: invert so that low confidence = high hardness.
    confidence_hardness = 1.0 - avg_conf

    hardness = (
      self.hardness_weights["score_spread"] * score_var
      + self.hardness_weights["strategy_disagreement"] * disagreement
      + self.hardness_weights["confidence_gap"] * confidence_hardness
      + self.hardness_weights["oracle_gap"] * oracle_gap_component
    )
    hardness = max(0.0, min(1.0, hardness))

    return TaskHardnessRecord(
      task_id=task.id,
      hardness_score=hardness,
      score_spread=score_var,
      strategy_disagreement=disagreement,
      avg_confidence=avg_conf,
      oracle_gap_verifier=oracle_gap_v,
      oracle_gap_judge=oracle_gap_j,
      verifier_result=verifier_result,
      judge_result=judge_result,
      elapsed_verifier_s=elapsed_v,
      elapsed_judge_s=elapsed_j,
    )

  # ── Report assembly ──────────────────────────────────────────────────────────

  def _build_report(
    self,
    records: list[TaskHardnessRecord],
    total_elapsed: float,
  ) -> HarnessReport:
    if not records:
      return HarnessReport(
        hard_task_threshold=self.hard_task_threshold,
        total_elapsed_s=total_elapsed,
      )

    verifier_acc = _accuracy_on(records, StrategyType.VERIFIER)
    judge_acc = _accuracy_on(records, StrategyType.JUDGE)
    avg_hardness = statistics.mean(r.hardness_score for r in records)

    return HarnessReport(
      task_records=records,
      verifier_accuracy=verifier_acc,
      judge_accuracy=judge_acc,
      avg_hardness=avg_hardness,
      hard_task_threshold=self.hard_task_threshold,
      total_elapsed_s=total_elapsed,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Module-level helper functions
# ──────────────────────────────────────────────────────────────────────────────


def _timed(fn, *args, **kwargs) -> tuple:
  """Call *fn* with timing.  Returns ``(result, elapsed_seconds)``."""
  t0 = time.perf_counter()
  result = fn(*args, **kwargs)
  return result, time.perf_counter() - t0


def _oracle_best(trajectories: TrajectoryList) -> str | None:
  """Return the highest-reward trajectory ID, or ``None`` when no reward is set."""
  rewarded = [t for t in trajectories if t.reward is not None]
  if not rewarded:
    return None
  return max(rewarded, key=lambda t: t.reward).id  # type: ignore[arg-type]


def _oracle_gap(
  oracle_id: str | None,
  result: EvaluationResult,
  trajectories: TrajectoryList,
) -> float:
  """Return the reward gap between oracle-best and selected-best in [0, 1].

  Uses ground-truth reward when available; falls back to strategy scores.
  """
  scores = result.trajectory_scores
  if not scores:
    return 0.0

  selected_id = result.best_trajectory_id

  if oracle_id is not None:
    reward_map = {t.id: t.reward for t in trajectories if t.reward is not None}
    oracle_reward = reward_map.get(oracle_id, 0.0)
    selected_reward = reward_map.get(selected_id, 0.0)
    return max(0.0, oracle_reward - selected_reward)

  # No ground truth: gap is 0 when the selected is already the highest scored.
  selected_score = scores.get(selected_id, ScoreResult(trajectory_id=selected_id, score=0.0)).score
  best_score = max(s.score for s in scores.values())
  return max(0.0, best_score - selected_score)


def _inter_strategy_score_spread(
  verifier_result: EvaluationResult,
  judge_result: EvaluationResult,
) -> float:
  """Mean absolute score difference between strategies per trajectory (absolute quality spread)."""
  v_scores = verifier_result.trajectory_scores
  j_scores = judge_result.trajectory_scores
  shared = [tid for tid in v_scores if tid in j_scores]
  if not shared:
    return 0.0
  diffs = [abs(v_scores[tid].score - j_scores[tid].score) for tid in shared]
  return statistics.mean(diffs)


def _pairwise_disagreement(
  trajectories: TrajectoryList,
  verifier_result: EvaluationResult,
  judge_result: EvaluationResult,
) -> float:
  """Fraction of trajectory pairs where strategies disagree on ranking."""
  v_scores = verifier_result.trajectory_scores
  j_scores = judge_result.trajectory_scores

  tids = [t.id for t in trajectories if t.id in v_scores and t.id in j_scores]
  if len(tids) < 2:
    return 0.0

  pairs = [(a, b) for idx, a in enumerate(tids) for b in tids[idx + 1 :]]
  if not pairs:
    return 0.0

  disagreements = 0
  for tid_a, tid_b in pairs:
    v_a, v_b = v_scores[tid_a].score, v_scores[tid_b].score
    j_a, j_b = j_scores[tid_a].score, j_scores[tid_b].score

    v_prefers_a = v_a > v_b
    j_prefers_a = j_a > j_b
    v_tie = abs(v_a - v_b) < 1e-9
    j_tie = abs(j_a - j_b) < 1e-9

    if not v_tie and not j_tie and (v_prefers_a != j_prefers_a):
      disagreements += 1

  return disagreements / len(pairs)


def _avg_confidence(
  verifier_result: EvaluationResult,
  judge_result: EvaluationResult,
) -> float:
  """Mean confidence across all trajectory scores in both strategy results."""
  all_confidences: list[float] = []
  for result in (verifier_result, judge_result):
    for score_result in result.trajectory_scores.values():
      all_confidences.append(score_result.confidence)
  if not all_confidences:
    return 1.0
  return statistics.mean(all_confidences)


def _accuracy_on(
  records: Sequence[TaskHardnessRecord],
  strategy: StrategyType,
) -> float:
  """Selection accuracy for a strategy: fraction of tasks with near-zero oracle gap."""
  if not records:
    return 0.0

  correct = 0
  for record in records:
    if strategy == StrategyType.VERIFIER:
      gap = record.oracle_gap_verifier
    else:
      gap = record.oracle_gap_judge

    # A near-zero gap means the correct trajectory was selected.
    if gap < 1e-6:
      correct += 1

  return correct / len(records)


def _validate_hardness_weights(weights: dict[str, float]) -> None:
  """Validate that hardness weights are non-negative and sum to 1.

  Raises:
    ValueError: If weights are invalid.
  """
  required_keys = {
    "score_spread",
    "strategy_disagreement",
    "confidence_gap",
    "oracle_gap",
  }
  missing = required_keys - weights.keys()
  if missing:
    raise ValueError(f"Missing hardness weight keys: {missing}")

  for key, val in weights.items():
    if val < 0:
      raise ValueError(f"Hardness weight '{key}' must be non-negative, got {val}")

  total = sum(weights[k] for k in required_keys)
  if abs(total - 1.0) > 1e-6:
    raise ValueError(
      f"Hardness weights must sum to 1.0, got {total:.6f}. Current weights: {weights}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Formatting utilities (used by HarnessReport)
# ──────────────────────────────────────────────────────────────────────────────


def _row(
  label: str,
  val_v: float,
  val_j: float,
  *,
  fmt: str = "pct",
  wins_v: int = 0,
  wins_j: int = 0,
) -> str:
  """Format one row of the summary table; fmt is one of ``"pct"``, ``"time"``, ``"count"``."""
  if fmt == "pct":
    v_str = f"{val_v:.1%}"
    j_str = f"{val_j:.1%}"
  elif fmt == "time":
    v_str = f"{val_v:.2f} s"
    j_str = f"{val_j:.2f} s"
  elif fmt == "count":
    v_str = str(wins_v)
    j_str = str(wins_j)
  else:
    v_str = f"{val_v:.4f}"
    j_str = f"{val_j:.4f}"

  return f"  │ {label:<28s}  │ {v_str:^10s} │ {j_str:^10s} │"
