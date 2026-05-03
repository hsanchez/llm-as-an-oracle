"""Side-by-side comparison harness for Verifier vs Judge, with composite hardness metrics.

Hardness is measured across four dimensions: score spread, strategy disagreement,
confidence gap, and oracle gap (selection error).
"""

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


@dataclass
class TaskHardnessRecord:
  """Hardness measurements and evaluation results for a single task."""

  task_id: str
  hardness_score: float = 0.0
  score_spread: float = 0.0
  strategy_disagreement: float = 0.0
  average_confidence: float = 1.0
  oracle_gap_verifier: float = 0.0
  oracle_gap_judge: float = 0.0
  verifier_result: EvaluationResult | None = None
  judge_result: EvaluationResult | None = None
  elapsed_verifier_seconds: float = 0.0
  elapsed_judge_seconds: float = 0.0

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


@dataclass
class HarnessReport:
  """Aggregated comparison report for all evaluated tasks."""

  task_records: list[TaskHardnessRecord] = field(default_factory=list)
  verifier_accuracy: float = 0.0
  judge_accuracy: float = 0.0
  average_hardness: float = 0.0
  hard_task_threshold: float = 0.6
  total_elapsed_seconds: float = 0.0

  @property
  def hard_tasks(self) -> list[TaskHardnessRecord]:
    """Tasks whose hardness score exceeds ``hard_task_threshold``."""
    return [
      record for record in self.task_records if record.hardness_score >= self.hard_task_threshold
    ]

  @property
  def easy_tasks(self) -> list[TaskHardnessRecord]:
    """Tasks whose hardness score is below ``hard_task_threshold``."""
    return [
      record for record in self.task_records if record.hardness_score < self.hard_task_threshold
    ]

  @property
  def verifier_wins_count(self) -> int:
    """Number of tasks where the verifier outperformed the judge."""
    return sum(1 for record in self.task_records if record.verifier_wins)

  @property
  def judge_wins_count(self) -> int:
    """Number of tasks where the judge outperformed the verifier."""
    return sum(1 for record in self.task_records if record.judge_wins)

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

  def summary(self) -> str:
    """Render a human-readable comparison summary table."""
    record_count = len(self.task_records)
    num_hard_tasks = len(self.hard_tasks)
    num_easy_tasks = len(self.easy_tasks)

    divider = "═" * 64

    lines = [
      "",
      divider,
      "  LLM Oracle — Evaluation Harness Report",
      divider,
      f"  Tasks evaluated : {record_count}",
      f"  Hard tasks      : {num_hard_tasks}  (threshold ≥ {self.hard_task_threshold:.2f})",
      f"  Easy tasks      : {num_easy_tasks}",
      f"  Total runtime   : {self.total_elapsed_seconds:.1f} s",
      "",
      "  ┌──────────────────────────────┬────────────┬────────────┐",
      "  │ Metric                       │  Verifier  │   Judge    │",
      "  ├──────────────────────────────┼────────────┼────────────┤",
      _format_row("Overall accuracy", self.verifier_accuracy, self.judge_accuracy),
      _format_row(
        "Accuracy on hard tasks",
        self.verifier_accuracy_on_hard(),
        self.judge_accuracy_on_hard(),
      ),
      _format_row(
        "Accuracy on easy tasks",
        self.verifier_accuracy_on_easy(),
        self.judge_accuracy_on_easy(),
      ),
      _format_row(
        "Wins (lower oracle gap)",
        self.verifier_wins_count / max(record_count, 1),
        self.judge_wins_count / max(record_count, 1),
        format_type="count",
        verifier_wins=self.verifier_wins_count,
        judge_wins=self.judge_wins_count,
      ),
      "  ├──────────────────────────────┼────────────┼────────────┤",
      _format_row(
        "Avg elapsed / task (s)",
        self._average_elapsed(StrategyType.VERIFIER),
        self._average_elapsed(StrategyType.JUDGE),
        format_type="time",
      ),
      "  └──────────────────────────────┴────────────┴────────────┘",
      "",
      f"  Average hardness score  : {self.average_hardness:.3f}",
      f"  Strategy agreement rate : {self._agreement_rate():.1%}",
      f"  Ties                    : {self.tie_count}",
      divider,
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

    for record in sorted(self.task_records, key=lambda item: -item.hardness_score):
      hard_flag = "✓" if record.hardness_score >= self.hard_task_threshold else " "
      winner = "Verifier" if record.verifier_wins else ("Judge" if record.judge_wins else "Tie")
      agree = "✓" if record.strategies_agree else "✗"
      rows.append(
        f"  {record.task_id:<30s}  {hard_flag:>5s}  "
        f"{record.oracle_gap_verifier:>6.3f}  "
        f"{record.oracle_gap_judge:>6.3f}  {winner:>9s}  {agree:>5s}"
      )

    return "\n".join(rows)

  def _average_elapsed(self, strategy: StrategyType) -> float:
    if not self.task_records:
      return 0.0
    if strategy == StrategyType.VERIFIER:
      values = [record.elapsed_verifier_seconds for record in self.task_records]
    else:
      values = [record.elapsed_judge_seconds for record in self.task_records]
    return statistics.mean(values) if values else 0.0

  def _agreement_rate(self) -> float:
    if not self.task_records:
      return 0.0
    return sum(1 for record in self.task_records if record.strategies_agree) / len(
      self.task_records
    )


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

    start_time = time.perf_counter()

    if parallel and len(task_trajectories) > 1:
      records = self._run_parallel(task_trajectories)
    else:
      records = list(self._run_sequential(task_trajectories))

    total_elapsed = time.perf_counter() - start_time
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

    verifier_result, elapsed_verifier = _timed(self.verifier.evaluate, task, trajectories)
    judge_result, elapsed_judge = _timed(self.judge.evaluate, task, trajectories)

    return self._compute_record(
      task,
      trajectories,
      verifier_result,
      judge_result,
      elapsed_verifier,
      elapsed_judge,
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
        pool.submit(self.run_single, task, trajectories): task.id
        for task, trajectories in task_trajectories
      }
      for future in as_completed(futures):
        records.append(future.result())
    return records

  def _compute_record(
    self,
    task: Task,
    trajectories: TrajectoryList,
    verifier_result: EvaluationResult,
    judge_result: EvaluationResult,
    elapsed_verifier: float = 0.0,
    elapsed_judge: float = 0.0,
  ) -> TaskHardnessRecord:
    oracle_id = _oracle_best(trajectories)

    score_var = _inter_strategy_score_spread(verifier_result, judge_result)
    disagreement = _pairwise_disagreement(trajectories, verifier_result, judge_result)
    average_confidence = _average_confidence(verifier_result, judge_result)
    oracle_gap_verifier = _oracle_gap(oracle_id, verifier_result, trajectories)
    oracle_gap_judge = _oracle_gap(oracle_id, judge_result, trajectories)

    oracle_gap_component = (oracle_gap_verifier + oracle_gap_judge) / 2.0

    confidence_hardness = 1.0 - average_confidence

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
      average_confidence=average_confidence,
      oracle_gap_verifier=oracle_gap_verifier,
      oracle_gap_judge=oracle_gap_judge,
      verifier_result=verifier_result,
      judge_result=judge_result,
      elapsed_verifier_seconds=elapsed_verifier,
      elapsed_judge_seconds=elapsed_judge,
    )

  def _build_report(
    self,
    records: list[TaskHardnessRecord],
    total_elapsed: float,
  ) -> HarnessReport:
    if not records:
      return HarnessReport(
        hard_task_threshold=self.hard_task_threshold,
        total_elapsed_seconds=total_elapsed,
      )

    verifier_acc = _accuracy_on(records, StrategyType.VERIFIER)
    judge_acc = _accuracy_on(records, StrategyType.JUDGE)
    average_hardness = statistics.mean(record.hardness_score for record in records)

    return HarnessReport(
      task_records=records,
      verifier_accuracy=verifier_acc,
      judge_accuracy=judge_acc,
      average_hardness=average_hardness,
      hard_task_threshold=self.hard_task_threshold,
      total_elapsed_seconds=total_elapsed,
    )


def _timed(fn, *args, **kwargs) -> tuple:
  """Call *fn* with timing.  Returns ``(result, elapsed_seconds)``."""
  start_time = time.perf_counter()
  result = fn(*args, **kwargs)
  return result, time.perf_counter() - start_time


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

  selected_score = scores.get(selected_id, ScoreResult(trajectory_id=selected_id, score=0.0)).score
  best_score = max(s.score for s in scores.values())
  return max(0.0, best_score - selected_score)


def _inter_strategy_score_spread(
  verifier_result: EvaluationResult,
  judge_result: EvaluationResult,
) -> float:
  """Mean absolute score difference between strategies per trajectory (absolute quality spread)."""
  verifier_scores = verifier_result.trajectory_scores
  judge_scores = judge_result.trajectory_scores
  shared = [trajectory_id for trajectory_id in verifier_scores if trajectory_id in judge_scores]
  if not shared:
    return 0.0
  differences = [
    abs(verifier_scores[trajectory_id].score - judge_scores[trajectory_id].score)
    for trajectory_id in shared
  ]
  return statistics.mean(differences)


def _pairwise_disagreement(
  trajectories: TrajectoryList,
  verifier_result: EvaluationResult,
  judge_result: EvaluationResult,
) -> float:
  """Fraction of trajectory pairs where strategies disagree on ranking."""
  verifier_scores = verifier_result.trajectory_scores
  judge_scores = judge_result.trajectory_scores

  trajectory_ids = [
    trajectory.id
    for trajectory in trajectories
    if trajectory.id in verifier_scores and trajectory.id in judge_scores
  ]
  if len(trajectory_ids) < 2:
    return 0.0

  pairs = [
    (trajectory_a_id, trajectory_b_id)
    for index, trajectory_a_id in enumerate(trajectory_ids)
    for trajectory_b_id in trajectory_ids[index + 1 :]
  ]
  if not pairs:
    return 0.0

  disagreements = 0
  for trajectory_a_id, trajectory_b_id in pairs:
    verifier_score_a = verifier_scores[trajectory_a_id].score
    verifier_score_b = verifier_scores[trajectory_b_id].score
    judge_score_a = judge_scores[trajectory_a_id].score
    judge_score_b = judge_scores[trajectory_b_id].score

    verifier_prefers_a = verifier_score_a > verifier_score_b
    judge_prefers_a = judge_score_a > judge_score_b
    verifier_tie = abs(verifier_score_a - verifier_score_b) < 1e-9
    judge_tie = abs(judge_score_a - judge_score_b) < 1e-9

    if not verifier_tie and not judge_tie and (verifier_prefers_a != judge_prefers_a):
      disagreements += 1

  return disagreements / len(pairs)


def _average_confidence(
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

  for key, value in weights.items():
    if value < 0:
      raise ValueError(f"Hardness weight '{key}' must be non-negative, got {value}")

  total = sum(weights[k] for k in required_keys)
  if abs(total - 1.0) > 1e-6:
    raise ValueError(
      f"Hardness weights must sum to 1.0, got {total:.6f}. Current weights: {weights}"
    )


def _format_row(
  label: str,
  verifier_value: float,
  judge_value: float,
  *,
  format_type: str = "pct",
  verifier_wins: int = 0,
  judge_wins: int = 0,
) -> str:
  """Format one row of the summary table; format_type is one of ``"pct"``, ``"time"``, ``"count"``."""
  if format_type == "pct":
    verifier_text = f"{verifier_value:.1%}"
    judge_text = f"{judge_value:.1%}"
  elif format_type == "time":
    verifier_text = f"{verifier_value:.2f} s"
    judge_text = f"{judge_value:.2f} s"
  elif format_type == "count":
    verifier_text = str(verifier_wins)
    judge_text = str(judge_wins)
  else:
    verifier_text = f"{verifier_value:.4f}"
    judge_text = f"{judge_value:.4f}"

  return f"  │ {label:<28s}  │ {verifier_text:^10s} │ {judge_text:^10s} │"
