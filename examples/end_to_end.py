#!/usr/bin/env python3
"""End-to-end example: LLM Oracle — Verifier · Judge · Harness · Router.

This script walks through every major feature of the llm-oracle package using
the built-in StubProvider so it runs completely offline with no API keys.

Sections
--------
1.  Setup       – shared model, config, and criteria
2.  Verifier    – score a single trajectory; run a pairwise tournament
3.  Judge       – pointwise rubric scoring; bias-mitigated pairwise comparison
4.  Harness     – side-by-side hardness analysis across multiple tasks
5.  Router      – default policy-chain routing; custom policy; feedback loop
6.  Summary     – print routing audit and harness report

Run
---
    cd llm-as-oracle
    uv run python examples/end_to_end.py
"""

from __future__ import annotations

import dataclasses
import textwrap

from llm_oracle import (
  EvaluationCriterion,
  EvaluationHarness,
  JudgeStrategy,
  OracleRouter,
  ScoringConfig,
  StubProvider,
  StubResponse,
  Task,
  TaskDifficulty,
  Trajectory,
  VerifierStrategy,
)
from llm_oracle.core.models import StrategyType
from llm_oracle.routing.router import (
  PolicyVote,
  RoutingPolicy,
  RoutingSignals,
  SignalExtractor,
)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def banner(title: str) -> None:
  width = 68
  print()
  print("╔" + "═" * (width - 2) + "╗")
  print("║" + title.center(width - 2) + "║")
  print("╚" + "═" * (width - 2) + "╝")


def section(title: str) -> None:
  print()
  print("┌─" + "─" * len(title) + "─┐")
  print(f"│ {title} │")
  print("└─" + "─" * len(title) + "─┘")


def indent(text: str, prefix: str = "  ") -> str:
  return textwrap.indent(text.strip(), prefix)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Setup
# ─────────────────────────────────────────────────────────────────────────────

banner("1 · Setup — shared model, config, and criteria")

# The StubProvider mimics real model responses without any network call.
# It synthesises log-probability distributions peaked near the target score
# letters, so the verifier's logprob extraction path is exercised too.
stub_responses = [
  StubResponse(score="B", score_a="B", score_b="G"),  # verifier-optimistic
  StubResponse(score="D", score_a="D", score_b="H"),  # verifier-neutral
  StubResponse(score="A", score_a="A", score_b="F"),  # judge-optimistic
  StubResponse(score="E", score_a="C", score_b="J"),  # judge-neutral
]

model = StubProvider(
  model_id="stub-oracle",
  default_score="D",
  default_score_a="D",
  default_score_b="H",
  responses=stub_responses,
  seed=42,
)

config = ScoringConfig(
  granularity=20,  # A–T scoring scale (matches LLM-as-a-Verifier paper)
  num_verifications=2,  # K=2 repeated verifications per criterion
  num_criteria=3,  # C=3 criteria
  use_logprobs=True,  # extract expected score from logprob distribution
  fuzzy_threshold=0.75,
)

criteria: list[EvaluationCriterion] = [
  EvaluationCriterion(
    id="correctness",
    name="Correctness",
    description=(
      "Does the solution produce the correct output for all inputs? "
      "Check edge cases and off-by-one errors."
    ),
    weight=2.0,
  ),
  EvaluationCriterion(
    id="efficiency",
    name="Efficiency",
    description=(
      "Is the time and space complexity optimal or near-optimal for this class of problem?"
    ),
    weight=1.5,
  ),
  EvaluationCriterion(
    id="readability",
    name="Readability",
    description=(
      "Is the code clean, well-named, and easy to understand without additional context?"
    ),
    weight=1.0,
  ),
]

print(f"  Model       : {model.model_id}")
print(f"  Granularity : {config.granularity}  (A=best … T=worst)")
print(f"  Verifications/criterion : {config.num_verifications}")
print(f"  Criteria    : {[c.name for c in criteria]}")


# ─────────────────────────────────────────────────────────────────────────────
# 2.  LLM-as-a-Verifier
# ─────────────────────────────────────────────────────────────────────────────

banner("2 · LLM-as-a-Verifier")

verifier = VerifierStrategy(model, config, criteria)

# ── Task & trajectories ───────────────────────────────────────────────────────

sort_task = Task(
  id="sort-integers",
  description="Sort a list of integers in ascending order",
  problem_statement=(
    "Implement a function `sort_list(lst: list[int]) -> list[int]` that "
    "returns a new sorted list without modifying the input.  The function "
    "must handle empty lists, negative numbers, and duplicate values."
  ),
  ground_truth="def sort_list(lst): return sorted(lst)",
  test_cases=[
    {"input": [3, 1, 2], "expected": [1, 2, 3]},
    {"input": [], "expected": []},
    {"input": [-1, 0, -5], "expected": [-5, -1, 0]},
  ],
  difficulty=TaskDifficulty.EASY,
)

trajectories: list[Trajectory] = [
  Trajectory(
    id="traj-optimal",
    task_id=sort_task.id,
    content="def sort_list(lst: list[int]) -> list[int]:\n    return sorted(lst)",
    output="[1, 2, 3] ✓ | [] ✓ | [-5, -1, 0] ✓",
    reward=1.0,
  ),
  Trajectory(
    id="traj-inplace",
    task_id=sort_task.id,
    content=(
      "def sort_list(lst: list[int]) -> list[int]:\n"
      "    lst.sort()  # modifies input — violates spec\n"
      "    return lst"
    ),
    output="[1, 2, 3] ✓ | [] ✓ | [-5, -1, 0] ✓ (but mutates input)",
    reward=0.5,
  ),
  Trajectory(
    id="traj-bubble",
    task_id=sort_task.id,
    content=(
      "def sort_list(lst):\n"
      "    n = len(lst)\n"
      "    for i in range(n):\n"
      "        for j in range(n - i - 1):\n"
      "            if lst[j] > lst[j + 1]:\n"
      "                lst[j], lst[j + 1] = lst[j + 1], lst[j]\n"
      "    return lst"
    ),
    output="[1, 2, 3] ✓ | [] ✓ | [-5, -1, 0] ✓ (O(n²), mutates input)",
    reward=0.0,
  ),
]

# ── Single-criterion score ────────────────────────────────────────────────────

section("Single-criterion scoring (Correctness)")

score = verifier.score_trajectory(sort_task, trajectories[0], criteria[0])
print(f"  Trajectory  : {score.trajectory_id}")
print(f"  Score       : {score.score:.4f}  (normalised 0–1)")
print(f"  Confidence  : {score.confidence:.4f}")
print(f"  Criterion   : {score.criterion_scores}")

# ── Pairwise comparison ───────────────────────────────────────────────────────

section("Pairwise comparison (optimal vs bubble)")

comp = verifier.compare_trajectories(sort_task, trajectories[0], trajectories[2], criteria[0])
print(f"  Score A ({comp.trajectory_a_id:<14s}) : {comp.score_a:.4f}")
print(f"  Score B ({comp.trajectory_b_id:<14s}) : {comp.score_b:.4f}")
winner_label = comp.winner or "tie"
print(f"  Winner      : {winner_label}")
print(f"  Confidence  : {comp.confidence:.4f}")

# ── Full tournament evaluation ────────────────────────────────────────────────

section("Tournament evaluation (3 trajectories)")

v_result = verifier.evaluate(sort_task, trajectories)
print(f"  Best trajectory : {v_result.best_trajectory_id}")
print(f"  Strategy        : {v_result.strategy_type.value}")
print(f"  Pairwise calls  : {len(v_result.pairwise_comparisons)}")
print()
print("  Per-trajectory scores:")
for tid, sr in sorted(v_result.trajectory_scores.items(), key=lambda x: -x[1].score):
  crit_str = "  ".join(f"{k}={v:.3f}" for k, v in sr.criterion_scores.items())
  print(f"    {tid:<16s}  overall={sr.score:.4f}   [{crit_str}]")

print()
meta = v_result.metadata
print(
  f"  Granularity g={meta.get('granularity')}  "
  f"verifications K={meta.get('num_verifications')}  "
  f"comparisons={meta.get('num_comparisons')}"
)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  LLM-as-a-Judge
# ─────────────────────────────────────────────────────────────────────────────

banner("3 · LLM-as-a-Judge")

judge = JudgeStrategy(
  model,
  config,
  criteria,
  score_min=1.0,
  score_max=10.0,
  swap_pairwise=True,  # mitigates positional bias
  reasoning_depth="detailed",  # "brief" | "detailed" | "chain_of_thought"
)

# ── Single-criterion pointwise score ─────────────────────────────────────────

section("Pointwise scoring (Efficiency, traj-optimal)")

j_score = judge.score_trajectory(sort_task, trajectories[0], criteria[1])
print(f"  Trajectory  : {j_score.trajectory_id}")
print(f"  Score       : {j_score.score:.4f}  (normalised 0–1)")
print(f"  Raw score   : {j_score.raw_score}")
print(f"  Confidence  : {j_score.confidence:.4f}")

# ── Positional-bias-mitigated pairwise ────────────────────────────────────────

section("Pairwise with position-swap (optimal vs in-place)")

j_comp = judge.compare_trajectories(sort_task, trajectories[0], trajectories[1], criteria[0])
print(f"  Score A ({j_comp.trajectory_a_id:<14s}) : {j_comp.score_a:.4f}")
print(f"  Score B ({j_comp.trajectory_b_id:<14s}) : {j_comp.score_b:.4f}")
print(f"  Winner      : {j_comp.winner or 'tie'}")
print(f"  Confidence  : {j_comp.confidence:.4f}")
print("  Note        : scores already averaged over both orderings (swap_pairwise=True)")

# ── Full judge evaluation ─────────────────────────────────────────────────────

section("Full judge evaluation (3 trajectories)")

j_result = judge.evaluate(sort_task, trajectories)
print(f"  Best trajectory : {j_result.best_trajectory_id}")
print(f"  Strategy        : {j_result.strategy_type.value}")
print(f"  Score range     : {j_result.metadata['score_range']}")
print(f"  Swap pairwise   : {j_result.metadata['swap_pairwise']}")
print()
print("  Per-trajectory scores:")
for tid, sr in sorted(j_result.trajectory_scores.items(), key=lambda x: -x[1].score):
  crit_str = "  ".join(f"{k}={v:.3f}" for k, v in sr.criterion_scores.items())
  print(f"    {tid:<16s}  overall={sr.score:.4f}   [{crit_str}]")


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Evaluation Harness — hardness comparison
# ─────────────────────────────────────────────────────────────────────────────

banner("4 · Evaluation Harness — hardness comparison")

# Create a broader task set to stress-test both strategies
tasks_and_trajectories = []

# Task 1: trivial sort (easy — judge should suffice)
tasks_and_trajectories.append((sort_task, trajectories))

# Task 2: binary search (medium, verifiable via test cases)
search_task = Task(
  id="binary-search",
  description="Binary search in a sorted array",
  problem_statement=(
    "Implement `binary_search(arr: list[int], target: int) -> int` that "
    "returns the index of `target` in the sorted `arr`, or -1 if absent. "
    "Must run in O(log n)."
  ),
  test_cases=[
    {"arr": [1, 3, 5, 7, 9], "target": 5, "expected": 2},
    {"arr": [1, 3, 5, 7, 9], "target": 6, "expected": -1},
    {"arr": [], "target": 1, "expected": -1},
  ],
  difficulty=TaskDifficulty.MEDIUM,
  ground_truth=(
    "def binary_search(arr, target):\n"
    "    lo, hi = 0, len(arr) - 1\n"
    "    while lo <= hi:\n"
    "        mid = (lo + hi) // 2\n"
    "        if arr[mid] == target: return mid\n"
    "        elif arr[mid] < target: lo = mid + 1\n"
    "        else: hi = mid - 1\n"
    "    return -1"
  ),
)
search_trajs = [
  Trajectory(
    "bs-correct",
    search_task.id,
    "def binary_search(arr, target):\n"
    "    lo, hi = 0, len(arr) - 1\n"
    "    while lo <= hi:\n"
    "        mid = (lo + hi) // 2\n"
    "        if arr[mid] == target: return mid\n"
    "        elif arr[mid] < target: lo = mid + 1\n"
    "        else: hi = mid - 1\n"
    "    return -1",
    reward=1.0,
  ),
  Trajectory(
    "bs-linear",
    search_task.id,
    "def binary_search(arr, target):\n"
    "    for i, v in enumerate(arr):\n"
    "        if v == target: return i\n"
    "    return -1  # O(n) — wrong complexity",
    reward=0.0,
  ),
]
tasks_and_trajectories.append((search_task, search_trajs))

# Task 3: open-ended essay (hard to verify — judge should shine)
essay_task = Task(
  id="raft-essay",
  description="Explain the Raft consensus algorithm",
  problem_statement=(
    "Write a technical essay (≈400 words) explaining the Raft consensus "
    "algorithm.  Cover leader election, log replication, and safety "
    "guarantees.  Compare briefly with Paxos."
  ),
  difficulty=TaskDifficulty.HARD,
)
essay_trajs = [
  Trajectory(
    "essay-good",
    essay_task.id,
    "Raft is a consensus algorithm designed for understandability …  "
    "[covers all required topics with clarity and depth]",
    reward=1.0,
  ),
  Trajectory(
    "essay-poor",
    essay_task.id,
    "Raft is a thing that makes servers agree.  Paxos is also a thing.",
    reward=0.0,
  ),
]
tasks_and_trajectories.append((essay_task, essay_trajs))

# ── Run the harness ───────────────────────────────────────────────────────────

harness = EvaluationHarness(
  verifier=verifier,
  judge=judge,
  max_workers=2,
  hard_task_threshold=0.50,
  hardness_weights={
    "score_spread": 0.25,
    "strategy_disagreement": 0.35,
    "confidence_gap": 0.20,
    "oracle_gap": 0.20,
  },
)

report = harness.run(tasks_and_trajectories, parallel=False)

print(report.summary())

section("Per-task breakdown")
print(report.per_task_table())

section("Hard vs. easy task accuracy")
print(f"  Hard tasks  ({len(report.hard_tasks)} tasks)")
print(f"    Verifier accuracy : {report.verifier_accuracy_on_hard():.1%}")
print(f"    Judge    accuracy : {report.judge_accuracy_on_hard():.1%}")
print(f"  Easy tasks  ({len(report.easy_tasks)} tasks)")
print(f"    Verifier accuracy : {report.verifier_accuracy_on_easy():.1%}")
print(f"    Judge    accuracy : {report.judge_accuracy_on_easy():.1%}")


# ─────────────────────────────────────────────────────────────────────────────
# 5.  OracleRouter
# ─────────────────────────────────────────────────────────────────────────────

banner("5 · OracleRouter — intelligent strategy selection")

# ── Default router with five built-in policies ────────────────────────────────

section("Default policy-chain router")

router = OracleRouter.default(
  verifier,
  judge,
  confidence_threshold=0.50,  # commit if aggregated confidence ≥ 50 %
)

print("  Built-in policies (in chain order):")
for i, policy in enumerate(router._chain.policies, 1):
  print(f"    {i}. {policy.name:<25s}  weight={policy.weight}")

# ── Route each task and inspect the decision ──────────────────────────────────

section("Routing decisions (cold — no prior hardness)")

for task, trajs in tasks_and_trajectories:
  decision = router.route(task, trajs)
  strategy_icon = "🔍" if decision.selected_strategy == StrategyType.VERIFIER else "⚖️ "
  print(
    f"  {strategy_icon} {task.id:<22s}  "
    f"→ {decision.selected_strategy.value:<9s}  "
    f"conf={decision.confidence:.3f}  "
    f"(⏱ {decision.elapsed_ms:.2f} ms)"
  )
  # Show the top-2 most influential policy votes
  top_votes = sorted(decision.policy_votes, key=lambda v: -v.confidence * v.weight)[:2]
  for vote in top_votes:
    arrow = "↑" if vote.preferred == decision.selected_strategy else "↓"
    print(
      f"       {arrow} [{vote.policy_name:<20s}]  "
      f"{vote.preferred.value:<9s}  conf={vote.confidence:.2f}  w={vote.weight}"
    )

# ── Hardness feedback loop ────────────────────────────────────────────────────

section("Feedback loop — inject harness hardness into router")

for record in report.task_records:
  router.update_hardness(record.task_id, record.hardness_score)
  print(f"  Updated cache: {record.task_id:<22s}  hardness={record.hardness_score:.3f}")

print()
print("  Routing decisions (warm — prior hardness available):")
for task, trajs in tasks_and_trajectories:
  decision = router.route(task, trajs)
  strategy_icon = "🔍" if decision.selected_strategy == StrategyType.VERIFIER else "⚖️ "
  ph = decision.signals.prior_hardness
  print(
    f"  {strategy_icon} {task.id:<22s}  "
    f"→ {decision.selected_strategy.value:<9s}  "
    f"conf={decision.confidence:.3f}  "
    + (f"prior_hardness={ph:.3f}" if ph is not None else "prior_hardness=N/A")
  )

# ── Custom routing policy ─────────────────────────────────────────────────────

section("Registering a custom routing policy")


class ShortProblemPolicy(RoutingPolicy):
  """Prefer the judge for very short problem statements.

  Short problems are usually self-explanatory; the judge's holistic
  chain-of-thought reasoning works well without needing fine-grained
  log-probability calibration.
  """

  name = "short_problem"
  weight = 0.7  # modest weight — supportive, not dominant

  _threshold_chars: int = 150

  def vote(
    self,
    task: Task,
    trajectories,
    signals: RoutingSignals,
  ) -> PolicyVote:
    is_short = signals.problem_length < (self._threshold_chars / 2000.0)
    if is_short:
      return PolicyVote(
        policy_name=self.name,
        preferred=StrategyType.JUDGE,
        confidence=0.63,
        weight=self.weight,
        signals_used=["problem_length"],
        reasoning=(
          f"Problem statement is short "
          f"(length_signal={signals.problem_length:.3f} < "
          f"{self._threshold_chars / 2000.0:.3f}); "
          "judge's rubric is sufficient."
        ),
      )
    return PolicyVote(
      policy_name=self.name,
      preferred=StrategyType.VERIFIER,
      confidence=0.53,
      weight=self.weight,
      signals_used=["problem_length"],
      reasoning="Problem is long enough that fine-grained scoring may help.",
    )


router.register_policy(ShortProblemPolicy())
print(f"  Registered '{ShortProblemPolicy.name}' policy  (weight={ShortProblemPolicy.weight})")
print(f"  Chain now has {len(router._chain.policies)} policies.")

# Re-route with the new policy active
print()
print("  Routing decisions (with ShortProblemPolicy):")
for task, trajs in tasks_and_trajectories:
  decision = router.route(task, trajs)
  short_vote = next(
    (v for v in decision.policy_votes if v.policy_name == "short_problem"),
    None,
  )
  strategy_icon = "🔍" if decision.selected_strategy == StrategyType.VERIFIER else "⚖️ "
  print(
    f"  {strategy_icon} {task.id:<22s}  "
    f"→ {decision.selected_strategy.value:<9s}  "
    f"conf={decision.confidence:.3f}  "
    + (f"(short_policy voted {short_vote.preferred.value})" if short_vote else "")
  )

# ── Route-and-evaluate in one call ───────────────────────────────────────────

section("router.evaluate() — route and evaluate in one call")

result, decision = router.evaluate(sort_task, trajectories)
print(f"  Task         : {result.task_id}")
print(f"  Strategy     : {decision.selected_strategy.value}")
print(f"  Confidence   : {decision.confidence:.3f}")
print(f"  Best traj    : {result.best_trajectory_id}")
print()
print("  Signal features:")
for _f in dataclasses.fields(decision.signals):
  print(f"    {_f.name:<35s} : {getattr(decision.signals, _f.name)}")


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Signal extractor standalone demo
# ─────────────────────────────────────────────────────────────────────────────

banner("6 · SignalExtractor — feature introspection")

extractor = SignalExtractor()

for task, trajs in tasks_and_trajectories:
  signals = extractor.extract(task, trajs, prior_hardness=None)
  print(f"  Task: {task.id}")
  print(
    f"    has_ground_truth          = {signals.has_ground_truth:.0f}  "
    f"has_test_cases              = {signals.has_test_cases:.0f}"
  )
  print(
    f"    trajectory_count          = {signals.trajectory_count}  "
    f"output_available            = {signals.output_available:.0f}"
  )
  print(
    f"    verifiable_kw_density     = {signals.verifiable_keyword_density:.3f}  "
    f"judgement_kw_density        = {signals.judgement_keyword_density:.3f}"
  )
  print(
    f"    stated_difficulty         = {signals.stated_difficulty:.2f}  "
    f"problem_length (norm)       = {signals.problem_length:.3f}"
  )
  print()


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Routing audit log
# ─────────────────────────────────────────────────────────────────────────────

banner("7 · Router audit log")
print(router.routing_summary())


# ─────────────────────────────────────────────────────────────────────────────
# Done
# ─────────────────────────────────────────────────────────────────────────────

banner("All examples completed successfully 🎉")
print(
  "  Next steps:\n"
  "  • Swap StubProvider for OpenAIProvider / AnthropicProvider / GeminiProvider\n"
  "  • Set OPENAI_API_KEY / ANTHROPIC_API_KEY / GEMINI_API_KEY in your environment\n"
  "  • Add your own RoutingPolicy subclasses to customise the routing chain\n"
  "  • Run the test suite:  pytest tests/ -v\n"
)
