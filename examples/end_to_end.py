#!/usr/bin/env python3
"""End-to-end example: LLM Oracle — Verifier, Judge, Harness, Router.

Runs completely offline using StubProvider (no API keys needed).
Swap StubProvider for OpenAIProvider, AnthropicProvider, or GeminiProvider
and set the corresponding API key environment variable when you are ready
to use a real model.

    cd llm-as-oracle
    uv run python examples/end_to_end.py
"""

from __future__ import annotations

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
from llm_oracle.routing.router import PolicyVote, RoutingPolicy, RoutingSignals, SignalExtractor

# ---------------------------------------------------------------------------
# 1. Setup — shared model, config, and criteria
# ---------------------------------------------------------------------------

# StubProvider mimics real model responses without any network call.
# Replace with OpenAIProvider("gpt-4o") or AnthropicProvider("claude-opus-4-5")
# once you have an API key.
model = StubProvider(
  model_id="stub-oracle",
  default_score="D",
  default_score_a="D",
  default_score_b="H",
  responses=[
    StubResponse(score="B", score_a="B", score_b="G"),
    StubResponse(score="D", score_a="D", score_b="H"),
    StubResponse(score="A", score_a="A", score_b="F"),
    StubResponse(score="E", score_a="C", score_b="J"),
  ],
  seed=42,
)

config = ScoringConfig(
  granularity=20,  # A–T scoring scale (LLM-as-a-Verifier paper)
  num_verifications=2,  # K=2 repeated verifications per criterion
  use_logprobs=True,  # extract score from logprob distribution
)

# Criteria are fully caller-defined — use any domain, not just code.
criteria = [
  EvaluationCriterion(
    id="correctness",
    name="Correctness",
    description="Does the solution produce the correct output for all inputs?",
    weight=2.0,
  ),
  EvaluationCriterion(
    id="efficiency",
    name="Efficiency",
    description="Is the time and space complexity optimal for this problem class?",
    weight=1.5,
  ),
  EvaluationCriterion(
    id="readability",
    name="Readability",
    description="Is the code clean and easy to understand without additional context?",
    weight=1.0,
  ),
]


# ---------------------------------------------------------------------------
# 2. LLM-as-a-Verifier
# ---------------------------------------------------------------------------

verifier = VerifierStrategy(model, config, criteria)

task = Task(
  id="sort-integers",
  description="Sort a list of integers in ascending order",
  problem_statement=(
    "Implement `sort_list(lst: list[int]) -> list[int]` that returns "
    "a new sorted list without modifying the input."
  ),
  ground_truth="def sort_list(lst): return sorted(lst)",
  test_cases=[
    {"input": [3, 1, 2], "expected": [1, 2, 3]},
    {"input": [], "expected": []},
    {"input": [-1, 0, -5], "expected": [-5, -1, 0]},
  ],
  difficulty=TaskDifficulty.EASY,
)

trajectories = [
  Trajectory(
    id="traj-optimal",
    task_id=task.id,
    content="def sort_list(lst: list[int]) -> list[int]:\n    return sorted(lst)",
    output="[1, 2, 3] | [] | [-5, -1, 0]",
    reward=1.0,
  ),
  Trajectory(
    id="traj-inplace",
    task_id=task.id,
    content=(
      "def sort_list(lst: list[int]) -> list[int]:\n"
      "    lst.sort()  # mutates input — violates spec\n"
      "    return lst"
    ),
    reward=0.5,
  ),
  Trajectory(
    id="traj-bubble",
    task_id=task.id,
    content=(
      "def sort_list(lst):\n"
      "    n = len(lst)\n"
      "    for i in range(n):\n"
      "        for j in range(n - i - 1):\n"
      "            if lst[j] > lst[j + 1]:\n"
      "                lst[j], lst[j + 1] = lst[j + 1], lst[j]\n"
      "    return lst"
    ),
    reward=0.0,
  ),
]

print("\n# Verifier — score a single trajectory against one criterion")
score = verifier.score_trajectory(task, trajectories[0], criteria[0])
print(f"  trajectory : {score.trajectory_id}")
print(f"  score      : {score.score:.4f}  (normalised 0-1)")
print(f"  confidence : {score.confidence:.4f}")

print("\n# Verifier — pairwise comparison of two trajectories")
comp = verifier.compare_trajectories(task, trajectories[0], trajectories[2], criteria[0])
print(f"  score_A ({comp.trajectory_a_id}) : {comp.score_a:.4f}")
print(f"  score_B ({comp.trajectory_b_id}) : {comp.score_b:.4f}")
print(f"  winner    : {comp.winner or 'tie'}")
print(f"  confidence: {comp.confidence:.4f}")

print("\n# Verifier — tournament evaluation across all trajectories")
v_result = verifier.evaluate(task, trajectories)
print(f"  best trajectory : {v_result.best_trajectory_id}")
print(f"  pairwise calls  : {len(v_result.pairwise_comparisons)}")
for tid, sr in sorted(v_result.trajectory_scores.items(), key=lambda x: -x[1].score):
  print(f"  {tid:<16s}  score={sr.score:.4f}")


# ---------------------------------------------------------------------------
# 3. LLM-as-a-Judge
# ---------------------------------------------------------------------------

judge = JudgeStrategy(
  model,
  config,
  criteria,
  score_min=1.0,
  score_max=10.0,
  swap_pairwise=True,  # runs each pairwise twice to cancel positional bias
  reasoning_depth="detailed",  # "brief" | "detailed" | "chain_of_thought"
)

print("\n# Judge — pointwise score for one trajectory and criterion")
j_score = judge.score_trajectory(task, trajectories[0], criteria[1])
print(f"  trajectory : {j_score.trajectory_id}")
print(f"  score      : {j_score.score:.4f}  (raw: {j_score.raw_score})")
print(f"  confidence : {j_score.confidence:.4f}")

print("\n# Judge — pairwise comparison (swap_pairwise cancels positional bias)")
j_comp = judge.compare_trajectories(task, trajectories[0], trajectories[1], criteria[0])
print(f"  score_A ({j_comp.trajectory_a_id}) : {j_comp.score_a:.4f}")
print(f"  score_B ({j_comp.trajectory_b_id}) : {j_comp.score_b:.4f}")
print(f"  winner    : {j_comp.winner or 'tie'}")

print("\n# Judge — full evaluation across all trajectories")
j_result = judge.evaluate(task, trajectories)
print(f"  best trajectory : {j_result.best_trajectory_id}")
for tid, sr in sorted(j_result.trajectory_scores.items(), key=lambda x: -x[1].score):
  print(f"  {tid:<16s}  score={sr.score:.4f}")


# ---------------------------------------------------------------------------
# 4. Evaluation Harness — run verifier and judge side-by-side
# ---------------------------------------------------------------------------

search_task = Task(
  id="binary-search",
  description="Binary search in a sorted array",
  problem_statement=(
    "Implement `binary_search(arr: list[int], target: int) -> int` "
    "returning the index of target or -1. Must run in O(log n)."
  ),
  test_cases=[
    {"arr": [1, 3, 5, 7, 9], "target": 5, "expected": 2},
    {"arr": [1, 3, 5, 7, 9], "target": 6, "expected": -1},
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
    "    return -1",
    reward=0.0,
  ),
]

essay_task = Task(
  id="raft-essay",
  description="Explain the Raft consensus algorithm",
  problem_statement=(
    "Write a technical essay explaining Raft consensus: leader election, "
    "log replication, safety guarantees, and a brief Paxos comparison."
  ),
  difficulty=TaskDifficulty.HARD,
)
essay_trajs = [
  Trajectory(
    "essay-good",
    essay_task.id,
    "Raft is a consensus algorithm designed for understandability ... "
    "[covers all required topics with clarity and depth]",
    reward=1.0,
  ),
  Trajectory(
    "essay-poor",
    essay_task.id,
    "Raft is a thing that makes servers agree. Paxos is also a thing.",
    reward=0.0,
  ),
]

tasks_and_trajectories = [
  (task, trajectories),
  (search_task, search_trajs),
  (essay_task, essay_trajs),
]

harness = EvaluationHarness(
  verifier=verifier,
  judge=judge,
  max_workers=2,
  hard_task_threshold=0.50,
)

print("\n# Harness — compare verifier and judge across multiple tasks")
report = harness.run(tasks_and_trajectories, parallel=False)
print(report.summary())
print(report.per_task_table())

print(f"  Hard tasks ({len(report.hard_tasks)})")
print(f"    verifier accuracy : {report.verifier_accuracy_on_hard():.1%}")
print(f"    judge    accuracy : {report.judge_accuracy_on_hard():.1%}")
print(f"  Easy tasks ({len(report.easy_tasks)})")
print(f"    verifier accuracy : {report.verifier_accuracy_on_easy():.1%}")
print(f"    judge    accuracy : {report.judge_accuracy_on_easy():.1%}")


# ---------------------------------------------------------------------------
# 5. OracleRouter — automatic strategy selection
# ---------------------------------------------------------------------------

router = OracleRouter.default(verifier, judge, confidence_threshold=0.50)

print("\n# Router — route each task (no prior hardness)")
for t, trajs in tasks_and_trajectories:
  decision = router.route(t, trajs)
  print(
    f"  {t.id:<22s}  -> {decision.selected_strategy.value:<9s}"
    f"  conf={decision.confidence:.3f}  ({decision.elapsed_ms:.1f} ms)"
  )

# Feed harness hardness scores back so the router learns from real results.
for record in report.task_records:
  router.update_hardness(record.task_id, record.hardness_score)

print("\n# Router — same tasks re-routed with prior hardness from the harness")
for t, trajs in tasks_and_trajectories:
  decision = router.route(t, trajs)
  ph = decision.signals.prior_hardness
  print(
    f"  {t.id:<22s}  -> {decision.selected_strategy.value:<9s}"
    f"  conf={decision.confidence:.3f}  prior_hardness={ph:.3f}"
  )


# Custom policy — extend the router with your own routing logic.
class ShortProblemPolicy(RoutingPolicy):
  """Prefer judge for very short problem statements."""

  name = "short_problem"
  weight = 0.7

  def vote(self, task, trajectories, signals: RoutingSignals) -> PolicyVote:
    is_short = signals.problem_length < (150 / 2000.0)
    return PolicyVote(
      policy_name=self.name,
      preferred=StrategyType.JUDGE if is_short else StrategyType.VERIFIER,
      confidence=0.63 if is_short else 0.53,
      weight=self.weight,
      signals_used=["problem_length"],
      reasoning=(
        "Short problem: judge is sufficient." if is_short else "Long problem: verifier preferred."
      ),
    )


router.register_policy(ShortProblemPolicy())

print("\n# Router — route-and-evaluate in one call")
result, decision = router.evaluate(task, trajectories)
print(f"  strategy   : {decision.selected_strategy.value}")
print(f"  confidence : {decision.confidence:.3f}")
print(f"  best traj  : {result.best_trajectory_id}")


# ---------------------------------------------------------------------------
# 6. Signal introspection — inspect what the router sees for any task
# ---------------------------------------------------------------------------

print("\n# SignalExtractor — routing signals for each task")
extractor = SignalExtractor()
for t, trajs in tasks_and_trajectories:
  signals = extractor.extract(t, trajs)
  print(f"  {t.id}")
  print(
    f"    has_ground_truth={signals.has_ground_truth:.0f}"
    f"  has_test_cases={signals.has_test_cases:.0f}"
    f"  trajectory_count={signals.trajectory_count}"
  )
  print(
    f"    verifiable_kw_density={signals.verifiable_keyword_density:.3f}"
    f"  judgement_kw_density={signals.judgement_keyword_density:.3f}"
  )

print("\n# Router audit log")
print(router.routing_summary())

print(
  "\nNext steps:\n"
  "  Swap StubProvider for OpenAIProvider / AnthropicProvider / GeminiProvider\n"
  "  Set OPENAI_API_KEY / ANTHROPIC_API_KEY / GEMINI_API_KEY in your environment\n"
  "  Define EvaluationCriterion objects for your domain (not just code tasks)\n"
  "  Add RoutingPolicy subclasses to extend the routing chain\n"
  "  Run the test suite: pytest tests/ -v"
)
