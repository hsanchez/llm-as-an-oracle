#!/usr/bin/env python3
"""Jupytext-style tutorial for LLM as an Oracle.

Run as a script:
    uv run python examples/llm_as_an_oracle_tutorial.py

Or open in a notebook environment that understands ``# %%`` cells.

Prerequisite:
    uv sync

For cell-by-cell execution, make sure your notebook or editor kernel is using
this project's ``.venv`` so ``from llm_oracle import ...`` resolves normally.
"""

# %% [markdown]
# # LLM as an Oracle: Tutorial
#
# This tutorial shows how to use the public API in `llm_oracle` to:
#
# - define a task and candidate trajectories
# - evaluate them with `LLM-as-a-Verifier`
# - evaluate them with `LLM-as-a-Judge`
# - route between them with the `Oracle` layer
# - compare both strategies with the evaluation harness
#
# The tutorial uses `StubProvider`, so it runs offline and does not require API
# keys. You can later swap it for `OpenAIProvider`, `AnthropicProvider`, or
# `GeminiProvider`.

# %% [markdown]
# ## 0. Import the public API
#
# This tutorial imports everything from the top-level `llm_oracle` package
# rather than from internal modules. That is the intended user-facing API.
#
# The imported names fall into a few groups:
#
# - data models such as `Task`, `Trajectory`, and `EvaluationCriterion`
# - evaluator implementations such as `VerifierStrategy` and `JudgeStrategy`
# - orchestration and analysis tools such as `OracleRouter` and
#   `EvaluationHarness`
# - the offline demo model `StubProvider`

# %%
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

# %% [markdown]
# ## 1. Build a shared model, scoring config, and evaluation criteria
#
# In this library, both the Judge and the Verifier consume:
#
# - a language model provider
# - a scoring configuration
# - a list of evaluation criteria
#
# For a tutorial, `StubProvider` is ideal because it exercises the API without
# requiring any external service.

# %% [markdown]
# ### 1a. Create a deterministic stand-in for a real model
#
# `StubProvider` is a fake language-model provider used for demos and tests.
# It lets us exercise the full Judge/Verifier/Router API without making network
# calls or requiring API keys.
#
# In this example:
#
# - `model_id` is just a display name so we can identify the provider in logs
#   and printed output.
# - `default_score` is the fallback single-score grade returned when a strategy
#   asks for one score and the scripted responses run out.
# - `default_score_a` and `default_score_b` are fallback pairwise grades for
#   comparisons between two candidate trajectories.
# - `responses` is a scripted sequence of `StubResponse` objects. Each call to
#   the provider consumes the next response, which makes the tutorial
#   deterministic and repeatable.
# - `seed` fixes any randomized behavior inside the stub so the demo is stable.
#
# The exact letter grades are not important here. What matters is that the
# tutorial has predictable model outputs, so we can focus on how the library is
# wired together.

# %% [markdown]
# ### 1b. Define the shared scoring configuration and rubric
#
# After creating the stub model, we define the evaluation settings that both
# strategies will share.
#
# - `ScoringConfig` controls how fine-grained and how repeated the evaluation
#   process should be.
# - `granularity=20` means the internal scoring scale is fairly fine rather than
#   extremely coarse.
# - `num_verifications=2` tells the verifier-oriented path to repeat parts of
#   its evaluation more than once.
# - `num_criteria=3` matches the three rubric dimensions we define below.
# - `use_logprobs=True` enables confidence-aware behavior when a provider
#   supports log probabilities. With the stub, this mainly keeps the example
#   aligned with the real API.
#
# The `criteria` list is the rubric shared by Judge and Verifier. Each
# `EvaluationCriterion` has:
#
# - an internal `id`
# - a human-readable `name`
# - a natural-language `description`
# - a `weight` that says how important that criterion is
#
# We print a short summary at the end so the audience can confirm the setup
# before moving on.

# %%
stub = StubProvider(
  model_id="tutorial-stub",
  default_score="C",
  default_score_a="C",
  default_score_b="H",
  responses=[
    StubResponse(score="B", score_a="B", score_b="G"),
    StubResponse(score="C", score_a="C", score_b="H"),
    StubResponse(score="A", score_a="A", score_b="F"),
    StubResponse(score="D", score_a="B", score_b="J"),
  ],
  seed=7,
)

config = ScoringConfig(
  granularity=20,
  num_verifications=2,
  num_criteria=3,
  use_logprobs=True,
)

criteria = [
  EvaluationCriterion(
    id="correctness",
    name="Correctness",
    description="Does the trajectory satisfy the task requirements and produce the correct result?",
    weight=2.0,
  ),
  EvaluationCriterion(
    id="efficiency",
    name="Efficiency",
    description="Is the solution computationally reasonable for the problem being solved?",
    weight=1.0,
  ),
  EvaluationCriterion(
    id="clarity",
    name="Clarity",
    description="Is the solution easy to read, explain, and maintain?",
    weight=1.0,
  ),
]

print("Model:", stub.model_id)
print("Granularity:", config.granularity)
print("Criteria:", [criterion.name for criterion in criteria])


# %% [markdown]
# ## 2. Define a task and a few candidate trajectories
#
# A `Task` captures the problem statement and optional evidence such as:
#
# - `ground_truth`
# - `test_cases`
# - `difficulty`
#
# A `Trajectory` is a candidate task-solving attempt. It may contain:
#
# - the main solution content
# - execution output
# - an optional reward signal

# %% [markdown]
# ### 2a. Build one verifiable task and three candidate solutions
#
# This cell creates the concrete data that the rest of the tutorial evaluates.
#
# The `Task` represents a binary-search programming problem. It includes:
#
# - `description` and `problem_statement` so the evaluator knows the task
# - `ground_truth`, which gives the verifier a strong reference solution
# - `test_cases`, which provide executable-style evidence
# - `difficulty`, which the router may use as a signal
#
# Then we create three `Trajectory` objects:
#
# - `traj-correct`: a proper binary-search implementation
# - `traj-linear`: correct behavior, but the wrong algorithmic idea
# - `traj-buggy`: an implementation with a boundary-condition bug
#
# This mix is intentional. It gives the Judge and Verifier something meaningful
# to separate: correctness, efficiency, and failure cases.

# %%
task = Task(
  id="binary-search",
  description="Implement binary search over a sorted integer array.",
  problem_statement=(
    "Write a function `binary_search(arr, target)` that returns the index of "
    "the target in a sorted array, or -1 if the target is not present."
  ),
  ground_truth=(
    "def binary_search(arr, target):\n"
    "    lo, hi = 0, len(arr) - 1\n"
    "    while lo <= hi:\n"
    "        mid = (lo + hi) // 2\n"
    "        if arr[mid] == target:\n"
    "            return mid\n"
    "        if arr[mid] < target:\n"
    "            lo = mid + 1\n"
    "        else:\n"
    "            hi = mid - 1\n"
    "    return -1"
  ),
  test_cases=[
    {"input": {"arr": [1, 3, 5, 7], "target": 5}, "expected": 2},
    {"input": {"arr": [1, 3, 5, 7], "target": 6}, "expected": -1},
  ],
  difficulty=TaskDifficulty.MEDIUM,
)

trajectories = [
  Trajectory(
    id="traj-correct",
    task_id=task.id,
    content=(
      "def binary_search(arr, target):\n"
      "    lo, hi = 0, len(arr) - 1\n"
      "    while lo <= hi:\n"
      "        mid = (lo + hi) // 2\n"
      "        if arr[mid] == target:\n"
      "            return mid\n"
      "        elif arr[mid] < target:\n"
      "            lo = mid + 1\n"
      "        else:\n"
      "            hi = mid - 1\n"
      "    return -1"
    ),
    output="[1,3,5,7],5 -> 2 | [1,3,5,7],6 -> -1",
    reward=1.0,
  ),
  Trajectory(
    id="traj-linear",
    task_id=task.id,
    content=(
      "def binary_search(arr, target):\n"
      "    for i, value in enumerate(arr):\n"
      "        if value == target:\n"
      "            return i\n"
      "    return -1"
    ),
    output="[1,3,5,7],5 -> 2 | [1,3,5,7],6 -> -1",
    reward=0.7,
  ),
  Trajectory(
    id="traj-buggy",
    task_id=task.id,
    content=(
      "def binary_search(arr, target):\n"
      "    lo, hi = 0, len(arr) - 1\n"
      "    while lo < hi:\n"
      "        mid = (lo + hi) // 2\n"
      "        if arr[mid] < target:\n"
      "            lo = mid + 1\n"
      "        else:\n"
      "            hi = mid\n"
      "    return lo"
    ),
    output="[1,3,5,7],5 -> 2 | [1,3,5,7],6 -> 3  # wrong",
    reward=0.0,
  ),
]

print("Task:", task.id)
print("Trajectories:", [trajectory.id for trajectory in trajectories])


# %% [markdown]
# ## 3. Use the Verifier directly
#
# The Verifier is the better fit when evaluation can use stronger evidence:
#
# - test cases
# - reference solutions
# - expected outputs
# - execution results
#
# It can score a single trajectory, compare two trajectories, or evaluate a
# whole candidate set.

# %% [markdown]
# ### 3a. Instantiate the Verifier and score one trajectory
#
# Here we build a `VerifierStrategy` using the shared `stub`, `config`, and
# `criteria`.
#
# Then we call `score_trajectory(...)` on just one candidate and one criterion:
# `criteria[0]`, which is `Correctness`.
#
# This is the smallest useful verifier example. It answers:
# "How does the verifier score this one trajectory on this one rubric
# dimension?"
#
# In the printed output, pay attention to:
#
# - `score`: the normalized numeric grade
# - `confidence`: how certain the evaluator is
# - `criterion_scores`: the breakdown by rubric criterion

# %%
verifier = VerifierStrategy(stub, config, criteria)

single_verifier_score = verifier.score_trajectory(task, trajectories[0], criteria[0])
print("Single verifier score:")
print("  trajectory:", single_verifier_score.trajectory_id)
print("  score:", round(single_verifier_score.score, 4))
print("  confidence:", round(single_verifier_score.confidence, 4))
print("  criterion_scores:", single_verifier_score.criterion_scores)


# %% [markdown]
# ### 3b. Compare two trajectories head-to-head with the Verifier
#
# Scoring one trajectory is useful, but many ranking problems are really about
# choosing between candidates.
#
# This cell compares:
#
# - trajectory A: the correct binary-search solution
# - trajectory B: the buggy solution
#
# We again focus on the `Correctness` criterion to keep the example simple.
#
# The result reports separate scores for A and B, a winner, and a confidence
# value. This is a good cell to pause on during the demo because it shows the
# library can do pairwise preference judgments, not just absolute scoring.

# %%
pairwise_verifier = verifier.compare_trajectories(
  task,
  trajectories[0],
  trajectories[2],
  criteria[0],
)

print("Verifier pairwise comparison:")
print("  A:", pairwise_verifier.trajectory_a_id, "score:", round(pairwise_verifier.score_a, 4))
print("  B:", pairwise_verifier.trajectory_b_id, "score:", round(pairwise_verifier.score_b, 4))
print("  winner:", pairwise_verifier.winner)
print("  confidence:", round(pairwise_verifier.confidence, 4))


# %% [markdown]
# ### 3c. Evaluate the full candidate set with the Verifier
#
# This cell runs the Verifier over the full list of trajectories and produces an
# `EvaluationResult`.
#
# That object includes:
#
# - `best_trajectory_id`: the overall winner
# - `strategy_type`: which evaluator produced the result
# - `metadata`: additional diagnostic information
# - `trajectory_scores`: the per-trajectory score table
#
# The loop at the end sorts candidates from best to worst so the ranking is easy
# to read live.

# %%
verifier_result = verifier.evaluate(task, trajectories)

print("Verifier full evaluation:")
print("  best trajectory:", verifier_result.best_trajectory_id)
print("  strategy:", verifier_result.strategy_type.value)
print("  metadata:", verifier_result.metadata)
for trajectory_id, score_result in sorted(
  verifier_result.trajectory_scores.items(),
  key=lambda item: -item[1].score,
):
  print(
    f"  {trajectory_id}: score={score_result.score:.4f} confidence={score_result.confidence:.4f}"
  )


# %% [markdown]
# ## 4. Use the Judge directly
#
# The Judge is useful when evaluation is more holistic or open-ended. Even in
# structured tasks, it can still provide a useful baseline.

# %% [markdown]
# ### 4a. Instantiate the Judge and score one trajectory
#
# The `JudgeStrategy` uses the same task, trajectories, and rubric, but its
# style is more holistic than the Verifier's evidence-heavy approach.
#
# A few Judge-specific options appear here:
#
# - `score_min` and `score_max` define the scoring range
# - `swap_pairwise=True` allows the implementation to compare candidates in both
#   orders to reduce position bias
# - `reasoning_depth="detailed"` asks for richer judge reasoning
#
# As with the Verifier, we start with a single-trajectory score before moving to
# a full ranking.

# %%
judge = JudgeStrategy(
  stub,
  config,
  criteria,
  score_min=1.0,
  score_max=10.0,
  swap_pairwise=True,
  reasoning_depth="detailed",
)

single_judge_score = judge.score_trajectory(task, trajectories[0], criteria[0])
print("Single judge score:")
print("  trajectory:", single_judge_score.trajectory_id)
print("  score:", round(single_judge_score.score, 4))
print("  confidence:", round(single_judge_score.confidence, 4))
print("  criterion_scores:", single_judge_score.criterion_scores)


# %% [markdown]
# ### 4b. Rank the full candidate set with the Judge
#
# This mirrors the earlier Verifier evaluation, but now the Judge is producing
# the ranking.
#
# Running both paths on the same task is useful because it makes the conceptual
# difference concrete:
#
# - the Verifier leans on evidence such as tests and reference answers
# - the Judge applies a more holistic scoring process
#
# Comparing their outputs side by side prepares us for the Oracle router, whose
# job is to choose between these strategies automatically.

# %%
judge_result = judge.evaluate(task, trajectories)

print("Judge full evaluation:")
print("  best trajectory:", judge_result.best_trajectory_id)
print("  strategy:", judge_result.strategy_type.value)
print("  metadata:", judge_result.metadata)
for trajectory_id, score_result in sorted(
  judge_result.trajectory_scores.items(),
  key=lambda item: -item[1].score,
):
  print(
    f"  {trajectory_id}: score={score_result.score:.4f} confidence={score_result.confidence:.4f}"
  )


# %% [markdown]
# ## 5. Use the Oracle router
#
# This is the main idea of the project.
#
# Rather than deciding manually whether a task should be evaluated by the Judge
# or the Verifier, we construct an `OracleRouter` and let it route the task
# based on task signals such as:
#
# - ground truth
# - test cases
# - difficulty
# - execution output
# - trajectory count

# %% [markdown]
# ### 5a. Build the default Oracle router and inspect its routing decision
#
# `OracleRouter.default(...)` creates a router with the library's built-in
# routing policy chain.
#
# The first call, `router.route(...)`, does not yet run the final evaluation.
# Instead, it answers:
# "Given this task and these trajectories, which strategy should we use?"
#
# The printed fields are worth explaining during the demo:
#
# - `selected_strategy`: Judge or Verifier
# - `confidence`: how strongly the router prefers that choice
# - `elapsed_ms`: routing overhead
# - `features`: the extracted signals used by the policy chain
# - `reasoning`: a textual explanation of the decision

# %%
router = OracleRouter.default(verifier, judge)

decision = router.route(task, trajectories)
print("Routing decision:")
print("  selected strategy:", decision.selected_strategy.value)
print("  confidence:", round(decision.confidence, 4))
print("  elapsed_ms:", round(decision.elapsed_ms, 3))
print("  features:", decision.features)
print("  reasoning:")
print(decision.reasoning)


# %% [markdown]
# ### 5b. Let the Oracle route and evaluate in one step
#
# `router.evaluate(...)` is the high-level convenience method.
#
# It first decides which evaluator to use, then delegates the actual scoring to
# that evaluator, and finally returns both:
#
# - `oracle_result`: the evaluation output
# - `oracle_decision`: the routing decision that led to it
#
# This is the most compact end-to-end API in the project.

# %%
oracle_result, oracle_decision = router.evaluate(task, trajectories)

print("Oracle evaluation:")
print("  selected strategy:", oracle_decision.selected_strategy.value)
print("  best trajectory:", oracle_result.best_trajectory_id)
print("  strategy used:", oracle_result.strategy_type.value)


# %% [markdown]
# ## 6. Compare Judge and Verifier with the evaluation harness
#
# The `EvaluationHarness` runs both strategies and computes comparison signals
# such as:
#
# - score spread
# - strategy disagreement
# - average confidence
# - oracle gap
# - composite hardness
#
# This is useful for benchmarking and analysis.

# %% [markdown]
# ### 6a. Run the evaluation harness on one task
#
# The `EvaluationHarness` is an analysis tool rather than a single evaluator.
# It runs both Judge and Verifier, then computes comparison metrics that help
# you understand how hard the task is and how much the strategies disagree.
#
# In this cell:
#
# - `max_workers=2` allows the harness to use limited concurrency when needed
# - `run_single(...)` produces one `TaskHardnessRecord`
#
# The printed metrics summarize the task from an evaluation perspective, not
# from the perspective of solving the original binary-search problem.

# %%
harness = EvaluationHarness(verifier=verifier, judge=judge, max_workers=2)

record = harness.run_single(task, trajectories)
print("Harness record:")
print("  task_id:", record.task_id)
print("  hardness_score:", round(record.hardness_score, 4))
print("  score_spread:", round(record.score_spread, 4))
print("  strategy_disagreement:", round(record.strategy_disagreement, 4))
print("  avg_confidence:", round(record.avg_confidence, 4))
print("  oracle_gap_verifier:", round(record.oracle_gap_verifier, 4))
print("  oracle_gap_judge:", round(record.oracle_gap_judge, 4))


# %% [markdown]
# ### 6b. Generate a small harness report
#
# `run(...)` is the batch version of the harness API. Even though we only pass
# one task here, it returns the same kind of report object you would use for a
# larger benchmark suite.
#
# We print:
#
# - `summary()`: a high-level textual overview
# - `per_task_table()`: a compact table for each task in the run
#
# This cell is useful for showing how the project scales from one-off demos to
# systematic evaluator comparison.

# %%
report = harness.run([(task, trajectories)], parallel=False)

print(report.summary())
print(report.per_task_table())


# %% [markdown]
# ## 7. A second task: when the Judge is more natural
#
# The Oracle becomes more useful when your workload mixes task types.
#
# Here is an open-ended task where holistic evaluation is more natural than
# evidence-based verification.

# %% [markdown]
# ### 7a. Create an open-ended task and route it
#
# Up to now, the tutorial used a strongly verifiable programming problem. This
# cell changes the task type.
#
# `essay_task` has no ground truth, no test cases, and no execution outputs.
# That means the Verifier has less concrete evidence to work with, while the
# Judge becomes a more natural fit.
#
# We create two essay trajectories:
#
# - `essay-a`: specific and comparative
# - `essay-b`: vague and shallow
#
# Then we ask the router to choose a strategy and evaluate the pair. This cell
# demonstrates the main promise of the Oracle layer: different task shapes can
# trigger different evaluators.

# %%
essay_task = Task(
  id="essay-compare-sorts",
  description="Compare merge sort and quicksort.",
  problem_statement=(
    "Write a concise essay comparing merge sort and quicksort, including time "
    "complexity, space complexity, and practical trade-offs."
  ),
  difficulty=TaskDifficulty.EASY,
)

essay_trajectories = [
  Trajectory(
    id="essay-a",
    task_id=essay_task.id,
    content=(
      "Merge sort guarantees O(n log n) time and is stable, but it uses extra "
      "memory. Quicksort is often faster in practice because of cache locality, "
      "but its worst case is O(n^2) unless implemented carefully."
    ),
  ),
  Trajectory(
    id="essay-b",
    task_id=essay_task.id,
    content=(
      "Merge sort and quicksort are both fast. Merge sort is recursive. "
      "Quicksort is also recursive. Both are useful."
    ),
  ),
]

essay_decision = router.route(essay_task, essay_trajectories)
essay_result, _ = router.evaluate(essay_task, essay_trajectories)

print("Essay task routing:")
print("  selected strategy:", essay_decision.selected_strategy.value)
print("  confidence:", round(essay_decision.confidence, 4))
print("  best trajectory:", essay_result.best_trajectory_id)


# %% [markdown]
# ## 8. Practical guidance
#
# Use `VerifierStrategy` directly when:
#
# - you already know the task is verifiable
# - you want explicit control over evidence-based evaluation
#
# Use `JudgeStrategy` directly when:
#
# - the task is open-ended
# - you want a holistic evaluator
#
# Use `OracleRouter` when:
#
# - your tasks vary in type
# - you want adaptive evaluator selection
# - you want routing decisions to be inspectable
#
# Use `EvaluationHarness` when:
#
# - you are benchmarking Judge vs Verifier
# - you want to measure evaluation hardness
# - you want to study when each strategy works better


# %% [markdown]
# ## 9. Swapping the model provider
#
# This tutorial used `StubProvider`, but the API is designed so you can swap in
# a real backend with the same strategy objects.
#
# Example sketch:
#
# ```python
# from llm_oracle import OpenAIProvider
#
# model = OpenAIProvider(model_id="gpt-4o")
# verifier = VerifierStrategy(model, config, criteria)
# judge = JudgeStrategy(model, config, criteria)
# router = OracleRouter.default(verifier, judge)
# ```
#
# The rest of the workflow stays the same.


# %% [markdown]
# ## 10. Summary
#
# You have now seen the main API layers:
#
# - `Task` and `Trajectory`
# - `VerifierStrategy`
# - `JudgeStrategy`
# - `OracleRouter`
# - `EvaluationHarness`
#
# The core idea is simple:
#
# - Judge and Verifier are different evaluators
# - Oracle is the orchestration layer that selects between them
