#!/usr/bin/env python3
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
# ### 1a. Understand the score scale used in this tutorial
#
# With `granularity=20`, the verifier-style scoring scale uses the letters
# `A` through `T`.
#
# - `A` is the best score
# - `T` is the worst score
# - intermediate letters represent progressively weaker evaluations
#
# A rough interpretation is:
#
# - `A-D`: strong success
# - `E-G`: mostly correct, with some issues
# - `H-J`: uncertain, but leaning toward success
# - `K-M`: uncertain, but leaning toward failure
# - `N-P`: significant issues remain
# - `Q-T`: failure
#
# Internally, these letters are ordinal score tokens. The strategies map them
# onto a numeric scale and then normalize the result to the `[0, 1]` range for
# reporting.

# %% [markdown]
# ### 1b. Create a deterministic stand-in for a real model
#
# `StubProvider` exercises the full API without network calls or API keys.
# `responses` is consumed in order, making the tutorial deterministic. Setting
# all three score fields on each `StubResponse` lets it handle both
# single-trajectory and pairwise prompts — the provider picks the right field
# at call time. Unset fields fall back to the `default_score*` values.

# %% [markdown]
# ### 1c. Define the shared scoring configuration and rubric
#
# `ScoringConfig` is shared by both strategies. `criteria` is the rubric each
# evaluator scores against — each entry has an `id`, `name`, `description`,
# and `weight`.
#
# > **Note:** `use_logprobs=True` tells the Verifier to extract token-level log
# > probabilities from the model response — the mechanism it uses to achieve
# > finer score discrimination between close candidates.

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
# Three trajectories give the evaluators something to rank:
#
# - `traj-correct`: proper binary search
# - `traj-linear`: correct output, wrong algorithm
# - `traj-buggy`: boundary-condition bug

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
# `score_trajectory` scores a single candidate against one criterion.
# The result includes `score`, `confidence`, and `criterion_scores`.

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
# `compare_trajectories` does a pairwise comparison on one criterion and returns
# separate scores for A and B, a winner, and a confidence value.

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
# `evaluate` returns an `EvaluationResult` with `best_trajectory_id`,
# `strategy_type`, `metadata`, and a per-trajectory `trajectory_scores` table.

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
# Judge-specific options: `score_min`/`score_max` set the numeric range,
# `swap_pairwise` reduces position bias, `reasoning_depth` controls output
# verbosity.

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
# Same call as the Verifier, different evaluator. Comparing both outputs
# side by side shows where they agree or diverge before we let the router
# choose between them.

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
# `router.route(...)` decides which strategy to use without running evaluation.
# The result includes `selected_strategy`, `confidence`, `elapsed_ms`,
# `signals`, and `reasoning`.

# %%
router = OracleRouter.default(verifier, judge)

decision = router.route(task, trajectories)
print("Routing decision:")
print("  selected strategy:", decision.selected_strategy.value)
print("  confidence:", round(decision.confidence, 4))
print("  elapsed_ms:", round(decision.elapsed_ms, 3))
print("  signals:", decision.signals)
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
# `run_single` runs both Judge and Verifier and returns a `TaskHardnessRecord`
# with comparison metrics: spread, disagreement, confidence, and oracle gap.

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
# `run(...)` is the batch version. It returns the same report object whether you
# pass one task or many. `summary()` gives a high-level overview;
# `per_task_table()` gives a per-task breakdown.

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
# `essay_task` has no ground truth or test cases, so the Verifier has little
# evidence to work with and the Judge becomes the natural fit. Two trajectories
# — one specific, one vague — show the router adapting to task shape.

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
