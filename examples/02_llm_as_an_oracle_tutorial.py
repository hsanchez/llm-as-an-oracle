#!/usr/bin/env python3

# %% [markdown]
# # LLM as an Oracle: Real-World Tutorial with Anthropic Claude
#
# This tutorial uses `AnthropicProvider` (claude-sonnet-4-6) to demonstrate
# why the Oracle approach is useful in practice.
#
# The key difference from the stub tutorial:
#
# - Trajectories are **close in quality** — all three agents produced plausible
#   fixes, but one has a latent bug that only surfaces on edge cases.
# - The **routing decision is non-trivial** — the task has test cases (a
#   Verifier signal) but subtle quality differences (a Judge signal). Both
#   evaluators add information.
# - A second, open-ended task shows the router switching to the Judge
#   automatically when there is no structured evidence to work with.
#
# This is closer to real Oracle usage: evaluating competing AI agent outputs
# where the correct answer is not immediately obvious.
#
# Prerequisites:
#
# - `ANTHROPIC_API_KEY` must be set in your environment.
# - Run `uv sync` to install dependencies.
#
# Cost note: `num_verifications=1` and two criteria are used to keep token
# usage minimal. Increase them for more rigorous evaluation.

# %% [markdown]
# ## 0. Import the public API

# %%
from __future__ import annotations

from dotenv import load_dotenv

from llm_oracle import (
  AnthropicProvider,
  EvaluationCriterion,
  EvaluationHarness,
  JudgeStrategy,
  OracleRouter,
  ScoringConfig,
  Task,
  TaskDifficulty,
  Trajectory,
  VerifierStrategy,
)

load_dotenv()  # reads .env from the repo root (or any parent directory)


# %% [markdown]
# ## 1. Provider, scoring config, and evaluation criteria
#
# `AnthropicProvider` reads `ANTHROPIC_API_KEY` from the environment.
#
# Anthropic does not expose token-level log probabilities, so the Verifier
# falls back to parsing its letter score from the model's text output.
# Set `use_logprobs=False` to use that path directly.

# %%
model = AnthropicProvider(model_id="claude-sonnet-4-6")

# granularity=20 → 20-level A–T letter scale for the Verifier.
# num_verifications=1 → one scoring pass per criterion (keep costs low).
# use_logprobs=False → required for Anthropic; score extracted from text.
config = ScoringConfig(
  granularity=20,
  num_verifications=1,
  num_criteria=2,
  use_logprobs=False,
)

# Evaluation criteria
criteria = [
  EvaluationCriterion(
    id="correctness",
    name="Correctness",
    description=(
      "Does the implementation produce the correct output for ALL inputs, "
      "including edge cases? Verify against the test cases, paying close "
      "attention to inputs with duplicate values."
    ),
    weight=2.0,
  ),
  EvaluationCriterion(
    id="quality",
    name="Code Quality",
    description=(
      "Is the code readable, idiomatic, and reasonably efficient? "
      "Prefer standard library idioms over manual loops. Avoid unnecessary "
      "intermediate data structures or redundant operations."
    ),
    weight=1.0,
  ),
]

print("Model:", model.model_id)
print("Criteria:", [c.name for c in criteria])


# %% [markdown]
# ## 2. Task and candidate trajectories
#
# The task simulates a code-review scenario: an agent was given a buggy Python
# function and asked to fix it. Three different agents produced three fixes.
#
# The original bug: `sorted(nums)[-k:]` returns the k largest values in
# **ascending** order, not descending. All three agents notice this.
#
# Where they differ:
#
# - `traj-clean`: idiomatic one-liner, handles all edge cases correctly.
# - `traj-convoluted`: correct but uses a manual O(n·k) loop where
#   `sorted(..., reverse=True)[:k]` would do.
# - `traj-latent-bug`: fixes the ordering but silently deduplicates via
#   `set()`, causing wrong output when duplicates should appear in the
#   top-k (e.g., `top_k([5, 5, 3], 2)` → `[5, 3]` instead of `[5, 5]`).
#
# All three pass the obvious test case. Only careful evaluation — or the
# edge-case test — catches the latent bug.

# %%
# bug: sorted(nums)[-k:] returns ascending order, not descending
buggy_original = """\
def top_k(nums: list[int], k: int) -> list[int]:
    # Returns the k largest numbers in descending order.
    return sorted(nums)[-k:]
"""

task = Task(
  id="top-k-fix",
  description="Fix top_k() so it returns the k largest numbers in descending order.",
  problem_statement=(
    "An AI agent was given the following buggy implementation and asked to fix it:\n\n"
    f"```python\n{buggy_original}```\n"
    "Three agents produced fixes. Evaluate which fix is best.\n\n"
    "Requirements:\n"
    "  1. Return the k largest numbers in descending order.\n"
    "  2. Preserve duplicates (e.g., top_k([5, 5, 3], 2) → [5, 5]).\n"
    "  3. Handle k > len(nums) gracefully (return all elements).\n"
    "  4. Prefer idiomatic, readable code."
  ),
  ground_truth=(
    "def top_k(nums: list[int], k: int) -> list[int]:\n    return sorted(nums, reverse=True)[:k]\n"
  ),
  test_cases=[
    {
      "input": "top_k([3, 1, 4, 1, 5, 9, 2, 6], 3)",
      "expected": "[9, 6, 5]",
      "note": "basic case",
    },
    {
      "input": "top_k([5, 5, 3], 2)",
      "expected": "[5, 5]",
      "note": "duplicates must be preserved — catches the set() bug",
    },
    {
      "input": "top_k([7], 1)",
      "expected": "[7]",
      "note": "single-element list",
    },
    {
      "input": "top_k([2, 1], 5)",
      "expected": "[2, 1]",
      "note": "k larger than list length",
    },
  ],
  difficulty=TaskDifficulty.MEDIUM,
)

trajectories = [
  Trajectory(
    id="traj-clean",
    task_id=task.id,
    content=(
      "def top_k(nums: list[int], k: int) -> list[int]:\n"
      "    # Sort descending and slice the first k elements.\n"
      "    return sorted(nums, reverse=True)[:k]\n"
    ),
    output=(
      "top_k([3, 1, 4, 1, 5, 9, 2, 6], 3) → [9, 6, 5]\n"
      "top_k([5, 5, 3], 2)                → [5, 5]\n"
      "top_k([7], 1)                       → [7]\n"
      "top_k([2, 1], 5)                    → [2, 1]"
    ),
    reward=1.0,
  ),
  Trajectory(
    id="traj-convoluted",
    task_id=task.id,
    content=(
      "def top_k(nums: list[int], k: int) -> list[int]:\n"
      "    result = []\n"
      "    remaining = list(nums)\n"
      "    for _ in range(min(k, len(nums))):\n"
      "        max_val = max(remaining)\n"
      "        result.append(max_val)\n"
      "        remaining.remove(max_val)\n"
      "    return result\n"
    ),
    output=(
      "top_k([3, 1, 4, 1, 5, 9, 2, 6], 3) → [9, 6, 5]\n"
      "top_k([5, 5, 3], 2)                → [5, 5]\n"
      "top_k([7], 1)                       → [7]\n"
      "top_k([2, 1], 5)                    → [2, 1]"
    ),
    reward=0.65,
  ),
  Trajectory(
    id="traj-latent-bug",
    task_id=task.id,
    content=(
      "def top_k(nums: list[int], k: int) -> list[int]:\n"
      "    # Deduplicate first, then sort descending.\n"
      "    return sorted(set(nums), reverse=True)[:k]\n"
    ),
    output=(
      "top_k([3, 1, 4, 1, 5, 9, 2, 6], 3) → [9, 6, 5]  ✓\n"
      "top_k([5, 5, 3], 2)                → [5, 3]      ✗ (should be [5, 5])\n"
      "top_k([7], 1)                       → [7]         ✓\n"
      "top_k([2, 1], 5)                    → [2, 1]      ✓"
    ),
    reward=0.1,
  ),
]

print("Task:", task.id)
print("Trajectories:", [t.id for t in trajectories])


# %% [markdown]
# ## 3. Use the Verifier directly
#
# The Verifier builds a prompt that includes the ground truth, test cases, and
# execution output, then asks Claude to assign a letter score. The test case for
# `top_k([5, 5, 3], 2)` gives it concrete evidence to detect the `set()` bug.

# %% [markdown]
# ### 3a. Score one trajectory against one criterion

# %%
verifier = VerifierStrategy(model, config, criteria)

single_score = verifier.score_trajectory(
  # bug fixing task
  task,
  trajectories[0],
  criteria[0],
)

print("Verifier — single trajectory score:")
print("  trajectory:", single_score.trajectory_id)
print("  score:", round(single_score.score, 4))
print("  confidence:", round(single_score.confidence, 4))


# %% [markdown]
# ### 3b. Compare the clean fix against the latent-bug fix head-to-head
#
# This is the most interesting comparison: both produce the same output on
# the obvious test case. The Verifier must use the duplicate test case and
# the execution output to detect the difference.

# %%
pairwise = verifier.compare_trajectories(
  task,
  trajectories[0],  # traj-clean
  trajectories[2],  # traj-latent-bug
  criteria[0],
)

print("Verifier — clean vs. latent-bug:")
print("  A:", pairwise.trajectory_a_id, "→ score:", round(pairwise.score_a, 4))
print("  B:", pairwise.trajectory_b_id, "→ score:", round(pairwise.score_b, 4))
print("  winner:", pairwise.winner)
print("  confidence:", round(pairwise.confidence, 4))


# %% [markdown]
# ### 3c. Evaluate all three candidates

# %%
verifier_result = verifier.evaluate(task, trajectories)

print("Verifier — full evaluation:")
print("  best trajectory:", verifier_result.best_trajectory_id)
for tid, sr in sorted(
  verifier_result.trajectory_scores.items(),
  key=lambda item: -item[1].score,
):
  print(f"  {tid}: score={sr.score:.4f}  confidence={sr.confidence:.4f}")


# %% [markdown]
# ## 4. Use the Judge directly
#
# The Judge evaluates holistically from the code alone, without structured test
# evidence. It can still reason about the `set()` deduplication problem by
# reading the implementation, but it does so through language reasoning rather
# than test-case matching. Comparing its verdict to the Verifier's is
# informative — this is exactly what the harness measures.

# %% [markdown]
# ### 4a. Score one trajectory

# %%
judge = JudgeStrategy(
  model,
  config,
  criteria,
  score_min=1.0,
  score_max=10.0,
  swap_pairwise=True,
  reasoning_depth="detailed",
)

single_judge = judge.score_trajectory(
  task,  # bug fixing task
  trajectories[0],
  criteria[0],
)

print("Judge — single trajectory score:")
print("  trajectory:", single_judge.trajectory_id)
print("  score:", round(single_judge.score, 4))
print("  confidence:", round(single_judge.confidence, 4))


# %% [markdown]
# ### 4b. Evaluate all three candidates

# %%
judge_result = judge.evaluate(task, trajectories)

print("Judge — full evaluation:")
print("  best trajectory:", judge_result.best_trajectory_id)
for tid, sr in sorted(
  judge_result.trajectory_scores.items(),
  key=lambda item: -item[1].score,
):
  print(f"  {tid}: score={sr.score:.4f}  confidence={sr.confidence:.4f}")


# %% [markdown]
# ## 5. Use the Oracle router
#
# The router examines task signals before making a decision:
#
# - Ground truth is present → Verifier signal.
# - Test cases are present → strong Verifier signal.
# - The task has execution output per trajectory → another Verifier signal.
# - Keyword density: "fix", "bug", "implement" → Verifier domain.
#
# Expected routing: **Verifier**, with high confidence. The interesting
# question is whether the Verifier correctly ranks the three candidates
# despite all three appearing superficially plausible.

# %% [markdown]
# ### 5a. Inspect the routing decision

# %%
router = OracleRouter.default(verifier, judge)

decision = router.route(task, trajectories)

print("Routing decision:")
print("  selected strategy:", decision.selected_strategy.value)
print("  confidence:", round(decision.confidence, 4))
print("  elapsed_ms:", round(decision.elapsed_ms, 3))
print()
print(decision.reasoning)


# %% [markdown]
# ### 5b. Route and evaluate in one call

# %%
oracle_result, oracle_decision = router.evaluate(task, trajectories)

print("Oracle evaluation:")
print("  strategy selected:", oracle_decision.selected_strategy.value)
print("  best trajectory:", oracle_result.best_trajectory_id)
print("  strategy used:", oracle_result.strategy_type.value)


# %% [markdown]
# ## 6. Evaluation harness
#
# The harness runs both strategies and surfaces four hardness signals. Here
# the interesting signal is **strategy disagreement**: does the Judge, which
# reasons from code structure alone, agree with the Verifier, which has test
# evidence?
#
# If they disagree on the ranking of `traj-convoluted` vs `traj-latent-bug`,
# that reveals the task is genuinely ambiguous without test evidence — which
# is exactly when the Oracle's routing decision matters most.

# %%
harness = EvaluationHarness(verifier=verifier, judge=judge, max_workers=1)

record = harness.run_single(task, trajectories)

print("Harness record:")
print("  hardness_score:", round(record.hardness_score, 4))
print("  score_spread:", round(record.score_spread, 4))
print("  strategy_disagreement:", round(record.strategy_disagreement, 4))
print("  avg_confidence:", round(record.avg_confidence, 4))
print("  oracle_gap_verifier:", round(record.oracle_gap_verifier, 4))
print("  oracle_gap_judge:", round(record.oracle_gap_judge, 4))
print("  strategies agree:", record.strategies_agree)
print()

report = harness.run([(task, trajectories)], parallel=False)
print(report.summary())


# %% [markdown]
# ## 7. Second task: open-ended question, Judge wins the routing
#
# The Oracle's value is clearest when your workload mixes task types. Here is
# a DevOps design question with no ground truth and no test cases — there is
# nothing for the Verifier to compare against, so the router should select
# the Judge.
#
# Two trajectories: one thorough, one superficial. The Judge's holistic
# rubric is the right tool for separating them.

# %%
ops_task = Task(
  id="zero-downtime-migrations",
  description="Explain how to run database schema migrations with zero downtime.",
  problem_statement=(
    "Your team needs to add a non-nullable column to a high-traffic PostgreSQL "
    "table with 50 million rows. The service cannot go offline. Describe the "
    "approach you would take to deploy this migration safely, covering: the "
    "migration steps, how you maintain backwards compatibility during the "
    "transition, how you handle rollback if something goes wrong, and any "
    "tooling or patterns you would use."
  ),
  difficulty=TaskDifficulty.HARD,
)

ops_trajectories = [
  Trajectory(
    id="response-thorough",
    task_id=ops_task.id,
    content=(
      "Use the expand-contract pattern across three deploys:\n\n"
      "Phase 1 — Expand: add the column as nullable with a DEFAULT. "
      "PostgreSQL can do this without a full table rewrite (metadata-only on "
      "Postgres 11+). Deploy the new application code that writes the default "
      "value for new rows, but still tolerates the column being absent "
      "(backwards-compatible read path).\n\n"
      "Phase 2 — Backfill: populate existing rows in batches "
      "(e.g., 1000 rows at a time with a short sleep between batches) to "
      "avoid lock contention. Use a background job, not a migration script "
      "that blocks your deploy pipeline.\n\n"
      "Phase 3 — Contract: once all rows are populated, add the NOT NULL "
      "constraint using `ALTER TABLE ... SET NOT NULL` — Postgres 12+ can "
      "validate this constraint without a full table lock if you first run "
      "`ADD CONSTRAINT ... NOT VALID` and then `VALIDATE CONSTRAINT` "
      "separately. Finally drop the DEFAULT if it is no longer needed and "
      "remove the backwards-compatibility shim from the application.\n\n"
      "Rollback: each phase is independently reversible. Phase 1 can be "
      "rolled back by dropping the column. Phase 3 can be rolled back by "
      "dropping the constraint. No deploy needs to be atomic with the "
      "migration.\n\n"
      "Tooling: Flyway or Liquibase for versioned migrations; "
      "pg_repack if a VACUUM FULL is later needed without locking."
    ),
  ),
  Trajectory(
    id="response-vague",
    task_id=ops_task.id,
    content=(
      "Database migrations with zero downtime are tricky. The main idea is "
      "to make changes backwards-compatible so old and new application code "
      "can run at the same time. You should add the column first, then "
      "deploy the application, then fill in the data. Make sure to test it "
      "in staging before production. Have a rollback plan ready in case "
      "something breaks. Tools like Flyway can help manage this."
    ),
  ),
]

ops_decision = router.route(ops_task, ops_trajectories)
ops_result, _ = router.evaluate(ops_task, ops_trajectories)

print("Ops task routing:")
print("  selected strategy:", ops_decision.selected_strategy.value)
print("  confidence:", round(ops_decision.confidence, 4))
print("  best trajectory:", ops_result.best_trajectory_id)
print()
print("Routing reasoning:")
print(ops_decision.reasoning)


# %% [markdown]
# ## 8. What this example shows
#
# The Oracle's value is not in making easy calls — any evaluator can rank
# a correct solution above one that returns the wrong type. Its value is in:
#
# - **Routing non-trivially**: this coding task had test-case evidence
#   (Verifier signal) *and* subtle structural quality differences (Judge
#   signal). The router weighed both and committed to the stronger signal.
#
# - **Catching latent bugs**: `traj-latent-bug` passed the obvious test. The
#   Verifier's prompt included the duplicate edge case, giving Claude the
#   evidence it needed to detect `set()` deduplication as incorrect.
#
# - **Adapting to task type**: the ops task had none of the Verifier's
#   signals. The router switched to the Judge without any configuration
#   change on your part.
#
# - **Measuring disagreement**: the harness `strategy_disagreement` metric
#   tells you when the Judge and Verifier diverge — which is when the Oracle
#   routing decision carries the most value.
#
# To use a different provider, swap `AnthropicProvider` for `OpenAIProvider`
# or `GeminiProvider` — the rest of the workflow is unchanged.
