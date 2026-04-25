#!/usr/bin/env python3

# %% [markdown]
# ## 1. LLMs evaluating other LLMs
#
# As AI systems produce more outputs — summaries, rewrites, responses —
# evaluation tends to become a bottleneck. This demo shows a practical framework
# for automating it: structured verification when evidence exists, holistic
# judgment when it doesn't, with automatic routing between them.
#
# This demo introduces a design pattern for automated LLM evaluation.
# It consists of three components:
#
# 1. **Verifier**: structured, evidence-sensitive scoring.
# 2. **Judge**: holistic scoring for open-ended tasks where evidence is absent.
# 3. **Oracle**: orchestration layer that routes between the two automatically.
#
# By the end you will know when to reach for each one and how to wire them
# into a pipeline that handles both types of tasks.

# %% [markdown]
# ## 2. The Scenario: Three LLMs fixed the same bug.
#
# Task: Which fix is correct?
#
# For this scenario,
# Two are correct. One has a latent bug that passes 3 of 4 test cases and
# looks perfectly plausible on first reading.
#
# Runs this pipeline entirely offline. No API keys needed.
#
# ```bash
# uv run python examples/03_hands_on_example.py
# ```

# %%
# Original buggy code:
#
# ```python
# def top_k(nums: list[int], k: int) -> list[int]:
#   return sorted(nums)[-k:]  # bug: returns ascending, not descending
# ```
#
# Three LLMs each produced a fix. On the obvious test they all look correct.
# The difference only surfaces on a duplicate-value input.
#
# | Trajectory              | top_k([5, 5, 3], 2) | Verdict      |
# |-------------------------|---------------------|--------------|
# | traj-clean-code         | [5, 5]              | Correct      |
# | traj-overengineered     | [5, 5]              | Correct      |
# | traj-latent-bug         | [5, 3]              | Wrong        |
#
# `traj-latent-bug` calls `set()` before sorting — silently removing
# duplicates. It passes 3 of 4 test cases and reads like a reasonable fix.

# %% [markdown]
# ## 3. Imports and Setup

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
# ### 3a. Shared model and config
#
# `StubProvider` runs the full evaluation pipeline offline — no network, no
# API key. To use a real model, replace it with:
#
# ```python
# from llm_oracle import AnthropicProvider
# model = AnthropicProvider("claude-sonnet-4-6")
# ```
#
# The stub is configured so the Verifier correctly identifies the latent bug.
# On a real model, the prompt includes test cases and execution output; the
# model uses the duplicate edge case as evidence to detect `set()`.

# %%
stub = StubProvider(
  model_id="stub-model",
  default_score="C",
  default_score_a="C",
  default_score_b="H",
  responses=[
    # Cycling pool: consumed in round-robin across all model.generate() calls.
    # 4 distinct entries give enough score variation for a non-trivial ranking.
    # Verifier reads the letter tokens; Judge ignores them (falls back to midpoint).
    StubResponse(score="B", score_a="B", score_b="G"),
    StubResponse(score="C", score_a="C", score_b="H"),
    StubResponse(score="A", score_a="A", score_b="F"),
    StubResponse(score="D", score_a="B", score_b="J"),
  ],
  seed=42,
)

# %%
# Both strategies share this config; Judge maps its scores to
# score_min/score_max instead of the letter scale.
#
# granularity=20       Verifier: A–T scoring scale (A is best, T is worst)
# num_verifications=2  Verifier: each criterion scored K=2 times and averaged
# use_logprobs=True    Verifier: weighted expectation over score tokens instead
#                      of a single discrete token — the continuous reward signal
#
config = ScoringConfig(
  granularity=20,
  num_verifications=2,
  use_logprobs=True,
)

# Both Verifier and Judge score against these criteria — it is the shared rubric.
# weight controls each criterion's contribution to the final score.
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
      "Is the code readable and idiomatic? Prefer standard library idioms over manual loops."
    ),
    weight=1.0,
  ),
]

# %% [markdown]
# ### 3b. The task and the three candidate trajectories

# %%
buggy = "def top_k(nums: list[int], k: int) -> list[int]:\n    return sorted(nums)[-k:]"

task = Task(
  id="top-k-fix",
  description="Fix top_k() so it returns the k largest numbers in descending order.",
  problem_statement=(
    f"Fix this buggy function:\n\n```python\n{buggy}\n```\n\n"
    "Requirements:\n"
    "  1. Return the k largest numbers in descending order.\n"
    "  2. Preserve duplicates — top_k([5, 5, 3], 2) must return [5, 5].\n"
    "  3. Handle k > len(nums) gracefully.\n"
    "  4. Prefer idiomatic, readable code."
  ),
  ground_truth="def top_k(nums, k):\n    return sorted(nums, reverse=True)[:k]",
  test_cases=[
    {"input": "top_k([3, 1, 4, 1, 5, 9], 3)", "expected": "[9, 5, 4]"},
    {
      "input": "top_k([5, 5, 3], 2)",
      "expected": "[5, 5]",
      "note": "duplicate values — catches the set() bug",
    },
    {"input": "top_k([7], 1)", "expected": "[7]"},
    {"input": "top_k([2, 1], 5)", "expected": "[2, 1]"},
  ],
  # difficulty is a routing signal for the Oracle's DifficultyPolicy.
  # Neither Verifier nor Judge use it directly for scoring.
  difficulty=TaskDifficulty.MEDIUM,
)

# A Trajectory is a candidate task-solving attempt — what one agent produced.
# content is the response; output is the execution result; reward is the
# oracle-best ground-truth score used by the harness to measure accuracy.
trajectories = [
  Trajectory(
    id="traj-clean-code",
    task_id=task.id,
    content="def top_k(nums, k):\n    return sorted(nums, reverse=True)[:k]",
    output=(
      "top_k([3, 1, 4, 1, 5, 9], 3) → [9, 5, 4]  ✓\ntop_k([5, 5, 3], 2)           → [5, 5]      ✓"
    ),
    reward=1.0,
  ),
  Trajectory(
    id="traj-overengineered",
    task_id=task.id,
    content=(
      "def top_k(nums, k):\n"
      "    result = []\n"
      "    remaining = list(nums)\n"
      "    for _ in range(min(k, len(nums))):\n"
      "        m = max(remaining)\n"
      "        result.append(m)\n"
      "        remaining.remove(m)\n"
      "    return result"
    ),
    output=(
      "top_k([3, 1, 4, 1, 5, 9], 3) → [9, 5, 4]  ✓\ntop_k([5, 5, 3], 2)           → [5, 5]      ✓"
    ),
    reward=0.65,
  ),
  Trajectory(
    id="traj-latent-bug",
    task_id=task.id,
    content="def top_k(nums, k):\n    return sorted(set(nums), reverse=True)[:k]",
    output=(
      "top_k([3, 1, 4, 1, 5, 9], 3) → [9, 5, 4]  ✓\n"
      "top_k([5, 5, 3], 2)           → [5, 3]      ✗  should be [5, 5]"
    ),
    reward=0.1,
  ),
]

print("\n# The latent bug — only the duplicate test case reveals it:")
print("  traj-clean-code:       top_k([5, 5, 3], 2) → [5, 5]  ✓")
print("  traj-overengineered:  top_k([5, 5, 3], 2) → [5, 5]  ✓")
print("  traj-latent-bug:  top_k([5, 5, 3], 2) → [5, 3]  ✗  (should be [5, 5])")


# %% [markdown]
# ## 4. LLM-as-a-Verifier
#
# Verifier, Judge, and Oracle are each independently usable — this session
# introduces them in order of increasing sophistication so you can see what
# each adds before combining them.
#
# The Verifier is the right tool here because we have concrete evidence:
# a ground-truth reference, test cases (including the duplicate edge case),
# and execution output per trajectory.
#
# **How it scores:** instead of picking a single score token, the Verifier
# reads the model's log-probability distribution over all score tokens
# (`A`–`T`) and computes a weighted expectation. This turns evaluation into
# a **continuous reward signal** — finer discrimination between close candidates.
#
# $$\text{Reward} = \frac{1}{CK} \sum_{c=1}^{C} \sum_{k=1}^{K} \sum_{g=1}^{G}
# p(\text{score}_g) \times \text{value}(\text{score}_g)$$
#
# The duplicate test case gives the Verifier the evidence it needs to detect
# `set()` deduplication as incorrect.

# %%
verifier = VerifierStrategy(stub, config, criteria)

print("\n# Verifier — full evaluation:")
v_result = verifier.evaluate(task, trajectories)
print(f"  best trajectory: {v_result.best_trajectory_id}")
for tid, sr in sorted(v_result.trajectory_scores.items(), key=lambda x: -x[1].score):
  print(f"  {tid:<20s}  score={sr.score:.4f}  confidence={sr.confidence:.4f}")

print()
print("  The Verifier correctly identifies the ranking:")
print("  traj-clean-code  >  traj-overengineered  >  traj-latent-bug")


# %% [markdown]
# ### 4a. Pairwise comparison: the most interesting matchup
#
# Pointwise scores can tie when two candidates are close. Pairwise comparison
# gives a direct head-to-head signal and is how the tournament inside
# `evaluate()` resolves those ties into a ranking.
#
# `traj-clean-code` vs `traj-latent-bug` — both produce identical output on the
# basic test. The Verifier uses the duplicate test case and execution output
# to decide.

# %%
pairwise = verifier.compare_trajectories(task, trajectories[0], trajectories[2], criteria[0])
print("\n# Verifier pairwise — clean vs. latent-bug:")
print(f"  {pairwise.trajectory_a_id:<20s}  score={pairwise.score_a:.4f}")
print(f"  {pairwise.trajectory_b_id:<20s}  score={pairwise.score_b:.4f}")
print(f"  winner: {pairwise.winner}  confidence={pairwise.confidence:.4f}")


# %% [markdown]
# ## 5. LLM-as-a-Judge
#
# The Judge evaluates holistically — reasoning from the code structure alone,
# without test-case matching. It _can_ reason about `set()` deduplication by
# reading the implementation, but it is working from language understanding,
# not concrete evidence.
#
# **With `StubProvider`:** the Judge receives letter tokens that do not match
# its numeric regex, so it falls back to the scale midpoint (0.5) for every
# trajectory. This faithfully represents what happens when the evaluator has
# no structured evidence to differentiate on — the scores tie.
#
# **With a real model:** the Judge can sometimes catch the bug through code
# reading, but it is less reliable than the Verifier when test-case evidence
# is available. The gap is smallest on easy bugs; largest on subtle ones like
# this duplicate-value edge case.

# %%
judge = JudgeStrategy(
  stub,
  config,
  criteria,
  score_min=1.0,  # raw model scores are mapped to this range before
  score_max=10.0,  # normalizing to [0, 1] for reporting
  swap_pairwise=True,  # run each pairwise in both orders; average to cancel positional bias
  reasoning_depth="detailed",  # "brief" | "detailed" | "chain_of_thought"
)

print("\n# Judge — full evaluation:")
j_result = judge.evaluate(task, trajectories)
print(f"  best trajectory: {j_result.best_trajectory_id}")
for tid, sr in sorted(j_result.trajectory_scores.items(), key=lambda x: -x[1].score):
  print(f"  {tid:<20s}  score={sr.score:.4f}  confidence={sr.confidence:.4f}")

print()
print("  The Judge produces equal scores — without test-case evidence it")
print("  cannot discriminate between the three fixes.")


# %% [markdown]
# ## 6. The Oracle Router
#
# Rather than manually picking Judge or Verifier for each task, `OracleRouter`
# examines task signals and decides automatically.
#
# The four-step algorithm:
#
# ```
# 1. SignalExtractor  →  RoutingSignals
# 2. PolicyChain      →  6 policies cast weighted votes
# 3. Aggregate        →  verifier_total vs judge_total
# 4. Select           →  winner, or Judge fallback if confidence < 0.60
# ```
#
# For this task the signals strongly favor Verifier:
# - `has_ground_truth = True`
# - `has_test_cases = True`  ← strongest single signal
# - keyword evidence: "fix", "bug", "implement" → verifiable domain
# - execution output attached to each trajectory
#
# Every policy vote, signal value, and confidence score is exposed in the
# routing decision — fully auditable.

# %%
router = OracleRouter.default(verifier, judge)

decision = router.route(task, trajectories)
print("\n# Routing decision:")
print(f"  selected:     {decision.selected_strategy.value}")
print(f"  confidence:   {decision.confidence:.4f}")
print(f"  elapsed_ms:   {decision.elapsed_ms:.1f}")
print()
print(decision.reasoning)


# %% [markdown]
# ### 6a. Route and evaluate in one call
#
# `router.evaluate(...)` routes, picks the winning strategy, runs evaluation,
# and returns both the result and the routing decision that produced it.
# This is the high-level convenience method for production use.

# %%
oracle_result, oracle_decision = router.evaluate(task, trajectories)
print("\n# Oracle evaluation:")
print(f"  strategy:        {oracle_decision.selected_strategy.value}")
print(f"  best trajectory: {oracle_result.best_trajectory_id}")
print(f"  strategy used:   {oracle_result.strategy_type.value}")

# --- natural stopping point for a short session ---
# Sections 1–6a cover the full story: Verifier differentiates (has evidence),
# Judge ties (no evidence), Oracle routes automatically. Sections 7–8 extend
# the picture but are not required to make the core point.

# %% [markdown]
# ## 7. Second Task: When the Judge Wins the Routing
#
# The Oracle's value is clearest when your workload mixes task types.
#
# This next task is an open-ended design question — no ground truth, no test
# cases, no execution output. The Verifier has nothing to compare against.
# The router should automatically switch to the Judge.

# %%
ops_task = Task(
  id="zero-downtime-migration",
  description="Explain how to run a PostgreSQL schema migration with zero downtime.",
  problem_statement=(
    "Your team needs to add a non-nullable column to a high-traffic table "
    "with 50 million rows. The service cannot go offline. Describe the "
    "migration steps, how you maintain backwards compatibility during the "
    "transition, and how you handle rollback if something goes wrong."
  ),
  difficulty=TaskDifficulty.HARD,
  # No ground_truth. No test_cases. Verifier signals are absent.
)

ops_trajectories = [
  Trajectory(
    id="response-thorough",
    task_id=ops_task.id,
    content=(
      "Use the expand-contract pattern across three deploys:\n"
      "1. Add column as nullable with DEFAULT — no table lock on Postgres 11+.\n"
      "2. Backfill existing rows in batches with a background job "
      "(avoid lock contention).\n"
      "3. Add NOT NULL using ADD CONSTRAINT ... NOT VALID, then "
      "VALIDATE CONSTRAINT separately to skip a full table lock.\n"
      "Rollback: each phase is independently reversible."
    ),
  ),
  Trajectory(
    id="response-vague",
    task_id=ops_task.id,
    content=(
      "Database migrations with zero downtime are tricky. The idea is "
      "to keep things backwards-compatible. Add the column, then deploy, "
      "then backfill the data. Test in staging first."
    ),
  ),
]

ops_decision = router.route(ops_task, ops_trajectories)
ops_result, _ = router.evaluate(ops_task, ops_trajectories)

print("\n# Open-ended task routing:")
print(f"  selected:        {ops_decision.selected_strategy.value}  (expected: JUDGE)")
print(f"  confidence:      {ops_decision.confidence:.4f}")
print(f"  best trajectory: {ops_result.best_trajectory_id}")
print()
print("Routing reasoning:")
print(ops_decision.reasoning)


# %% [markdown]
# ## 8. Harness: Measuring Evaluation Hardness
#
# `EvaluationHarness` runs both strategies and surfaces comparison signals.
# The most informative metric here is `strategy_disagreement`: how often do
# Judge and Verifier rank the same pair of trajectories differently?
#
# High disagreement on the coding task is expected — the Verifier has test
# evidence to differentiate candidates; the Judge (stub mode) produces ties.
# That gap is exactly when the Oracle routing decision carries the most value:
# picking the Verifier on a task where the Judge is blind.

# %%
harness = EvaluationHarness(verifier=verifier, judge=judge, max_workers=2)

record = harness.run_single(task, trajectories)
print("\n# Harness — coding task:")
print(f"  hardness_score:         {record.hardness_score:.4f}")
print(f"  score_spread:           {record.score_spread:.4f}")
print(f"  strategy_disagreement:  {record.strategy_disagreement:.4f}")
print(f"  avg_confidence:         {record.avg_confidence:.4f}")
print(f"  oracle_gap_verifier:    {record.oracle_gap_verifier:.4f}")
print(f"  oracle_gap_judge:       {record.oracle_gap_judge:.4f}")
print(f"  strategies agree:       {record.strategies_agree}")

report = harness.run([(task, trajectories)], parallel=False)
print()
print(report.summary())


# %% [markdown]
# ## 9. Takeaways
#
# **Use Verifier** when evaluation can be grounded in evidence:
# test cases, reference answers, execution output.
#
# **Use Judge** when evaluation is open-ended, holistic, or subjective.
#
# **Use Oracle** when your pipeline sees both task types — the router
# selects automatically and exposes its reasoning.
#
# **Use Harness** for benchmarking: `strategy_disagreement` identifies
# the tasks where the routing decision actually matters.
#
# Key principle: **verification does not require the largest model.**
# A smaller, efficient model (e.g., Gemini 2.5 Flash) can evaluate and rank
# outputs from a larger model (e.g., Claude Opus 4.7) as long as it can apply
# structured reasoning against the available evidence.
#
# ---
#
# To run against a real model, replace `StubProvider` with:
#
# ```python
# from llm_oracle import AnthropicProvider
# model = AnthropicProvider("claude-sonnet-4-6")  # reads ANTHROPIC_API_KEY
# ```
#
# Everything else stays the same.
