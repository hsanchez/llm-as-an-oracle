#!/usr/bin/env python3

# %% [markdown]
# # LLM as an Oracle: Full Feature Demo
#
# ## The evaluation problem
#
# Human evaluation is the gold standard — but it does not scale. It is slow,
# expensive, and cannot keep pace with LLM-aided workflows.
#
# Automated metrics like BLEU and ROUGE are cheap but shallow: they miss
# reasoning quality, correctness of approach, and usefulness.
#
# Around 2023, frontier models crossed a reliability threshold that opened a
# better path. This gave rise to a family of `LLM-as-*` patterns:
# `LLM-as-a-Judge`, `LLM-as-a-Critic`, `LLM-as-a-Verifier`, and others.
#
# But which pattern is the right one for a given task? That is the question
# I am exploring in this demo.
#
# ## What the Oracle is
#
# The Oracle is an evaluation orchestrator. It sits above the Judge and the
# Verifier and decides which one is the better fit for the task at hand.
# "Oracle" here means *adaptive evaluation layer* — not an all-knowing model.
# It is a system design pattern.
#
# What it evaluates are **trajectories** — full candidate task-solving attempts.
# A trajectory is not just a final answer. It can include intermediate reasoning,
# tool calls, code, execution output, and an optional reward signal. The
# evaluator scores the entire attempt.
#
# ## When the oracle needs help
#
# Even an adaptive evaluation layer has a limit. Sometimes the task is
# under-specified — the correct answer depends on facts that were never
# included. When the score spread across trajectories falls below a confidence
# threshold, the oracle recognizes it cannot make a reliable judgment and
# escalates to a **Human Oracle**: a host-provided adapter that delivers a
# targeted clarifying question and returns the answer. The oracle then applies
# the clarification and re-evaluates. It does not escalate speculatively — only
# at the exact point where evaluation evidence runs out.
#
# ## What we'll see
#
# **Task 1** puts three agents' bug fixes under the oracle's lens. Two fixes
# are correct. One looks reasonable but doesn't actually solve the problem —
# it hides it. We'll see whether the oracle catches it, and whether the answer
# changes depending on which evaluation strategy is used.
#
# **Task 2** is a different kind of hard. Three architecture recommendations,
# all defensible — but which one is right depends on facts about the team that
# nobody included in the task description. We'll watch the oracle evaluate,
# realize it's stuck, and escalate to the Human Oracle for the one fact it
# needs.
#
# Prerequisites:
#
# - `ANTHROPIC_API_KEY` must be set in your environment.
# - Run `uv sync` to install dependencies.
#
# ```bash
# uv run python examples/07_full_demo.py
# ```
#
# Cost note: `num_verifications=1` and `max_workers=1` minimize API calls.

# %% [markdown]
# ## 1. Imports
#
# Everything we need is in `llm_oracle`. The four evaluation components —
# `VerifierStrategy`, `JudgeStrategy`, `OracleRouter`, and `EvaluationHarness`
# — plus the human escalation types.

# %%
from __future__ import annotations

from dataclasses import dataclass

from dotenv import load_dotenv

from llm_oracle import (
  AnthropicProvider,
  EvaluationCriterion,
  EvaluationHarness,
  EvaluationResult,
  HumanRequest,
  HumanResponse,
  JudgeStrategy,
  OracleRouter,
  ScoringConfig,
  Task,
  TaskDifficulty,
  Trajectory,
  VerifierStrategy,
)

load_dotenv()

# %% [markdown]
# ## 2. The Human Oracle
#
# Before we get to the tasks, let's talk about the human side of this system.
#
# The library defines what a question looks like (`HumanRequest`) and what an
# answer looks like (`HumanResponse`). What it does *not* define is how you
# deliver the question or collect the answer — that's deliberately left to the
# host application. In production you'd wire this to a Slack bot, a ticket
# queue, or an async approval workflow. Here, we block on standard input so
# you can type the answer yourself and see the re-evaluation happen live.


# %%
@dataclass
class ConsoleHumanOracle:
  """Blocks and asks the current interactive user."""

  responder_id: str = "architect"

  def ask(self, request: HumanRequest) -> HumanResponse:
    print("\n── Human escalation ──────────────────────────────────────")
    print(f"  Task    : {request.task_id}")
    print(f"  Reason  : {request.reason}")
    print(f"  Question: {request.question}")

    answer = input("  Answer> ").strip()
    if not answer:
      raise ValueError("Human answer cannot be empty.")

    return HumanResponse(
      request_id=request.id,
      answer=answer,
      responder_id=self.responder_id,
      metadata={"source": "console"},
    )


# %% [markdown]
# ## 3. Task 1 — The N+1 Bug
#
# The original function issues one SQL query per order to fetch its items.
# With 50 orders, that's 51 queries. With 5,000 orders, it's 5,001. Three
# agents were asked to fix it.
#
# Two of them actually did: `traj-join` rewrites the query with a JOIN
# (one round-trip, idiomatic SQL), and `traj-prefetch` batch-fetches items
# using `WHERE IN` (two queries, more code). Both are correct.
#
# The third — `traj-cache` — wraps the inner query in an `@lru_cache`
# decorator. It reads like a performance optimization. But on a cold cache it
# still issues N queries. It doesn't fix the bug; it just amortizes it across
# repeated calls. This is exactly the kind of output that sounds plausible and
# may even fool a casual reviewer.
#
# We've included ground truth and test cases that make the query-count
# requirement explicit. Let's see if that evidence changes what the oracle does.

# %%
_buggy = """\
def get_orders_with_items(user_id: int) -> list[dict]:
    orders = db.execute("SELECT * FROM orders WHERE user_id = ?", [user_id])
    for order in orders:
        order["items"] = db.execute(          # N+1: one query per order
            "SELECT * FROM items WHERE order_id = ?", [order["id"]]
        )
    return orders
"""

n1_task = Task(
  id="fix-n1-query",
  description="Fix the N+1 query bug in get_orders_with_items().",
  problem_statement=(
    "The following function issues one SQL query per order to fetch its items, "
    "causing N+1 queries under load. Fix it so the data is retrieved in a "
    "constant number of queries regardless of result size.\n\n"
    f"```python\n{_buggy}```\n"
    "Requirements:\n"
    "  1. All items for all orders must be returned correctly.\n"
    "  2. The fix must work for a user with zero orders.\n"
    "  3. The total number of SQL queries must not grow with the number of orders.\n"
    "  4. Prefer idiomatic, readable SQL over application-layer workarounds."
  ),
  ground_truth=(
    "def get_orders_with_items(user_id: int) -> list[dict]:\n"
    "    rows = db.execute(\n"
    "        'SELECT o.*, i.* FROM orders o '\n"
    "        'LEFT JOIN items i ON i.order_id = o.id '\n"
    "        'WHERE o.user_id = ?', [user_id]\n"
    "    )\n"
    "    return _group_by_order(rows)\n"
  ),
  test_cases=[
    {
      "input": "get_orders_with_items(user_id=1) where user has 3 orders with 2 items each",
      "expected": "1 SQL query; returns 3 orders each containing 2 items",
    },
    {
      "input": "get_orders_with_items(user_id=99) where user has no orders",
      "expected": "1 SQL query; returns empty list",
    },
    {
      "input": "get_orders_with_items(user_id=2) where user has 50 orders",
      "expected": "1 SQL query regardless of order count",
    },
  ],
  difficulty=TaskDifficulty.MEDIUM,
)

n1_trajectories = [
  Trajectory(
    id="traj-join",
    task_id=n1_task.id,
    content=(
      "def get_orders_with_items(user_id: int) -> list[dict]:\n"
      "    rows = db.execute(\n"
      "        'SELECT o.*, i.* FROM orders o '\n"
      "        'LEFT JOIN items i ON i.order_id = o.id '\n"
      "        'WHERE o.user_id = ?', [user_id]\n"
      "    )\n"
      "    return _group_by_order(rows)\n"
    ),
    output="1 query. Returns correct nested structure for all test cases.",
    reward=1.0,
  ),
  Trajectory(
    id="traj-prefetch",
    task_id=n1_task.id,
    content=(
      "def get_orders_with_items(user_id: int) -> list[dict]:\n"
      "    orders = db.execute('SELECT * FROM orders WHERE user_id = ?', [user_id])\n"
      "    if not orders:\n"
      "        return []\n"
      "    order_ids = [o['id'] for o in orders]\n"
      "    placeholders = ','.join('?' * len(order_ids))\n"
      "    items = db.execute(\n"
      "        f'SELECT * FROM items WHERE order_id IN ({placeholders})', order_ids\n"
      "    )\n"
      "    items_by_order = {}\n"
      "    for item in items:\n"
      "        items_by_order.setdefault(item['order_id'], []).append(item)\n"
      "    for order in orders:\n"
      "        order['items'] = items_by_order.get(order['id'], [])\n"
      "    return orders\n"
    ),
    output="2 queries. Returns correct nested structure for all test cases.",
    reward=0.75,
  ),
  Trajectory(
    id="traj-cache",
    task_id=n1_task.id,
    content=(
      "@lru_cache(maxsize=256)\n"
      "def _fetch_items(order_id: int) -> list[dict]:\n"
      "    return db.execute('SELECT * FROM items WHERE order_id = ?', [order_id])\n"
      "\n"
      "def get_orders_with_items(user_id: int) -> list[dict]:\n"
      "    orders = db.execute('SELECT * FROM orders WHERE user_id = ?', [user_id])\n"
      "    for order in orders:\n"
      "        order['items'] = _fetch_items(order['id'])\n"
      "    return orders\n"
    ),
    output=(
      "Still N+1 queries on cold cache. Cached hits on warm cache. "
      "Returns correct data but does not eliminate the N+1 problem."
    ),
    reward=0.1,
  ),
]

# %% [markdown]
# ## 4. Task 2 — The Architecture Decision
#
# The second task is a different kind of problem. The team needs to pick a
# storage backend for their event log. Three agents gave three different
# recommendations: PostgreSQL with partitioning, a time-series database, and
# a Kafka sink into a columnar store. All three are reasonable. All three could
# be the right answer for *someone*.
#
# But which one is right for *this* team? That depends on whether they're
# already on PostgreSQL, how many events per day they expect, and whether Kafka
# is already in their stack. None of that was in the task description.
#
# We embed those facts as `human_clarifications` — a map from the architect's
# answer to the missing task context. When the oracle escalates, whatever the
# architect types unlocks the right branch:
#
# - `existing-postgres` → constrained to PostgreSQL → partitioning wins
# - `high-volume`       → 100M+ events/day → time-series wins
# - `event-streaming`   → Kafka already in stack → columnar sink wins
# - `unclear`           → not enough information yet → Judge evaluates as-is

# %%
event_log_task = Task(
  id="event-log-storage",
  description="Select a storage backend for the team's high-volume event log.",
  problem_statement=(
    "The team needs to store and query a continuous stream of application "
    "events (user actions, system metrics, audit records). Three AI agents "
    "produced architecture recommendations. Evaluate which recommendation "
    "is most appropriate.\n\n"
    "Note: The team's existing infrastructure and expected event volume were "
    "not provided."
  ),
  difficulty=TaskDifficulty.HARD,
  metadata={
    "human_clarifications": {
      "existing-postgres": {
        "problem_statement": (
          "The team runs PostgreSQL and has no budget to introduce new "
          "infrastructure. Expected volume: ~1M events/day. Query pattern: "
          "daily aggregations, not real-time. Select the best storage approach."
        ),
        "ground_truth": (
          "PostgreSQL with declarative table partitioning (PARTITION BY RANGE on "
          "event_time) is the correct choice. It requires no new operational "
          "overhead, fits the team's existing expertise, and handles 1M events/day "
          "with routine index maintenance. A time-series or streaming solution "
          "would introduce unnecessary complexity."
        ),
        "test_cases": [
          {
            "input": "daily aggregation query over 90 days of events",
            "expected": "partition pruning limits scan to relevant partitions; no full table scan",
          },
        ],
      },
      "high-volume": {
        "problem_statement": (
          "The team expects 100M+ events/day and needs sub-second query latency "
          "for real-time dashboards. Existing infrastructure is flexible. "
          "Select the best storage approach."
        ),
        "ground_truth": (
          "A purpose-built time-series database (TimescaleDB or InfluxDB) is the "
          "correct choice at this scale. Time-series databases compress temporal "
          "data efficiently, support continuous aggregates, and are optimized for "
          "the write-heavy, time-ordered workload. PostgreSQL partitioning would "
          "struggle with 100M events/day without significant tuning."
        ),
        "test_cases": [
          {
            "input": "rolling 5-minute window aggregation at 1M events/minute",
            "expected": "continuous aggregate refreshes in under 1 second",
          },
        ],
      },
      "event-streaming": {
        "problem_statement": (
          "Events already flow through Apache Kafka. The team needs analytical "
          "queries over the event history with flexible schema. "
          "Select the best storage approach."
        ),
        "ground_truth": (
          "A Kafka sink into a columnar store (ClickHouse or Apache Parquet on S3 "
          "with Athena) is the correct choice. It reuses the existing Kafka "
          "pipeline, eliminates dual-write complexity, and columnar storage excels "
          "at the aggregation-heavy query patterns typical of event analytics."
        ),
        "test_cases": [
          {
            "input": "ad-hoc aggregation across 12 months of event history",
            "expected": "columnar scan returns result in seconds without index hints",
          },
        ],
      },
      "unclear": {
        "problem_statement": (
          "The team's existing infrastructure and expected event volume are not "
          "yet determined. Provide guidance that applies across a range of scales "
          "and infrastructure starting points."
        ),
      },
    },
  },
)

event_log_trajectories = [
  Trajectory(
    id="traj-postgres",
    task_id=event_log_task.id,
    content=(
      "Recommendation: Use PostgreSQL with declarative table partitioning "
      "(PARTITION BY RANGE on event_time). Benefits: no new infrastructure, "
      "team expertise already in place, automatic partition pruning on "
      "time-range queries, pg_partman for automated partition management. "
      "Limitation: requires regular VACUUM and index maintenance at scale."
    ),
  ),
  Trajectory(
    id="traj-timeseries",
    task_id=event_log_task.id,
    content=(
      "Recommendation: Adopt a purpose-built time-series database such as "
      "TimescaleDB (PostgreSQL-compatible) or InfluxDB. Benefits: automated "
      "data compression for time-ordered writes, continuous aggregates for "
      "real-time dashboards, retention policies. TimescaleDB is a drop-in "
      "PostgreSQL extension, reducing migration friction. Best for high-volume "
      "or real-time query workloads."
    ),
  ),
  Trajectory(
    id="traj-kafka-sink",
    task_id=event_log_task.id,
    content=(
      "Recommendation: Sink the Kafka event stream into a columnar store "
      "(ClickHouse or S3 + Athena). Benefits: reuses existing Kafka pipeline, "
      "columnar compression for analytical queries, decouples event collection "
      "from storage. ClickHouse delivers sub-second aggregations at billions "
      "of rows. Best when Kafka is already in the stack and query patterns "
      "are aggregation-heavy."
    ),
  ),
]

# %% [markdown]
# ## 5. Provider, Scoring Config, and Criteria
#
# We're using Claude Sonnet as the evaluating model.
#
# The `granularity=20` setting is worth explaining. Rather than asking the
# model to pick a single score token (e.g., "4 out of 5"), the Verifier uses
# the full log probability distribution over 20 scoring tokens and computes a
# weighted expectation. This turns evaluation from a discrete decision into a
# **continuous reward signal** — finer-grained, less noisy, and more
# discriminative on close candidates. Anthropic's API does not expose raw log
# probabilities, so we fall back to parsing the letter score from the model's
# text output (`use_logprobs=False`); the granularity still applies to how the
# score is interpreted.
#
# The two criteria — correctness and quality — are deliberately broad so they
# apply to both a code fix and an architecture recommendation without needing
# task-specific wiring.

# %%
model = AnthropicProvider(model_id="claude-sonnet-4-6")

config = ScoringConfig(
  granularity=20,
  num_verifications=1,
  use_logprobs=False,  # required for Anthropic; score extracted from text
)

# Criteria apply to both tasks. Descriptions are broad enough to cover
# both structured code fixes and open-ended architecture recommendations.
criteria = [
  EvaluationCriterion(
    id="correctness",
    name="Correctness",
    description=(
      "Does the output correctly solve the stated problem? For code: does it "
      "satisfy all requirements and test cases? For recommendations: does it "
      "accurately address the constraints and trade-offs?"
    ),
    weight=2.0,
  ),
  EvaluationCriterion(
    id="quality",
    name="Quality",
    description=(
      "Is the output idiomatic, clear, and maintainable? Prefer standard "
      "patterns over ad-hoc workarounds. Penalize unnecessary complexity."
    ),
    weight=1.0,
  ),
]

print(f"Model    : {model.model_id}")
print(f"Criteria : {[c.name for c in criteria]}")

# %% [markdown]
# ## 6. Wiring Up the Strategies
#
# We instantiate both strategies and hand them to two routers — one per task.
#
# For Task 1 we use a tight `uncertainty_threshold=0.15`. The Verifier should
# produce a clearly discriminating spread across the three trajectories, so
# escalation should not fire and the oracle returns a clean result.
#
# For Task 2 we set `uncertainty_threshold=1.1`, which guarantees escalation
# always fires (all [0, 1] score spreads are below the threshold). This is a
# demo convenience — in production you would tune the threshold to your task
# distribution rather than forcing it.

# %%
verifier = VerifierStrategy(model, config, criteria)
judge = JudgeStrategy(model, config, criteria, reasoning_depth="detailed")

router = OracleRouter.default(
  verifier=verifier,
  judge=judge,
  uncertainty_threshold=0.15,
)

router_with_escalation = OracleRouter.default(
  verifier=verifier,
  judge=judge,
  human_oracle=ConsoleHumanOracle(),
  uncertainty_threshold=1.1,  # always escalate — demo convenience
)

# %% [markdown]
# ## 7. The Verifier
#
# The Verifier is designed for tasks where correctness can be grounded in
# evidence — test cases, reference solutions, execution output. It decomposes
# evaluation across criteria, repeats verification to reduce noise, and scores
# each trajectory against that structured evidence.
#
# Let's run it now. The Verifier will include the ground truth, the three test
# cases, and each trajectory's execution output in its prompt. The test case
# that matters most is the third: "1 SQL query regardless of order count."
# That's the requirement `traj-cache` violates on a cold cache. Watch where it
# lands relative to the other two trajectories.

# %%
print("\n" + "=" * 60)
print("VERIFIER — Task 1 (N+1 fix)")
print("=" * 60)

verifier_result = verifier.evaluate(n1_task, n1_trajectories)

print(f"  Best trajectory : {verifier_result.best_trajectory_id}")
for tid, score in sorted(verifier_result.trajectory_scores.items(), key=lambda kv: -kv[1].score):
  print(f"  {tid:<20s}  score={score.score:.3f}  confidence={score.confidence:.3f}")

# %% [markdown]
# ## 8. The Judge
#
# The Judge is designed for tasks where quality is holistic and correctness is
# not directly executable — essays, explanations, design proposals, qualitative
# comparisons. It applies a rubric and reasons about the trajectory as a whole,
# without requiring structured evidence.
#
# Now let's run the same task through the Judge and see if the verdict changes.
# It only sees the code — no ground truth, no test cases. It can still reason
# about query counts by reading the implementation, but it has to rely on
# general knowledge rather than a concrete requirement. The `@lru_cache` pattern
# is a recognized optimization technique; without a test case pinning the
# requirement to "always 1 query," the Judge may give `traj-cache` more credit.
#
# Compare the two rankings. If they agree, the task was straightforward enough
# that either strategy would have worked. If they disagree, the routing decision
# is doing real work — and that's exactly what we'll measure next.

# %%
print("\n" + "=" * 60)
print("JUDGE — Task 1 (N+1 fix)")
print("=" * 60)

judge_result = judge.evaluate(n1_task, n1_trajectories)

print(f"  Best trajectory : {judge_result.best_trajectory_id}")
for tid, score in sorted(judge_result.trajectory_scores.items(), key=lambda kv: -kv[1].score):
  print(f"  {tid:<20s}  score={score.score:.3f}  confidence={score.confidence:.3f}")

# %% [markdown]
# ## 9. The Router
#
# We've seen both strategies in isolation. Now let's let the oracle choose.
#
# Before evaluating anything, the router reads the task and looks for signals:
# Is there a ground truth? Test cases? Execution output attached to the
# trajectories? What does the task description's vocabulary tell us about the
# domain? Each signal is a vote. The router weighs them and commits.
#
# For this task, the signals are strong: ground truth, three test cases,
# execution output, and a description full of "fix", "bug", "query". We expect
# the router to pick the Verifier with high confidence and print its reasoning.

# %%
print("\n" + "=" * 60)
print("ROUTER — Task 1 (N+1 fix)")
print("=" * 60)

decision = router.route(n1_task, n1_trajectories)
print(f"  Selected strategy : {decision.selected_strategy.value}")
print(f"  Confidence        : {decision.confidence:.3f}")
print(f"  Reasoning         : {decision.reasoning}")

oracle_result, oracle_decision = router.evaluate(n1_task, n1_trajectories)
if not isinstance(oracle_result, EvaluationResult):
  raise RuntimeError("Evaluation is waiting on a human response.")
print(f"\n  Oracle best trajectory : {oracle_result.best_trajectory_id}")

# %% [markdown]
# ## 10. The Harness — Measuring What It Costs to Get This Wrong
#
# The harness runs both strategies against the same task and computes four
# hardness signals. Two are especially worth watching here:
#
# `strategy_disagreement` — if this is non-zero, the Verifier and the Judge
# picked different winners. That means routing to the wrong strategy would have
# returned the wrong answer. This is the metric that tells you how much the
# routing decision is actually worth.
#
# `verifier_wins` — if True, the Verifier's selection is closer to the ground
# truth. Combined with a non-zero disagreement, this confirms that for this
# task, the Verifier is not just different — it's better.

# %%
print("\n" + "=" * 60)
print("HARNESS — Task 1 hardness metrics")
print("=" * 60)

harness = EvaluationHarness(verifier=verifier, judge=judge, max_workers=1)
record = harness.run_single(n1_task, n1_trajectories)

print(f"  hardness_score        : {record.hardness_score:.4f}")
print(f"  score_spread          : {record.score_spread:.4f}")
print(f"  strategy_disagreement : {record.strategy_disagreement:.4f}")
print(f"  average_confidence    : {record.average_confidence:.4f}")
print(f"  strategies_agree      : {record.strategies_agree}")
print(f"  verifier_wins         : {record.verifier_wins}")

# %% [markdown]
# ## 11. The Human Oracle — When the Evidence Runs Out
#
# Task 2 is where things get interesting.
#
# There's no ground truth and no test cases — the router selects the Judge.
# The Judge evaluates the three architecture recommendations and produces
# scores, but the three trajectories are all defensible. The score spread
# is likely to be tight — tighter than our `uncertainty_threshold` of 0.15.
#
# When that happens, the oracle doesn't guess. It generates a question from
# its own evaluation reasoning — the specific thing it couldn't determine —
# and routes that question to the human oracle. You'll see it print the
# question and wait.
#
# Type one of these answers and watch what happens next:
#
#   existing-postgres  → unlocks ground truth + test case → re-routes to Verifier
#   high-volume        → unlocks ground truth + test case → re-routes to Verifier
#   event-streaming    → unlocks ground truth + test case → re-routes to Verifier
#   unclear            → no new evidence → re-evaluates with Judge as-is
#
# In every case, the oracle re-evaluates with the now-clarified task and
# returns a result it can actually stand behind.

# %%
print("\n" + "=" * 60)
print("ROUTER + HUMAN ORACLE — Task 2 (event log storage)")
print("=" * 60)

el_result, el_decision = router_with_escalation.evaluate(event_log_task, event_log_trajectories)
if not isinstance(el_result, EvaluationResult):
  raise RuntimeError("Evaluation is waiting on a human response.")

print(f"\n  Strategy   : {el_decision.selected_strategy.value}")
print(f"  Confidence : {el_decision.confidence:.3f}")
print(f"  Escalated  : {el_decision.metadata.get('human_escalated', False)}")
print(f"  Answer     : {el_decision.metadata.get('human_response', '—')}")
print(f"\n  Best trajectory : {el_result.best_trajectory_id}")
for tid, score in el_result.trajectory_scores.items():
  print(f"  {tid:<22s}  score={score.score:.3f}")

# %% [markdown]
# ## 12. What We Just Saw
#
# The Judge and the Verifier solve related but different evaluation problems.
# The Judge is strong for holistic, open-ended assessment. The Verifier is
# strong for structured, evidence-grounded correctness. Using one everywhere
# creates failure modes: Judge-only systems can be too coarse or too subjective;
# Verifier-only systems are unnecessarily rigid on open-ended tasks.
#
# On Task 1, the Verifier had a test case that pinned the query-count
# requirement explicitly. That's what let it confidently rank `traj-cache`
# below the other two. The Judge, without that anchor, may have given it more
# credit. The harness confirmed what was at stake: if `strategy_disagreement`
# was non-zero, routing to the wrong strategy would have returned the wrong
# winner.
#
# Task 2 showed the third layer. Even a correctly configured oracle can reach
# a point where no strategy can make the call — not because it's confused, but
# because the task itself is under-specified. The human oracle exists for exactly
# that moment: the oracle asks the one question that unlocks the decision,
# applies the answer, and re-evaluates with the context it was missing.
#
# One important caveat: the Oracle is a system design pattern, not ground truth
# itself. Routing quality depends on the signals and policies. The Judge and
# Verifier can both still be wrong. Task formulation strongly affects outcomes.
# The right mental model is *adaptive evaluation* — not omniscient evaluation.
