#!/usr/bin/env python3

# %% [markdown]
# # HumanOracle: Uncertainty-Based Human Escalation
#
# This example demonstrates selective escalation:
#
# 1. The router evaluates trajectories using the selected strategy.
# 2. If the score spread across trajectories falls below an uncertainty
#    threshold, the oracle cannot distinguish the candidates and asks the
#    human for clarification.
# 3. The clarified task is re-routed and re-evaluated.
#
# `HumanOracle` is not an evaluation strategy. It is the boundary where the
# host application decides how and whom to ask. The oracle owns uncertainty
# detection and question generation; the host owns delivery.
#
# Run as a script:
#
# ```bash
# uv run python examples/04_human_oracle.py
# ```
#
# Or run cell by cell in an editor that supports `# %%` Python cells.

# %% [markdown]
# ## 1. Imports

# %%
from __future__ import annotations

from dataclasses import dataclass

from llm_oracle import (
  EvaluationCriterion,
  EvaluationResult,
  HumanRequest,
  HumanResponse,
  JudgeStrategy,
  OracleRouter,
  ScoringConfig,
  StubProvider,
  Task,
  TaskDifficulty,
  Trajectory,
  VerifierStrategy,
)

# %% [markdown]
# ## 2. Host-Side HumanOracle
#
# The library owns the boundary types. The application owns the implementation.
#
# `ConsoleHumanOracle` blocks, asks the current user, and returns a
# `HumanResponse`. In production, replace this with a webhook, ticket
# integration, or a simulated benchmark human.


# %%
@dataclass
class ConsoleHumanOracle:
  """HumanOracle that asks the current interactive user."""

  responder_id: str = "console-user"

  def ask(self, request: HumanRequest) -> HumanResponse:
    """Ask a question with standard input.

    Raises:
      ValueError: If the user submits an empty answer.
    """
    print("\nHuman input needed")
    print(f"Task    : {request.task_id}")
    print(f"Reason  : {request.reason}")
    print(f"Question: {request.question}")

    answer = input("Answer> ").strip()
    if not answer:
      raise ValueError("Human answer cannot be empty.")

    return HumanResponse(
      request_id=request.id,
      answer=answer,
      responder_id=self.responder_id,
      metadata={"source": "console"},
    )


# %% [markdown]
# ## 3. A Task With Ambiguous Requirements
#
# The task has contradictory source notes. The oracle evaluates first; if the
# score spread across trajectories is below the uncertainty threshold, it
# auto-generates a clarifying question from the evaluation reasoning and asks
# the human.
#
# `human_clarifications` maps human answer keys to task field overrides.
# When the human types "order-form", the task gains a ground truth and test
# cases that allow the verifier to produce a discriminating result.

# %%
task = Task(
  id="refund-policy-summary",
  description="Summarize the refund policy for a customer-facing help article.",
  problem_statement=(
    "Write a concise summary of the refund policy. The source notes say "
    "refunds are allowed within 30 days, but a later note says enterprise "
    "contracts follow their signed order form. Clarify which rule applies "
    "to enterprise customers before evaluating candidate summaries."
  ),
  difficulty=TaskDifficulty.MEDIUM,
  metadata={
    "human_clarifications": {
      "order-form": {
        "problem_statement": (
          "Write a concise summary of the refund policy. Enterprise customers "
          "are governed by their signed order form, not the standard 30-day rule."
        ),
        "ground_truth": (
          "Enterprise customers follow their signed order form for refunds. "
          "The 30-day policy applies to non-enterprise accounts only."
        ),
        "test_cases": [
          {
            "input": "enterprise customer requests refund after 45 days",
            "expected": "refer to signed order form",
          }
        ],
      },
      "30-day": {
        "problem_statement": (
          "Write a concise summary of the refund policy. All customers, including "
          "enterprise accounts, are eligible for refunds within 30 days of purchase."
        ),
        "ground_truth": (
          "All customers including enterprise accounts may request a refund within "
          "30 days of purchase."
        ),
        "test_cases": [
          {
            "input": "enterprise customer requests refund after 15 days",
            "expected": "refund approved under 30-day policy",
          }
        ],
      },
      "unclear": {
        "problem_statement": (
          "Write a concise summary of the refund policy. The applicable rule for "
          "enterprise customers is ambiguous; note this uncertainty in your summary."
        ),
      },
    },
  },
)

# %% [markdown]
# ## 4. Candidate Trajectories

# %%
trajectories = [
  Trajectory(
    id="summary-order-form",
    task_id=task.id,
    content=(
      "Refund policy: standard accounts receive refunds within 30 days. "
      "Enterprise customers follow the terms in their signed order form."
    ),
  ),
  Trajectory(
    id="summary-30-day",
    task_id=task.id,
    content=(
      "Refund policy: all customers, including enterprise accounts, are eligible "
      "for a full refund within 30 days of purchase."
    ),
  ),
]

# %% [markdown]
# ## 5. Build the Router
#
# `uncertainty_threshold=1.1` guarantees the escalation fires in this example
# (since all [0, 1] score spreads are below the threshold). In production, use
# the default (0.10) or tune to your task distribution.
#
# When the oracle cannot distinguish trajectories, it auto-generates a question
# from the evaluation reasoning and asks `ConsoleHumanOracle`.
#
# Try each answer to see different routing outcomes:
#
# - `order-form` or `30-day` → adds `ground_truth` and `test_cases` → Verifier
# - `unclear`               → no evidence added → Judge

# %%
criteria = [
  EvaluationCriterion(
    id="accuracy",
    name="Accuracy",
    description="Does the summary accurately reflect the applicable refund policy?",
    weight=2.0,
  ),
  EvaluationCriterion(
    id="clarity",
    name="Clarity",
    description="Is the summary clear and easy for a customer to understand?",
    weight=1.0,
  ),
]

stub = StubProvider(model_id="stub", seed=0)
config = ScoringConfig(granularity=20, num_verifications=2)

verifier = VerifierStrategy(stub, config, criteria)
judge = JudgeStrategy(stub, config, criteria)

router = OracleRouter.default(
  verifier=verifier,
  judge=judge,
  human_oracle=ConsoleHumanOracle(),
  uncertainty_threshold=1.1,  # always escalate for demo purposes
)

# %% [markdown]
# ## 6. Evaluate
#
# The router evaluates first. If the score spread is below the uncertainty
# threshold, it auto-generates a `HumanRequest` from the evaluation reasoning,
# asks `ConsoleHumanOracle`, applies the matching clarification, then
# re-routes and re-evaluates with the clarified task.

# %%
result, decision = router.evaluate(task, trajectories)

print("\n# Routing decision")
print(f"  Strategy   : {decision.selected_strategy.value}")
print(f"  Confidence : {decision.confidence:.3f}")
print(f"  Escalated  : {decision.metadata.get('human_escalated', False)}")
print(f"  Answer     : {decision.metadata.get('human_response', '—')}")

print("\n# Evaluation result")
if not isinstance(result, EvaluationResult):
  raise RuntimeError("Evaluation is waiting on a human response.")
print(f"  Best trajectory : {result.best_trajectory_id}")
for tid, score in result.trajectory_scores.items():
  print(f"  {tid:<30s}  score={score.score:.3f}")

# %% [markdown]
# ## 7. Principle

# %%
print("\n# Principle")
print("  The oracle evaluates first, then detects when human input is needed.")
print("  The host application decides how and whom to ask.")
