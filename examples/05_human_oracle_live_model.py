#!/usr/bin/env python3

# %% [markdown]
# # HumanOracle with Anthropic Claude: Jurisdiction-Gated Compliance Review
#
# This example is drawn from the selective-escalation framing in HiL-Bench
# (arxiv 2604.09408): an oracle that evaluates AI agent outputs should recognize
# when it cannot make a reliable judgment and ask a targeted clarifying question,
# rather than guessing.
#
# **Scenario**: A legal-tech assistant asked three AI agents to assess whether a
# SaaS company's practice of processing EU user data on US servers complies with
# applicable regulations. All three answers are internally consistent, but their
# correctness depends entirely on whether the customer's Data Processing
# Agreement includes EU Standard Contractual Clauses (SCCs) — information that
# was not supplied in the original task.
#
# Without that fact, the oracle cannot distinguish the candidates. It escalates
# to a compliance officer, applies the clarification, and re-evaluates.
#
# Prerequisites:
#
# - `ANTHROPIC_API_KEY` must be set in your environment.
# - Run `uv sync` to install dependencies.
#
# Run as a script:
#
# ```bash
# uv run python examples/05_human_oracle_anthropic.py
# ```
#
# Cost note: `num_verifications=1` keeps token usage minimal.

# %% [markdown]
# ## 1. Imports

# %%
from __future__ import annotations

from dataclasses import dataclass

from dotenv import load_dotenv

from llm_oracle import (
  AnthropicProvider,
  EvaluationCriterion,
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
# ## 2. Host-Side HumanOracle
#
# In production this would be a webhook to your compliance ticket system or an
# async approval queue. Here it blocks on standard input.


# %%
@dataclass
class ConsoleHumanOracle:
  """HumanOracle that asks the current interactive user."""

  responder_id: str = "compliance-officer"

  def ask(self, request: HumanRequest) -> HumanResponse:
    print("\n── Human escalation ──────────────────────────────────")
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
# ## 3. Task: EU–US Data Transfer Compliance Assessment
#
# A B2B SaaS customer asked: "Does our practice of storing EU user data on
# US-based servers comply with applicable regulations?"
#
# Three AI agents produced assessments. All are defensible in isolation, but
# each implicitly assumes a different answer to the SCC question — which the
# original task did not supply.
#
# `human_clarifications` maps the compliance officer's answer to task field
# overrides:
#
# - `scc`      → SCCs are in place → transfer is lawful → Verifier path
# - `no-scc`   → no SCCs in place  → transfer is unlawful → Verifier path
# - `unclear`  → DPA under legal review → no ground truth → Judge path

# %%
task = Task(
  id="eu-us-data-transfer",
  description=(
    "Assess whether the customer's EU–US data transfer practice complies "
    "with applicable regulations."
  ),
  problem_statement=(
    "A SaaS customer stores personal data of EU users on servers located in "
    "the United States. They ask: 'Does this comply with applicable data "
    "protection regulations?' Three AI compliance agents produced assessments. "
    "Evaluate which assessment is most accurate and actionable.\n\n"
    "Note: The customer's Data Processing Agreement (DPA) status regarding "
    "EU Standard Contractual Clauses (SCCs) was not provided."
  ),
  difficulty=TaskDifficulty.HARD,
  metadata={
    "human_clarifications": {
      "scc": {
        "problem_statement": (
          "A SaaS customer stores personal data of EU users on US servers. "
          "Their DPA includes EU Standard Contractual Clauses (SCCs) approved "
          "under GDPR Article 46. Assess compliance."
        ),
        "ground_truth": (
          "The transfer is lawful. EU SCCs under Article 46 GDPR provide a valid "
          "legal mechanism for EU–US personal data transfers. The customer must "
          "also complete a Transfer Impact Assessment (TIA) and implement "
          "supplementary measures if US surveillance laws pose a residual risk."
        ),
        "test_cases": [
          {
            "input": "Customer has SCCs in DPA and conducts annual TIA",
            "expected": "transfer is compliant; recommend documenting TIA results",
          },
          {
            "input": "Customer has SCCs but no TIA on file",
            "expected": "compliant in mechanism but TIA required to confirm",
          },
        ],
      },
      "no-scc": {
        "problem_statement": (
          "A SaaS customer stores personal data of EU users on US servers. "
          "Their DPA does not include Standard Contractual Clauses or any other "
          "Article 46 transfer mechanism. Assess compliance."
        ),
        "ground_truth": (
          "The transfer is unlawful under GDPR Chapter V. Without SCCs, Binding "
          "Corporate Rules, or an adequacy decision covering the US, the customer "
          "has no valid legal basis for the transfer. They must either execute SCCs "
          "immediately or cease transferring EU personal data to US servers."
        ),
        "test_cases": [
          {
            "input": "Customer relies on consent for the transfer",
            "expected": "consent is not a valid Article 46 mechanism; SCCs required",
          },
        ],
      },
      "unclear": {
        "problem_statement": (
          "A SaaS customer stores personal data of EU users on US servers. "
          "Their DPA status is under review by legal counsel and the applicable "
          "transfer mechanism has not been confirmed. Assess what guidance can "
          "be offered given this uncertainty."
        ),
      },
    },
  },
)

# %% [markdown]
# ## 4. Candidate Trajectories
#
# - `traj-compliant`: assumes SCCs are in place → recommends proceed with TIA.
# - `traj-non-compliant`: assumes no valid mechanism → recommends halt transfer.
# - `traj-conditional`: avoids the SCC assumption → recommends legal review first.

# %%
trajectories = [
  Trajectory(
    id="traj-compliant",
    task_id=task.id,
    content=(
      "Assessment: The EU–US data transfer is lawful provided the customer's "
      "DPA includes Standard Contractual Clauses (SCCs) under GDPR Article 46. "
      "SCCs are the standard mechanism used by US-based cloud providers. "
      "Recommendation: confirm SCCs are executed and conduct a Transfer Impact "
      "Assessment (TIA) to document that supplementary measures address any "
      "residual risk from US surveillance law (FISA 702). Archive the TIA "
      "for regulator requests."
    ),
  ),
  Trajectory(
    id="traj-non-compliant",
    task_id=task.id,
    content=(
      "Assessment: The EU–US data transfer is not lawful. Following Schrems II "
      "(C-311/18), the EU–US Privacy Shield was invalidated. Without a valid "
      "Article 46 mechanism — SCCs, Binding Corporate Rules, or an adequacy "
      "decision — there is no legal basis for transferring EU personal data to "
      "US servers. Recommendation: halt the transfer immediately and execute "
      "Standard Contractual Clauses before resuming. Failure to act exposes the "
      "customer to GDPR fines of up to 4% of global annual turnover."
    ),
  ),
  Trajectory(
    id="traj-conditional",
    task_id=task.id,
    content=(
      "Assessment: Compliance depends on whether a valid transfer mechanism is "
      "in place. GDPR Chapter V requires one of: SCCs (Article 46(2)(c)), "
      "Binding Corporate Rules (Article 47), or an adequacy decision. The "
      "EU–US Data Privacy Framework (2023) provides adequacy for certified "
      "US organizations. Without knowing which mechanism, if any, the customer "
      "relies on, a definitive compliance determination cannot be made. "
      "Recommendation: legal counsel should audit the DPA before any assessment "
      "is issued."
    ),
  ),
]

# %% [markdown]
# ## 5. Provider, Scoring Config, and Criteria

# %%
model = AnthropicProvider(model_id="claude-sonnet-4-6")

config = ScoringConfig(
  granularity=20,
  num_verifications=1,
  use_logprobs=False,  # required for Anthropic; score extracted from text
)

criteria = [
  EvaluationCriterion(
    id="accuracy",
    name="Accuracy",
    description=(
      "Is the legal analysis correct given the applicable regulation and the "
      "facts supplied? Penalize assumptions that contradict the stated facts."
    ),
    weight=2.0,
  ),
  EvaluationCriterion(
    id="actionability",
    name="Actionability",
    description=(
      "Does the assessment give the customer a clear, concrete next step? "
      "Vague recommendations ('consult a lawyer') score lower than specific "
      "procedural guidance."
    ),
    weight=1.0,
  ),
]

# %% [markdown]
# ## 6. Build the Router
#
# `uncertainty_threshold=1.1` guarantees escalation fires in this example —
# all [0, 1] score spreads are below the threshold. In production, tune this
# to your task distribution (the default is 0.10).
#
# When the oracle cannot distinguish trajectories, it generates a HumanRequest
# from the evaluation reasoning and routes it to `ConsoleHumanOracle`.
#
# Try each answer to see different outcomes:
#
# - `scc`     → adds ground truth and test cases → Verifier re-evaluates
# - `no-scc`  → adds ground truth and test cases → Verifier re-evaluates
# - `unclear` → no structured evidence added     → Judge re-evaluates

# %%
verifier = VerifierStrategy(model, config, criteria)
judge = JudgeStrategy(model, config, criteria)

router = OracleRouter.default(
  verifier=verifier,
  judge=judge,
  human_oracle=ConsoleHumanOracle(),
  uncertainty_threshold=1.1,  # always escalate for demo purposes
)

# %% [markdown]
# ## 7. Evaluate
#
# Flow:
#   1. Oracle routes to Judge (no ground truth yet) and evaluates.
#   2. Score spread is below the threshold → oracle cannot distinguish.
#   3. Oracle generates a HumanRequest citing the ambiguity.
#   4. Compliance officer answers; clarification is applied to the task.
#   5. Oracle re-routes (Verifier if SCC status supplied) and re-evaluates.

# %%
result, decision = router.evaluate(task, trajectories)

print("\n── Routing decision ──────────────────────────────────")
print(f"  Strategy   : {decision.selected_strategy.value}")
print(f"  Confidence : {decision.confidence:.3f}")
print(f"  Escalated  : {decision.metadata.get('human_escalated', False)}")
print(f"  Answer     : {decision.metadata.get('human_response', '—')}")

print("\n── Evaluation result ─────────────────────────────────")
print(f"  Best trajectory : {result.best_trajectory_id}")
for tid, score in result.trajectory_scores.items():
  print(f"  {tid:<28s}  score={score.score:.3f}")

# %% [markdown]
# ## 8. What this example shows
#
# The oracle evaluated three legally defensible answers but could not pick one
# without external knowledge (the SCC status). Rather than guessing, it:
#
# - Generated a targeted question from its own evaluation reasoning.
# - Delegated the one unknowable fact to a human.
# - Applied the clarification and re-evaluated with the now-sufficient context.
#
# This is the HiL-Bench selective-escalation pattern in practice: the oracle
# does not escalate speculatively before evaluation — it escalates at the
# exact point where evaluation evidence runs out.
