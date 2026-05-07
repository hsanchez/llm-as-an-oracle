# 001 Human Escalation

## Status

Revised

## Context

The oracle routes evaluation work between two strategies:

- Judge, for rubric-based or semantic evaluation.
- Verifier, for evidence-grounded evaluation.

Some tasks cannot be evaluated correctly from the available context. They may
contain missing requirements, ambiguous instructions, or contradictory
information. When this happens, both strategies produce low-spread scores —
the oracle cannot distinguish which trajectory is better — rather than wrong
scores.

HiL-Bench frames this as a selective escalation problem: agents should
recognize when they cannot make a reliable judgment and ask targeted
clarifying questions, rather than guessing.

## Decision

Human escalation is triggered by evaluation uncertainty, not by pre-flight
task analysis.

The oracle evaluates first. If the score spread across trajectories falls
below an uncertainty threshold, the oracle cannot distinguish the candidates
and asks the human for clarification. The clarified task is then re-evaluated.

The flow is:

```
task + trajectories
       │
       ▼
  route → Judge or Verifier
       │
       ▼
  evaluate
       │
       ├── score spread ≥ threshold ──→ return result
       │
       └── score spread < threshold (oracle uncertain)
              │
              ├── no human_oracle ──→ return uncertain result
              │
              └── human_oracle configured
                     │
                     ▼
               build HumanRequest from evaluation result
               (task context + reasoning from evaluation)
                     │
                     ▼
               HumanOracle.ask(request)
                     │
                     ▼
               apply clarification to task
                     │
                     ▼
               re-route → Judge or Verifier
                     │
                     ▼
               re-evaluate → return clarified result
```

The `HumanRequest` is generated from the evaluation result — the task
description and the oracle's own reasoning about why it could not distinguish
the trajectories. Task authors do not pre-author escalation metadata.

The host application owns how and whom to ask by providing a `HumanOracle`.
The oracle owns the uncertainty detection and question generation.

The core package provides:

- `HumanRequest` — targeted clarifying question with reason
- `HumanResponse` — the human's answer
- `HumanResponsePending` — handle for deferred responses (not yet supported)
- `HumanOracle` — host-implemented adapter for delivering questions and receiving answers

Pre-authored `human_clarifications` in `task.metadata` are supported as a
testing and demo convenience — they map normalized answer keys to task field
overrides. When absent, the human's free-form answer is appended to the
problem statement.

## Consequences

Judge and Verifier remain the only evaluation strategies.

Human escalation surfaces at the natural point of uncertainty — after the
oracle has tried to evaluate and failed to produce a discriminating result —
rather than as a speculative pre-flight classifier.

The host application remains free to implement `HumanOracle` as a console
prompt, async webhook, ticket system, or simulated benchmark human.

`HumanResponsePending` is recognized but not resumed; deferred workflows are
out of scope for this revision.

This is a minimal scaffold. It does not implement Ask-F1, multi-blocker
scoring, or the full HiL-Bench evaluation protocol. Those require a blocker
registry per task and a separate benchmark harness.
