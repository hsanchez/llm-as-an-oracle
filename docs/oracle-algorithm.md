# Oracle Algorithm

This document defines what `LLM-as-an-Oracle` means in this repository.

In this codebase, the Oracle is not a third evaluator. It is the routing
process that decides whether a task should be evaluated by `VerifierStrategy`
or `JudgeStrategy`.

## Purpose

The Oracle exists because `Judge` and `Verifier` are useful on different task
shapes:

- `Verifier` is stronger when evaluation can use concrete evidence such as
  reference solutions, test cases, expected outputs, or execution results.
- `Judge` is stronger when evaluation is more holistic, open-ended, or
  subjective.

The Oracle tries to choose the better evaluator for each `(task, trajectories)`
pair instead of forcing a single strategy for all workloads.

## Algorithm

The default Oracle algorithm is implemented by `OracleRouter.default(...)` in
[`src/llm_oracle/routing/router.py`](/Users/huascar/dev/github.com/hsanchez/llm-as-oracle/src/llm_oracle/routing/router.py:833).

It works in four stages.

```text
Input: Task + Trajectories
            |
            v
+----------------------- LLM-as-an-Oracle -----------------------+
|                                                               |
|  1. SignalExtractor ──► RoutingSignals                        |
|                               |                               |
|  2.             PolicyChain (6 policies, weighted votes)      |
|                               |                               |
|  3.          verifier_total vs judge_total                    |
|                               |                               |
|  4.      confidence ≥ 0.60? ──no──► Judge (fallback)         |
|                    |                                          |
|                   yes                                         |
|             ┌─────┴─────┐                                     |
|          Verifier      Judge  (winning strategy only)         |
+-------------|-------------|------------------------------------|
              └──────┬──────┘
                     v
       EvaluationResult + DetailedRoutingDecision
```

The router first extracts structured signals from the task and trajectories
(step 1), then runs a fixed chain of six deterministic policies that each cast
a weighted vote for `Verifier` or `Judge` (step 2). Those votes are aggregated
into a confidence score for each side (step 3). If the winning side clears the
0.60 threshold the corresponding strategy is selected; otherwise the router
falls back to `Judge` (step 4). Only the selected strategy runs; the other
branch is never invoked.

### 1. Extract routing signals

`SignalExtractor` converts raw task and trajectory fields into a structured
`RoutingSignals` object.

The default signals are:

- `has_ground_truth`
- `has_test_cases`
- `trajectory_count`
- `stated_difficulty`
- `verifiable_keyword_density`
- `judgement_keyword_density`
- `problem_length`
- `output_available`
- `prior_hardness`

These are computed from fields already present on `Task` and `Trajectory`, plus
an optional cached hardness score from the evaluation harness.

### 2. Collect policy votes

The default router runs a fixed chain of routing policies:

1. `PriorHardnessPolicy`
2. `GroundTruthPolicy`
3. `KeywordDomainPolicy`
4. `DifficultyPolicy`
5. `OutputAvailabilityPolicy`
6. `TrajectoryCountPolicy`

Each policy emits a `PolicyVote`:

- preferred strategy: `VERIFIER` or `JUDGE`
- confidence in `[0, 1]`
- policy weight
- signals used
- reasoning text

The policies are deterministic and do not call an LLM in the hot path.

### 3. Aggregate weighted confidence

`PolicyChain` sums `confidence * weight` for each side:

- verifier total
- judge total

The winning side is the one with the larger weighted total. Its normalized
share becomes the final routing confidence.

If the aggregate confidence is below the router threshold, the chain falls back
to `Judge`. The default threshold is `0.60`.

## Default policy behavior

The intent of each built-in policy is:

- `PriorHardnessPolicy`: use prior harness hardness when available; high
  hardness pushes toward `Verifier`, low hardness toward `Judge`.
- `GroundTruthPolicy`: prefer `Verifier` when reference answers or test cases
  exist.
- `KeywordDomainPolicy`: prefer `Verifier` for code-like or executable tasks,
  and `Judge` for open-ended writing or analysis tasks.
- `DifficultyPolicy`: prefer `Judge` for easy tasks and `Verifier` for hard
  tasks.
- `OutputAvailabilityPolicy`: prefer `Verifier` when trajectories include
  execution output.
- `TrajectoryCountPolicy`: prefer `Judge` for single-trajectory or large-`n`
  candidate sets, and slight `Verifier` preference for small comparisons.

These policies are heuristics. The Oracle is not claiming to recover a
ground-truth optimal evaluator in every case; it is a practical routing
strategy that makes its assumptions explicit and inspectable.

## Outputs

`OracleRouter.route(...)` returns a `DetailedRoutingDecision` containing:

- `selected_strategy`
- `confidence`
- `reasoning`
- `policy_votes`
- `signals`
- `elapsed_ms`

`OracleRouter.evaluate(...)` then:

1. calls `route(...)`
2. resolves the winning strategy
3. delegates to that strategy's `evaluate(...)`
4. returns both the evaluation result and the routing decision

This separation matters because the routing step is independently inspectable.

## Relationship to the Tutorial

The tutorial in
[`examples/llm_as_an_oracle_tutorial.py`](/Users/huascar/dev/github.com/hsanchez/llm-as-oracle/examples/llm_as_an_oracle_tutorial.py:1)
shows how to use the Oracle API end to end.

That file is intentionally practical. This document is the canonical place to
describe the routing algorithm itself.

## When to Read What

- Read `README.md` for the short project-level definition.
- Read this document for the routing logic and design assumptions.
- Read the tutorial for usage examples and expected outputs.
