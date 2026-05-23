# Architecture

## Overview

LLM-as-an-Oracle is an evaluation orchestrator. It sits above
`LLM-as-a-Judge`, `LLM-as-a-Verifier`, and optional reliability strategies such
as adversarial claim verification. It decides which evaluator is the better fit
for a given task.

The Oracle is not a third evaluator. Its job is routing:

1. inspect the task and candidate trajectories
2. choose a configured evaluation strategy
3. return the selected evaluator's result together with the routing decision

This architecture exists because the two evaluation styles are complementary:

- `Judge` is stronger for holistic, open-ended, or subjective evaluation
- `Verifier` is stronger when evaluation can use concrete evidence such as test
  cases, reference answers, expected outputs, or execution results
- `AdversarialVerifier` is stronger when a trajectory is a claim that needs
  evidence-based confirmation or rejection

## System Shape

```text
CLI / API
   |
   v
OracleRouter
   |
   +--> SignalExtractor
   |      Produces RoutingSignals from Task + Trajectories
   |
   +--> PolicyChain
   |      Aggregates weighted votes from routing policies
   |
   v
Selected strategy
   |
   +--> JudgeStrategy
   |      Holistic, rubric-based evaluation
   |
   +--> VerifierStrategy
   |      Structured, evidence-sensitive evaluation
   |
   +--> AdversarialVerifierStrategy
          Claim verification with confirmation and challenge passes
   |
   v
EvaluationResult + DetailedRoutingDecision


Separate research path:
EvaluationHarness
   |
   v
Runs both strategies on the same task and compares them
```

## Main Components

### Oracle Router

`OracleRouter` is the core orchestration layer. In the default configuration it
does not call an LLM to decide. It uses deterministic signals and policy votes
to select the evaluation strategy.

The default routing flow is:

1. `SignalExtractor` derives `RoutingSignals`
2. `PolicyChain` runs the configured routing policies
3. weighted vote totals are aggregated per configured strategy
4. the higher-confidence strategy wins
5. if confidence is below `0.60`, the router falls back to `Judge`

`OracleRouter.route(...)` returns a `DetailedRoutingDecision` with:

- selected strategy
- aggregate confidence
- reasoning text
- per-policy votes
- extracted signals
- elapsed routing time

`OracleRouter.evaluate(...)` then runs only the selected strategy and returns:

- `EvaluationResult`, or `HumanResponsePending` when human escalation is
  deferred
- `DetailedRoutingDecision`

### Routing Signals

The default router derives these signals from `Task` and `Trajectory` data:

- `has_ground_truth`
- `has_test_cases`
- `trajectory_count`
- `stated_difficulty`
- `verifiable_keyword_density`
- `judgment_keyword_density`
- `problem_length`
- `output_available`
- `is_claim_verification`
- `prior_hardness`

These are simple, inspectable features. The router is designed to make its
assumptions explicit.

### Routing Policies

The default `PolicyChain` uses six deterministic base policies:

1. `PriorHardnessPolicy`
2. `GroundTruthPolicy`
3. `KeywordDomainPolicy`
4. `DifficultyPolicy`
5. `OutputAvailabilityPolicy`
6. `TrajectoryCountPolicy`

When an adversarial strategy is configured, `OracleRouter.default(...)` also
installs `ClaimVerificationPolicy`. It strongly prefers
`StrategyType.ADVERSARIAL` only when
`task.metadata["evaluation_mode"] == "claim_verification"`.

Each policy emits a `PolicyVote`:

- preferred strategy
- confidence
- weight
- signals used
- reasoning

The chain aggregates `confidence * weight` per strategy and normalizes the
winning strategy into the final routing confidence.

### Judge Strategy

`JudgeStrategy` implements holistic evaluation for open-ended tasks.

Current characteristics:

- rubric-based scoring
- text-based model evaluation
- pairwise comparison support
- weighted aggregation across provided criteria

The strategy consumes criteria supplied to it. It does not define the Oracle;
it is one branch the Oracle may choose.

### Verifier Strategy

`VerifierStrategy` implements more structured evaluation for evidence-grounded
tasks.

Current characteristics:

- repeated verification
- pairwise tournament-style comparison
- logprob-aware score extraction when the provider supports it
- aggregation across criteria and repeated checks

This strategy is useful when task correctness can be grounded in stronger
evidence than holistic judgment alone.

### Adversarial Verifier Strategy

`AdversarialVerifierStrategy` verifies a claim represented by a trajectory. It
wraps two verifier strategies:

- confirmation verifier: checks whether the claim is supported
- challenge verifier: checks whether evidence supports rejecting the claim

It returns a normal `EvaluationResult` and stores the adversarial decision in
`ScoreResult.metadata["decision"]` as `confirmed`, `rejected`, or `uncertain`.
The `uncertain` decision is intended for host-managed human escalation.

### Evaluation Harness

`EvaluationHarness` is the comparison and benchmarking layer. It is separate
from routing.

Its role is:

- run both strategies for the same task
- compute comparison metrics
- estimate task hardness
- support research on when `Judge` or `Verifier` performs better

The harness supports parallelism across tasks. It is not the production routing
path.

### Providers

`core/providers.py` exposes a common `LanguageModel` interface across model
backends. Current providers include:

- OpenAI
- Anthropic
- Gemini
- StubProvider

This keeps strategy code provider-agnostic.

## Package Layout

```text
src/llm_oracle/
├── core/
│   ├── models.py
│   ├── providers.py
│   └── strategy.py
├── strategies/
│   ├── adversarial.py
│   ├── judge.py
│   └── verifier.py
├── routing/
│   └── router.py
├── evaluation/
│   └── harness.py
├── _cli.py
└── __main__.py
```

## Design Properties

- Adaptive evaluation: the system chooses an evaluator instead of forcing one
  method for every task
- Deterministic routing: default Oracle routing is heuristic and auditable
- Clear separation of concerns: routing, evaluation, benchmarking, and provider
  integration are separate layers
- Extensibility: policies, strategies, and providers can be extended without
  changing the whole stack
- Inspectability: routing outputs expose both decision and rationale

## Relationship Between Oracle and Harness

`OracleRouter` and `EvaluationHarness` serve different purposes.

- `OracleRouter`: choose one evaluator for a task and run it
- `EvaluationHarness`: run both evaluators and compare them

The router is the production decision layer. The harness is the research and
benchmarking layer.

## Further Reading

- [`docs/core/oracle-algorithm.md`](core/oracle-algorithm.md) for the routing
  algorithm
- [`README.md`](../README.md) for the project-level definition
- [`examples/llm_as_an_oracle_tutorial.py`](../examples/llm_as_an_oracle_tutorial.py)
  for end-to-end usage
