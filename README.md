## LLM AS AN ORACLE

This repository combines two related evaluation paradigms:

- `LLM-as-a-Judge`: a model evaluates candidate answers or trajectories holistically and assigns a score or preference.
- `LLM-as-a-Verifier`: a more structured evaluation framework that improves discrimination by using finer scoring granularity, repeated verification, and criteria decomposition.

The `LLM-as-a-Verifier` framing used here is inspired by **"LLM-as-a-Verifier: A
General-Purpose Verification Framework"**:

- [LLM-as-a-Test-Oracle](https://dl.acm.org/doi/10.1145/3650212.3680308)
- [LLM-as-a-Verifier](https://llm-as-a-verifier.notion.site/)

That project presents Verifier as an improvement over standard Judge-style
evaluation for difficult trajectory comparison, especially when coarse scoring
produces too many ties.

In practice, these approaches are best treated as complementary:

- Use `Judge` when evaluation is open-ended, rubric-based, holistic, or subjective.
- Use `Verifier` when evaluation can be grounded in stronger evidence such as reference solutions, test cases, expected outputs, or execution results.
- Use `AdversarialVerifier` when a trajectory is a claim that needs confirmation and challenge passes.
- Use the `Oracle` layer when you want an evaluation orchestrator that routes to the better configured strategy for the task.

This repository therefore treats `LLM-as-an-Oracle` as an orchestration layer:

- it exposes both `Judge` and `Verifier`
- it supports adversarial claim verification as an optional strategy
- it compares them in a shared evaluation harness
- it routes tasks based on explicit, inspectable task signals

## Public API

`llm_oracle` exposes a broad top-level public API on purpose.

Most users should start with this common import set:

```python
from llm_oracle import (
  AdversarialVerifierStrategy,
  EvaluationCriterion,
  EvaluationHarness,
  JudgeStrategy,
  OracleRouter,
  ScoringConfig,
  StubProvider,
  Task,
  TaskDifficulty,
  Trajectory,
  VerifierStrategy,
)
```

Advanced names remain available from the top level when you need them:

- routing internals such as `RoutingPolicy`, `PolicyChain`, `SignalExtractor`, and `DetailedRoutingDecision`
- provider extension hooks such as `create_provider`, `get_provider`, and `register_provider`
- lower-level abstractions such as `BaseStrategy` and `LanguageModel`

The policy is convenience first:

- the common set is the default user-facing surface
- advanced names are still public, but not the first thing new users need to learn

## Installation

Install directly from GitHub with `uv`:

```bash
uv add "llm-as-an-oracle @ git+https://github.com/hsanchez/llm-as-an-oracle.git"
```

Install provider SDKs only when you need them:

```bash
uv add "llm-as-an-oracle[openai] @ git+https://github.com/hsanchez/llm-as-an-oracle.git"
uv add "llm-as-an-oracle[anthropic] @ git+https://github.com/hsanchez/llm-as-an-oracle.git"
uv add "llm-as-an-oracle[gemini] @ git+https://github.com/hsanchez/llm-as-an-oracle.git"
uv add "llm-as-an-oracle[all] @ git+https://github.com/hsanchez/llm-as-an-oracle.git"
```

With `pip`:

```bash
pip install "llm-as-an-oracle @ git+https://github.com/hsanchez/llm-as-an-oracle.git"
```

From a local clone:

```bash
git clone https://github.com/hsanchez/llm-as-an-oracle.git
cd llm-as-an-oracle
uv pip install .
```

## How the Oracle Works

`OracleRouter` is a deterministic routing layer that decides whether a task
should be evaluated by `VerifierStrategy`, `JudgeStrategy`, or a configured
`AdversarialVerifierStrategy`.

At a high level, the algorithm does four things:

1. Extract routing signals from the task and trajectories.
2. Ask a small set of routing policies to cast weighted votes.
3. Aggregate the votes into per-strategy confidence scores.
4. Select the winning strategy, or fall back to `Judge` if confidence is too low.

The default router uses these signals:

- presence of `ground_truth`
- presence of `test_cases`
- keyword evidence for verifiable vs open-ended tasks
- stated task difficulty
- whether execution output is available
- number of candidate trajectories
- explicit claim-verification mode in task metadata
- optional prior hardness from the evaluation harness

The routing process is fully auditable. Each decision includes the extracted
signals, each policy's vote, the final confidence, and a human-readable
reasoning trace.

For the full algorithm and policy details, see
[`docs/oracle-algorithm.md`](docs/oracle-algorithm.md).

## Adversarial Claim Verification

`AdversarialVerifierStrategy` verifies a claim represented by a trajectory. It
wraps two `VerifierStrategy` instances:

- a confirmation verifier that asks whether the claim is supported
- a challenge verifier that asks whether evidence supports rejecting the claim

The two passes are controlled by separate criteria:

```python
confirmation_criterion = EvaluationCriterion(
  id="claim_supported",
  name="Claim supported",
  description=(
    "Score high only when the original claim is clearly supported by the "
    "task evidence and rubric. Score low when the evidence does not support "
    "the claim or the claim requires assumptions not present in the task."
  ),
)

challenge_criterion = EvaluationCriterion(
  id="claim_challenged",
  name="Claim challenged",
  description=(
    "Score high only when there is an evidence-based reason the original "
    "claim is wrong. Score low when the challenge would require speculation "
    "or when the original claim is well supported."
  ),
)
```

The decision policy is:

```text
confirmation high, challenge low, both confident -> confirmed
confirmation low, challenge high, both confident -> rejected
otherwise -> uncertain
```

Use it directly when every trajectory is already a claim to verify:

```python
confirmation_verifier = VerifierStrategy(model, confirmation_config, [confirmation_criterion])
challenge_verifier = VerifierStrategy(model, challenge_config, [challenge_criterion])

adversarial = AdversarialVerifierStrategy(
  confirmation_verifier=confirmation_verifier,
  challenge_verifier=challenge_verifier,
  confirmation_criterion=confirmation_criterion,
  challenge_criterion=challenge_criterion,
)

result = adversarial.evaluate(task, [trajectory])
decision = result.trajectory_scores[trajectory.id].metadata["decision"]
```

To use it through the Oracle, pass it as the optional adversarial strategy and
mark claim-verification tasks explicitly:

```python
router = OracleRouter.default(
  verifier=verifier,
  judge=judge,
  adversarial=adversarial,
)

task = Task(
  id="claim-review",
  description="Verify a claim.",
  problem_statement="Was the original complied? label correct?",
  metadata={"evaluation_mode": "claim_verification"},
)
```

The router only considers the adversarial strategy when it is configured and
the task metadata explicitly marks the task as claim verification. Ordinary
single-trajectory tasks continue through the existing Judge/Verifier routing
path.

## Evaluation Metrics

This project has two layers of evaluation metrics.

- Strategy-level metrics answer: "How did a given evaluator score and rank the trajectories?"
- Harness-level metrics answer: "How well did Judge and Verifier perform relative to each other and to oracle-best selection?"

At the strategy level, `JudgeStrategy` and `VerifierStrategy` produce:

- `score`: normalized trajectory score in `[0, 1]`
- `criterion_scores`: per-criterion score breakdown
- `best_trajectory_id`: the selected best trajectory
- `pairwise_comparisons`: head-to-head comparisons when applicable
- `confidence`: confidence attached to scores and pairwise decisions

At the harness level, `EvaluationHarness` computes:

- `verifier_accuracy` and `judge_accuracy`: whether each strategy selected the oracle-best trajectory
- `oracle_gap_verifier` and `oracle_gap_judge`: how far each selected trajectory is from the oracle-best one
- `score_spread`: absolute score difference between Judge and Verifier on shared trajectories
- `strategy_disagreement`: fraction of trajectory pairs ranked differently by the two strategies
- `average_confidence`: average confidence across both strategies
- `hardness_score`: weighted composite of score spread, disagreement, inverted confidence, and oracle gap
- `elapsed_verifier_seconds` and `elapsed_judge_seconds`: runtime per strategy

## Glossary

- `Trajectory`: a candidate task-solving attempt, including the model's produced solution and any associated outputs or intermediate steps used for evaluation.
- `Judge`: an evaluator that scores or ranks trajectories holistically, usually with rubric-based or preference-based reasoning.
- `Verifier`: an evaluator that scores trajectories using a more structured verification process, often grounded in criteria decomposition, repeated verification, reference solutions, test cases, or execution evidence.
- `AdversarialVerifier`: an evaluator that verifies a claim through confirmation and challenge passes.
- `Oracle`: the orchestration layer that routes to the better configured evaluation mode for the task.

## CLI USAGE

```bash
# Print package information
uv run python -m llm_oracle info

# Route a task and inspect the decision
uv run python -m llm_oracle route --task "Implement quicksort" --difficulty hard --ground-truth

# Compare both strategies on a task
uv run python -m llm_oracle compare --task "Explain merge sort" --trajectories 3
```

## Development Setup

```bash
uv sync
```

This repository uses a local `.venv`. If your editor reports imports like
`pytest` as unresolved, make sure it is using:

```bash
.venv/bin/python
```

In Zed or VSCode, set the Python interpreter path in your project settings to `.venv/bin/python`.

## 📌 Citation

Please cite the `llm-as-an-oracle` tool itself following [CITATION.cff](./CITATION.cff) file.
