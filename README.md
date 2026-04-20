## LLM AS AN ORACLE

This repository combines two related evaluation paradigms:

- `LLM-as-a-Judge`: a model evaluates candidate answers or trajectories holistically and assigns a score or preference.
- `LLM-as-a-Verifier`: a more structured evaluation framework that improves discrimination by using finer scoring granularity, repeated verification, and criteria decomposition.

The `LLM-as-a-Verifier` framing used here is inspired by **"LLM-as-a-Verifier: A
General-Purpose Verification Framework"**:

- [LLM-as-a-Verifier](https://llm-as-a-verifier.notion.site/)

That project presents Verifier as an improvement over standard Judge-style
evaluation for difficult trajectory comparison, especially when coarse scoring
produces too many ties.

In practice, these approaches are best treated as complementary:

- Use `Judge` when evaluation is open-ended, rubric-based, holistic, or subjective.
- Use `Verifier` when evaluation can be grounded in stronger evidence such as reference solutions, test cases, expected outputs, or execution results.
- Use the `Oracle` layer when you want an evaluation orchestrator that routes between Judge and Verifier and selects the better evaluation mode for the task.

This repository therefore treats `LLM-as-an-Oracle` as an orchestration layer:

- it exposes both `Judge` and `Verifier`
- it compares them in a shared evaluation harness
- it routes tasks to one or the other based on task signals

## How the Oracle Works

`OracleRouter` is a deterministic routing layer that decides whether a task
should be evaluated by `VerifierStrategy` or `JudgeStrategy`.

At a high level, the algorithm does four things:

1. Extract routing signals from the task and trajectories.
2. Ask a small set of routing policies to cast weighted votes.
3. Aggregate the votes into Judge-vs-Verifier confidence scores.
4. Select the winning strategy, or fall back to `Judge` if confidence is too low.

The default router uses these signals:

- presence of `ground_truth`
- presence of `test_cases`
- keyword evidence for verifiable vs open-ended tasks
- stated task difficulty
- whether execution output is available
- number of candidate trajectories
- optional prior hardness from the evaluation harness

The routing process is fully auditable. Each decision includes the extracted
signals, each policy's vote, the final confidence, and a human-readable
reasoning trace.

For the full algorithm and policy details, see
[`docs/oracle-algorithm.md`](docs/oracle-algorithm.md).

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
- `avg_confidence`: average confidence across both strategies
- `hardness_score`: weighted composite of score spread, disagreement, inverted confidence, and oracle gap
- `elapsed_verifier_s` and `elapsed_judge_s`: runtime per strategy

## Glossary

- `Trajectory`: a candidate task-solving attempt, including the model's produced solution and any associated outputs or intermediate steps used for evaluation.
- `Judge`: an evaluator that scores or ranks trajectories holistically, usually with rubric-based or preference-based reasoning.
- `Verifier`: an evaluator that scores trajectories using a more structured verification process, often grounded in criteria decomposition, repeated verification, reference solutions, test cases, or execution evidence.
- `Oracle`: the orchestration layer that routes between Judge and Verifier and selects the better evaluation mode for the task.

## CLI USAGE

```bash
# Run the offline demo (no API keys needed)
uv run python main.py demo

# Route a task and inspect the decision
uv run python main.py route --task "Implement quicksort" --difficulty hard --ground-truth

# Compare both strategies on a task
uv run python main.py compare --task "Explain merge sort" --trajectories 3

# Run the test suite (148 tests)
uv run python main.py test
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

The workspace is configured for VS Code / Cursor to prefer that interpreter.

## 📌 Citation

Please cite the `llm-as-an-oracle` tool itself following [CITATION.cff](./CITATION.cff) file.
