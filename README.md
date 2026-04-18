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
- Use `Oracle` when you want a system that can route between them and apply the better evaluation mode for the task.

This repository therefore treats `LLM-as-an-Oracle` as an orchestration layer:

- it exposes both `Judge` and `Verifier`
- it compares them in a shared evaluation harness
- it routes tasks to one or the other based on task signals

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
