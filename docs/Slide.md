## Slide 0: LLM as an Oracle

Combining `LLM-as-a-Judge` and `LLM-as-a-Verifier` into an adaptive evaluation layer.

- Audience: Computer Scientists, AI Researchers, and AI Engineers
- Goal: explain the idea, the motivation, the architecture, and how to use it
- Framing: `Oracle` is an evaluation orchestrator, not a magical all-knowing model


## Slide 1: The Problem

Modern LLM systems often produce multiple candidate solutions, trajectories, or
tool-using attempts for the same task.

The hard question is not only:

- "How do we generate candidates?"

It is also:

- "How do we evaluate them reliably?"
- "Which candidate is actually best?"
- "Should we use the same evaluator for every task?"

The answer is often no.


## Slide 2: Why One Evaluator Is Not Enough

Different tasks demand different evaluation styles.

- Some tasks are open-ended, subjective, or rubric-based.
- Some tasks are verifiable against test cases, references, or execution output.
- Some tasks are ambiguous enough that a single evaluation method is brittle.

Using one evaluator everywhere creates failure modes:

- Judge-only systems can be too coarse or too subjective.
- Verifier-only systems can be unnecessarily rigid or less useful for open-ended tasks.

This leads to the next question:

- what exactly are we evaluating?
- not just answers, but full candidate attempts (i.e., a trajectory)


## Slide 3: What Exactly Is Being Evaluated?

In this setting, a `trajectory` is a candidate task-solving attempt.

A trajectory may include:

- the produced solution
- intermediate reasoning or steps
- tool calls
- code
- execution output
- an optional reward or success signal

The evaluator does not just score a final answer. It scores a full attempt.


## Slide 4: LLM-as-a-Judge

`LLM-as-a-Judge` evaluates trajectories holistically.

Typical characteristics:

- rubric-based scoring
- preference comparison between candidates
- qualitative reasoning
- useful for open-ended tasks

Good fit:

- essays
- explanations
- design proposals
- qualitative comparisons
- tasks without clean ground truth

Main strength:

- broad, flexible, human-like evaluation


## Slide 5: LLM-as-a-Verifier

`LLM-as-a-Verifier` is a more structured evaluation framework. It is quite
useful when you need more reliable comparison and evidence-sensitive scoring.
(Arguably, it is an advancement over standard `Judge` for difficult evaluation.)

In this repo, the Verifier is inspired by:

- [LLM-as-a-Verifier](https://llm-as-a-verifier.notion.site/)

Core ideas:

- finer scoring granularity
- repeated verification
- criteria decomposition
- evidence-sensitive evaluation

Good fit:

- coding tasks
- tasks with reference solutions
- test-case-based evaluation
- expected outputs
- execution evidence

Main strength:

- more discriminative evaluation on hard, evidence-grounded tasks


## Slide 6: Are Judge and Verifier Replacements?

Not usually. They solve related but different evaluation problems.

- `Judge` is strong for holistic, open-ended assessment.
- `Verifier` is strong for structured, evidence-based correctness assessment.
- The `Oracle` layer exists to decide which evaluator is the better fit for the task.


## Slide 7: What Is LLM-as-an-Oracle?

`LLM-as-an-Oracle` is the orchestration layer I've built above Judge and Verifier.

In this repo, the Oracle layer:

- exposes both evaluators
- routes tasks to one of them
- records the routing decision
- supports side-by-side comparison

Important definition:

- The `Oracle` layer is an evaluation orchestrator that routes between Judge and
  Verifier and selects the better evaluation mode for the task.


## Slide 8: Oracle Architecture

Conceptual pipeline:

1. Receive a task and one or more candidate trajectories
2. Extract task signals
3. Decide whether to use Judge or Verifier
4. Evaluate trajectories with the selected strategy
5. Return scores, comparisons, and the selected best trajectory

Key components in this repo:

- `JudgeStrategy`
- `VerifierStrategy`
- `OracleRouter`
- `EvaluationHarness`
- provider layer for model backends


## Slide 9: Routing Intuition

The Oracle does not route randomly.

It uses task signals such as:

- presence of ground truth
- presence of test cases
- execution output availability
- task difficulty
- trajectory count
- prior hardness
- domain cues in the problem statement

Typical intuition:

- open-ended essay -> Judge
- executable coding task with tests -> Verifier
- ambiguous cases -> use routing policy and confidence


## Slide 10: When To Use Judge

Choose `Judge` when:

- evaluation is open-ended
- quality is holistic
- correctness is not directly executable
- style, usefulness, argument quality, or clarity matter

Examples:

- "Which explanation is clearer?"
- "Which design proposal is more compelling?"
- "Which summary is more useful to a human reader?"


## Slide 11: When To Use Verifier

Choose `Verifier` when:

- correctness can be grounded in evidence
- you have test cases or expected outputs
- execution output is informative
- you need finer discrimination between close candidates

Examples:

- "Which implementation actually solves the task?"
- "Which extracted structure matches the reference best?"
- "Which agent trajectory is more likely correct under formal constraints?"


## Slide 12: When To Use Oracle

Choose `Oracle` when:

- tasks vary in type
- you do not want to hand-pick Judge vs Verifier every time
- you want an adaptive evaluation layer
- you want better operational clarity around evaluator choice

This is especially useful in:

- agent benchmarks
- coding agents
- multi-step tool-using systems
- evaluation pipelines with mixed task types


## Slide 13: How To Use It in This Repo

The repository exposes a CLI and Python abstractions.

CLI examples:

```bash
uv run python main.py demo
uv run python main.py route --task "Implement quicksort" --difficulty hard --ground-truth
uv run python main.py compare --task "Explain merge sort" --trajectories 3
uv run python main.py test
```

What each command does:

- `demo`: runs the end-to-end offline example
- `route`: shows the Oracle routing decision
- `compare`: compares Judge and Verifier side by side
- `test`: runs the test suite


## Slide 14: How To Read a Routing Decision

A routing decision should be interpretable.

What to inspect:

- selected strategy
- confidence
- extracted signals
- policy votes
- reasoning for the decision

This matters because Oracle is not only selecting an evaluator.
It is also making that choice auditable.


## Slide 15: Suggested Tutorial Flow for New Users

If you are new to the repo, use this sequence:

1. Run the offline demo
2. Route a clearly verifiable task
3. Route a clearly open-ended task
4. Compare Judge and Verifier on the same task
5. Read the routing signals and reasoning
6. Inspect the tests to understand expected behavior

This teaches both the concept and the mechanics.


## Slide 16: Research Framing

For researchers, the interesting question is not just:

- "Can an LLM evaluate another LLM?"

It is:

- "Which evaluation regime is appropriate for which task?"
- "Can routing improve evaluation reliability?"
- "What signals predict when one evaluator outperforms another?"
- "How should we measure evaluation hardness?"

This repo treats Oracle as a hypothesis:

- adaptive evaluator selection is better than evaluator monoculture


## Slide 17: Engineering Framing

For engineers, the value is operational.

Oracle gives you:

- a common interface over multiple evaluators
- policy-based routing
- inspectable decisions
- harness-based comparison
- a path to productionizing mixed evaluation workloads

This is useful when you care about:

- reliability
- debuggability
- cost-quality tradeoffs
- evaluator specialization


## Slide 18: Limitations

This framing is useful, but not magic.

Limitations include:

- routing quality depends on the signals and policies
- Judge and Verifier can both still be wrong
- provider quality matters
- task formulations strongly affect outcomes
- "Oracle" is a system design pattern, not ground truth itself

The right mental model is:

- adaptive evaluation, not omniscient evaluation


## Slide 19: Takeaways

- `Judge` and `Verifier` solve related but different evaluation problems.
- `Verifier` is stronger for structured, evidence-grounded evaluation.
- `Judge` remains valuable for open-ended, rubric-driven tasks.
- `Oracle` is the orchestration layer that routes between them.
- The main idea is not replacement. It is adaptive evaluator selection.


## Slide 20: Discussion

Questions worth discussing with this audience:

- What task features best predict Judge vs Verifier success?
- Should Oracle route to one evaluator, or combine both?
- How should evaluation hardness be measured?
- What are the failure modes of evaluator routing?
- How should this scale to multi-agent or tool-augmented systems?
