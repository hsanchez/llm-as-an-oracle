## Slide 0: LLM as an Oracle

Combining `LLM-as-a-Judge` and `LLM-as-a-Verifier` into an adaptive evaluation layer.

- Audience: Computer Scientists, AI Researchers, and AI Engineers
- Goal: explain the idea, when to use it, and how to use it


## Slide 1: The Core Problem

LLM systems often produce multiple candidate solutions or trajectories for the same task.

The hard question is:

- how do we evaluate those candidates reliably?

Using one evaluator for every task is often the wrong abstraction.


## Slide 2: Two Evaluation Paradigms

`LLM-as-a-Judge`

- holistic
- rubric-based
- preference-oriented
- good for open-ended tasks

`LLM-as-a-Verifier`

- structured
- finer-grained
- repeated verification
- criteria decomposition
- good for evidence-grounded tasks

This leads to the next question:

- what exactly are we evaluating?
- not just answers, but full candidate attempts (i.e., a trajectory)

## Slide 3: What Is a Trajectory?

A `trajectory` is a candidate task-solving attempt.

It may include:

- the produced solution
- intermediate steps
- tool calls
- code
- execution output

The evaluator is scoring the attempt, not only the final answer.


## Slide 4: Judge vs Verifier

Use `Judge` when:

- evaluation is qualitative
- correctness is open-ended
- style, clarity, or usefulness matter

Use `Verifier` when:

- you have reference solutions
- you have test cases
- you have expected outputs
- execution evidence matters


## Slide 5: What Is the Oracle Layer?

`LLM-as-an-Oracle` is the orchestration layer above Judge and Verifier.

Definition:

- The `Oracle` layer is an evaluation orchestrator that routes between Judge and Verifier and selects the better evaluation mode for the task.

This repo treats Oracle as:

- a router
- an evaluator selector
- an auditable decision layer


## Slide 6: How Routing Works

The Oracle uses task signals such as:

- ground truth
- test cases
- execution outputs
- task difficulty
- trajectory count
- prior hardness
- domain cues in the prompt

Typical intuition:

- essay task -> Judge
- coding task with tests -> Verifier


## Slide 7: When To Use Oracle

Use `Oracle` when:

- tasks vary in type
- you want adaptive evaluator selection
- you do not want to manually choose Judge vs Verifier each time
- you want routing decisions to be inspectable

This is especially useful for:

- coding agents
- benchmark pipelines
- tool-using agents
- mixed evaluation workloads


## Slide 8: How To Use This Repo

CLI examples:

```bash
uv run python main.py demo
uv run python main.py route --task "Implement quicksort" --difficulty hard --ground-truth
uv run python main.py compare --task "Explain merge sort" --trajectories 3
uv run python main.py test
```

Suggested flow:

1. run the demo
2. inspect a routing decision
3. compare Judge and Verifier
4. read the routing signals and confidence


## Slide 9: Main Takeaways

- `Judge` and `Verifier` are complementary, not identical.
- `Verifier` is stronger for structured, evidence-based evaluation.
- `Judge` remains useful for holistic, open-ended evaluation.
- `Oracle` is the adaptive layer that selects between them.
- The key idea is evaluator orchestration, not evaluator monoculture.
