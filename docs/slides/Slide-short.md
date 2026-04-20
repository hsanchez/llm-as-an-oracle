## Slide 0: The Evaluation Problem

Human evaluation is the gold standard — but it does not scale.

- slow and expensive to run at the pace of model iteration
- automated metrics (BLEU, ROUGE, exact match) are shallow
- they miss reasoning quality, correctness of approach, and usefulness

The field needed a better path.


## Slide 1: The LLM-as-* Idiom

Around 2023, frontier models crossed a reliability threshold:

- their judgments on text quality and reasoning correlate strongly with human judgments
- they are instruction-following enough to apply a rubric consistently
- a model call is cheap enough to run many evaluations per task

This gave rise to a family of evaluation patterns:

- `LLM-as-a-Judge` — holistic, preference-based scoring
- `LLM-as-a-Verifier` — structured, evidence-sensitive scoring
- `LLM-as-a-Critic`, `LLM-as-a-Ranker`, and others

The key insight: if a model is capable enough to produce output, it is capable enough to evaluate it.

But which evaluation pattern is the right one for a given task?

That is the question `LLM-as-an-Oracle` answers.


## Slide 2: LLM as an Oracle

`LLM-as-an-Oracle` is an evaluation orchestrator. It sits above the
`LLM-as-a-Judge` and the `LLM-as-a-Verifier` and decides which one is the better
fit for a given task.

The term "Oracle" here means adaptive evaluation layer, not an all-knowing
model. *It is a system design pattern.*


## Slide 3: The Core Problem

LLM systems often produce multiple candidate solutions or trajectories for the same task.

The hard question is:

- how do we evaluate those candidates reliably?

Using one evaluator for every task is often the wrong abstraction.


## Slide 4: Two Evaluation Paradigms

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

## Slide 5: What Is a Trajectory?

A `trajectory` is a candidate task-solving attempt.

It may include:

- the produced solution
- intermediate steps
- tool calls
- code
- execution output

The evaluator is scoring the attempt, not only the final answer.


## Slide 6: Judge vs Verifier

Use `Judge` when:

- evaluation is qualitative
- correctness is open-ended
- style, clarity, or usefulness matter

Use `Verifier` when:

- you have reference solutions
- you have test cases
- you have expected outputs
- execution evidence matters


## Slide 7: What Is the Oracle Layer?

`LLM-as-an-Oracle` is the orchestration layer above Judge and Verifier.

Definition:

- The `Oracle` layer is an evaluation orchestrator that routes between Judge and Verifier and selects the better evaluation mode for the task.

This repo treats Oracle as:

- a router
- an evaluator selector
- an auditable decision layer


## Slide 8: How Routing Works

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


## Slide 9: When To Use Oracle

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


## Slide 10: How To Use This Repo

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
