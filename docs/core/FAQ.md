# FAQ

## What is an Oracle?

The Oracle is an evaluation orchestrator. It sits above the Judge and the
Verifier and decides which one is the better fit for a given task. It does
not evaluate trajectories itself — it routes to the evaluator that is most
likely to produce a reliable result, then returns that evaluator's output
alongside the routing decision that led to it.

The term "Oracle" here means adaptive evaluation layer, not an all-knowing
model. It is a system design pattern.

---

## How does the Oracle work?

When you call `OracleRouter.evaluate(task, trajectories)`, it runs three steps:

1. **Signal extraction** — the router reads the task and trajectories and
   extracts features: whether ground truth is present, whether test cases
   exist, whether execution outputs are available, task difficulty, trajectory
   count, and keyword density.

2. **Policy voting** — each routing policy inspects those signals and casts a
   weighted vote for either the Verifier or the Judge. The votes are aggregated
   into a confidence score for each strategy.

3. **Dispatch** — if the winning strategy's confidence exceeds the threshold
   (default 0.60), the router dispatches to it. Otherwise it falls back to the
   Judge. The result includes the evaluation output and a `DetailedRoutingDecision`
   that records every signal value, every policy vote, and the final reasoning.

---

## What is a Verifier?

The Verifier (`VerifierStrategy`) is an evidence-sensitive evaluator. It scores
trajectories using token-level log probabilities rather than a single sampled
score, which gives it finer discrimination between close candidates.

It is inspired by the [LLM-as-a-Verifier](https://llm-as-a-verifier.notion.site/)
framework and implements the following formula:

```
R(t, τ) = (1 / C·K) · Σ_c Σ_k Σ_g  p_θ(v_g | t, c, τ) · φ(v_g)
```

Where `C` is the number of criteria, `K` is the number of repeated
verifications, `G` is the scoring granularity, `p_θ` is the model's token
probability, and `φ` maps score tokens to scalar values.

---

## How does the Verifier work?

For a single trajectory it scores the trajectory against each criterion `K`
times and averages the results.

For multiple trajectories it runs a round-robin pairwise tournament: every
pair of trajectories is compared on every criterion across `K` verification
rounds. The trajectory that wins the most head-to-head matchups is returned
as the best.

The logprob-based scoring requires a provider that exposes token probabilities.
`OpenAIProvider` and `GeminiProvider` support this. `AnthropicProvider` does
not expose logprobs, so the Verifier falls back to text-based score extraction
when used with Anthropic models.

---

## What is a Judge?

The Judge (`JudgeStrategy`) is a holistic evaluator. It scores trajectories
using a rubric and structured chain-of-thought reasoning. It does not require
log probabilities, so it works with any provider that supports plain text
generation.

It scores each trajectory pointwise on a 1–10 scale across all criteria, then
uses pairwise comparisons to break ties. To reduce positional bias, each
pairwise comparison is run twice with the trajectory order swapped and the
scores averaged.

---

## How does the Judge work?

For each trajectory, the Judge calls the model once per criterion per
verification round, parses the numeric score from the `<score>` tag in the
response, normalizes it to `[0, 1]`, and averages across repetitions.

When multiple trajectories are present, it also runs pairwise comparisons for
tie-breaking. The final score is a weighted combination of the pointwise score
(70%) and the pairwise win rate (30%).

The `reasoning_depth` parameter controls how much chain-of-thought the model
is asked to produce: `"brief"`, `"detailed"`, or `"chain_of_thought"`.

---

## What is the Router?

The Router (`OracleRouter`) is the decision layer that selects between the
Judge and the Verifier for each task. It is composed of a chain of routing
policies, each of which inspects task signals and casts a weighted vote.

The default policy chain includes six policies:

| Policy | Weight | Signal used |
|---|---|---|
| `ground_truth` | 2.0 | presence of ground truth or test cases |
| `prior_hardness` | 1.8 | cached hardness score from a prior harness run |
| `keyword_domain` | 1.5 | keyword density in the problem statement |
| `difficulty` | 1.0 | stated task difficulty |
| `output_availability` | 0.9 | presence of execution outputs |
| `trajectory_count` | 0.8 | number of candidate trajectories |

---

## How does the Router work?

Each policy produces a `PolicyVote` with a preferred strategy and a confidence
value. The chain aggregates votes by computing a weighted confidence total for
each strategy and picks the winner. If the winner's normalized confidence is
below the threshold, the router falls back to the Judge.

Every routing decision is recorded in a `DetailedRoutingDecision` that includes
the extracted signals, every policy vote, the final confidence, and a
human-readable reasoning string. The full history is accessible via
`router.decision_log` and `router.routing_summary()`.

You can extend the router with your own policy by subclassing `RoutingPolicy`
and calling `router.register_policy(MyPolicy())`.

---

## How do I choose between Judge, Verifier, and Oracle?

**Use the Verifier directly** when:
- you know the task is verifiable
- you have ground truth, test cases, or execution outputs
- you need fine-grained discrimination between close candidates

**Use the Judge directly** when:
- evaluation is open-ended or qualitative
- correctness cannot be grounded in concrete evidence
- style, clarity, argument quality, or usefulness matter

**Use the Oracle** when:
- your workload mixes task types
- you do not want to manually choose an evaluator for each task
- you want routing decisions to be inspectable and auditable
- you want a single evaluation interface regardless of task type

If you are unsure, start with the Oracle. Its routing decisions are fully
transparent, so you can inspect which evaluator it chose and why.
