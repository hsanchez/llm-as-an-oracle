# FAQ

## What is an Oracle?

The Oracle is an evaluation orchestrator. It sits above configured evaluation
strategies and decides which one is the better fit for a given task. It does
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
   weighted vote for a strategy. The votes are aggregated into a confidence
   score for each strategy.

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

## What is an Adversarial Verifier?

The Adversarial Verifier (`AdversarialVerifierStrategy`) is a claim verifier
built from two `VerifierStrategy` instances.

It is useful when a single trajectory represents a claim that needs
confirmation, such as "this model complied", "this answer satisfies the
rubric", or "this output refused the unsafe request".

It runs two passes:

1. **Confirmation** — checks whether the claim is supported by the evidence and
   criteria.
2. **Challenge** — checks whether there is an evidence-based reason the claim
   is wrong.

The confirmation criterion should reward support for the original claim. For
example: "Score high only when the original claim is clearly supported by the
task evidence and rubric."

The challenge criterion should reward evidence against the original claim. For
example: "Score high only when there is an evidence-based reason the original
claim is wrong; score low when the challenge would require speculation."

The policy is:

```text
confirmation high, challenge low, both confident -> confirmed
confirmation low, challenge high, both confident -> rejected
otherwise -> uncertain
```

The decision is stored in `ScoreResult.metadata["decision"]`. The metadata also
includes confirmation and challenge scores, confidences, thresholds, and a
human-readable decision reason. Callers can escalate `uncertain` results to a
human oracle.

To let the Oracle route to the Adversarial Verifier, configure the router with
an adversarial strategy and mark the task explicitly:

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

The router does not send every single-trajectory task to the Adversarial
Verifier. The task metadata must opt in.

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

The Router (`OracleRouter`) is the decision layer that selects between
configured strategies for each task. It is composed of a chain of routing
policies, each of which inspects task signals and casts a weighted vote.

The default policy chain includes six base policies:

| Policy | Weight | Signal used |
|---|---|---|
| `ground_truth` | 2.0 | presence of ground truth or test cases |
| `prior_hardness` | 1.8 | cached hardness score from a prior harness run |
| `keyword_domain` | 1.5 | keyword density in the problem statement |
| `difficulty` | 1.0 | stated task difficulty |
| `output_availability` | 0.9 | presence of execution outputs |
| `trajectory_count` | 0.8 | number of candidate trajectories |

When an adversarial strategy is configured, the router also installs
`claim_verification`, which strongly prefers the Adversarial Verifier for tasks
with `metadata["evaluation_mode"] == "claim_verification"`.

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

## What happens when human review is pending?

`HumanOracle.ask()` may return `HumanResponsePending` for asynchronous review
queues such as Slack, GitHub, Linear, or ticket systems. In that case,
`OracleRouter.evaluate()` returns the pending handle with the routing decision:

```python
result, decision = router.evaluate(task, trajectories)
if isinstance(result, HumanResponsePending):
  external_id = result.external_id
```

The decision metadata records `human_escalated=True`, `human_pending=True`, the
request id, external id, and pending message. The host application is
responsible for persisting that state and resuming the workflow when the human
answer arrives.

---

## How do I choose between Judge, Verifier, and Oracle?

**Use the Verifier directly** when:
- you know the task is verifiable
- you have ground truth, test cases, or execution outputs
- you need fine-grained discrimination between close candidates

**Use the Adversarial Verifier directly** when:
- one trajectory represents a claim that needs confirmation
- you want an evidence-based challenge before accepting or rejecting the claim
- uncertain outcomes should trigger human review

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
