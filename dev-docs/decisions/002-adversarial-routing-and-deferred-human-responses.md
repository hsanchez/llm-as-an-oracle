# 002 Adversarial Routing and Deferred Human Responses

## Status

Accepted

## Context

The router originally selected between Judge and Verifier. That works for
candidate selection, but claim verification has a different shape: a single
trajectory often represents a claim that should be confirmed or challenged, not
ranked against alternatives.

The human escalation boundary also allowed `HumanResponsePending`, but the
router raised `RuntimeError` when a host returned it. That made asynchronous
review queues valid at the protocol layer and unusable at the router layer.

## Decision

Add `StrategyType.ADVERSARIAL` and let `OracleRouter` hold an optional
`AdversarialVerifierStrategy`.

Adversarial routing is explicit. `SignalExtractor` sets
`RoutingSignals.is_claim_verification` only when:

```python
task.metadata.get("evaluation_mode") == "claim_verification"
```

When an adversarial strategy is configured, `OracleRouter.default(...)` installs
`ClaimVerificationPolicy`. The policy strongly prefers
`StrategyType.ADVERSARIAL` for claim-verification tasks and abstains for normal
tasks.

`OracleRouter.evaluate(...)` now returns:

```python
tuple[EvaluationResult | HumanResponsePending, DetailedRoutingDecision]
```

When human review is deferred, the router returns the pending handle and records
the request metadata on the routing decision. The host application owns
persistence and resume behavior.

## Consequences

The router keeps the same ingestion API: callers still pass a `Task` and
trajectories. The strategy set expands from Judge/Verifier to configured
strategies.

Single-trajectory tasks are not inferred to be claim verification. The caller
must opt in through task metadata.

Deferred human responses no longer crash the router, but the library still does
not implement workflow resumption. Resumption remains host-application
responsibility.
