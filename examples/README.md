# Examples

The examples are ordered as a hybrid learning path. The first notebooks build
the core workflow, then later notebooks focus on specific production patterns.

## Learning Path

| Example | Purpose |
|---|---|
| `00_quickstart.py` | Smallest offline end-to-end path: task, trajectories, Verifier, Judge, harness, router, and signals. |
| `01_core_tutorial.py` | Cumulative beginner tutorial using `StubProvider`; introduces the public API step by step. |
| `02_live_provider_tutorial.py` | Same core workflow with a live provider (`AnthropicProvider`) and close candidate outputs. |
| `03_hands_on_example.py` | Narrative hands-on workflow for evaluating LLM-generated bug fixes. |
| `04_human_oracle.py` | Synchronous human escalation with a host-provided `HumanOracle`. |
| `05_human_oracle_live_model.py` | Human escalation with a live provider in a compliance-review scenario. |
| `06_adversarial_claim_verification.py` | Claim verification with confirmation/challenge verifiers, router opt-in, and deferred human escalation. |
| `07_full_demo.py` | Capstone demo that ties together live-provider evaluation, routing, harness metrics, and human escalation. |

All files use Jupytext-style percent cells (`# %%`) where notebook-style
execution is useful. They can also be run as normal Python scripts.
