"""Routing package for LLM Oracle.

Exports the intelligent task router and all supporting components:
  - OracleRouter: Main router that dispatches to configured strategies
  - PolicyChain: Aggregates multiple routing policy votes
  - RoutingPolicy: Abstract base for custom routing policies
  - Built-in policies: GroundTruthPolicy, ClaimVerificationPolicy, etc.
  - SignalExtractor: Extracts routing signals from tasks
  - DetailedRoutingDecision: Auditable routing decision with per-policy votes
"""

from llm_oracle.routing.router import (
  ClaimVerificationPolicy,
  DetailedRoutingDecision,
  DifficultyPolicy,
  GroundTruthPolicy,
  KeywordDomainPolicy,
  OracleRouter,
  OutputAvailabilityPolicy,
  PolicyChain,
  PolicyVote,
  PriorHardnessPolicy,
  RoutingPolicy,
  RoutingSignals,
  SignalExtractor,
  TrajectoryCountPolicy,
)

__all__ = [
  # Core router
  "OracleRouter",
  # Policy chain
  "PolicyChain",
  # Abstract base
  "RoutingPolicy",
  # Built-in policies
  "GroundTruthPolicy",
  "ClaimVerificationPolicy",
  "KeywordDomainPolicy",
  "DifficultyPolicy",
  "TrajectoryCountPolicy",
  "PriorHardnessPolicy",
  "OutputAvailabilityPolicy",
  # Data types
  "RoutingSignals",
  "PolicyVote",
  "DetailedRoutingDecision",
  # Signal extraction
  "SignalExtractor",
]
