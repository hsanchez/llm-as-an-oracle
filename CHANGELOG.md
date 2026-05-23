# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- `OracleRouter`: deterministic routing layer across configured evaluation strategies
- `JudgeStrategy`: holistic rubric-based trajectory evaluation
- `VerifierStrategy`: structured criteria-decomposition evaluation
- `AdversarialVerifierStrategy`: confirmation/challenge claim verification strategy
- `EvaluationHarness`: harness-level metrics comparing both strategies
- Optional adversarial claim-verification routing via `metadata["evaluation_mode"] == "claim_verification"`
- `HumanResponsePending` router return path for deferred human escalation workflows
- Hybrid Jupytext examples learning path, including adversarial claim verification
- CLI commands: `demo`, `route`, `compare`, `test`
