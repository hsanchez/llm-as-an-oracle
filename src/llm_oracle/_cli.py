"""LLM Oracle CLI implementation."""

from __future__ import annotations

import argparse
import subprocess
import sys
import textwrap


def _banner(title: str, width: int = 68) -> None:
  print()
  print("╔" + "═" * (width - 2) + "╗")
  print("║" + title.center(width - 2) + "║")
  print("╚" + "═" * (width - 2) + "╝")


def _section(title: str) -> None:
  print()
  print(f"  ── {title} {'─' * max(0, 58 - len(title))}")


def _info(msg: str) -> None:
  print(f"  {msg}")


def _ok(msg: str) -> None:
  print(f"  ✓ {msg}")


def _err(msg: str) -> None:
  print(f"  ✗ {msg}", file=sys.stderr)


def _import_oracle():
  """Import llm_oracle, printing a helpful message on failure."""
  try:
    import llm_oracle  # noqa: PLC0415

    return llm_oracle
  except ImportError:
    _err("Could not import 'llm_oracle'.")
    _err(
      "Make sure you are running from the project root and that the package is on your Python path."
    )
    _err("Try:  uv run python -m llm_oracle <command>")
    sys.exit(1)


def cmd_info(_args: argparse.Namespace) -> None:
  """Print package version and component summary."""
  oracle = _import_oracle()

  _banner("LLM Oracle — Package Information")
  _info(f"Version  : {oracle.__version__}")
  _info(f"Author   : {oracle.__author__}")
  print()
  _info("Components:")
  components = [
    ("VerifierStrategy", "LLM-as-a-Verifier with logprob fine-grained scoring"),
    ("JudgeStrategy", "LLM-as-a-Judge with rubric chain-of-thought evaluation"),
    ("EvaluationHarness", "Side-by-side hardness comparison harness"),
    ("OracleRouter", "Signal-based policy-chain intelligent router"),
    ("StubProvider", "Offline stub provider for testing (no API key needed)"),
    ("OpenAIProvider", "OpenAI chat-completions (requires openai package)"),
    ("AnthropicProvider", "Anthropic Messages API (requires anthropic package)"),
    ("GeminiProvider", "Google Gemini via google-genai (requires google-genai)"),
  ]
  for name, desc in components:
    _info(f"  {name:<22s}  {desc}")

  print()
  _info("Built-in routing policies:")
  policies = [
    ("PriorHardnessPolicy", "w=1.8", "Uses cached harness hardness score"),
    ("GroundTruthPolicy", "w=2.0", "Prefers verifier when ground truth is present"),
    ("KeywordDomainPolicy", "w=1.5", "Code keywords → verifier; essay keywords → judge"),
    ("DifficultyPolicy", "w=1.0", "HARD → verifier; EASY → judge"),
    ("OutputAvailabilityPolicy", "w=0.9", "Execution outputs → verifier"),
    ("TrajectoryCountPolicy", "w=0.8", "Many candidates → judge (quadratic cost)"),
  ]
  for name, weight, desc in policies:
    _info(f"  {name:<28s}  {weight}  {desc}")
  print()


def cmd_demo(_args: argparse.Namespace) -> None:
  """Run the full end-to-end demo (offline, no API keys)."""
  _banner("LLM Oracle — End-to-End Demo")
  _info("Running examples/end_to_end.py …")
  print()

  result = subprocess.run(
    [sys.executable, "examples/end_to_end.py"],
    check=False,
  )
  sys.exit(result.returncode)


def cmd_route(args: argparse.Namespace) -> None:
  """Route a task description to verifier or judge and show the decision."""
  oracle = _import_oracle()

  task_text: str = args.task
  difficulty_map = {
    "easy": oracle.TaskDifficulty.EASY,
    "medium": oracle.TaskDifficulty.MEDIUM,
    "hard": oracle.TaskDifficulty.HARD,
    "unknown": oracle.TaskDifficulty.UNKNOWN,
  }
  difficulty = difficulty_map.get(
    (args.difficulty or "unknown").lower(), oracle.TaskDifficulty.UNKNOWN
  )

  n_trajs: int = max(1, int(args.trajectories or 2))

  _banner("LLM Oracle — Routing Decision")

  task = oracle.Task(
    id="cli-task",
    description=task_text[:80],
    problem_statement=task_text,
    difficulty=difficulty,
    ground_truth="<ground truth not provided>" if args.ground_truth else None,
    test_cases=[{"example": True}] if args.test_cases else None,
  )

  trajs = [
    oracle.Trajectory(
      id=f"traj-{i + 1}",
      task_id="cli-task",
      content=f"[Placeholder trajectory {i + 1} for: {task_text[:40]}…]",
      output="[output]" if args.with_output else None,
    )
    for i in range(n_trajs)
  ]

  stub = oracle.StubProvider()
  config = oracle.ScoringConfig()
  criteria = [
    oracle.EvaluationCriterion("c1", "Correctness", "Is it correct?"),
    oracle.EvaluationCriterion("c2", "Clarity", "Is it readable?"),
  ]
  verifier = oracle.VerifierStrategy(stub, config, criteria)
  judge = oracle.JudgeStrategy(stub, config, criteria)
  router = oracle.OracleRouter.default(verifier, judge, confidence_threshold=0.50)

  if args.prior_hardness is not None:
    hardness = float(args.prior_hardness)
    router.update_hardness("cli-task", hardness)
    _info(f"Injected prior hardness : {hardness:.3f}")

  decision = router.route(task, trajs)

  _section("Task")
  _info(f"Text       : {task_text[:70]}{'…' if len(task_text) > 70 else ''}")
  _info(f"Difficulty : {difficulty.value}")
  _info(f"Trajectories : {n_trajs}")
  _info(f"Ground truth : {'yes' if task.ground_truth else 'no'}")
  _info(f"Test cases   : {'yes' if task.test_cases else 'no'}")
  _info(f"With output  : {'yes' if args.with_output else 'no'}")

  _section("Routing Decision")
  icon = "🔍 Verifier" if decision.selected_strategy == oracle.StrategyType.VERIFIER else "⚖️  Judge"
  _info(f"Selected strategy : {icon}")
  _info(f"Confidence        : {decision.confidence:.4f}")
  _info(f"Routing latency   : {decision.elapsed_ms:.2f} ms")

  _section("Signals")
  signals = decision.signals
  if signals:
    rows = [
      ("has_ground_truth", signals.has_ground_truth),
      ("has_test_cases", signals.has_test_cases),
      ("trajectory_count", signals.trajectory_count),
      ("stated_difficulty", signals.stated_difficulty),
      ("verifiable_keyword_density", signals.verifiable_keyword_density),
      ("judgement_keyword_density", signals.judgement_keyword_density),
      ("problem_length (norm)", signals.problem_length),
      ("output_available", signals.output_available),
      ("prior_hardness", signals.prior_hardness),
    ]
    for k, v in rows:
      val_str = f"{v:.4f}" if isinstance(v, float) else str(v)
      _info(f"  {k:<35s} : {val_str}")

  _section("Policy Votes")
  for vote in sorted(decision.policy_votes, key=lambda v: -v.confidence * v.weight):
    arrow = "→" if vote.preferred == decision.selected_strategy else "←"
    _info(
      f"  {arrow} [{vote.policy_name:<26s}]  "
      f"{vote.preferred.value:<9s}  conf={vote.confidence:.3f}  "
      f"weight={vote.weight}"
    )

  _section("Reasoning")
  for line in decision.reasoning.splitlines():
    _info(f"  {line}")
  print()


def cmd_compare(args: argparse.Namespace) -> None:
  """Score a task with both strategies and print a side-by-side comparison."""
  oracle = _import_oracle()

  task_text: str = args.task
  n_trajs: int = max(1, int(args.trajectories or 2))

  _banner("LLM Oracle — Verifier vs Judge Comparison")

  task = oracle.Task(
    id="compare-task",
    description=task_text[:80],
    problem_statement=task_text,
    difficulty=oracle.TaskDifficulty.MEDIUM,
  )
  trajs = [
    oracle.Trajectory(
      id=f"traj-{i + 1}",
      task_id="compare-task",
      content=f"[Candidate solution {i + 1}]",
      reward=1.0 if i == 0 else 0.0,
    )
    for i in range(n_trajs)
  ]

  stub = oracle.StubProvider(seed=7)
  config = oracle.ScoringConfig(granularity=20, num_verifications=2)
  criteria = [
    oracle.EvaluationCriterion("correctness", "Correctness", "Is it correct?", weight=2.0),
    oracle.EvaluationCriterion("clarity", "Clarity", "Is it readable?", weight=1.0),
  ]
  verifier = oracle.VerifierStrategy(stub, config, criteria)
  judge = oracle.JudgeStrategy(stub, config, criteria, swap_pairwise=True)
  harness = oracle.EvaluationHarness(verifier=verifier, judge=judge)

  _info(f"Task         : {task_text[:70]}")
  _info(f"Trajectories : {n_trajs}")
  _info("Running harness …")

  record = harness.run_single(task, trajs)

  _section("Verifier Result")
  if record.verifier_result:
    vr = record.verifier_result
    _info(f"Best trajectory : {vr.best_trajectory_id}")
    _info("Scores:")
    for tid, sr in sorted(vr.trajectory_scores.items(), key=lambda x: -x[1].score):
      _info(f"  {tid:<12s}  overall={sr.score:.4f}  confidence={sr.confidence:.4f}")

  _section("Judge Result")
  if record.judge_result:
    jr = record.judge_result
    _info(f"Best trajectory : {jr.best_trajectory_id}")
    _info("Scores:")
    for tid, sr in sorted(jr.trajectory_scores.items(), key=lambda x: -x[1].score):
      _info(f"  {tid:<12s}  overall={sr.score:.4f}  confidence={sr.confidence:.4f}")

  _section("Hardness Metrics")
  _info(f"Composite hardness score   : {record.hardness_score:.4f}")
  _info(f"Score spread               : {record.score_spread:.4f}")
  _info(f"Strategy disagreement      : {record.strategy_disagreement:.4f}")
  _info(f"Average confidence         : {record.avg_confidence:.4f}")
  _info(f"Oracle gap (verifier)      : {record.oracle_gap_verifier:.4f}")
  _info(f"Oracle gap (judge)         : {record.oracle_gap_judge:.4f}")
  _info(f"Strategies agree           : {record.strategies_agree}")
  _info(f"Verifier wins              : {record.verifier_wins}")
  _info(f"Judge wins                 : {record.judge_wins}")
  _info(f"Elapsed (verifier)         : {record.elapsed_verifier_s:.3f} s")
  _info(f"Elapsed (judge)            : {record.elapsed_judge_s:.3f} s")
  print()


def cmd_test(_args: argparse.Namespace) -> None:
  """Run the test suite via pytest."""
  _banner("LLM Oracle — Test Suite")
  _info("Running pytest …")
  print()

  result = subprocess.run(
    [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"],
    check=False,
  )
  sys.exit(result.returncode)


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
    prog="llm-oracle",
    description=textwrap.dedent("""\
            LLM Oracle — LLM-as-a-Verifier · LLM-as-a-Judge · Intelligent Router

            Commands
            --------
              info     Print package version and component overview
              demo     Run the full end-to-end offline demo
              route    Route a task to verifier or judge and inspect the decision
              compare  Score a task with both strategies side-by-side
              test     Run the test suite
        """),
    formatter_class=argparse.RawDescriptionHelpFormatter,
  )
  sub = parser.add_subparsers(dest="command", metavar="<command>")

  sub.add_parser("info", help="Print package version and component overview")
  sub.add_parser("demo", help="Run the full end-to-end offline demo")

  route_p = sub.add_parser("route", help="Route a task and inspect the decision")
  route_p.add_argument("--task", "-t", required=True, help="Natural-language task description")
  route_p.add_argument(
    "--difficulty",
    "-d",
    choices=["easy", "medium", "hard", "unknown"],
    default="unknown",
  )
  route_p.add_argument("--trajectories", "-n", type=int, default=2, metavar="N")
  route_p.add_argument("--ground-truth", action="store_true", default=False)
  route_p.add_argument("--test-cases", action="store_true", default=False)
  route_p.add_argument("--with-output", action="store_true", default=False)
  route_p.add_argument("--prior-hardness", type=float, default=None, metavar="0.0-1.0")

  compare_p = sub.add_parser("compare", help="Score a task with both strategies side-by-side")
  compare_p.add_argument("--task", "-t", required=True, help="Natural-language task description")
  compare_p.add_argument("--trajectories", "-n", type=int, default=2, metavar="N")

  sub.add_parser("test", help="Run the test suite via pytest")

  return parser


def main(argv: list[str] | None = None) -> None:
  parser = build_parser()
  args = parser.parse_args(argv)

  dispatch = {
    "info": cmd_info,
    "demo": cmd_demo,
    "route": cmd_route,
    "compare": cmd_compare,
    "test": cmd_test,
  }

  if args.command is None:
    parser.print_help()
    sys.exit(0)

  handler = dispatch.get(args.command)
  if handler is None:
    _err(f"Unknown command: {args.command!r}")
    parser.print_help()
    sys.exit(1)

  handler(args)
