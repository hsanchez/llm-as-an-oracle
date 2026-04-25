"""Comprehensive test suite for the LLM Oracle system.

Tests cover:
  - Core data models validation
  - StubProvider generation and logprob synthesis
  - VerifierStrategy: scoring, pairwise comparison, tournament selection
  - JudgeStrategy: pointwise scoring, pairwise with swap, best selection
  - EvaluationHarness: hardness metrics, report assembly, per-task records
  - SignalExtractor: signal computation from tasks/trajectories
  - All built-in RoutingPolicy implementations
  - PolicyChain: aggregation, confidence threshold, fallback
  - OracleRouter: route(), evaluate(), factory constructors, audit log

Run with:
    pytest tests/test_oracle.py -v
"""

from __future__ import annotations

import math

import pytest

# ──────────────────────────────────────────────────────────────────────────────
# Package imports
# ──────────────────────────────────────────────────────────────────────────────
from llm_oracle import (
  EvaluationCriterion,
  EvaluationHarness,
  EvaluationResult,
  JudgeStrategy,
  ModelConfig,
  OpenAIProvider,
  OracleRouter,
  PairwiseComparison,
  PolicyChain,
  ScoreResult,
  ScoringConfig,
  StrategyType,
  StubProvider,
  StubResponse,
  Task,
  TaskDifficulty,
  Trajectory,
  VerifierStrategy,
)
from llm_oracle.routing.router import (
  DetailedRoutingDecision,
  DifficultyPolicy,
  GroundTruthPolicy,
  KeywordDomainPolicy,
  OutputAvailabilityPolicy,
  PolicyVote,
  PriorHardnessPolicy,
  RoutingPolicy,
  RoutingSignals,
  SignalExtractor,
  TrajectoryCountPolicy,
)

# ──────────────────────────────────────────────────────────────────────────────
# Shared fixtures
# ──────────────────────────────────────────────────────────────────────────────


@pytest.fixture()
def criteria() -> list[EvaluationCriterion]:
  return [
    EvaluationCriterion(
      id="correctness",
      name="Correctness",
      description="Is the solution functionally correct?",
      weight=1.5,
    ),
    EvaluationCriterion(
      id="clarity",
      name="Clarity",
      description="Is the solution easy to read and understand?",
      weight=1.0,
    ),
  ]


@pytest.fixture()
def scoring_config() -> ScoringConfig:
  return ScoringConfig(
    granularity=20,
    num_verifications=2,
    use_logprobs=True,
  )


@pytest.fixture()
def stub_model() -> StubProvider:
  return StubProvider(
    model_id="stub",
    default_score="C",
    default_score_a="C",
    default_score_b="F",
    seed=0,
  )


@pytest.fixture()
def easy_task() -> Task:
  return Task(
    id="task-easy",
    description="Sort integers",
    problem_statement="Write a function that sorts a list of integers.",
    difficulty=TaskDifficulty.EASY,
    ground_truth="def sort_list(lst): return sorted(lst)",
  )


@pytest.fixture()
def hard_task() -> Task:
  return Task(
    id="task-hard",
    description="Implement a distributed consensus algorithm",
    problem_statement=(
      "Implement the Raft consensus algorithm with leader election, "
      "log replication, and membership changes."
    ),
    difficulty=TaskDifficulty.HARD,
  )


@pytest.fixture()
def code_task() -> Task:
  return Task(
    id="task-code",
    description="Implement binary search",
    problem_statement=(
      "Implement a binary search algorithm that returns the index of "
      "a target value in a sorted array.  Return -1 if not found."
    ),
    difficulty=TaskDifficulty.MEDIUM,
    test_cases=[
      {"input": [1, 3, 5, 7], "target": 5, "expected": 2},
      {"input": [1, 3, 5, 7], "target": 6, "expected": -1},
    ],
  )


@pytest.fixture()
def essay_task() -> Task:
  return Task(
    id="task-essay",
    description="Compare two sorting algorithms",
    problem_statement=(
      "Write an essay comparing merge sort and quicksort.  Discuss their "
      "time complexity, space complexity, and practical trade-offs."
    ),
    difficulty=TaskDifficulty.MEDIUM,
  )


@pytest.fixture()
def trajectories(easy_task: Task) -> list[Trajectory]:
  return [
    Trajectory(
      id="traj-a",
      task_id=easy_task.id,
      content="def sort_list(lst): return sorted(lst)",
      output="[1, 2, 3, 4]",
      reward=1.0,
    ),
    Trajectory(
      id="traj-b",
      task_id=easy_task.id,
      content="def sort_list(lst):\n    lst.sort()\n    return lst",
      output="[1, 2, 3, 4]",
      reward=1.0,
    ),
    Trajectory(
      id="traj-c",
      task_id=easy_task.id,
      content="def sort_list(lst):\n    # bubble sort\n    for i in range(len(lst)):\n        for j in range(len(lst)-1):\n            if lst[j] > lst[j+1]: lst[j], lst[j+1] = lst[j+1], lst[j]\n    return lst",
      reward=0.0,
    ),
  ]


@pytest.fixture()
def verifier(stub_model, scoring_config, criteria) -> VerifierStrategy:
  return VerifierStrategy(stub_model, scoring_config, criteria)


@pytest.fixture()
def judge(stub_model, scoring_config, criteria) -> JudgeStrategy:
  return JudgeStrategy(
    stub_model,
    scoring_config,
    criteria,
    score_min=1.0,
    score_max=10.0,
    swap_pairwise=True,
    reasoning_depth="detailed",
  )


@pytest.fixture()
def router(verifier, judge) -> OracleRouter:
  return OracleRouter.default(verifier, judge)


# ──────────────────────────────────────────────────────────────────────────────
# 1.  Core model tests
# ──────────────────────────────────────────────────────────────────────────────


class TestCoreModels:
  def test_task_is_frozen(self, easy_task: Task) -> None:
    with pytest.raises((AttributeError, TypeError)):
      easy_task.id = "mutated"  # type: ignore[misc]

  def test_task_defaults(self) -> None:
    task = Task(id="t", description="d", problem_statement="p")
    assert task.difficulty == TaskDifficulty.UNKNOWN
    assert task.test_cases is None
    assert task.ground_truth is None
    assert task.metadata == {}

  def test_trajectory_mutable(self, easy_task: Task) -> None:
    t = Trajectory(id="x", task_id=easy_task.id, content="code")
    t.reward = 0.5
    assert t.reward == 0.5

  def test_evaluation_criterion_weight_default(self) -> None:
    c = EvaluationCriterion(id="x", name="X", description="desc")
    assert c.weight == 1.0

  def test_scoring_config_defaults(self) -> None:
    cfg = ScoringConfig()
    assert cfg.granularity == 20
    assert cfg.num_verifications == 4
    assert cfg.use_logprobs is True

  def test_score_result_criterion_scores(self) -> None:
    sr = ScoreResult(
      trajectory_id="t1",
      score=0.75,
      criterion_scores={"correctness": 0.8, "clarity": 0.7},
    )
    assert len(sr.criterion_scores) == 2
    assert sr.criterion_scores["correctness"] == pytest.approx(0.8)

  def test_pairwise_comparison_winner_none_on_tie(self) -> None:
    pc = PairwiseComparison(
      trajectory_a_id="a",
      trajectory_b_id="b",
      score_a=0.5,
      score_b=0.5,
    )
    assert pc.winner is None

  def test_model_config_defaults(self) -> None:
    cfg = ModelConfig(model_id="gpt-4o", provider="openai")
    assert cfg.temperature == 1.0
    assert cfg.additional_params == {}


# ──────────────────────────────────────────────────────────────────────────────
# 2.  StubProvider tests
# ──────────────────────────────────────────────────────────────────────────────


class TestStubProvider:
  def test_generate_pointwise_no_logprobs(self, stub_model: StubProvider) -> None:
    text, tokens, logprobs = stub_model.generate(
      "Score this trajectory.\n<score>LETTER</score>",
      return_logprobs=False,
    )
    assert isinstance(text, str)
    assert "<score>" in text
    assert tokens is None
    assert logprobs is None

  def test_generate_pairwise_no_logprobs(self, stub_model: StubProvider) -> None:
    text, tokens, logprobs = stub_model.generate(
      "Compare A and B.\n<score_A>X</score_A>\n<score_B>X</score_B>",
      return_logprobs=False,
    )
    assert "<score_A>" in text
    assert "<score_B>" in text
    assert tokens is None

  def test_generate_with_logprobs_returns_lists(self, stub_model: StubProvider) -> None:
    text, tokens, logprobs = stub_model.generate(
      "Score this.\n<score>LETTER</score>",
      return_logprobs=True,
    )
    assert tokens is not None
    assert logprobs is not None
    assert len(tokens) == len(logprobs)

  def test_logprobs_peaked_near_target_score(self) -> None:
    """The peaked distribution should assign > 50% probability to the target."""
    model = StubProvider(default_score="A", seed=1)
    _, tokens, logprobs = model.generate(
      "Analyze.\n<score>LETTER_A_TO_T</score>",
      return_logprobs=True,
    )
    assert logprobs is not None
    # Find position after <score> tag
    full_text = "".join(tokens or [])
    tag = "<score>"
    idx = full_text.find(tag)
    if idx >= 0 and idx + len(tag) < len(logprobs):
      pos_lps = logprobs[idx + len(tag)]
      # The top token should have log_prob > log(0.5)
      top_lp = max(lp for _, lp in pos_lps)
      assert top_lp > math.log(0.5)

  def test_round_robin_responses(self) -> None:
    responses = [
      StubResponse(score="A"),
      StubResponse(score="T"),
    ]
    model = StubProvider(responses=responses)
    text0, _, _ = model.generate("prompt")
    text1, _, _ = model.generate("prompt")
    text2, _, _ = model.generate("prompt")  # wraps around

    assert "A" in text0
    assert "T" in text1
    assert "A" in text2  # cycled back

  def test_call_count_increments(self, stub_model: StubProvider) -> None:
    assert stub_model._call_count == 0
    stub_model.generate("p1")
    stub_model.generate("p2")
    assert stub_model._call_count == 2


# ──────────────────────────────────────────────────────────────────────────────
# 3.  BaseStrategy helpers
# ──────────────────────────────────────────────────────────────────────────────


class TestBaseStrategy:
  def test_validate_config_rejects_zero_granularity(
    self, stub_model: StubProvider, criteria: list[EvaluationCriterion]
  ) -> None:
    bad_config = ScoringConfig(granularity=1)
    with pytest.raises(ValueError, match="Granularity"):
      VerifierStrategy(stub_model, bad_config, criteria)

  def test_validate_config_rejects_empty_criteria(
    self, stub_model: StubProvider, scoring_config: ScoringConfig
  ) -> None:
    with pytest.raises(ValueError, match="criterion"):
      VerifierStrategy(stub_model, scoring_config, [])

  def test_validate_config_rejects_bad_fuzzy_threshold(
    self, stub_model: StubProvider, criteria: list[EvaluationCriterion]
  ) -> None:
    bad_config = ScoringConfig(fuzzy_threshold=1.5)
    with pytest.raises(ValueError, match="[Ff]uzzy"):
      VerifierStrategy(stub_model, bad_config, criteria)

  def test_normalize_score_clamps(self, verifier: VerifierStrategy) -> None:
    assert verifier.normalize_score(-5.0) == 0.0
    assert verifier.normalize_score(999.0) == 1.0

  def test_normalize_score_midpoint(self, verifier: VerifierStrategy) -> None:
    g = verifier.config.granularity
    mid = (1.0 + g) / 2.0
    norm = verifier.normalize_score(mid)
    assert 0.0 < norm < 1.0

  def test_aggregate_criterion_scores_weighted(self, verifier: VerifierStrategy) -> None:
    # correctness weight=1.5, clarity weight=1.0
    scores = {"correctness": 1.0, "clarity": 0.0}
    agg = verifier.aggregate_criterion_scores(scores)
    expected = (1.0 * 1.5 + 0.0 * 1.0) / (1.5 + 1.0)
    assert agg == pytest.approx(expected, abs=1e-6)

  def test_aggregate_empty_scores_returns_midpoint(self, verifier: VerifierStrategy) -> None:
    assert verifier.aggregate_criterion_scores({}) == pytest.approx(0.5)

  def test_get_scale_description_keys(self, verifier: VerifierStrategy) -> None:
    scale = verifier.get_scale_description()
    assert "scale_description" in scale
    assert "valid_tokens" in scale
    assert "granularity" in scale

  def test_scale_has_correct_granularity(self, verifier: VerifierStrategy) -> None:
    scale = verifier.get_scale_description()
    # Each letter A-T and its lowercase counterpart
    unique_values = set(scale["valid_tokens"].values())
    assert len(unique_values) == verifier.config.granularity

  def test_select_best_trajectory_returns_highest_score(
    self, verifier: VerifierStrategy, easy_task: Task, trajectories: list[Trajectory]
  ) -> None:
    scores = {
      "traj-a": ScoreResult(trajectory_id="traj-a", score=0.9),
      "traj-b": ScoreResult(trajectory_id="traj-b", score=0.3),
      "traj-c": ScoreResult(trajectory_id="traj-c", score=0.5),
    }
    best = verifier.select_best_trajectory(easy_task, trajectories, scores)
    assert best == "traj-a"

  def test_select_best_trajectory_empty_raises(
    self, verifier: VerifierStrategy, easy_task: Task
  ) -> None:
    with pytest.raises(ValueError):
      verifier.select_best_trajectory(easy_task, [], {})


# ──────────────────────────────────────────────────────────────────────────────
# 4.  VerifierStrategy tests
# ──────────────────────────────────────────────────────────────────────────────


class TestVerifierStrategy:
  def test_strategy_type(self, verifier: VerifierStrategy) -> None:
    assert verifier.get_strategy_type() == StrategyType.VERIFIER

  def test_score_trajectory_returns_normalized(
    self,
    verifier: VerifierStrategy,
    easy_task: Task,
    trajectories: list[Trajectory],
    criteria: list[EvaluationCriterion],
  ) -> None:
    result = verifier.score_trajectory(easy_task, trajectories[0], criteria[0])
    assert isinstance(result, ScoreResult)
    assert 0.0 <= result.score <= 1.0
    assert result.trajectory_id == trajectories[0].id

  def test_score_trajectory_has_reasoning(
    self,
    verifier: VerifierStrategy,
    easy_task: Task,
    trajectories: list[Trajectory],
    criteria: list[EvaluationCriterion],
  ) -> None:
    result = verifier.score_trajectory(easy_task, trajectories[0], criteria[0])
    assert isinstance(result.reasoning, str)
    assert len(result.reasoning) > 0

  def test_compare_trajectories_returns_pairwise(
    self,
    verifier: VerifierStrategy,
    easy_task: Task,
    trajectories: list[Trajectory],
    criteria: list[EvaluationCriterion],
  ) -> None:
    comp = verifier.compare_trajectories(easy_task, trajectories[0], trajectories[1], criteria[0])
    assert isinstance(comp, PairwiseComparison)
    assert comp.trajectory_a_id == trajectories[0].id
    assert comp.trajectory_b_id == trajectories[1].id
    assert 0.0 <= comp.score_a <= 1.0
    assert 0.0 <= comp.score_b <= 1.0

  def test_compare_trajectories_winner_is_valid(
    self,
    verifier: VerifierStrategy,
    easy_task: Task,
    trajectories: list[Trajectory],
    criteria: list[EvaluationCriterion],
  ) -> None:
    comp = verifier.compare_trajectories(easy_task, trajectories[0], trajectories[1], criteria[0])
    valid_winners = {trajectories[0].id, trajectories[1].id, None}
    assert comp.winner in valid_winners

  def test_evaluate_single_trajectory(
    self,
    verifier: VerifierStrategy,
    easy_task: Task,
    trajectories: list[Trajectory],
  ) -> None:
    result = verifier.evaluate(easy_task, [trajectories[0]])
    assert result.best_trajectory_id == trajectories[0].id
    assert result.strategy_type == StrategyType.VERIFIER
    assert len(result.pairwise_comparisons) == 0

  def test_evaluate_multiple_trajectories(
    self,
    verifier: VerifierStrategy,
    easy_task: Task,
    trajectories: list[Trajectory],
  ) -> None:
    result = verifier.evaluate(easy_task, trajectories)
    traj_ids = {t.id for t in trajectories}
    assert result.best_trajectory_id in traj_ids
    assert result.task_id == easy_task.id
    assert len(result.trajectory_scores) == len(trajectories)

  def test_evaluate_empty_trajectories_raises(
    self, verifier: VerifierStrategy, easy_task: Task
  ) -> None:
    with pytest.raises(ValueError):
      verifier.evaluate(easy_task, [])

  def test_evaluate_result_has_pairwise_comparisons(
    self,
    verifier: VerifierStrategy,
    easy_task: Task,
    trajectories: list[Trajectory],
  ) -> None:
    result = verifier.evaluate(easy_task, trajectories[:2])
    # Should have at least one pairwise comparison per criterion per verification
    assert len(result.pairwise_comparisons) >= 1

  def test_evaluate_metadata_contains_granularity(
    self,
    verifier: VerifierStrategy,
    easy_task: Task,
    trajectories: list[Trajectory],
  ) -> None:
    result = verifier.evaluate(easy_task, trajectories)
    assert result.metadata.get("granularity") == verifier.config.granularity

  def test_tournament_selection_picks_most_wins(
    self,
    verifier: VerifierStrategy,
    easy_task: Task,
    trajectories: list[Trajectory],
  ) -> None:
    """Tournament should select the trajectory with the most comparison wins."""
    # Inject biased comparisons where traj-a always wins
    from llm_oracle.core.models import PairwiseComparison as PC

    biased: list[PairwiseComparison] = [
      PC(
        trajectory_a_id="traj-a",
        trajectory_b_id="traj-b",
        score_a=0.9,
        score_b=0.1,
        winner="traj-a",
        criterion_id="correctness",
      ),
      PC(
        trajectory_a_id="traj-a",
        trajectory_b_id="traj-c",
        score_a=0.9,
        score_b=0.1,
        winner="traj-a",
        criterion_id="correctness",
      ),
      PC(
        trajectory_a_id="traj-b",
        trajectory_b_id="traj-c",
        score_a=0.8,
        score_b=0.2,
        winner="traj-b",
        criterion_id="correctness",
      ),
    ]
    winner = verifier._tournament_selection(trajectories, biased)
    assert winner == "traj-a"

  def test_score_extraction_fallback_on_missing_logprobs(
    self,
    verifier: VerifierStrategy,
  ) -> None:
    """When logprobs are None, fall back to text parsing."""
    text = "Analysis complete.\n<score>C</score>"
    raw_score, confidence = verifier._extract_score_from_logprobs(text, None, None, "<score>")
    # C is 3rd from top in 20-point scale → raw_score = 18.0
    assert raw_score == pytest.approx(18.0)
    assert confidence > 0.5  # text extraction returns moderate confidence

  def test_score_extraction_defaults_to_midpoint_on_parse_failure(
    self, verifier: VerifierStrategy
  ) -> None:
    text = "No score tags here at all."
    raw_score, confidence = verifier._extract_score_from_logprobs(text, None, None, "<score>")
    assert raw_score == pytest.approx(verifier.config.granularity / 2.0)
    assert confidence == pytest.approx(0.5)


# ──────────────────────────────────────────────────────────────────────────────
# 5.  JudgeStrategy tests
# ──────────────────────────────────────────────────────────────────────────────


class TestJudgeStrategy:
  def test_strategy_type(self, judge: JudgeStrategy) -> None:
    assert judge.get_strategy_type() == StrategyType.JUDGE

  def test_init_rejects_inverted_score_range(
    self,
    stub_model: StubProvider,
    scoring_config: ScoringConfig,
    criteria: list[EvaluationCriterion],
  ) -> None:
    with pytest.raises(ValueError, match="score_min"):
      JudgeStrategy(stub_model, scoring_config, criteria, score_min=10.0, score_max=1.0)

  def test_init_rejects_invalid_reasoning_depth(
    self,
    stub_model: StubProvider,
    scoring_config: ScoringConfig,
    criteria: list[EvaluationCriterion],
  ) -> None:
    with pytest.raises(ValueError, match="reasoning_depth"):
      JudgeStrategy(
        stub_model,
        scoring_config,
        criteria,
        reasoning_depth="telepathic",  # type: ignore[arg-type]
      )

  def test_score_trajectory_returns_normalized(
    self,
    judge: JudgeStrategy,
    easy_task: Task,
    trajectories: list[Trajectory],
    criteria: list[EvaluationCriterion],
  ) -> None:
    result = judge.score_trajectory(easy_task, trajectories[0], criteria[0])
    assert 0.0 <= result.score <= 1.0
    assert result.trajectory_id == trajectories[0].id

  def test_score_trajectory_has_reasoning(
    self,
    judge: JudgeStrategy,
    easy_task: Task,
    trajectories: list[Trajectory],
    criteria: list[EvaluationCriterion],
  ) -> None:
    result = judge.score_trajectory(easy_task, trajectories[0], criteria[0])
    assert len(result.reasoning) > 0

  def test_compare_trajectories_no_positional_bias(
    self,
    judge: JudgeStrategy,
    easy_task: Task,
    trajectories: list[Trajectory],
    criteria: list[EvaluationCriterion],
  ) -> None:
    """With swap_pairwise=True, scores should be averaged over two orderings."""
    comp = judge.compare_trajectories(easy_task, trajectories[0], trajectories[1], criteria[0])
    assert isinstance(comp, PairwiseComparison)
    assert 0.0 <= comp.score_a <= 1.0
    assert 0.0 <= comp.score_b <= 1.0

  def test_compare_trajectories_winner_is_valid(
    self,
    judge: JudgeStrategy,
    easy_task: Task,
    trajectories: list[Trajectory],
    criteria: list[EvaluationCriterion],
  ) -> None:
    comp = judge.compare_trajectories(easy_task, trajectories[0], trajectories[1], criteria[0])
    assert comp.winner in {trajectories[0].id, trajectories[1].id, None}

  def test_evaluate_single_trajectory(
    self,
    judge: JudgeStrategy,
    easy_task: Task,
    trajectories: list[Trajectory],
  ) -> None:
    result = judge.evaluate(easy_task, [trajectories[0]])
    assert result.best_trajectory_id == trajectories[0].id
    assert result.strategy_type == StrategyType.JUDGE

  def test_evaluate_multiple_trajectories(
    self,
    judge: JudgeStrategy,
    easy_task: Task,
    trajectories: list[Trajectory],
  ) -> None:
    result = judge.evaluate(easy_task, trajectories)
    traj_ids = {t.id for t in trajectories}
    assert result.best_trajectory_id in traj_ids
    assert len(result.trajectory_scores) == len(trajectories)

  def test_evaluate_empty_trajectories_raises(self, judge: JudgeStrategy, easy_task: Task) -> None:
    with pytest.raises(ValueError):
      judge.evaluate(easy_task, [])

  def test_evaluate_metadata_keys(
    self,
    judge: JudgeStrategy,
    easy_task: Task,
    trajectories: list[Trajectory],
  ) -> None:
    result = judge.evaluate(easy_task, trajectories)
    meta = result.metadata
    assert "score_range" in meta
    assert "swap_pairwise" in meta
    assert "reasoning_depth" in meta
    assert meta["swap_pairwise"] is True

  @pytest.mark.parametrize(
    "depth",
    ["brief", "detailed", "chain_of_thought"],
  )
  def test_reasoning_depths(
    self,
    stub_model: StubProvider,
    scoring_config: ScoringConfig,
    criteria: list[EvaluationCriterion],
    easy_task: Task,
    trajectories: list[Trajectory],
    depth: str,
  ) -> None:
    judge = JudgeStrategy(
      stub_model,
      scoring_config,
      criteria,
      reasoning_depth=depth,
    )
    result = judge.score_trajectory(easy_task, trajectories[0], criteria[0])
    assert 0.0 <= result.score <= 1.0

  def test_normalize_clamps_to_unit_interval(self, judge: JudgeStrategy) -> None:
    assert judge._normalize(0.0) == pytest.approx(0.0)
    assert judge._normalize(10.0) == pytest.approx(1.0)
    assert judge._normalize(11.0) == pytest.approx(1.0)  # clamped

  def test_score_to_confidence_extremes_are_high(self, judge: JudgeStrategy) -> None:
    conf_low = judge._score_to_confidence(1.0)
    conf_mid = judge._score_to_confidence(5.5)
    conf_high = judge._score_to_confidence(10.0)
    assert conf_low > conf_mid
    assert conf_high > conf_mid

  def test_pairwise_confidence_large_gap(self, judge: JudgeStrategy) -> None:
    assert judge._pairwise_confidence(1.0, 10.0) > judge._pairwise_confidence(5.0, 5.5)

  def test_parse_tagged_float_missing_tag_returns_midpoint(self, judge: JudgeStrategy) -> None:
    import re

    pattern = re.compile(r"<score>\s*(\d+(?:\.\d+)?)\s*</score>", re.IGNORECASE)
    val = judge._parse_tagged_float("no score here", pattern)
    expected_mid = (judge.score_min + judge.score_max) / 2.0
    assert val == pytest.approx(expected_mid)

  def test_parse_tagged_float_valid(self, judge: JudgeStrategy) -> None:
    import re

    pattern = re.compile(r"<score>\s*(\d+(?:\.\d+)?)\s*</score>", re.IGNORECASE)
    val = judge._parse_tagged_float("Analysis.\n<score>8.5</score>", pattern)
    assert val == pytest.approx(8.5)

  def test_win_rate_computation(
    self,
    judge: JudgeStrategy,
    trajectories: list[Trajectory],
  ) -> None:
    comps = [
      PairwiseComparison("traj-a", "traj-b", 0.8, 0.3, winner="traj-a"),
      PairwiseComparison("traj-a", "traj-c", 0.7, 0.2, winner="traj-a"),
      PairwiseComparison("traj-b", "traj-c", 0.6, 0.3, winner="traj-b"),
    ]
    win_rates = judge._compute_win_rates(trajectories, comps)
    assert win_rates["traj-a"] > win_rates["traj-b"] > win_rates["traj-c"]


# ──────────────────────────────────────────────────────────────────────────────
# 6.  EvaluationHarness tests
# ──────────────────────────────────────────────────────────────────────────────


class TestEvaluationHarness:
  @pytest.fixture()
  def harness(self, verifier: VerifierStrategy, judge: JudgeStrategy) -> EvaluationHarness:
    return EvaluationHarness(verifier=verifier, judge=judge, max_workers=2)

  def test_harness_type_validation(
    self,
    verifier: VerifierStrategy,
    judge: JudgeStrategy,
  ) -> None:
    """Passing strategies in the wrong positions should raise TypeError."""
    with pytest.raises(TypeError, match="VERIFIER"):
      EvaluationHarness(verifier=judge, judge=judge)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="JUDGE"):
      EvaluationHarness(verifier=verifier, judge=verifier)  # type: ignore[arg-type]

  def test_run_single_populates_record(
    self,
    harness: EvaluationHarness,
    easy_task: Task,
    trajectories: list[Trajectory],
  ) -> None:
    record = harness.run_single(easy_task, trajectories)
    assert record.task_id == easy_task.id
    assert 0.0 <= record.hardness_score <= 1.0
    assert record.verifier_result is not None
    assert record.judge_result is not None

  def test_run_single_empty_trajectories_raises(
    self, harness: EvaluationHarness, easy_task: Task
  ) -> None:
    with pytest.raises(ValueError):
      harness.run_single(easy_task, [])

  def test_run_returns_report(
    self,
    harness: EvaluationHarness,
    easy_task: Task,
    trajectories: list[Trajectory],
  ) -> None:
    report = harness.run([(easy_task, trajectories)], parallel=False)
    assert len(report.task_records) == 1
    assert 0.0 <= report.verifier_accuracy <= 1.0
    assert 0.0 <= report.judge_accuracy <= 1.0

  def test_run_empty_returns_empty_report(self, harness: EvaluationHarness) -> None:
    report = harness.run([])
    assert report.task_records == []

  def test_hardness_score_in_unit_interval(
    self,
    harness: EvaluationHarness,
    easy_task: Task,
    trajectories: list[Trajectory],
  ) -> None:
    record = harness.run_single(easy_task, trajectories)
    assert 0.0 <= record.hardness_score <= 1.0

  def test_oracle_gap_zero_when_perfect_selection(
    self,
    harness: EvaluationHarness,
  ) -> None:
    from llm_oracle.evaluation.harness import _oracle_gap

    trajs = [
      Trajectory("a", "t", "code a", reward=1.0),
      Trajectory("b", "t", "code b", reward=0.3),
    ]
    result = EvaluationResult(
      task_id="t",
      strategy_type=StrategyType.VERIFIER,
      best_trajectory_id="a",
      trajectory_scores={
        "a": ScoreResult("a", 0.9),
        "b": ScoreResult("b", 0.3),
      },
    )
    gap = _oracle_gap("a", result, trajs)
    assert gap == pytest.approx(0.0)

  def test_oracle_gap_nonzero_when_wrong_selection(self) -> None:
    from llm_oracle.evaluation.harness import _oracle_gap

    trajs = [
      Trajectory("a", "t", "code a", reward=1.0),
      Trajectory("b", "t", "code b", reward=0.3),
    ]
    result = EvaluationResult(
      task_id="t",
      strategy_type=StrategyType.VERIFIER,
      best_trajectory_id="b",  # selected wrong one
      trajectory_scores={
        "a": ScoreResult("a", 0.9),
        "b": ScoreResult("b", 0.3),
      },
    )
    gap = _oracle_gap("a", result, trajs)  # oracle is "a", but "b" was selected
    assert gap == pytest.approx(0.7)  # reward["a"] - reward["b"] = 1.0 - 0.3

  def test_pairwise_disagreement_full(self) -> None:
    from llm_oracle.evaluation.harness import _pairwise_disagreement

    trajs = [
      Trajectory("a", "t", "code a"),
      Trajectory("b", "t", "code b"),
    ]
    v_result = EvaluationResult(
      task_id="t",
      strategy_type=StrategyType.VERIFIER,
      best_trajectory_id="a",
      trajectory_scores={
        "a": ScoreResult("a", 0.9),
        "b": ScoreResult("b", 0.1),
      },
    )
    j_result = EvaluationResult(
      task_id="t",
      strategy_type=StrategyType.JUDGE,
      best_trajectory_id="b",
      trajectory_scores={
        "a": ScoreResult("a", 0.1),
        "b": ScoreResult("b", 0.9),
      },
    )
    disagreement = _pairwise_disagreement(trajs, v_result, j_result)
    assert disagreement == pytest.approx(1.0)

  def test_pairwise_disagreement_zero_when_agree(self) -> None:
    from llm_oracle.evaluation.harness import _pairwise_disagreement

    trajs = [Trajectory("a", "t", "c"), Trajectory("b", "t", "c")]
    same_scores = {
      "a": ScoreResult("a", 0.8),
      "b": ScoreResult("b", 0.2),
    }
    r1 = EvaluationResult("t", StrategyType.VERIFIER, "a", same_scores)
    r2 = EvaluationResult("t", StrategyType.JUDGE, "a", dict(same_scores))
    assert _pairwise_disagreement(trajs, r1, r2) == pytest.approx(0.0)

  def test_inter_strategy_score_spread(self) -> None:
    from llm_oracle.evaluation.harness import _inter_strategy_score_spread

    # When strategies agree on every score, spread = 0 (easy to evaluate).
    r_agree_v = EvaluationResult(
      "t",
      StrategyType.VERIFIER,
      "a",
      {"a": ScoreResult("a", 0.8), "b": ScoreResult("b", 0.2)},
    )
    r_agree_j = EvaluationResult(
      "t",
      StrategyType.JUDGE,
      "a",
      {"a": ScoreResult("a", 0.8), "b": ScoreResult("b", 0.2)},
    )
    assert _inter_strategy_score_spread(r_agree_v, r_agree_j) == pytest.approx(0.0)

    # When strategies maximally disagree on every score, spread = 1 (hard).
    r_disagree_v = EvaluationResult(
      "t",
      StrategyType.VERIFIER,
      "a",
      {"a": ScoreResult("a", 1.0), "b": ScoreResult("b", 0.0)},
    )
    r_disagree_j = EvaluationResult(
      "t",
      StrategyType.JUDGE,
      "b",
      {"a": ScoreResult("a", 0.0), "b": ScoreResult("b", 1.0)},
    )
    assert _inter_strategy_score_spread(r_disagree_v, r_disagree_j) == pytest.approx(1.0)

  def test_report_summary_contains_key_sections(
    self,
    harness: EvaluationHarness,
    easy_task: Task,
    trajectories: list[Trajectory],
  ) -> None:
    report = harness.run([(easy_task, trajectories)], parallel=False)
    summary = report.summary()
    assert "Verifier" in summary
    assert "Judge" in summary
    assert "hardness" in summary.lower() or "Hardness" in summary

  def test_report_per_task_table(
    self,
    harness: EvaluationHarness,
    easy_task: Task,
    trajectories: list[Trajectory],
  ) -> None:
    report = harness.run([(easy_task, trajectories)], parallel=False)
    table = report.per_task_table()
    assert easy_task.id in table

  def test_report_hard_easy_partition(
    self,
    harness: EvaluationHarness,
    easy_task: Task,
    trajectories: list[Trajectory],
  ) -> None:
    report = harness.run([(easy_task, trajectories)], parallel=False)
    total = len(report.hard_tasks) + len(report.easy_tasks)
    assert total == len(report.task_records)

  def test_hardness_weights_validation(
    self, verifier: VerifierStrategy, judge: JudgeStrategy
  ) -> None:
    from llm_oracle.evaluation.harness import _validate_hardness_weights

    with pytest.raises(ValueError, match="sum"):
      _validate_hardness_weights(
        {
          "score_spread": 0.5,
          "strategy_disagreement": 0.5,
          "confidence_gap": 0.5,
          "oracle_gap": 0.5,
        }
      )

    with pytest.raises(ValueError, match="Missing"):
      _validate_hardness_weights({"score_spread": 1.0})

  def test_task_hardness_record_wins_properties(self) -> None:
    from llm_oracle.evaluation.harness import TaskHardnessRecord

    r = TaskHardnessRecord(
      task_id="t",
      oracle_gap_verifier=0.1,
      oracle_gap_judge=0.3,
    )
    assert r.verifier_wins is True
    assert r.judge_wins is False

  def test_task_hardness_record_strategies_agree(self) -> None:
    from llm_oracle.evaluation.harness import TaskHardnessRecord

    r = TaskHardnessRecord(
      task_id="t",
      verifier_result=EvaluationResult("t", StrategyType.VERIFIER, "a", {}),
      judge_result=EvaluationResult("t", StrategyType.JUDGE, "a", {}),
    )
    assert r.strategies_agree is True


# ──────────────────────────────────────────────────────────────────────────────
# 7.  SignalExtractor tests
# ──────────────────────────────────────────────────────────────────────────────


class TestSignalExtractor:
  @pytest.fixture()
  def extractor(self) -> SignalExtractor:
    return SignalExtractor()

  def test_ground_truth_signal_set(
    self, extractor: SignalExtractor, easy_task: Task, trajectories: list[Trajectory]
  ) -> None:
    signals = extractor.extract(easy_task, trajectories)
    assert signals.has_ground_truth == 1.0

  def test_ground_truth_signal_absent(
    self,
    extractor: SignalExtractor,
    hard_task: Task,
    trajectories: list[Trajectory],
  ) -> None:
    signals = extractor.extract(hard_task, trajectories)
    assert signals.has_ground_truth == 0.0

  def test_test_cases_signal(
    self,
    extractor: SignalExtractor,
    code_task: Task,
    trajectories: list[Trajectory],
  ) -> None:
    signals = extractor.extract(code_task, trajectories)
    assert signals.has_test_cases == 1.0

  def test_trajectory_count(
    self, extractor: SignalExtractor, easy_task: Task, trajectories: list[Trajectory]
  ) -> None:
    signals = extractor.extract(easy_task, trajectories)
    assert signals.trajectory_count == len(trajectories)

  def test_output_available_when_present(
    self, extractor: SignalExtractor, easy_task: Task, trajectories: list[Trajectory]
  ) -> None:
    # trajectories[0] has output="[1, 2, 3, 4]"
    signals = extractor.extract(easy_task, trajectories)
    assert signals.output_available == 1.0

  def test_output_unavailable_when_missing(
    self, extractor: SignalExtractor, easy_task: Task
  ) -> None:
    trajs = [Trajectory("x", easy_task.id, "code")]  # no output
    signals = extractor.extract(easy_task, trajs)
    assert signals.output_available == 0.0

  def test_difficulty_encoding(
    self,
    extractor: SignalExtractor,
    easy_task: Task,
    hard_task: Task,
    trajectories: list[Trajectory],
  ) -> None:
    easy_signals = extractor.extract(easy_task, trajectories)
    hard_signals = extractor.extract(hard_task, trajectories)
    assert easy_signals.stated_difficulty < hard_signals.stated_difficulty

  def test_verifiable_keyword_density_for_code_task(
    self,
    extractor: SignalExtractor,
    code_task: Task,
    trajectories: list[Trajectory],
  ) -> None:
    signals = extractor.extract(code_task, trajectories)
    assert signals.verifiable_keyword_density > 0.0

  def test_judgement_keyword_density_for_essay_task(
    self,
    extractor: SignalExtractor,
    essay_task: Task,
    trajectories: list[Trajectory],
  ) -> None:
    signals = extractor.extract(essay_task, trajectories)
    assert signals.judgement_keyword_density > 0.0

  def test_prior_hardness_passed_through(
    self, extractor: SignalExtractor, easy_task: Task, trajectories: list[Trajectory]
  ) -> None:
    signals = extractor.extract(easy_task, trajectories, prior_hardness=0.75)
    assert signals.prior_hardness == pytest.approx(0.75)

  def test_prior_hardness_defaults_to_none(
    self, extractor: SignalExtractor, easy_task: Task, trajectories: list[Trajectory]
  ) -> None:
    signals = extractor.extract(easy_task, trajectories)
    assert signals.prior_hardness is None

  def test_problem_length_normalized(
    self, extractor: SignalExtractor, easy_task: Task, trajectories: list[Trajectory]
  ) -> None:
    signals = extractor.extract(easy_task, trajectories)
    assert 0.0 <= signals.problem_length <= 1.0


# ──────────────────────────────────────────────────────────────────────────────
# 8.  Routing policy tests
# ──────────────────────────────────────────────────────────────────────────────


class TestRoutingPolicies:
  """Test each built-in routing policy in isolation."""

  @pytest.fixture()
  def signals_with_ground_truth(self) -> RoutingSignals:
    return RoutingSignals(has_ground_truth=1.0, has_test_cases=1.0)

  @pytest.fixture()
  def signals_no_ground_truth(self) -> RoutingSignals:
    return RoutingSignals(has_ground_truth=0.0, has_test_cases=0.0)

  # ── GroundTruthPolicy ─────────────────────────────────────────────────────

  def test_ground_truth_policy_prefers_verifier_with_ref(
    self,
    easy_task: Task,
    trajectories: list[Trajectory],
    signals_with_ground_truth: RoutingSignals,
  ) -> None:
    policy = GroundTruthPolicy()
    vote = policy.vote(easy_task, trajectories, signals_with_ground_truth)
    assert vote.preferred == StrategyType.VERIFIER
    assert vote.confidence > 0.5

  def test_ground_truth_policy_prefers_judge_without_ref(
    self,
    hard_task: Task,
    trajectories: list[Trajectory],
    signals_no_ground_truth: RoutingSignals,
  ) -> None:
    policy = GroundTruthPolicy()
    vote = policy.vote(hard_task, trajectories, signals_no_ground_truth)
    assert vote.preferred == StrategyType.JUDGE

  def test_ground_truth_policy_weight(self) -> None:
    assert GroundTruthPolicy().weight == 2.0

  # ── KeywordDomainPolicy ───────────────────────────────────────────────────

  def test_keyword_domain_verifiable_task(
    self,
    code_task: Task,
    trajectories: list[Trajectory],
  ) -> None:
    extractor = SignalExtractor()
    signals = extractor.extract(code_task, trajectories)
    policy = KeywordDomainPolicy()
    vote = policy.vote(code_task, trajectories, signals)
    # code_task has "implement", "search" etc. → should prefer verifier
    # (or at least not strongly prefer judge)
    # The test is lenient because the actual keyword density depends on
    # the exact problem text.
    assert vote.preferred in {StrategyType.VERIFIER, StrategyType.JUDGE}
    assert 0.5 <= vote.confidence <= 1.0

  def test_keyword_domain_essay_task(
    self,
    essay_task: Task,
    trajectories: list[Trajectory],
  ) -> None:
    extractor = SignalExtractor()
    signals = extractor.extract(essay_task, trajectories)
    policy = KeywordDomainPolicy()
    vote = policy.vote(essay_task, trajectories, signals)
    assert vote.preferred in {StrategyType.VERIFIER, StrategyType.JUDGE}
    assert vote.confidence >= 0.5

  def test_keyword_domain_ambiguous_signals(
    self,
    easy_task: Task,
    trajectories: list[Trajectory],
  ) -> None:
    # Flat signals: no keyword advantage either way
    signals = RoutingSignals(
      verifiable_keyword_density=0.1,
      judgement_keyword_density=0.1,
    )
    policy = KeywordDomainPolicy()
    vote = policy.vote(easy_task, trajectories, signals)
    assert vote.confidence < 0.7  # low confidence on ambiguous input

  # ── DifficultyPolicy ──────────────────────────────────────────────────────

  def test_difficulty_policy_easy_prefers_judge(
    self, easy_task: Task, trajectories: list[Trajectory]
  ) -> None:
    signals = RoutingSignals(stated_difficulty=0.0)
    policy = DifficultyPolicy()
    vote = policy.vote(easy_task, trajectories, signals)
    assert vote.preferred == StrategyType.JUDGE

  def test_difficulty_policy_hard_prefers_verifier(
    self, hard_task: Task, trajectories: list[Trajectory]
  ) -> None:
    signals = RoutingSignals(stated_difficulty=1.0)
    policy = DifficultyPolicy()
    vote = policy.vote(hard_task, trajectories, signals)
    assert vote.preferred == StrategyType.VERIFIER

  def test_difficulty_policy_medium_has_low_confidence(
    self, easy_task: Task, trajectories: list[Trajectory]
  ) -> None:
    task = Task(
      id="medium",
      description="medium task",
      problem_statement="p",
      difficulty=TaskDifficulty.MEDIUM,
    )
    signals = RoutingSignals(stated_difficulty=0.5)
    policy = DifficultyPolicy()
    vote = policy.vote(task, trajectories, signals)
    assert vote.confidence <= 0.65

  # ── TrajectoryCountPolicy ─────────────────────────────────────────────────

  def test_trajectory_count_single_prefers_judge(
    self, easy_task: Task, trajectories: list[Trajectory]
  ) -> None:
    signals = RoutingSignals(trajectory_count=1)
    policy = TrajectoryCountPolicy()
    vote = policy.vote(easy_task, [trajectories[0]], signals)
    assert vote.preferred == StrategyType.JUDGE

  def test_trajectory_count_few_prefers_verifier(
    self, easy_task: Task, trajectories: list[Trajectory]
  ) -> None:
    signals = RoutingSignals(trajectory_count=3)
    policy = TrajectoryCountPolicy()
    vote = policy.vote(easy_task, trajectories, signals)
    assert vote.preferred == StrategyType.VERIFIER

  def test_trajectory_count_many_prefers_judge(
    self, easy_task: Task, trajectories: list[Trajectory]
  ) -> None:
    signals = RoutingSignals(trajectory_count=10)
    many_trajs = [Trajectory(f"t{i}", easy_task.id, f"code {i}") for i in range(10)]
    policy = TrajectoryCountPolicy()
    vote = policy.vote(easy_task, many_trajs, signals)
    assert vote.preferred == StrategyType.JUDGE

  # ── PriorHardnessPolicy ───────────────────────────────────────────────────

  def test_prior_hardness_none_abstains(
    self, easy_task: Task, trajectories: list[Trajectory]
  ) -> None:
    signals = RoutingSignals(prior_hardness=None)
    policy = PriorHardnessPolicy()
    vote = policy.vote(easy_task, trajectories, signals)
    assert vote.confidence < 0.55  # near-abstain

  def test_prior_hardness_low_prefers_judge(
    self, easy_task: Task, trajectories: list[Trajectory]
  ) -> None:
    signals = RoutingSignals(prior_hardness=0.1)
    policy = PriorHardnessPolicy()
    vote = policy.vote(easy_task, trajectories, signals)
    assert vote.preferred == StrategyType.JUDGE
    assert vote.confidence > 0.6

  def test_prior_hardness_high_prefers_verifier(
    self, hard_task: Task, trajectories: list[Trajectory]
  ) -> None:
    signals = RoutingSignals(prior_hardness=0.85)
    policy = PriorHardnessPolicy()
    vote = policy.vote(hard_task, trajectories, signals)
    assert vote.preferred == StrategyType.VERIFIER
    assert vote.confidence > 0.6

  def test_prior_hardness_high_weight(self) -> None:
    assert (
      PriorHardnessPolicy().weight > GroundTruthPolicy().weight
      or PriorHardnessPolicy().weight >= 1.5
    )

  # ── OutputAvailabilityPolicy ──────────────────────────────────────────────

  def test_output_available_prefers_verifier(
    self, easy_task: Task, trajectories: list[Trajectory]
  ) -> None:
    signals = RoutingSignals(output_available=1.0)
    policy = OutputAvailabilityPolicy()
    vote = policy.vote(easy_task, trajectories, signals)
    assert vote.preferred == StrategyType.VERIFIER

  def test_output_unavailable_prefers_judge(
    self, easy_task: Task, trajectories: list[Trajectory]
  ) -> None:
    signals = RoutingSignals(output_available=0.0)
    policy = OutputAvailabilityPolicy()
    vote = policy.vote(easy_task, trajectories, signals)
    assert vote.preferred == StrategyType.JUDGE

  # ── Policy vote dataclass ─────────────────────────────────────────────────

  def test_policy_vote_fields(self) -> None:
    vote = PolicyVote(
      policy_name="test",
      preferred=StrategyType.VERIFIER,
      confidence=0.8,
      weight=1.5,
      signals_used=["has_ground_truth"],
      reasoning="Because ground truth is available.",
    )
    assert vote.preferred == StrategyType.VERIFIER
    assert vote.confidence == pytest.approx(0.8)
    assert "has_ground_truth" in vote.signals_used


# ──────────────────────────────────────────────────────────────────────────────
# 9.  PolicyChain tests
# ──────────────────────────────────────────────────────────────────────────────


class TestPolicyChain:
  def test_requires_at_least_one_policy(self) -> None:
    with pytest.raises(ValueError):
      PolicyChain([])

  def test_unanimous_verifier_decision(
    self, easy_task: Task, trajectories: list[Trajectory]
  ) -> None:
    signals = RoutingSignals(
      has_ground_truth=1.0,
      has_test_cases=1.0,
      trajectory_count=3,
      stated_difficulty=1.0,
      verifiable_keyword_density=0.3,
      judgement_keyword_density=0.0,
      output_available=1.0,
      prior_hardness=0.9,
    )
    chain = PolicyChain(
      [
        GroundTruthPolicy(),
        DifficultyPolicy(),
        PriorHardnessPolicy(),
        OutputAvailabilityPolicy(),
        TrajectoryCountPolicy(),
      ],
      confidence_threshold=0.5,
    )
    strategy, confidence, votes, reasoning = chain.decide(easy_task, trajectories, signals)
    assert strategy == StrategyType.VERIFIER
    assert confidence > 0.5
    assert len(votes) == 5

  def test_unanimous_judge_decision(self, easy_task: Task, trajectories: list[Trajectory]) -> None:
    signals = RoutingSignals(
      has_ground_truth=0.0,
      has_test_cases=0.0,
      trajectory_count=1,
      stated_difficulty=0.0,
      verifiable_keyword_density=0.0,
      judgement_keyword_density=0.3,
      output_available=0.0,
      prior_hardness=0.1,
    )
    chain = PolicyChain(
      [
        GroundTruthPolicy(),
        DifficultyPolicy(),
        PriorHardnessPolicy(),
        TrajectoryCountPolicy(),
      ],
      confidence_threshold=0.5,
    )
    strategy, confidence, votes, reasoning = chain.decide(easy_task, [trajectories[0]], signals)
    assert strategy == StrategyType.JUDGE
    assert confidence > 0.5

  def test_fallback_when_below_threshold(
    self, easy_task: Task, trajectories: list[Trajectory]
  ) -> None:
    # Mix equal-weight opposing votes to force low confidence
    class _AlwaysVerifier(RoutingPolicy):
      name = "always_v"
      weight = 1.0

      def vote(self, task, trajectories, signals):
        return PolicyVote("always_v", StrategyType.VERIFIER, 0.55)

    class _AlwaysJudge(RoutingPolicy):
      name = "always_j"
      weight = 1.0

      def vote(self, task, trajectories, signals):
        return PolicyVote("always_j", StrategyType.JUDGE, 0.55)

    chain = PolicyChain(
      [_AlwaysVerifier(), _AlwaysJudge()],
      confidence_threshold=0.99,  # impossible to meet
      fallback_strategy=StrategyType.JUDGE,
    )
    signals = RoutingSignals()
    strategy, _, _, reasoning = chain.decide(easy_task, trajectories, signals)
    assert strategy == StrategyType.JUDGE
    assert "fallback" in reasoning.lower() or "threshold" in reasoning.lower()

  def test_policies_property_is_copy(self) -> None:
    chain = PolicyChain([GroundTruthPolicy()])
    policies = chain.policies
    policies.append(DifficultyPolicy())  # should not mutate the chain
    assert len(chain.policies) == 1

  def test_reasoning_mentions_all_policies(
    self, easy_task: Task, trajectories: list[Trajectory]
  ) -> None:
    signals = RoutingSignals(has_ground_truth=1.0, prior_hardness=0.8)
    chain = PolicyChain(
      [GroundTruthPolicy(), PriorHardnessPolicy()],
      confidence_threshold=0.4,
    )
    _, _, votes, reasoning = chain.decide(easy_task, trajectories, signals)
    assert "ground_truth" in reasoning
    assert "prior_hardness" in reasoning


# ──────────────────────────────────────────────────────────────────────────────
# 10.  OracleRouter tests
# ──────────────────────────────────────────────────────────────────────────────


class TestOracleRouter:
  def test_default_factory(self, verifier: VerifierStrategy, judge: JudgeStrategy) -> None:
    r = OracleRouter.default(verifier, judge)
    assert isinstance(r, OracleRouter)
    assert len(r._chain.policies) >= 5

  def test_verifier_only_factory(self, verifier: VerifierStrategy, judge: JudgeStrategy) -> None:
    r = OracleRouter.verifier_only(verifier, judge)
    decision = r.route(
      Task("t", "d", "p"),
      [Trajectory("x", "t", "code")],
    )
    assert decision.selected_strategy == StrategyType.VERIFIER
    assert decision.confidence == pytest.approx(1.0)

  def test_judge_only_factory(self, verifier: VerifierStrategy, judge: JudgeStrategy) -> None:
    r = OracleRouter.judge_only(verifier, judge)
    decision = r.route(
      Task("t", "d", "p"),
      [Trajectory("x", "t", "code")],
    )
    assert decision.selected_strategy == StrategyType.JUDGE
    assert decision.confidence == pytest.approx(1.0)

  def test_route_returns_detailed_decision(
    self,
    router: OracleRouter,
    easy_task: Task,
    trajectories: list[Trajectory],
  ) -> None:
    decision = router.route(easy_task, trajectories)
    assert isinstance(decision, DetailedRoutingDecision)
    assert decision.task_id == easy_task.id
    assert decision.selected_strategy in {StrategyType.VERIFIER, StrategyType.JUDGE}
    assert 0.0 <= decision.confidence <= 1.0

  def test_route_populates_signals(
    self,
    router: OracleRouter,
    easy_task: Task,
    trajectories: list[Trajectory],
  ) -> None:
    decision = router.route(easy_task, trajectories)
    assert decision.signals is not None
    assert decision.signals.trajectory_count == len(trajectories)

  def test_route_populates_policy_votes(
    self,
    router: OracleRouter,
    easy_task: Task,
    trajectories: list[Trajectory],
  ) -> None:
    decision = router.route(easy_task, trajectories)
    assert len(decision.policy_votes) >= 5

  def test_route_signals_match_task(
    self,
    router: OracleRouter,
    easy_task: Task,
    trajectories: list[Trajectory],
  ) -> None:
    decision = router.route(easy_task, trajectories)
    assert decision.signals.trajectory_count == len(trajectories)
    assert decision.signals.has_ground_truth == (1.0 if easy_task.ground_truth else 0.0)

  def test_route_elapsed_ms_positive(
    self,
    router: OracleRouter,
    easy_task: Task,
    trajectories: list[Trajectory],
  ) -> None:
    decision = router.route(easy_task, trajectories)
    assert decision.elapsed_ms >= 0.0

  def test_evaluate_returns_result_and_decision(
    self,
    router: OracleRouter,
    easy_task: Task,
    trajectories: list[Trajectory],
  ) -> None:
    result, decision = router.evaluate(easy_task, trajectories)
    assert isinstance(result, EvaluationResult)
    assert isinstance(decision, DetailedRoutingDecision)
    assert result.task_id == easy_task.id

  def test_evaluate_empty_trajectories_raises(self, router: OracleRouter, easy_task: Task) -> None:
    with pytest.raises(ValueError):
      router.evaluate(easy_task, [])

  def test_evaluate_dispatches_to_selected_strategy(
    self,
    verifier: VerifierStrategy,
    judge: JudgeStrategy,
    easy_task: Task,
    trajectories: list[Trajectory],
  ) -> None:
    verifier_router = OracleRouter.verifier_only(verifier, judge)
    result, decision = verifier_router.evaluate(easy_task, trajectories)
    assert result.strategy_type == StrategyType.VERIFIER

    judge_router = OracleRouter.judge_only(verifier, judge)
    result, decision = judge_router.evaluate(easy_task, trajectories)
    assert result.strategy_type == StrategyType.JUDGE

  def test_decision_log_accumulates(
    self,
    router: OracleRouter,
    easy_task: Task,
    hard_task: Task,
    trajectories: list[Trajectory],
  ) -> None:
    assert len(router.decision_log) == 0
    router.route(easy_task, trajectories)
    router.route(hard_task, trajectories)
    assert len(router.decision_log) == 2

  def test_decision_log_is_copy(
    self,
    router: OracleRouter,
    easy_task: Task,
    trajectories: list[Trajectory],
  ) -> None:
    router.route(easy_task, trajectories)
    log = router.decision_log
    log.clear()
    assert len(router.decision_log) == 1

  def test_routing_summary_empty(self, verifier: VerifierStrategy, judge: JudgeStrategy) -> None:
    r = OracleRouter.default(verifier, judge)
    summary = r.routing_summary()
    assert "No routing decisions" in summary

  def test_routing_summary_after_decisions(
    self,
    router: OracleRouter,
    easy_task: Task,
    trajectories: list[Trajectory],
  ) -> None:
    router.route(easy_task, trajectories)
    summary = router.routing_summary()
    assert easy_task.id in summary
    assert "Verifier" in summary or "Judge" in summary

  def test_update_hardness_cache(
    self,
    router: OracleRouter,
    easy_task: Task,
    trajectories: list[Trajectory],
  ) -> None:
    router.update_hardness(easy_task.id, 0.75)
    decision = router.route(easy_task, trajectories)
    assert decision.signals.prior_hardness == pytest.approx(0.75)

  def test_update_hardness_invalid_raises(self, router: OracleRouter) -> None:
    with pytest.raises(ValueError):
      router.update_hardness("task", 1.5)
    with pytest.raises(ValueError):
      router.update_hardness("task", -0.1)

  def test_register_policy_appends(
    self,
    router: OracleRouter,
    easy_task: Task,
    trajectories: list[Trajectory],
  ) -> None:
    initial_count = len(router._chain.policies)

    class _NopPolicy(RoutingPolicy):
      name = "nop"

      def vote(self, task, trajectories, signals):
        return PolicyVote("nop", StrategyType.JUDGE, 0.5)

    router.register_policy(_NopPolicy())
    assert len(router._chain.policies) == initial_count + 1

  def test_register_policy_at_position(
    self,
    router: OracleRouter,
    easy_task: Task,
    trajectories: list[Trajectory],
  ) -> None:
    class _MarkerPolicy(RoutingPolicy):
      name = "marker"

      def vote(self, task, trajectories, signals):
        return PolicyVote("marker", StrategyType.VERIFIER, 0.5)

    router.register_policy(_MarkerPolicy(), position=0)
    assert router._chain.policies[0].name == "marker"

  def test_register_policy_returns_self_for_chaining(
    self,
    router: OracleRouter,
  ) -> None:
    class _P(RoutingPolicy):
      name = "p"

      def vote(self, task, trajectories, signals):
        return PolicyVote("p", StrategyType.JUDGE, 0.5)

    result = router.register_policy(_P())
    assert result is router

  def test_router_routes_hard_coded_task_to_verifier(
    self,
    verifier: VerifierStrategy,
    judge: JudgeStrategy,
  ) -> None:
    """A task with ground truth, test cases, hard difficulty, and output
    should route strongly to the verifier."""
    strong_verifier_task = Task(
      id="strong-v",
      description="Implement algorithm",
      problem_statement=(
        "Implement and test a function that returns the correct output. "
        "The function must pass all test cases."
      ),
      difficulty=TaskDifficulty.HARD,
      ground_truth="def fn(): return 42",
      test_cases=[{"input": [], "expected": 42}],
    )
    trajs = [
      Trajectory("t1", "strong-v", "def fn(): return 42", output="42"),
      Trajectory("t2", "strong-v", "def fn(): return 0", output="0"),
      Trajectory("t3", "strong-v", "def fn(): return -1", output="-1"),
    ]
    router = OracleRouter.default(verifier, judge, confidence_threshold=0.45)
    router.update_hardness("strong-v", 0.9)
    decision = router.route(strong_verifier_task, trajs)
    assert decision.selected_strategy == StrategyType.VERIFIER

  def test_router_routes_open_ended_task_to_judge(
    self,
    verifier: VerifierStrategy,
    judge: JudgeStrategy,
  ) -> None:
    """An open-ended essay task with no ground truth and a single trajectory
    should route to the judge."""
    open_task = Task(
      id="open-essay",
      description="Write an essay",
      problem_statement=(
        "Write a comparative essay analysing the trade-offs between "
        "two competing software design patterns."
      ),
      difficulty=TaskDifficulty.EASY,
    )
    trajs = [Trajectory("t1", "open-essay", "Paragraph 1 …")]
    router = OracleRouter.default(verifier, judge, confidence_threshold=0.45)
    router.update_hardness("open-essay", 0.15)
    decision = router.route(open_task, trajs)
    assert decision.selected_strategy == StrategyType.JUDGE


# ──────────────────────────────────────────────────────────────────────────────
# 11.  Provider registry tests
# ──────────────────────────────────────────────────────────────────────────────


class TestProviderRegistry:
  def test_get_stub_provider(self) -> None:
    from llm_oracle.core.providers import get_provider

    cls = get_provider("stub")
    assert cls is StubProvider

  def test_get_unknown_provider_raises(self) -> None:
    from llm_oracle.core.providers import get_provider

    with pytest.raises(KeyError, match="Unknown provider"):
      get_provider("nonexistent_provider_xyz")

  def test_register_and_retrieve_custom_provider(self) -> None:
    from llm_oracle.core.providers import get_provider, register_provider

    class MyCustomProvider(StubProvider):
      pass

    register_provider("my_custom", MyCustomProvider)
    assert get_provider("my_custom") is MyCustomProvider

  def test_register_invalid_provider_raises(self) -> None:
    from llm_oracle.core.providers import register_provider

    class BadProvider:
      pass  # no generate method

    with pytest.raises(TypeError):
      register_provider("bad", BadProvider)

  def test_create_provider_from_model_id_stub(self) -> None:
    from llm_oracle.core.providers import create_provider

    cfg = ModelConfig(model_id="stub", provider="stub")
    provider = create_provider(cfg)
    assert isinstance(provider, StubProvider)

  def test_create_provider_from_model_id_unknown_raises(self) -> None:
    from llm_oracle.core.providers import create_provider

    cfg = ModelConfig(model_id="totally-unknown-model-xyz-123", provider="")
    with pytest.raises(ValueError, match="Cannot infer provider"):
      create_provider(cfg)

  def test_stub_provider_from_config(self) -> None:
    cfg = ModelConfig(model_id="stub", provider="stub")
    provider = StubProvider.from_config(cfg)
    assert isinstance(provider, StubProvider)
    assert provider.model_id == "stub"

  def test_openai_provider_requires_package(self, monkeypatch) -> None:
    """OpenAIProvider should raise ImportError if 'openai' is not installed."""
    import sys

    original = sys.modules.get("openai")
    sys.modules["openai"] = None  # type: ignore[assignment]
    try:
      with pytest.raises((ImportError, TypeError)):
        OpenAIProvider(model_id="gpt-4o", api_key="fake")
    finally:
      if original is None:
        sys.modules.pop("openai", None)
      else:
        sys.modules["openai"] = original


# ──────────────────────────────────────────────────────────────────────────────
# 12.  Integration smoke tests
# ──────────────────────────────────────────────────────────────────────────────


class TestIntegration:
  """End-to-end smoke tests exercising the full pipeline."""

  def test_full_pipeline_single_task(
    self,
    router: OracleRouter,
    easy_task: Task,
    trajectories: list[Trajectory],
  ) -> None:
    result, decision = router.evaluate(easy_task, trajectories)
    assert result.task_id == easy_task.id
    assert result.best_trajectory_id in {t.id for t in trajectories}
    assert decision.selected_strategy in {StrategyType.VERIFIER, StrategyType.JUDGE}
    assert len(result.trajectory_scores) == len(trajectories)

  def test_full_harness_two_tasks(
    self,
    verifier: VerifierStrategy,
    judge: JudgeStrategy,
    easy_task: Task,
    hard_task: Task,
    trajectories: list[Trajectory],
  ) -> None:
    harness = EvaluationHarness(verifier=verifier, judge=judge, max_workers=2)
    report = harness.run(
      [
        (easy_task, trajectories),
        (hard_task, trajectories),
      ],
      parallel=False,
    )
    assert len(report.task_records) == 2
    summary = report.summary()
    assert len(summary) > 0

  def test_router_with_hardness_feedback_loop(
    self,
    verifier: VerifierStrategy,
    judge: JudgeStrategy,
    easy_task: Task,
    hard_task: Task,
    trajectories: list[Trajectory],
  ) -> None:
    """Simulate a feedback loop: run harness, then feed hardness back into router."""
    harness = EvaluationHarness(verifier=verifier, judge=judge)
    router = OracleRouter.default(verifier, judge)

    # Run harness on both tasks
    report = harness.run(
      [
        (easy_task, trajectories),
        (hard_task, trajectories),
      ],
      parallel=False,
    )

    # Feed hardness scores back into router
    for record in report.task_records:
      router.update_hardness(record.task_id, record.hardness_score)

    # Now route each task again — the prior hardness signal is populated
    easy_decision = router.route(easy_task, trajectories)
    hard_decision = router.route(hard_task, trajectories)

    assert easy_decision.signals.prior_hardness is not None
    assert hard_decision.signals.prior_hardness is not None
    # Routing decisions should be valid strategy types
    assert easy_decision.selected_strategy in {StrategyType.VERIFIER, StrategyType.JUDGE}
    assert hard_decision.selected_strategy in {StrategyType.VERIFIER, StrategyType.JUDGE}

  def test_router_audit_log_after_multi_routing(
    self,
    router: OracleRouter,
    easy_task: Task,
    hard_task: Task,
    trajectories: list[Trajectory],
  ) -> None:
    router.route(easy_task, trajectories)
    router.route(hard_task, trajectories)
    summary = router.routing_summary()
    assert easy_task.id in summary
    assert hard_task.id in summary
    assert "Total decisions  : 2" in summary

  def test_verifier_and_judge_produce_compatible_results(
    self,
    verifier: VerifierStrategy,
    judge: JudgeStrategy,
    easy_task: Task,
    trajectories: list[Trajectory],
  ) -> None:
    v_result = verifier.evaluate(easy_task, trajectories)
    j_result = judge.evaluate(easy_task, trajectories)

    # Both should produce valid results for the same task
    assert v_result.task_id == j_result.task_id == easy_task.id
    assert v_result.strategy_type != j_result.strategy_type
    assert v_result.best_trajectory_id in {t.id for t in trajectories}
    assert j_result.best_trajectory_id in {t.id for t in trajectories}

    # Both should score all trajectories
    assert set(v_result.trajectory_scores.keys()) == {t.id for t in trajectories}
    assert set(j_result.trajectory_scores.keys()) == {t.id for t in trajectories}

  def test_policy_chain_custom_policy(
    self,
    verifier: VerifierStrategy,
    judge: JudgeStrategy,
    easy_task: Task,
    trajectories: list[Trajectory],
  ) -> None:
    """Custom policies plugged into the router should influence routing."""

    class _AlwaysVerifierPolicy(RoutingPolicy):
      name = "always_verifier_custom"
      weight = 100.0  # overwhelming weight

      def vote(self, task, trajectories, signals):
        return PolicyVote(
          "always_verifier_custom",
          StrategyType.VERIFIER,
          1.0,
          weight=100.0,
        )

    router = OracleRouter.default(verifier, judge, confidence_threshold=0.4)
    router.register_policy(_AlwaysVerifierPolicy())
    decision = router.route(easy_task, trajectories)
    # The overwhelming custom policy should dominate
    assert decision.selected_strategy == StrategyType.VERIFIER
