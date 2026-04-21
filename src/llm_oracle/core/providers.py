"""Concrete LanguageModel implementations for OpenAI, Anthropic, Gemini, and a test stub.

All providers normalize vendor responses into ``(text, tokens, position_logprobs)``
and use lazy SDK imports so missing optional dependencies don't break module import.
"""

from __future__ import annotations

import math
import os
import random
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field  # noqa: F401
from typing import Any

from llm_oracle.core.models import ModelConfig

# Type alias matching BaseStrategy.LanguageModel protocol
PositionLogprobs = list[list[tuple[str, float]]]  # [(token_str, log_prob)]


# ──────────────────────────────────────────────────────────────────────────────
# Base provider
# ──────────────────────────────────────────────────────────────────────────────


class BaseProvider(ABC):
  """Abstract base for all provider implementations.

  Subclasses must implement ``generate`` with the same signature as the
  ``LanguageModel`` protocol — providers are structural subtypes.
  """

  def __init__(self, model_id: str, **kwargs: Any) -> None:
    self.model_id = model_id
    self._extra: dict[str, Any] = kwargs

  # ── Core API ─────────────────────────────────────────────────────────────────

  @abstractmethod
  def generate(
    self,
    prompt: str,
    temperature: float = 1.0,
    max_tokens: int = 4096,
    return_logprobs: bool = False,
    **kwargs: Any,
  ) -> tuple[str, list[str] | None, PositionLogprobs | None]:
    """Generate a completion for *prompt*.

    Returns:
      ``(text, tokens, position_logprobs)`` — logprob fields are ``None``
      when ``return_logprobs=False`` or the provider doesn't support them.
    """

  # ── Factory helper ───────────────────────────────────────────────────────────

  @classmethod
  def from_config(cls, config: ModelConfig) -> BaseProvider:
    """Instantiate a provider from a ModelConfig."""
    kwargs: dict[str, Any] = dict(config.additional_params)
    if config.api_key:
      kwargs["api_key"] = config.api_key
    return cls(model_id=config.model_id, **kwargs)  # type: ignore[call-arg]

  def __repr__(self) -> str:
    return f"{self.__class__.__name__}(model_id={self.model_id!r})"


# ──────────────────────────────────────────────────────────────────────────────
# OpenAI provider
# ──────────────────────────────────────────────────────────────────────────────


class OpenAIProvider(BaseProvider):
  """OpenAI chat completion provider (openai >= 1.0).

  Supports gpt-* and o* model families, top_logprobs extraction, and
  Azure OpenAI via ``base_url`` + ``api_version`` kwargs.
  """

  def __init__(
    self,
    model_id: str = "gpt-4o",
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    api_version: str | None = None,
    top_logprobs: int = 20,
    **kwargs: Any,
  ) -> None:
    super().__init__(model_id, **kwargs)
    try:
      import openai  # type: ignore[import-untyped]  # noqa: PLC0415
    except ImportError as exc:
      raise ImportError(
        "openai package is required for OpenAIProvider. Install it with: pip install openai"
      ) from exc

    client_kwargs: dict[str, Any] = {
      "api_key": api_key or os.getenv("OPENAI_API_KEY"),
    }
    if base_url:
      client_kwargs["base_url"] = base_url
    if api_version:
      client_kwargs["default_headers"] = {"api-version": api_version}

    self._client = openai.OpenAI(**client_kwargs)
    self._top_logprobs = min(max(1, top_logprobs), 20)

  def generate(
    self,
    prompt: str,
    temperature: float = 1.0,
    max_tokens: int = 4096,
    return_logprobs: bool = False,
    **kwargs: Any,
  ) -> tuple[str, list[str] | None, PositionLogprobs | None]:
    """Call the OpenAI chat completions API."""
    request: dict[str, Any] = {
      "model": self.model_id,
      "messages": [{"role": "user", "content": prompt}],
      "temperature": temperature,
      "max_tokens": max_tokens,
      **kwargs,
    }

    if return_logprobs:
      request["logprobs"] = True
      request["top_logprobs"] = self._top_logprobs

    response = self._client.chat.completions.create(**request)
    choice = response.choices[0]
    text: str = choice.message.content or ""

    tokens: list[str] | None = None
    position_logprobs: PositionLogprobs | None = None

    if return_logprobs and choice.logprobs and choice.logprobs.content:
      tokens = []
      position_logprobs = []
      for token_info in choice.logprobs.content:
        tokens.append(token_info.token)
        top_lps: list[tuple[str, float]] = [
          (tl.token, tl.logprob) for tl in (token_info.top_logprobs or [])
        ]
        position_logprobs.append(top_lps)

    return text, tokens, position_logprobs


# ──────────────────────────────────────────────────────────────────────────────
# Anthropic provider
# ──────────────────────────────────────────────────────────────────────────────


class AnthropicProvider(BaseProvider):
  """Anthropic Messages API provider (anthropic >= 0.20).

  Anthropic does not expose token-level log probabilities, so
  ``position_logprobs`` is always ``None``; the verifier falls back to
  text-based score extraction.
  """

  def __init__(
    self,
    model_id: str = "claude-opus-4-5",
    *,
    api_key: str | None = None,
    system: str | None = None,
    thinking: bool = False,
    budget_tokens: int = 8000,
    **kwargs: Any,
  ) -> None:
    super().__init__(model_id, **kwargs)
    try:
      import anthropic  # type: ignore[import-untyped]  # noqa: PLC0415
    except ImportError as exc:
      raise ImportError(
        "anthropic package is required for AnthropicProvider. "
        "Install it with: pip install anthropic"
      ) from exc

    self._client = anthropic.Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))
    self._system = system
    self._thinking = thinking
    self._budget_tokens = budget_tokens

  def generate(
    self,
    prompt: str,
    temperature: float = 1.0,
    max_tokens: int = 4096,
    return_logprobs: bool = False,
    **kwargs: Any,
  ) -> tuple[str, list[str] | None, PositionLogprobs | None]:
    """Call the Anthropic Messages API; always returns ``(text, None, None)``."""
    request: dict[str, Any] = {
      "model": self.model_id,
      "max_tokens": max_tokens,
      "messages": [{"role": "user", "content": prompt}],
      **kwargs,
    }

    # Extended thinking uses a fixed temperature of 1; otherwise pass through.
    if self._thinking:
      request["thinking"] = {
        "type": "enabled",
        "budget_tokens": self._budget_tokens,
      }
      request["temperature"] = 1.0  # Required for extended thinking
    else:
      request["temperature"] = temperature

    if self._system:
      request["system"] = self._system

    response = self._client.messages.create(**request)

    # Collect only text blocks (skip thinking blocks)
    text_parts: list[str] = []
    for block in response.content:
      if getattr(block, "type", None) == "text":
        text_parts.append(block.text)

    return "".join(text_parts), None, None


# ──────────────────────────────────────────────────────────────────────────────
# Gemini provider
# ──────────────────────────────────────────────────────────────────────────────


class GeminiProvider(BaseProvider):
  """Google Gemini provider via the google-genai SDK (>= 0.8).

  Uses Vertex AI when ``vertexai=True`` or ``VERTEX_API_KEY`` is set;
  otherwise uses ``GEMINI_API_KEY``. Supports logprob extraction.
  """

  def __init__(
    self,
    model_id: str = "gemini-2.5-flash",
    *,
    api_key: str | None = None,
    vertexai: bool = False,
    top_logprobs: int = 20,
    thinking_budget: int = 0,
    **kwargs: Any,
  ) -> None:
    super().__init__(model_id, **kwargs)
    try:
      from google import genai  # type: ignore[import-untyped]  # noqa: PLC0415
    except ImportError as exc:
      raise ImportError(
        "google-genai package is required for GeminiProvider. "
        "Install it with: pip install google-genai"
      ) from exc

    self._genai = genai

    vertex_key = api_key or os.getenv("VERTEX_API_KEY")
    gemini_key = api_key or os.getenv("GEMINI_API_KEY")

    if vertexai or vertex_key:
      self._client = genai.Client(vertexai=True, api_key=vertex_key)
    elif gemini_key:
      self._client = genai.Client(api_key=gemini_key)
    else:
      raise OSError(
        "No Gemini API key found. Set GEMINI_API_KEY or VERTEX_API_KEY, "
        "or pass api_key= to GeminiProvider."
      )

    self._top_logprobs = min(max(1, top_logprobs), 20)
    self._thinking_budget = thinking_budget

  def generate(
    self,
    prompt: str,
    temperature: float = 1.0,
    max_tokens: int = 4096,
    return_logprobs: bool = False,
    **kwargs: Any,
  ) -> tuple[str, list[str] | None, PositionLogprobs | None]:
    """Call the Gemini generate content API."""
    from google.genai.types import (  # type: ignore[import-untyped]  # noqa: PLC0415
      Content,
      GenerateContentConfig,
      Part,
      ThinkingConfig,
    )

    config_kwargs: dict[str, Any] = {
      "max_output_tokens": max_tokens,
      "temperature": temperature,
      **kwargs,
    }

    if return_logprobs:
      config_kwargs["response_logprobs"] = True
      config_kwargs["logprobs"] = self._top_logprobs

    config_kwargs["thinking_config"] = ThinkingConfig(thinking_budget=self._thinking_budget)

    config = GenerateContentConfig(**config_kwargs)
    response = self._client.models.generate_content(
      model=self.model_id,
      contents=[Content(role="user", parts=[Part(text=prompt)])],
      config=config,
    )

    text: str = response.text or ""
    tokens: list[str] | None = None
    position_logprobs: PositionLogprobs | None = None

    if return_logprobs:
      candidate = response.candidates[0] if response.candidates else None
      if candidate and candidate.logprobs_result:
        lr = candidate.logprobs_result
        if lr.top_candidates:
          position_logprobs = []
          for pos in lr.top_candidates:
            alts: list[tuple[str, float]] = [
              (lp.token, lp.log_probability) for lp in pos.candidates
            ]
            position_logprobs.append(alts)
        if lr.chosen_candidates:
          tokens = [c.token for c in lr.chosen_candidates]

    return text, tokens, position_logprobs


# ──────────────────────────────────────────────────────────────────────────────
# Stub provider (testing / offline use)
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class StubResponse:
  """Pre-programmed response for the stub provider.

  Mode is detected from the prompt: ``<score_A>``/``<score_B>`` tags → pairwise
  (uses ``score_a``/``score_b``); otherwise pointwise (uses ``score``).
  All three fields can be set so one instance works in either mode.
  Unset fields fall back to provider-level defaults.
  """

  text: str = ""
  score: str | None = None
  score_a: str | None = None
  score_b: str | None = None


class StubProvider(BaseProvider):
  """In-process stub for unit tests and offline use; no network calls."""

  def __init__(
    self,
    model_id: str = "stub",
    *,
    default_score: str = "E",
    default_score_a: str = "E",
    default_score_b: str = "G",
    responses: list[StubResponse] | None = None,
    seed: int = 42,
  ) -> None:
    super().__init__(model_id)
    self._default_score = default_score
    self._default_score_a = default_score_a
    self._default_score_b = default_score_b
    self._responses: list[StubResponse] = responses or []
    self._call_count: int = 0
    self._rng = random.Random(seed)

  def generate(
    self,
    prompt: str,
    temperature: float = 1.0,
    max_tokens: int = 4096,
    return_logprobs: bool = False,
    **kwargs: Any,
  ) -> tuple[str, list[str] | None, PositionLogprobs | None]:
    """Return a pre-programmed or synthesized response."""
    # Pick response template
    if self._responses:
      stub = self._responses[self._call_count % len(self._responses)]
    else:
      stub = StubResponse()

    self._call_count += 1

    # Determine scores based on prompt type
    is_pairwise = "<score_A>" in prompt or "<score_B>" in prompt
    score_a = stub.score_a or self._default_score_a
    score_b = stub.score_b or self._default_score_b
    score = stub.score or self._default_score

    # Build response text with embedded score tags
    if is_pairwise:
      text = stub.text or (
        f"Analysis:\n"
        f"Trajectory A demonstrates solid progress toward the goal.\n"
        f"Trajectory B shows a slightly different but effective approach.\n\n"
        f"<score_A>{score_a}</score_A>\n"
        f"<score_B>{score_b}</score_B>\n"
        f"<verdict>{'A' if score_a <= score_b else 'B'}</verdict>"
      )
    else:
      text = stub.text or (
        f"Analysis:\nThe trajectory shows reasonable progress.\n\n<score>{score}</score>"
      )

    if not return_logprobs:
      return text, None, None

    # Synthesise log-probability distributions
    tokens, position_logprobs = self._synthesise_logprobs(
      text, score, score_a, score_b, is_pairwise
    )
    return text, tokens, position_logprobs

  # ── Stub helpers ─────────────────────────────────────────────────────────────

  def _synthesise_logprobs(
    self,
    text: str,
    score: str,
    score_a: str,
    score_b: str,
    is_pairwise: bool,
  ) -> tuple[list[str], PositionLogprobs]:
    """Generate synthetic token + logprob arrays (~70% mass on the target score token)."""
    # Naïve character-level tokenisation — good enough for the stub.
    raw_tokens = list(text)
    tokens: list[str] = raw_tokens
    position_logprobs: PositionLogprobs = []

    alphabet = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
    tag_positions = self._find_tag_token_positions(text, is_pairwise)

    for i, tok in enumerate(tokens):
      if i in tag_positions:
        # At a scoring position: emit a peaked distribution.
        target_letter = tag_positions[i]
        probs = self._peaked_distribution(target_letter, alphabet, peak=0.70)
        lps: list[tuple[str, float]] = [
          (letter, math.log(p + 1e-12)) for letter, p in probs.items()
        ]
        # Sort by descending probability
        lps.sort(key=lambda x: -x[1])
        position_logprobs.append(lps[:20])
      else:
        # Non-scoring position: flat distribution over ASCII printable chars.
        base_lp = math.log(1.0 / max(len(alphabet), 1))
        position_logprobs.append([(tok, base_lp)])

    return tokens, position_logprobs

  def _find_tag_token_positions(
    self,
    text: str,
    is_pairwise: bool,
  ) -> dict[int, str]:
    """Return ``{char_index: target_letter}`` for each score tag position."""
    positions: dict[int, str] = {}

    tags = (
      [("<score_A>", self._default_score_a), ("<score_B>", self._default_score_b)]
      if is_pairwise
      else [("<score>", self._default_score)]
    )

    for tag, target in tags:
      for match in re.finditer(re.escape(tag), text):
        positions[match.end()] = target

    return positions

  def _peaked_distribution(
    self,
    target: str,
    alphabet: list[str],
    peak: float = 0.70,
  ) -> dict[str, float]:
    """Return a ``{token: probability}`` dict summing to 1, peaked at *target*."""
    remaining = 1.0 - peak
    others = [a for a in alphabet if a != target]
    if not others:
      return {target: 1.0}

    # Spread remaining mass uniformly among a random subset of neighbours.
    n_neighbours = min(self._rng.randint(3, 8), len(others))
    neighbours = self._rng.sample(others, n_neighbours)
    per_neighbour = remaining / n_neighbours

    dist: dict[str, float] = {tok: 0.0 for tok in alphabet}
    dist[target] = peak
    for nb in neighbours:
      dist[nb] = per_neighbour

    return dist


# ──────────────────────────────────────────────────────────────────────────────
# Provider registry
# ──────────────────────────────────────────────────────────────────────────────

_PROVIDER_REGISTRY: dict[str, type] = {
  "openai": OpenAIProvider,
  "anthropic": AnthropicProvider,
  "gemini": GeminiProvider,
  "google": GeminiProvider,
  "stub": StubProvider,
}


def get_provider(name: str) -> type:
  """Look up a provider class by registry name (case-insensitive).

  Raises:
    KeyError: If *name* is not registered.
  """
  key = name.strip().lower()
  if key not in _PROVIDER_REGISTRY:
    available = ", ".join(sorted(_PROVIDER_REGISTRY))
    raise KeyError(f"Unknown provider {name!r}. Available providers: {available}")
  return _PROVIDER_REGISTRY[key]


def register_provider(name: str, provider_class: type) -> None:
  """Register a custom provider class under *name* (case-insensitive).

  Raises:
    TypeError: If *provider_class* does not implement ``generate``.
  """
  if not callable(getattr(provider_class, "generate", None)):
    raise TypeError(f"{provider_class!r} must implement a callable 'generate' method.")
  _PROVIDER_REGISTRY[name.strip().lower()] = provider_class


def create_provider(config: ModelConfig) -> BaseProvider:
  """Instantiate the correct provider from a ModelConfig.

  Resolution order: explicit ``config.provider`` → model_id prefix
  (``gpt-``/``o1``/``o3`` → OpenAI, ``claude`` → Anthropic, ``gemini`` →
  Gemini, ``stub`` → Stub) → raises ValueError.
  """
  # Explicit provider name takes priority.
  if config.provider:
    provider_cls = get_provider(config.provider)
    return provider_cls.from_config(config)

  # Infer from model_id prefix.
  model_lower = (config.model_id or "").lower()
  if model_lower.startswith("gpt") or model_lower.startswith("o1") or model_lower.startswith("o3"):
    return OpenAIProvider.from_config(config)
  if model_lower.startswith("claude"):
    return AnthropicProvider.from_config(config)
  if model_lower.startswith("gemini"):
    return GeminiProvider.from_config(config)
  if model_lower == "stub" or not model_lower:
    return StubProvider(model_id=config.model_id or "stub")

  raise ValueError(
    f"Cannot infer provider from model_id {config.model_id!r}. "
    "Set config.provider explicitly or register a custom provider."
  )
