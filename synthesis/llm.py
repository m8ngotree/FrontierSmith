"""Thin LLM adapter layer.

Reuses the repo's existing client wrappers in
``Frontier-CS/src/frontier_cs/gen/llm_interface.py`` (``GPT`` for OpenAI,
``ClaudeBase``/``Claude*`` for Anthropic) -- the valuable part with timeout and
error handling -- rather than re-implementing them.

We load ``llm_interface.py`` directly by file path so we avoid the heavy
``frontier_cs`` package ``__init__`` chain, and we install lightweight stubs for
optional SDKs that may be absent (``google.generativeai`` is never used here;
``anthropic`` is stubbed only if missing so OpenAI-only runs still import). A
compact provider router then mirrors ``gen/llm.py:instantiate_llm_client``.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, List, Optional, TypeVar

from .config import PipelineConfig, REPO_ROOT

_LLM_INTERFACE_PATH = REPO_ROOT / "Frontier-CS" / "src" / "frontier_cs" / "gen" / "llm_interface.py"


def _install_optional_stubs() -> None:
    """Provide stub modules for optional SDKs so importing llm_interface.py
    never fails on dependencies we do not actually use (Gemini), or that may be
    installed later (Anthropic)."""
    # google.generativeai (Gemini) -- never used by this pipeline.
    if importlib.util.find_spec("google.generativeai") is None and "google.generativeai" not in sys.modules:
        google_pkg = sys.modules.get("google") or types.ModuleType("google")
        genai = types.ModuleType("google.generativeai")
        genai.configure = lambda *a, **k: None  # type: ignore[attr-defined]
        genai.GenerativeModel = object  # type: ignore[attr-defined]
        google_pkg.generativeai = genai  # type: ignore[attr-defined]
        sys.modules.setdefault("google", google_pkg)
        sys.modules["google.generativeai"] = genai

    # anthropic -- stub ONLY if absent; a real call would then error clearly.
    if importlib.util.find_spec("anthropic") is None and "anthropic" not in sys.modules:
        anthropic = types.ModuleType("anthropic")

        class _MissingAnthropic:
            def __init__(self, *a, **k):
                raise ImportError(
                    "The 'anthropic' package is required for the solver/test/verifier "
                    "stages. Install it (pip install anthropic) and re-run."
                )

        class APITimeoutError(Exception):
            pass

        anthropic.Anthropic = _MissingAnthropic  # type: ignore[attr-defined]
        anthropic.APITimeoutError = APITimeoutError  # type: ignore[attr-defined]
        sys.modules["anthropic"] = anthropic


def _load_llm_interface():
    _install_optional_stubs()
    spec = importlib.util.spec_from_file_location("_fs_llm_interface", _LLM_INTERFACE_PATH)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


_iface = _load_llm_interface()


def _detect_provider(model: str) -> str:
    """Compact provider detection (mirrors gen/llm.py:detect_provider)."""
    m = model.strip()
    hint, actual = (m.split("/", 1) if "/" in m else ("", m))
    hint, actual_lower = hint.lower(), actual.strip().lower()
    if hint in {"", "openai", "azure", "azure_openai"} and actual_lower.startswith("gpt"):
        return "openai"
    if hint in {"gemini", "google"} or "gemini" in actual_lower:
        return "google"
    if hint == "anthropic" or "claude" in actual_lower:
        return "anthropic"
    if hint == "xai" or "grok" in actual_lower:
        return "xai"
    if hint == "deepseek" or "deepseek" in actual_lower:
        return "deepseek"
    return hint or "openai"


def _instantiate(model: str, *, is_reasoning: bool, timeout: float):
    """Create an underlying LLMInterface client for the given model string,
    reusing the wrapper classes from llm_interface.py."""
    _, actual = (model.split("/", 1) if "/" in model else ("", model))
    provider = _detect_provider(model)

    if provider == "openai":
        effort = "high" if is_reasoning else None
        return _iface.GPT(model=actual, reasoning_effort=effort, timeout=timeout)
    if provider == "anthropic":
        al = actual.lower()
        if "claude-sonnet-4-5" in al:
            return _iface.Claude_Sonnet_4_5(model=actual)
        if "claude-opus" in al:
            return _iface.Claude_Opus(model=actual)
        return _iface.Claude(model=actual)
    if provider == "xai":
        effort = "high" if is_reasoning else None
        return _iface.Grok(model=actual, reasoning_effort=effort, timeout=timeout)
    if provider == "deepseek":
        return _iface.DeepSeek(model=actual, timeout=timeout)
    if provider == "google":
        return _iface.Gemini(model=actual, timeout=timeout)
    # Fallback: treat as an OpenAI-compatible chat endpoint.
    effort = "high" if is_reasoning else None
    return _iface.GPT(model=actual, reasoning_effort=effort, timeout=timeout)


class LLMClient:
    """Uniform wrapper exposing ``generate(prompt) -> str`` over one client."""

    def __init__(self, model: str, *, is_reasoning: bool, timeout: float) -> None:
        self.model = model
        self.client = _instantiate(model, is_reasoning=is_reasoning, timeout=timeout)

    def generate(self, prompt: str) -> str:
        """Send one user prompt; return text (empty string on any API error)."""
        text, _meta = self.client.call_llm(prompt)
        return text or ""


class Clients:
    """Bundle of logical clients, named by role from the paper."""

    def __init__(self, mutation: LLMClient, judge: LLMClient, solver: LLMClient) -> None:
        self.mutation = mutation   # mutation + coarse filter (OpenAI reasoning)
        self.judge = judge         # LLM-as-a-judge for divergence (OpenAI reasoning)
        self.solver = solver       # solutions + test-case + verifier agents (Anthropic)


def build_clients(config: PipelineConfig) -> Clients:
    """Instantiate the three logical clients used across the pipeline."""
    mutation = LLMClient(config.mutation_model, is_reasoning=True, timeout=config.llm_timeout)
    # Factory defaults reasoning models to "high"; the paper uses "medium".
    if hasattr(mutation.client, "reasoning_effort"):
        mutation.client.reasoning_effort = config.mutation_reasoning_effort

    solver = LLMClient(config.solver_model, is_reasoning=True, timeout=config.llm_timeout)
    if hasattr(solver.client, "thinking_budget"):
        solver.client.thinking_budget = config.solver_thinking_budget or None

    # Coarse filter + pairwise judge share the OpenAI reasoning client (Sec 4.1).
    return Clients(mutation=mutation, judge=mutation, solver=solver)


T = TypeVar("T")
R = TypeVar("R")


def map_concurrent(fn: Callable[[T], R], items: List[T], *, max_workers: int) -> List[R]:
    """Apply ``fn`` to each item across a thread pool, preserving input order.

    Individual tasks should handle their own errors and return a sentinel
    (e.g. ``None``); results are returned in the same order as ``items``.
    """
    if not items:
        return []
    results: List[Optional[R]] = [None] * len(items)
    workers = max(1, min(max_workers, len(items)))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        future_to_idx = {ex.submit(fn, item): i for i, item in enumerate(items)}
        for fut in as_completed(future_to_idx):
            results[future_to_idx[fut]] = fut.result()
    return results  # type: ignore[return-value]
