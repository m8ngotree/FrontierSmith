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
import json
import os
import sys
import threading
import time
import types
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, List, Optional, TypeVar

from .config import PipelineConfig, REPO_ROOT

_LLM_INTERFACE_PATH = REPO_ROOT / "Frontier-CS" / "src" / "frontier_cs" / "gen" / "llm_interface.py"


def _module_available(name: str) -> bool:
    """Whether ``name`` is importable, treating a missing PARENT package as
    'not available'. ``importlib.util.find_spec`` raises ModuleNotFoundError
    (rather than returning None) when an intermediate parent package is absent,
    so we guard against that here."""
    if name in sys.modules:
        return True
    try:
        return importlib.util.find_spec(name) is not None
    except ModuleNotFoundError:
        return False


def _install_optional_stubs() -> None:
    """Provide stub modules for optional SDKs so importing llm_interface.py
    never fails on dependencies we do not actually use (Gemini), or that may be
    installed later (Anthropic)."""
    # google.generativeai (Gemini) -- never used by this pipeline.
    if not _module_available("google.generativeai"):
        google_pkg = sys.modules.get("google") or types.ModuleType("google")
        genai = types.ModuleType("google.generativeai")
        genai.configure = lambda *a, **k: None  # type: ignore[attr-defined]
        genai.GenerativeModel = object  # type: ignore[attr-defined]
        google_pkg.generativeai = genai  # type: ignore[attr-defined]
        sys.modules.setdefault("google", google_pkg)
        sys.modules["google.generativeai"] = genai

    # python-dotenv -- only a convenience (load_dotenv); keys are read straight
    # from the environment, so a no-op stub is sufficient when it is absent.
    if not _module_available("dotenv"):
        dotenv = types.ModuleType("dotenv")
        dotenv.load_dotenv = lambda *a, **k: False  # type: ignore[attr-defined]
        dotenv.find_dotenv = lambda *a, **k: ""  # type: ignore[attr-defined]
        sys.modules["dotenv"] = dotenv

    # anthropic -- stub ONLY if absent; a real call would then error clearly.
    if not _module_available("anthropic"):
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

# ── API call log ──────────────────────────────────────────────────────────────
# One JSON-Lines file per process, written to synthesis/logs/.
# Each line is a self-contained JSON object with the full prompt and response.
_LOG_DIR = REPO_ROOT / "synthesis" / "logs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = _LOG_DIR / f"api_calls_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
_log_lock = threading.Lock()
_call_counter = 0


def _log_call(record: dict) -> None:
    """Append one JSON record to the log file (thread-safe)."""
    global _call_counter
    with _log_lock:
        _call_counter += 1
        record["call_index"] = _call_counter
        with open(_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


print(f"[llm] API call log → {_LOG_FILE}")


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
    """Uniform wrapper exposing ``generate(prompt) -> str`` over one client.

    ``max_concurrent`` controls how many threads may be inside ``generate``
    simultaneously for this client.  Set to 1 to make all calls sequential
    (useful for Anthropic when on a low rate-limit tier).
    """

    def __init__(
        self,
        model: str,
        *,
        is_reasoning: bool,
        timeout: float,
        max_concurrent: int = 8,
    ) -> None:
        self.model = model
        self.provider = _detect_provider(model)
        self.client = _instantiate(model, is_reasoning=is_reasoning, timeout=timeout)
        self.raw = getattr(self.client, "client", None)        # underlying SDK client
        self.model_id = getattr(self.client, "model", model)
        self._sem = threading.Semaphore(max_concurrent)

    def generate(self, prompt: str) -> str:
        """Send one user prompt; return text (empty string on any API error).

        Logs the request, waits for the semaphore (enforces max_concurrent),
        makes the call, then logs the outcome with latency and token estimates.
        """
        prompt_chars = len(prompt)
        prompt_tok_est = prompt_chars // 4
        ts = time.strftime("%H:%M:%S")
        print(f"[llm {ts}] --> {self.model_id}  prompt ~{prompt_tok_est} tok ({prompt_chars} chars)")

        t0 = time.monotonic()
        with self._sem:
            text, _meta = self.client.call_llm(prompt)
        elapsed = time.monotonic() - t0

        result = text or ""
        ts2 = time.strftime("%H:%M:%S")
        ok = bool(result)
        if ok:
            resp_chars = len(result)
            resp_tok_est = resp_chars // 4
            print(
                f"[llm {ts2}] <-- {self.model_id}  response ~{resp_tok_est} tok "
                f"({resp_chars} chars)  {elapsed:.1f}s"
            )
        else:
            print(f"[llm {ts2}] <-- {self.model_id}  EMPTY/ERROR  {elapsed:.1f}s")

        _log_call({
            "timestamp": ts,
            "model": self.model_id,
            "provider": self.provider,
            "prompt": prompt,
            "response": result,
            "prompt_chars": prompt_chars,
            "response_chars": len(result),
            "latency_s": round(elapsed, 2),
            "ok": ok,
        })

        return result

    def preflight(self, timeout: float = 30.0) -> "tuple[bool, str]":
        """Make one tiny call directly against the SDK to confirm the model is
        reachable. Unlike ``generate``, this does NOT swallow errors: it returns
        ``(False, "<ErrorType>: <message>")`` so a bad model / network stall /
        auth failure is reported instead of silently hanging the pipeline.
        """
        try:
            if self.provider in {"openai", "xai", "deepseek"}:
                kwargs = {
                    "model": self.model_id,
                    "messages": [{"role": "user", "content": "ping"}],
                    "timeout": timeout,
                }
                effort = getattr(self.client, "reasoning_effort", None)
                if effort:
                    kwargs["reasoning_effort"] = effort
                self.raw.chat.completions.create(**kwargs)
                return True, "ok"
            if self.provider == "anthropic":
                # Plain (no extended thinking) so a tiny max_tokens is valid.
                self.raw.messages.create(
                    model=self.model_id,
                    max_tokens=16,
                    messages=[{"role": "user", "content": "ping"}],
                    timeout=timeout,
                )
                return True, "ok"
            # Other providers: fall back to the wrapped call (best-effort).
            return (True, "ok") if self.client.call_llm("ping")[0] else (False, "empty response")
        except Exception as e:  # noqa: BLE001 - we want the real error string
            return False, f"{type(e).__name__}: {e}"


class Clients:
    """Bundle of logical clients, named by role from the paper."""

    def __init__(self, mutation: LLMClient, judge: LLMClient, solver: LLMClient) -> None:
        self.mutation = mutation   # mutation + coarse filter (OpenAI reasoning)
        self.judge = judge         # LLM-as-a-judge for divergence (OpenAI reasoning)
        self.solver = solver       # solutions + test-case + verifier agents (Anthropic)


def build_clients(config: PipelineConfig) -> Clients:
    """Instantiate the three logical clients used across the pipeline."""
    mutation = LLMClient(
        config.mutation_model,
        is_reasoning=True,
        timeout=config.llm_timeout,
        max_concurrent=config.max_workers,   # OpenAI: parallel is fine
    )
    # Factory defaults reasoning models to "high"; the paper uses "medium".
    if hasattr(mutation.client, "reasoning_effort"):
        mutation.client.reasoning_effort = config.mutation_reasoning_effort

    # Anthropic solver: always sequential (max_concurrent=1) so we never burst
    # past the per-minute output-token rate limit on low-tier API keys.
    solver = LLMClient(
        config.solver_model,
        is_reasoning=True,
        timeout=config.llm_timeout,
        max_concurrent=1,
    )
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
