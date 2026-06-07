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
import re
import sys
import threading
import time
import types
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple, TypeVar

from openai import APITimeoutError, OpenAI

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
# One JSON-Lines file per run. Each line is a self-contained JSON object with the
# full prompt + response, tagged with the pipeline stage and client role so a run
# can be reconstructed/filtered after the fact (e.g. `grep '"stage":"stage4'`).
_LOG_DIR = REPO_ROOT / "synthesis" / "logs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = _LOG_DIR / f"api_calls_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
_log_lock = threading.Lock()
_call_counter = 0

# Stages run sequentially, so a plain module global is a safe "current stage"
# label even for calls issued from worker threads inside a stage.
_CURRENT_STAGE = "init"


def set_log_file(path) -> None:
    """Redirect the API-call JSONL to ``path`` (called once at run start)."""
    global _LOG_FILE
    _LOG_FILE = Path(path)
    _LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    print(f"[llm] API call log → {_LOG_FILE}")


def set_stage(name: str) -> None:
    """Tag subsequent API-call log records with the active pipeline stage."""
    global _CURRENT_STAGE
    _CURRENT_STAGE = name


# Running totals across the whole process (tokens + estimated USD), reported at
# the end of a run. Guarded by ``_log_lock``.
_USAGE_TOTALS = {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0, "calls": 0}

# Approximate prices in USD per 1M tokens, as (input, output). Keys are matched
# as substrings of the model id (longest match wins). DeepSeek values are the
# cache-miss prices from the provider's pricing page; others are best-effort.
# A model with no match logs token counts but ``cost_usd: null`` (unknown).
_PRICING: dict[str, "tuple[float, float]"] = {
    # DeepSeek (cache-miss input prices from the provider pricing page).
    "deepseek-v4-pro": (0.435, 0.87),
    "deepseek-v4-flash": (0.14, 0.28),
    "deepseek-reasoner": (0.14, 0.28),
    "deepseek-chat": (0.14, 0.28),
    # OpenAI.
    "gpt-5.4": (2.5, 15.0),
    "o4-mini": (1.1, 4.4),
    "o3-mini": (1.1, 4.4),
    "o3": (2.0, 8.0),
    "gpt-4o-mini": (0.15, 0.6),
    "gpt-4o": (2.5, 10.0),
    # Anthropic (base input / output; longest-match wins). Current Opus 4.5-4.8
    # are $5/$25; the dated/4.1 Opus snapshots are the deprecated $15/$75 tier.
    "claude-opus": (5.0, 25.0),
    "claude-opus-4-1": (15.0, 75.0),
    "claude-opus-4-2025": (15.0, 75.0),
    "claude-sonnet": (3.0, 15.0),
    "claude-haiku": (1.0, 5.0),
    "claude-haiku-3": (0.8, 4.0),
}

# Field names used by the various SDK usage objects, mapped to in/out buckets.
_INPUT_TOKEN_FIELDS = ("prompt_tokens", "input_tokens")
_OUTPUT_TOKEN_FIELDS = ("completion_tokens", "output_tokens")


def _parse_usage(meta: str) -> dict:
    """Extract every ``*_tokens=<int>`` field from the stringified SDK response.

    Works across providers because OpenAI/DeepSeek emit
    ``usage=CompletionUsage(prompt_tokens=.., completion_tokens=.., total_tokens=..)``
    and Anthropic emits ``usage=Usage(input_tokens=.., output_tokens=..)`` -- both
    as stable Pydantic reprs inside ``str(completion)``.
    """
    usage: dict = {}
    if not isinstance(meta, str):
        return usage
    for field, value in re.findall(r"(\w*tokens)=(\d+)", meta):
        usage[field] = int(value)
    return usage


def _price_for(model_id: str) -> "Optional[tuple[float, float]]":
    """Longest-substring match into ``_PRICING`` (None if the model is unknown)."""
    m = model_id.lower()
    best: "Optional[tuple[float, float]]" = None
    best_len = -1
    for key, price in _PRICING.items():
        if key in m and len(key) > best_len:
            best, best_len = price, len(key)
    return best


def _estimate_cost(model_id: str, usage: dict) -> "Optional[float]":
    """USD cost from token usage and the price table (None if model unknown)."""
    price = _price_for(model_id)
    if price is None or not usage:
        return None
    in_price, out_price = price
    in_tok = next((usage[f] for f in _INPUT_TOKEN_FIELDS if f in usage), 0)
    out_tok = next((usage[f] for f in _OUTPUT_TOKEN_FIELDS if f in usage), 0)
    return round(in_tok / 1e6 * in_price + out_tok / 1e6 * out_price, 6)


def get_usage_totals() -> dict:
    """Snapshot of cumulative token/cost totals for this process."""
    with _log_lock:
        return dict(_USAGE_TOTALS)


def _log_call(record: dict) -> None:
    """Append one JSON record to the log file (thread-safe) and update totals."""
    global _call_counter
    with _log_lock:
        _call_counter += 1
        record["call_index"] = _call_counter
        record.setdefault("stage", _CURRENT_STAGE)
        usage = record.get("usage") or {}
        _USAGE_TOTALS["calls"] += 1
        _USAGE_TOTALS["input_tokens"] += next(
            (usage[f] for f in _INPUT_TOKEN_FIELDS if f in usage), 0
        )
        _USAGE_TOTALS["output_tokens"] += next(
            (usage[f] for f in _OUTPUT_TOKEN_FIELDS if f in usage), 0
        )
        if record.get("cost_usd"):
            _USAGE_TOTALS["cost_usd"] = round(
                _USAGE_TOTALS["cost_usd"] + record["cost_usd"], 6
            )
        with open(_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


print(f"[llm] API call log → {_LOG_FILE}")


def _map_deepseek_effort(effort: Optional[str]) -> str:
    """Map effort to DeepSeek's supported values (low/medium -> high, xhigh -> max)."""
    if not effort:
        return "high"
    e = effort.lower()
    if e in {"max", "xhigh"}:
        return "max"
    if e in {"low", "medium", "high"}:
        return "high"
    return effort


class _SynthesisDeepSeek:
    """DeepSeek client using the OpenAI-compatible endpoint (thinking disabled).

    Sends ``reasoning_effort`` but does NOT enable thinking mode, so the full
    ``max_tokens`` budget is available for response content rather than being
    consumed by chain-of-thought reasoning tokens.
    """

    def __init__(
        self,
        model: str,
        *,
        timeout: float,
        reasoning_effort: str = "high",
        max_tokens: int = 32000,
        base_url: str = "https://api.deepseek.com",
        api_key: Optional[str] = None,
    ) -> None:
        resolved_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.client = OpenAI(api_key=resolved_key, base_url=base_url, timeout=timeout)
        self.name = "deepseek"
        self.model = model
        self.max_tokens = max_tokens
        self.reasoning_effort = _map_deepseek_effort(reasoning_effort)

    def call_llm(self, user_prompt: str) -> Tuple[str, Any]:
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": user_prompt}],
                max_tokens=self.max_tokens,
                # Do NOT send reasoning_effort — on thinking-capable DeepSeek models
                # (e.g. v4-pro) that parameter alone triggers chain-of-thought mode,
                # consuming the entire token budget on reasoning and leaving nothing
                # for the actual response. Explicitly disable thinking instead.
                extra_body={"thinking": {"type": "disabled"}},
            )
            msg = completion.choices[0].message
            return msg.content or "", str(completion)
        except APITimeoutError as e:
            return "", str(e)
        except Exception as e:  # noqa: BLE001
            return "", str(e)


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


def _timeout_for(model: str, config: PipelineConfig) -> float:
    """Provider-specific request timeout."""
    if _detect_provider(model) == "deepseek":
        return config.deepseek_llm_timeout
    return config.llm_timeout


def _instantiate(model: str, *, is_reasoning: bool, timeout: float, reasoning_effort: Optional[str] = None):
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
        effort = reasoning_effort or ("high" if is_reasoning else "high")
        return _SynthesisDeepSeek(
            actual,
            timeout=timeout,
            reasoning_effort=effort,
        )
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
        reasoning_effort: Optional[str] = None,
    ) -> None:
        self.model = model
        self.role = "llm"                                      # set by build_clients
        self.provider = _detect_provider(model)
        self.client = _instantiate(
            model,
            is_reasoning=is_reasoning,
            timeout=timeout,
            reasoning_effort=reasoning_effort,
        )
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
            text, meta = self.client.call_llm(prompt)
        elapsed = time.monotonic() - t0

        result = text or ""
        ok = bool(result)
        usage = _parse_usage(meta if isinstance(meta, str) else str(meta))
        cost = _estimate_cost(self.model_id, usage)

        in_tok = next((usage[f] for f in _INPUT_TOKEN_FIELDS if f in usage), None)
        out_tok = next((usage[f] for f in _OUTPUT_TOKEN_FIELDS if f in usage), None)
        ts2 = time.strftime("%H:%M:%S")
        if ok:
            tok_str = (
                f"in={in_tok} out={out_tok} tok"
                if in_tok is not None
                else f"~{len(result) // 4} tok"
            )
            cost_str = f" ${cost:.4f}" if cost is not None else ""
            print(
                f"[llm {ts2}] <-- {self.model_id}  {tok_str}  {elapsed:.1f}s{cost_str}"
            )
        else:
            print(f"[llm {ts2}] <-- {self.model_id}  EMPTY/ERROR  {elapsed:.1f}s")

        _log_call({
            "timestamp": ts,
            "role": self.role,
            "model": self.model_id,
            "provider": self.provider,
            "prompt": prompt,
            "response": result,
            "prompt_chars": prompt_chars,
            "response_chars": len(result),
            "usage": usage,
            "cost_usd": cost,
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
                    "max_tokens": 16,
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
    mutation_timeout = _timeout_for(config.mutation_model, config)
    solver_timeout = _timeout_for(config.solver_model, config)
    solver_is_deepseek = _detect_provider(config.solver_model) == "deepseek"
    # Anthropic keys often hit per-minute output-token limits; DeepSeek allows
    # much higher concurrency (500 for v4-pro), so parallelize solver there.
    solver_concurrent = config.max_workers if solver_is_deepseek else 1

    mutation = LLMClient(
        config.mutation_model,
        is_reasoning=True,
        timeout=mutation_timeout,
        max_concurrent=config.max_workers,
        reasoning_effort=config.mutation_reasoning_effort,
    )
    # OpenAI GPT: factory defaults to "high"; override with config.
    if hasattr(mutation.client, "reasoning_effort") and _detect_provider(config.mutation_model) != "deepseek":
        mutation.client.reasoning_effort = config.mutation_reasoning_effort

    solver = LLMClient(
        config.solver_model,
        is_reasoning=True,
        timeout=solver_timeout,
        max_concurrent=solver_concurrent,
        reasoning_effort=config.mutation_reasoning_effort if solver_is_deepseek else None,
    )
    if hasattr(solver.client, "thinking_budget"):
        solver.client.thinking_budget = config.solver_thinking_budget or None

    mutation.role = "mutation"   # also used for coarse filter + pairwise judge
    solver.role = "solver"       # also used for test-case + verifier agents
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
