"""Central configuration for the FrontierSmith synthesis pipeline.

All hyperparameters from the paper (Algorithm 1 + Section 4.1 "Problem
synthesis") live here in one ``PipelineConfig`` dataclass, alongside the model
identifiers and filesystem paths. Two presets are provided:

* ``PipelineConfig.paper()``  - the exact paper-scale settings (expensive).
* ``PipelineConfig.smoke()``  - a tiny configuration for development/testing.

The paper uses "GPT-5.4 Thinking" (mutation, coarse filter, LLM-divergence
judge) and "Claude Sonnet 4.6" (solver, test-case agent, verifier agent).
All model IDs are overridable via constructor kwargs or the CLI flags
(``--mutation-model`` / ``--solver-model``), e.g. to route everything through
a cheaper provider like DeepSeek for development runs.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict


# Repo root = two levels up from this file (synthesis/config.py -> synthesis -> repo).
REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class PipelineConfig:
    # --- Algorithm 1 budgets ---
    B: int = 1000                       # seed problems sampled per iteration (line 2)
    num_iterations: int = 4             # outer-loop iterations (Section 4.1)
    n_solutions: int = 10               # solutions sampled per candidate (line 6)
    N_div: int = 100                    # keep top-N_div after LLM divergence (line 9)
    N_final: int = 50                   # keep top-N_final after exec divergence (line 14)

    # --- Stage 3 (idea divergence) ---
    judge_batch_size: int = 5           # pairs scored per LLM-as-a-judge call (batched O(n^2))
    solver_temperature: float = 1.0     # sampling temperature for diverse solutions

    # --- Stage 4 (testing infrastructure) ---
    n_test_cases: int = 10              # test inputs generated per problem (matches shipped n_cases)
    cross_validation_max_rounds: int = 5  # max test/verifier revision rounds before discarding

    # --- Models (provider is auto-detected from the string by gen/llm.py) ---
    # Mutation / coarse filter / LLM-as-a-judge -> OpenAI reasoning model.
    # Paper's "GPT-5.4 Thinking". Override via --mutation-model.
    mutation_model: str = "gpt-5.4-2026-03-05"
    mutation_reasoning_effort: str = "medium"   # paper uses "medium thinking effort"
    # Solver / test-case agent / verifier agent -> Anthropic with extended thinking.
    solver_model: str = "claude-sonnet-4-6"
    solver_thinking_budget: int = 20000         # tokens of extended thinking (0/None disables)

    # --- Concurrency / timeouts ---
    max_workers: int = 8                # thread-pool width for parallel LLM calls
    llm_timeout: float = 120.0          # per-request timeout (seconds)
    deepseek_llm_timeout: float = 600.0 # DeepSeek thinking responses can be slow
    preflight_timeout: float = 30.0     # per-request timeout for the startup preflight check

    # --- Judge ---
    judge_url: str = "http://localhost:8082"
    judge_max_wait: float = 300.0       # max seconds to poll a submission
    judge_poll_interval: float = 2.0

    # --- Paths ---
    seed_list_path: str = str(REPO_ROOT / "data" / "sample_lists" / "hardtest_hard_sampled_200.json")
    problems_root: str = str(REPO_ROOT / "data" / "problems" / "hardtest")          # where seed statements live (download_hardtest.py writes here)
    output_dir: str = str(REPO_ROOT / "synthesis" / "output" / "problems")          # generated packages
    artifacts_dir: str = str(REPO_ROOT / "artifacts")                               # per-problem artifact bundles
    checkpoint_dir: str = str(REPO_ROOT / "synthesis" / "checkpoints")              # per-stage JSON

    # --- Misc ---
    random_seed: int = 42

    # ------------------------------------------------------------------ presets
    @classmethod
    def paper(cls, **overrides: Any) -> "PipelineConfig":
        """Exact paper-scale configuration. Very expensive to run end-to-end."""
        return cls(**overrides)

    @classmethod
    def smoke(cls, **overrides: Any) -> "PipelineConfig":
        """Tiny configuration for development and end-to-end smoke testing."""
        defaults: Dict[str, Any] = dict(
            B=5,
            num_iterations=1,
            n_solutions=4,
            N_div=3,
            N_final=2,
            judge_batch_size=3,
            n_test_cases=4,
            cross_validation_max_rounds=2,
            max_workers=4,
            seed_list_path=str(REPO_ROOT / "data" / "sample_lists" / "smoke_seeds.json"),
        )
        defaults.update(overrides)
        return cls(**defaults)

    # ------------------------------------------------------------------ helpers
    def ensure_dirs(self) -> None:
        """Create output/checkpoint directories if they do not yet exist."""
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
