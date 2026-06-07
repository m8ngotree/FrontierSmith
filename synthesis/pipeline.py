"""Top-level orchestrator - FrontierSmith Algorithm 1.

    S <- S u D                                  (seed pool init, line 1)
    for iteration in 1..num_iterations:
        B  <- sample B problems from S          (line 2)
        C  <- Mutate(B)                         (line 3)
        C  <- CoarseFilter(C)                   (line 4)
        C  <- IdeaDivergenceFilter(C)  -> top N_div   (lines 5-9)
        C  <- BuildAndValidate(C)               (lines 10-13)
        P  <- topNfinal(C, exec divergence)     (line 14)
        S  <- S u P                             (line 15)
    return all validated P

Each stage is checkpointed to ``synthesis/checkpoints/iterN_stageM.json`` so a
run can resume after a crash. Use ``--fresh`` to ignore existing checkpoints.
"""

from __future__ import annotations

import argparse
import random
from typing import Callable, List, Optional

from .config import PipelineConfig
from .judge_client import JudgeClient
from .llm import Clients, build_clients
from .stage0_seed import load_seed_pool
from .stage1_mutate import mutate
from .stage2_coarse_filter import coarse_filter
from .stage3_idea_divergence import idea_divergence_filter
from .stage4_build_env import build_environment
from .stage5_output import select_and_write
from .types import Candidate, SeedProblem
from .utils import load_checkpoint, save_checkpoint


def _stage_with_checkpoint(
    config: PipelineConfig,
    name: str,
    fresh: bool,
    produce: Callable[[], List[Candidate]],
) -> List[Candidate]:
    """Run a candidate-producing stage, caching its output to a checkpoint."""
    if not fresh:
        cached = load_checkpoint(config, name)
        if cached is not None:
            print(f"[pipeline] resuming '{name}' from checkpoint ({len(cached)} candidates).")
            return [Candidate.from_dict(d) for d in cached]
    result = produce()
    save_checkpoint(config, name, [c.to_dict() for c in result])
    return result


def preflight_clients(clients: Clients, config: PipelineConfig) -> bool:
    """Verify each model is reachable before launching concurrent calls.

    Prints a clear PASS/FAIL with the real error for each client. Returns True
    only if the mutation client (needed first) is reachable.
    """
    print("[pipeline] preflight: probing models (one tiny call each)...")
    checks = [("mutation", clients.mutation), ("solver", clients.solver)]
    ok_mutation = True
    for name, client in checks:
        ok, msg = client.preflight(timeout=config.preflight_timeout)
        status = "OK" if ok else "FAILED"
        print(f"[pipeline]   {name:8s} {client.model}: {status} - {msg}")
        if name == "mutation" and not ok:
            ok_mutation = False
    if not ok_mutation:
        print(
            "[pipeline] Mutation model is unreachable. Common causes: model id does "
            "not exist (NotFoundError), network/proxy blocked (APIConnectionError/"
            "APITimeoutError), or bad key (AuthenticationError). Fix and re-run."
        )
    return ok_mutation


def _sample_batch(pool: List[SeedProblem], b: int, rng: random.Random) -> List[SeedProblem]:
    """Sample B problems from the pool (without replacement when possible)."""
    if not pool:
        return []
    if len(pool) <= b:
        return list(pool)
    return rng.sample(pool, b)


def run(config: PipelineConfig, *, fresh: bool = False, skip_preflight: bool = False) -> List[Candidate]:
    """Execute the full multi-iteration synthesis pipeline."""
    config.ensure_dirs()
    rng = random.Random(config.random_seed)

    clients: Clients = build_clients(config)

    judge = JudgeClient(config)
    if not judge.health():
        print(
            f"[pipeline] WARNING: judge at {config.judge_url} is not reachable. "
            f"Stage 4 will fail until it is running."
        )

    seed_pool: List[SeedProblem] = load_seed_pool(config)
    all_validated: List[Candidate] = []
    next_problem_index = 1

    for it in range(config.num_iterations):
        print(f"\n===== Iteration {it + 1}/{config.num_iterations} =====")
        batch = _sample_batch(seed_pool, config.B, rng)
        if not batch:
            print("[pipeline] seed pool is empty; stopping.")
            break

        candidates = _stage_with_checkpoint(
            config, f"iter{it}_stage1_mutate", fresh,
            lambda: mutate(batch, clients, config),
        )
        candidates = _stage_with_checkpoint(
            config, f"iter{it}_stage2_filter", fresh,
            lambda: coarse_filter(candidates, clients, config),
        )
        candidates = _stage_with_checkpoint(
            config, f"iter{it}_stage3_divergence", fresh,
            lambda: idea_divergence_filter(candidates, clients, config),
        )
        validated = _stage_with_checkpoint(
            config, f"iter{it}_stage4_buildenv", fresh,
            lambda: build_environment(candidates, clients, judge, config),
        )

        selected, new_seeds, next_problem_index = select_and_write(
            validated, config, start_index=next_problem_index
        )
        seed_pool.extend(new_seeds)          # line 15: expand the pool
        all_validated.extend(selected)
        print(f"[pipeline] iteration {it + 1} produced {len(selected)} validated problems "
              f"(total {len(all_validated)}).")

    print(f"\n[pipeline] DONE. {len(all_validated)} validated open-ended problems written to "
          f"{config.output_dir}.")
    return all_validated


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FrontierSmith synthesis pipeline (Algorithm 1).")
    p.add_argument("--preset", choices=["smoke", "paper"], default="smoke",
                   help="smoke = tiny dev run; paper = full paper-scale settings.")
    p.add_argument("--fresh", action="store_true", help="Ignore existing stage checkpoints.")
    p.add_argument("--judge-url", default=None, help="Override judge URL.")
    p.add_argument("--iterations", type=int, default=None, help="Override num_iterations.")
    p.add_argument("--batch-size", type=int, default=None, help="Override B (seeds per iteration).")
    p.add_argument("--mutation-model", default=None, help="Override mutation/judge model id.")
    p.add_argument("--solver-model", default=None, help="Override solver model id.")
    p.add_argument("--seed-list", default=None, help="Override path to the seed-list JSON manifest.")
    p.add_argument("--preflight-only", action="store_true",
                   help="Only probe the models and exit (no pipeline run).")
    p.add_argument("--no-preflight", action="store_true", help="Skip the startup model probe.")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)
    overrides = {}
    if args.judge_url is not None:
        overrides["judge_url"] = args.judge_url
    if args.iterations is not None:
        overrides["num_iterations"] = args.iterations
    if args.batch_size is not None:
        overrides["B"] = args.batch_size
    if args.mutation_model is not None:
        overrides["mutation_model"] = args.mutation_model
    if args.solver_model is not None:
        overrides["solver_model"] = args.solver_model
    if args.seed_list is not None:
        overrides["seed_list_path"] = args.seed_list

    config = (
        PipelineConfig.paper(**overrides)
        if args.preset == "paper"
        else PipelineConfig.smoke(**overrides)
    )

    if args.preflight_only:
        clients = build_clients(config)
        preflight_clients(clients, config)
        return

    run(config, fresh=args.fresh, skip_preflight=args.no_preflight)


if __name__ == "__main__":
    main()
