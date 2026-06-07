"""Stage 5 - Final selection and output (line 14; format = FrontierCS).

Re-rank validated candidates by execution-grounded divergence, keep the top
N_final, and write each as a complete FrontierCS problem package under the
synthesis output directory (NOT over the shipped frontiersmith_* references).
Selected problems are also returned as SeedProblems so the orchestrator can
expand the pool for the next iteration (line 15).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple

from .config import PipelineConfig
from .types import Candidate, SeedProblem
from .utils import write_problem_directory


def _problem_dir_name(index: int) -> str:
    return f"frontiersmith_synth_{index}"


def select_and_write(
    candidates: List[Candidate], config: PipelineConfig, *, start_index: int = 1
) -> Tuple[List[Candidate], List[SeedProblem], int]:
    """Select the top N_final validated candidates and write their packages.

    Returns (selected_candidates, seed_problems_for_pool, next_index).
    """
    validated = [c for c in candidates if c.validated and c.exec_divergence_score is not None]
    ranked = sorted(validated, key=lambda c: c.exec_divergence_score or 0.0, reverse=True)
    selected = ranked[: config.N_final]

    os.makedirs(config.output_dir, exist_ok=True)
    seeds: List[SeedProblem] = []
    index = start_index
    for cand in selected:
        dir_name = _problem_dir_name(index)
        out_dir = str(Path(config.output_dir) / dir_name)
        write_problem_directory(
            out_dir,
            statement=cand.mutated_statement,
            checker_code=cand.checker_code or "",
            generator_code=cand.generator_code or "",
            test_inputs=cand.test_inputs,
        )
        seeds.append(
            SeedProblem(
                problem_id=dir_name,
                tier="synthesized",
                folder_name=dir_name,
                statement=cand.mutated_statement,
                origin="synthesized",
            )
        )
        index += 1

    print(
        f"[stage5] Wrote {len(selected)}/{len(validated)} validated problems to "
        f"{config.output_dir} (N_final={config.N_final})."
    )
    return selected, seeds, index
