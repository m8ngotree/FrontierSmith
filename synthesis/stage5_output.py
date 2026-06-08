"""Stage 5 - Final selection and output (line 14; format = FrontierCS).

Re-rank validated candidates by execution-grounded divergence, keep the top
N_final, and write each as a complete FrontierCS problem package under the
synthesis output directory (NOT over the shipped frontiersmith_* references).
Selected problems are also returned as SeedProblems so the orchestrator can
expand the pool for the next iteration (line 15).

Each validated problem also gets a self-contained artifact bundle under
``config.artifacts_dir/<dir_name>/`` containing:
  problem/               - full FrontierCS package (statement, chk.cc, gen.cpp, testdata)
  candidate.json         - all pipeline metadata (mutation, scores, build log, etc.)
  solutions/             - the N sampled C++ solutions as individual .cpp files
  judge_feedback.json    - per-solution, per-test judge verdicts (status, msg, time, score)
  original_statement.txt - the seed problem statement this was mutated from
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import List, Tuple

from .config import PipelineConfig
from .types import Candidate, SeedProblem
from .utils import write_problem_directory


def _problem_dir_name(index: int) -> str:
    return f"frontiersmith_synth_{index}"


def _write_artifact(
    cand: Candidate,
    problem_out_dir: str,
    artifact_dir: str,
    problems_root: str,
) -> None:
    """Write a self-contained artifact bundle for one validated problem."""
    art = Path(artifact_dir)
    art.mkdir(parents=True, exist_ok=True)

    # 1) Copy the problem package.
    problem_src = Path(problem_out_dir)
    problem_dst = art / "problem"
    if problem_dst.exists():
        shutil.rmtree(problem_dst)
    shutil.copytree(problem_src, problem_dst)

    # 2) candidate.json — all pipeline metadata.
    meta = asdict(cand)
    (art / "candidate.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # 3) solutions/ — each sampled C++ solution as its own file.
    sol_dir = art / "solutions"
    sol_dir.mkdir(exist_ok=True)
    for i, code in enumerate(cand.solutions or []):
        (sol_dir / f"sol_{i}.cpp").write_text(code, encoding="utf-8")

    # 4) judge_feedback.json — per-solution, per-test judge verdicts captured
    #    during stage4 cross-validation (no re-submission needed).
    if cand.case_details:
        feedback: list = []
        for si, (score_row, detail_row) in enumerate(
            zip(cand.score_matrix, cand.case_details)
        ):
            sol_entry = {"solution_index": si, "tests": []}
            for ti, (score, detail) in enumerate(zip(score_row, detail_row)):
                sol_entry["tests"].append({
                    "test_index": ti + 1,
                    "score": score,
                    "status": detail.get("status", "?"),
                    "msg": detail.get("msg", ""),
                    "time_s": detail.get("time_s", 0.0),
                })
            feedback.append(sol_entry)
        (art / "judge_feedback.json").write_text(
            json.dumps(feedback, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    # 5) original_statement.txt — the seed problem this was mutated from.
    seed_stmt_path = Path(problems_root) / cand.seed_problem_id / "statement.txt"
    if seed_stmt_path.exists():
        shutil.copy2(seed_stmt_path, art / "original_statement.txt")
    else:
        # If the seed was itself synthesized (origin="synthesized"), there is no
        # original statement on disk; skip silently.
        pass

    print(f"[stage5]   artifact → {artifact_dir}")


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

        # Write the FrontierCS problem package.
        out_dir = str(Path(config.output_dir) / dir_name)
        write_problem_directory(
            out_dir,
            statement=cand.mutated_statement,
            checker_code=cand.checker_code or "",
            generator_code=cand.generator_code or "",
            test_inputs=cand.test_inputs,
        )

        # Write the artifact bundle.
        artifact_dir = str(Path(config.artifacts_dir) / dir_name)
        _write_artifact(cand, out_dir, artifact_dir, config.problems_root)

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
