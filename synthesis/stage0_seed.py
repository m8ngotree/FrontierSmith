"""Stage 0 - Seed pool (Algorithm 1, line 1).

Loads closed-ended seed problems from a sample-list JSON manifest (e.g.
``data/sample_lists/hardtest_hard_sampled_200.json``) and reads each problem's
``statement.txt`` from the local problems root. The pipeline initializes the
seed pool from this set; validated synthesized problems are appended on later
iterations (line 15).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List

from .config import PipelineConfig
from .types import SeedProblem


def _candidate_statement_paths(problems_root: str, tier: str, folder_name: str) -> List[str]:
    """Plausible locations of statement.txt, since HardTests packages can be
    laid out a few different ways depending on the download/install script."""
    root = Path(problems_root)
    return [
        str(root / f"hardtest_{tier}" / folder_name / "statement.txt"),
        str(root / tier / folder_name / "statement.txt"),
        str(root / folder_name / "statement.txt"),
        str(root / f"hardtest_{tier}" / folder_name / "problem.txt"),
    ]


def _read_statement(problems_root: str, tier: str, folder_name: str) -> str:
    for path in _candidate_statement_paths(problems_root, tier, folder_name):
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
    return ""


def load_seed_pool(config: PipelineConfig, *, limit: int | None = None) -> List[SeedProblem]:
    """Load seed problems listed in ``config.seed_list_path``.

    Problems whose ``statement.txt`` cannot be found locally are skipped (with a
    warning) -- run ``scripts/download_hardtest.py`` to populate the problems
    root. ``limit`` optionally caps how many entries are loaded (useful for
    smoke runs).
    """
    if not os.path.exists(config.seed_list_path):
        raise FileNotFoundError(
            f"Seed list not found: {config.seed_list_path}. "
            f"Point config.seed_list_path at a valid manifest."
        )

    with open(config.seed_list_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    entries = manifest.get("valid_problems", [])
    if limit is not None:
        entries = entries[:limit]

    pool: List[SeedProblem] = []
    missing = 0
    for entry in entries:
        problem_id = entry.get("problem_id", "")
        tier = entry.get("tier", "")
        folder_name = entry.get("folder_name", problem_id)
        statement = _read_statement(config.problems_root, tier, folder_name)
        if not statement:
            missing += 1
            continue
        pool.append(
            SeedProblem(
                problem_id=problem_id,
                tier=tier,
                folder_name=folder_name,
                statement=statement,
                origin="seed",
            )
        )

    if missing:
        print(
            f"[stage0] WARNING: {missing}/{len(entries)} seed statements not found under "
            f"{config.problems_root}; they were skipped. Loaded {len(pool)} seed problems."
        )
    else:
        print(f"[stage0] Loaded {len(pool)} seed problems.")
    return pool
