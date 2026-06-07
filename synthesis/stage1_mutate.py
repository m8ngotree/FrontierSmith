"""Stage 1 - Mutation (Algorithm 1, line 3; paper Section 3.1).

Each seed problem is mutated into an open-ended candidate by the OpenAI
reasoning model. The model returns a JSON object containing the extracted
formulation, the mutations applied, and a full problem statement in the
FrontierCS markdown format. Failures (API errors, malformed JSON) are skipped.
"""

from __future__ import annotations

import re
import uuid
from typing import List, Optional

from .config import PipelineConfig
from .llm import Clients, map_concurrent
from .prompts import build_mutation_prompt
from .types import Candidate, SeedProblem
from .utils import parse_json_response


def _unwrap_markdown(statement: str) -> str:
    """Strip a surrounding ```markdown ... ``` fence if the model included one."""
    s = statement.strip()
    fence = re.match(r"^```(?:markdown)?\s*\n(.*)```\s*$", s, re.DOTALL)
    if fence:
        return fence.group(1).strip()
    return s


def _mutate_one(seed: SeedProblem, clients: Clients) -> Optional[Candidate]:
    prompt = build_mutation_prompt(seed.statement)
    response = clients.mutation.generate(prompt)
    if not response:
        return None

    parsed = parse_json_response(response)
    if not parsed:
        return None

    statement = _unwrap_markdown(str(parsed.get("problem_statement", "")))
    if not statement:
        return None

    return Candidate(
        candidate_id=f"{seed.problem_id}__{uuid.uuid4().hex[:8]}",
        seed_problem_id=seed.problem_id,
        mutated_statement=statement,
        mutation_types=list(parsed.get("mutations_applied", [])),
        original_formulation=dict(parsed.get("original_formulation", {})),
        mutated_formulation=dict(parsed.get("mutated_formulation", {})),
    )


def mutate(seeds: List[SeedProblem], clients: Clients, config: PipelineConfig) -> List[Candidate]:
    """Mutate a batch of seed problems into open-ended candidates (parallel)."""
    results = map_concurrent(
        lambda s: _mutate_one(s, clients),
        seeds,
        max_workers=config.max_workers,
    )
    candidates = [c for c in results if c is not None]
    print(f"[stage1] Mutated {len(candidates)}/{len(seeds)} seeds into candidates.")
    return candidates
