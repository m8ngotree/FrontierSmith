"""Stage 2 - Coarse LLM-as-a-judge filter (Algorithm 1, line 4; Section 3.2).

Removes candidates that are not genuinely open-ended. A candidate is kept only
if all three conditions pass: optimization objective with no known optimum,
multiple distinct strategies plausible, and a meaningful continuous scoring
function. The verdict (per-condition reasons + overall) is recorded on the
candidate.
"""

from __future__ import annotations

from typing import List, Optional

from .config import PipelineConfig
from .llm import Clients, map_concurrent
from .prompts import build_coarse_filter_prompt
from .types import Candidate
from .utils import parse_json_response


def _filter_one(candidate: Candidate, clients: Clients) -> Candidate:
    prompt = build_coarse_filter_prompt(candidate.mutated_statement)
    response = clients.mutation.generate(prompt)
    parsed = parse_json_response(response)

    if not parsed:
        # Treat an unparseable verdict as a rejection (conservative).
        candidate.coarse_filter_passed = False
        candidate.coarse_filter_reasons = {"error": "unparseable judge response"}
        return candidate

    candidate.coarse_filter_reasons = parsed
    candidate.coarse_filter_passed = bool(parsed.get("overall_pass", False))
    return candidate


def coarse_filter(
    candidates: List[Candidate], clients: Clients, config: PipelineConfig
) -> List[Candidate]:
    """Run the coarse filter and return only passing candidates."""
    judged = map_concurrent(
        lambda c: _filter_one(c, clients),
        candidates,
        max_workers=config.max_workers,
    )
    kept = [c for c in judged if c.coarse_filter_passed]
    print(f"[stage2] Coarse filter kept {len(kept)}/{len(candidates)} candidates.")
    return kept
