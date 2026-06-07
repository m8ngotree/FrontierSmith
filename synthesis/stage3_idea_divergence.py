"""Stage 3 - Idea-divergence filter, LLM-based estimate (lines 5-9; Section 3.2).

For each surviving candidate we draw n independent solutions from the solver,
then an LLM-as-a-judge labels every solution pair as same- or different-strategy.
The divergence estimate is the fraction of pairs judged "different" (Eq. 2):

    d_hat(c) = (1 / C(n,2)) * sum_{i<j} 1[strategy(s_i) != strategy(s_j)]

We then keep the top N_div candidates by d_hat.

Implementation note: to avoid nesting thread pools, all solution-sampling calls
across all candidates are flattened into one pool, and likewise all pairwise
judging batches.
"""

from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Tuple

from .config import PipelineConfig
from .llm import Clients, map_concurrent
from .prompts import build_pairwise_judge_prompt, build_solution_prompt
from .types import Candidate, PairwiseJudgment
from .utils import extract_cpp, parse_json_response


def _sample_solutions(
    candidates: List[Candidate], clients: Clients, config: PipelineConfig
) -> None:
    """Populate ``candidate.solutions`` with extracted C++ for each candidate."""
    # One task per (candidate index, sample slot). Each returns (ci, code).
    tasks: List[int] = []
    for ci, _ in enumerate(candidates):
        tasks.extend([ci] * config.n_solutions)

    def sample(ci: int) -> Tuple[int, str]:
        prompt = build_solution_prompt(candidates[ci].mutated_statement)
        code = extract_cpp(clients.solver.generate(prompt))
        return ci, code

    results = map_concurrent(sample, tasks, max_workers=config.max_workers)

    by_candidate: Dict[int, List[str]] = {ci: [] for ci in range(len(candidates))}
    for ci, code in results:
        if code.strip():
            by_candidate[ci].append(code)
    for ci, cand in enumerate(candidates):
        cand.solutions = by_candidate[ci]


def _batch_pairs(pairs: List[Tuple[int, int]], batch_size: int) -> List[List[Tuple[int, int]]]:
    return [pairs[i : i + batch_size] for i in range(0, len(pairs), batch_size)]


def _judge_divergence(
    candidates: List[Candidate], clients: Clients, config: PipelineConfig
) -> None:
    """Run batched pairwise judging and set ``candidate.llm_divergence_score``."""
    # Build a flat list of judging tasks: (candidate index, batch of pairs).
    tasks: List[Tuple[int, List[Tuple[int, int]]]] = []
    for ci, cand in enumerate(candidates):
        n = len(cand.solutions)
        if n < 2:
            continue
        all_pairs = list(combinations(range(n), 2))
        for batch in _batch_pairs(all_pairs, config.judge_batch_size):
            tasks.append((ci, batch))

    def judge(task: Tuple[int, List[Tuple[int, int]]]) -> Tuple[int, List[PairwiseJudgment]]:
        ci, batch = task
        prompt = build_pairwise_judge_prompt(
            candidates[ci].mutated_statement, candidates[ci].solutions, batch
        )
        parsed = parse_json_response(clients.judge.generate(prompt))
        judgments: List[PairwiseJudgment] = []
        verdicts: Dict[str, dict] = {}
        if parsed:
            for item in parsed.get("judgments", []):
                verdicts[str(item.get("pair", ""))] = item
        for (i, j) in batch:
            item = verdicts.get(f"{i}-{j}", {})
            judgments.append(
                PairwiseJudgment(
                    solution_i=i,
                    solution_j=j,
                    # Missing/unparseable verdict defaults to "same" (0), the
                    # conservative choice that does not inflate divergence.
                    different=bool(item.get("different", False)),
                    reason=str(item.get("reason", "")),
                )
            )
        return ci, judgments

    results = map_concurrent(judge, tasks, max_workers=config.max_workers)

    per_candidate: Dict[int, List[PairwiseJudgment]] = {ci: [] for ci in range(len(candidates))}
    for ci, judgments in results:
        per_candidate[ci].extend(judgments)

    for ci, cand in enumerate(candidates):
        judgments = per_candidate[ci]
        cand.pairwise_judgments = judgments
        if judgments:
            n_different = sum(1 for j in judgments if j.different)
            cand.llm_divergence_score = n_different / len(judgments)
        else:
            cand.llm_divergence_score = 0.0


def idea_divergence_filter(
    candidates: List[Candidate], clients: Clients, config: PipelineConfig
) -> List[Candidate]:
    """Sample solutions, estimate LLM divergence, keep the top N_div candidates."""
    _sample_solutions(candidates, clients, config)
    _judge_divergence(candidates, clients, config)

    ranked = sorted(
        candidates,
        key=lambda c: (c.llm_divergence_score or 0.0),
        reverse=True,
    )
    kept = ranked[: config.N_div]
    print(
        f"[stage3] LLM divergence computed for {len(candidates)} candidates; "
        f"kept top {len(kept)} (N_div={config.N_div})."
    )
    return kept
