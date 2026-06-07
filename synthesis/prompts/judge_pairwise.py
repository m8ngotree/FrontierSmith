"""Batched LLM-as-a-Judge pairwise strategy-comparison prompt (Stage 3, Eq. 2).

A naive divergence estimate needs O(n^2) judge calls. We batch several pairs
into one query: the prompt lists all solutions once (numbered) and asks the
judge to label a given set of pairs as same- or different-strategy.
"""

from __future__ import annotations

from typing import List, Tuple

_HEADER = r"""You are an expert algorithm analyst. You are given several C++ solutions to the
SAME open-ended optimization problem. For specified pairs of solutions, judge
whether the two solutions use the SAME core algorithmic strategy or DIFFERENT
core strategies.

"Strategy" means the core algorithmic idea (e.g. greedy, local search /
simulated annealing, dynamic programming, flow/matching, ILP-style,
brute-force/randomized), regardless of low-level implementation differences.
Two solutions that differ only in constants, I/O, or minor tuning use the SAME
strategy. Two solutions built on fundamentally different ideas are DIFFERENT.

# Problem statement
<<STATEMENT>>

# Solutions
<<SOLUTIONS>>

# Pairs to judge
<<PAIRS>>

Respond with a SINGLE JSON object mapping each pair (as "i-j") to a verdict:
{
  "judgments": [
    {"pair": "i-j", "different": true/false, "reason": "..."},
    ...
  ]
}
Include exactly one entry for each requested pair. Use the 0-based indices shown.
"""


def build_pairwise_judge_prompt(
    statement: str,
    solutions: List[str],
    pairs: List[Tuple[int, int]],
) -> str:
    sol_blocks = []
    for idx, code in enumerate(solutions):
        sol_blocks.append(f"## Solution {idx}\n```cpp\n{code}\n```")
    solutions_text = "\n\n".join(sol_blocks)

    pairs_text = "\n".join(f"- {i}-{j}" for (i, j) in pairs)

    return (
        _HEADER
        .replace("<<STATEMENT>>", statement)
        .replace("<<SOLUTIONS>>", solutions_text)
        .replace("<<PAIRS>>", pairs_text)
    )
