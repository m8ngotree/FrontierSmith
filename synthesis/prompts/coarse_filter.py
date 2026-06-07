"""Coarse LLM-as-a-judge filter prompt (Stage 2, paper Section 3.2).

Checks three conditions; a candidate must pass ALL three to survive:
  1. defines an optimization objective with no known (efficiently certifiable) optimum
  2. multiple distinct strategies are plausible
  3. a scoring function can meaningfully rank submissions
"""

from __future__ import annotations

_TEMPLATE = r"""You are a strict reviewer of OPEN-ended optimization problems. Decide whether
the problem below qualifies as a high-quality open-ended problem.

Check these three conditions independently:
  1. OPTIMIZATION_NO_OPTIMUM: The problem defines an optimization objective for
     which there is no known way to efficiently certify the optimum at the
     stated scale (i.e. it is not a closed-ended task with a single correct
     answer or a known polynomial-time exact solution).
  2. MULTIPLE_STRATEGIES: Multiple genuinely distinct algorithmic strategies
     are plausible (e.g. greedy vs. local search vs. DP vs. ILP-style), rather
     than one dominant approach.
  3. MEANINGFUL_SCORING: The scoring function can meaningfully and continuously
     rank submissions by quality (not just pass/fail).

Be conservative: it is better to reject a borderline problem than to admit a
closed-ended one.

Respond with a SINGLE JSON object and nothing else:
{
  "condition_1": {"pass": true/false, "reason": "..."},
  "condition_2": {"pass": true/false, "reason": "..."},
  "condition_3": {"pass": true/false, "reason": "..."},
  "overall_pass": true/false
}
"overall_pass" MUST be true only if all three conditions pass.

# Problem statement
<<STATEMENT>>
"""


def build_coarse_filter_prompt(statement: str) -> str:
    return _TEMPLATE.replace("<<STATEMENT>>", statement)
