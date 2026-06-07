"""Verifier agent prompt (Stage 4, paper Section 3.3 + Eq. 4).

The agent writes a testlib-style C++ checker (chk.cc) that turns the problem's
objective O into a NORMALIZED continuous score in [0,1]. Two conventions are
non-negotiable because the FrontierCS judge depends on them:

1. The checker MUST emit a literal token ``Ratio: <float>`` inside its quitp
   message. The judge parses the per-test score with the regex /Ratio: ([0-9.]+)/
   and stores it as ``scoreRatio`` in [0,1]. (Confirmed in judge_engine.js and
   in the shipped frontiersmith_1/chk.cc.)
2. Infeasible / unparseable / crashing outputs MUST receive score 0 -- emit
   ``quitf(_wa, ...)`` (which carries no Ratio token, so the judge records 0).
"""

from __future__ import annotations

from typing import List

_TEMPLATE = r"""You are a verifier agent for an OPEN-ended optimization problem. Write a single
self-contained C++ testlib checker (chk.cc) that reads the test input and the
participant's output, validates feasibility, computes the objective, and emits
a NORMALIZED score in [0, 1].

# How scoring must work (paper Eq. 4)
- Translate the problem's objective O into a scoring program V(s, t) in [0, 1].
- Normalize against a BASELINE solution s* that you compute INSIDE the checker
  (it need not be strong: a trivial valid output, a random valid solution, or a
  simple greedy is fine). Ensure O(., t) > 0 (add a constant offset if needed).
- Let sigma = +1 for a MAXIMIZATION objective and -1 for MINIMIZATION. Then:
      V = max(0,  sigma * (O(s,t) - O(s*,t)) / max(O(s,t), O(s*,t)) )
  Clamp V into [0, 1]. A submission no better than the baseline scores 0; better
  submissions score toward 1.

# Hard requirements (the judge depends on these EXACTLY)
- Include testlib: `#include "testlib.h"` and `#include <bits/stdc++.h>`.
- In main: `registerTestlibCmd(argc, argv);`
- Read the test input from `inf` (e.g. `inf.readInt()`), the participant output
  from `ouf`. Reject extra trailing output with `if (!ouf.seekEof()) quitf(_wa, "...");`.
- If the output is INFEASIBLE or malformed: `quitf(_wa, "reason");`  (this yields score 0).
- For a feasible output, compute `double ratio` in [0,1] and finish with:
      quitp(ratio, "OK ... Ratio: %.6f", ratio);
  The substring "Ratio: <number>" MUST appear in the message -- the judge parses it.

# Reference pattern (adapt to THIS problem's objective)
```cpp
#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
int main(int argc, char* argv[]) {
    registerTestlibCmd(argc, argv);
    // 1) read input from inf
    // 2) read participant output from ouf; validate feasibility -> quitf(_wa, ...) on failure
    // 3) compute objective O_sub for the submission
    // 4) compute baseline objective O_base from a simple valid baseline
    // 5) normalize to ratio in [0,1] (Eq. 4); clamp
    double ratio = /* in [0,1] */ 0.0;
    quitp(ratio, "OK O_sub=... O_base=... Ratio: %.6f", ratio);
}
```

<<FEEDBACK>>

# Problem statement
<<STATEMENT>>

# Sampled solutions (so your checker can correctly parse valid outputs)
<<SOLUTIONS>>

Output ONLY the chk.cc code in a single ```cpp fenced block, with no other text.
"""


def build_verifier_prompt(
    statement: str,
    solutions: List[str],
    feedback: str = "",
) -> str:
    sol_blocks = []
    for idx, code in enumerate(solutions):
        sol_blocks.append(f"## Solution {idx}\n```cpp\n{code}\n```")
    solutions_text = "\n\n".join(sol_blocks)

    feedback_block = ""
    if feedback:
        feedback_block = (
            "# Feedback from the previous round (FIX THESE ISSUES)\n" + feedback + "\n"
        )

    return (
        _TEMPLATE
        .replace("<<FEEDBACK>>", feedback_block)
        .replace("<<STATEMENT>>", statement)
        .replace("<<SOLUTIONS>>", solutions_text)
    )
