"""Test-case agent prompt (Stage 4, paper Section 3.3).

The agent writes a testlib-style C++ generator program (gen.cpp). It is given
the problem statement and the sampled solutions so it can craft adversarial
inputs that expose where specific strategies break down. The generator must
follow the exact convention used by the judge's /exec/gen endpoint and the
shipped problems: registerGen + a testId read from argv[1].
"""

from __future__ import annotations

from typing import List

_TEMPLATE = r"""You are a test-case agent for an OPEN-ended optimization problem. Write a single
self-contained C++ testlib generator program (gen.cpp) that emits problem
INPUTS to standard output.

# Hard requirements (the judge runs your generator like the example below)
- Include testlib: `#include "testlib.h"` and `#include <bits/stdc++.h>`.
- In main: call `registerGen(argc, argv, 1);` then `int testId = atoi(argv[1]);`.
- Branch on `testId` from 1 to <<N>> and print ONE valid input per testId to
  stdout (use `cout`). Use testlib's `rnd` for randomness so runs are reproducible.
- Every generated input MUST satisfy the problem's Input format and Constraints.

# Design goals (vary inputs across the <<N>> test ids)
- Vary SIZE: from tiny corner cases up to near the maximum constraints.
- Vary STRUCTURE: e.g. sparse vs. dense graphs, uniform vs. skewed distributions,
  adversarial layouts.
- Craft ADVERSARIAL inputs that distinguish the strategies in the sampled
  solutions below (inputs where a greedy approach fails but local search wins,
  inputs that stress timing, etc.).

# Skeleton (follow this structure)
```cpp
#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
int main(int argc, char* argv[]) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);
    if (testId == 1) {
        // smallest corner case
    } else if (testId == 2) {
        // ...
    }
    // ... up to testId == <<N>>
    return 0;
}
```

<<FEEDBACK>>

# Problem statement
<<STATEMENT>>

# Sampled solutions (for crafting adversarial inputs)
<<SOLUTIONS>>

Output ONLY the gen.cpp code in a single ```cpp fenced block, with no other text.
"""


def build_test_case_prompt(
    statement: str,
    solutions: List[str],
    n_test_cases: int,
    feedback: str = "",
) -> str:
    sol_blocks = []
    # Cap how much solution code we include to keep the prompt manageable.
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
        .replace("<<N>>", str(n_test_cases))
        .replace("<<FEEDBACK>>", feedback_block)
        .replace("<<STATEMENT>>", statement)
        .replace("<<SOLUTIONS>>", solutions_text)
    )
