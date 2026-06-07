"""Solution-sampling prompt (Stage 3, paper Section 3.2).

Draws independent solver samples for a candidate. Diversity comes from
independent sampling (extended thinking enabled); we ask for a single complete
C++17 program so it can be compiled and run by the judge.
"""

from __future__ import annotations

_TEMPLATE = r"""You are a strong competitive programmer. Solve the OPEN-ended optimization
problem below in C++17. There is no single correct answer: aim for the best
quality score you can under the time and memory limits. Choose whatever
algorithmic strategy you think is best (greedy, local search, dynamic
programming, randomization, etc.).

Read from standard input and write to standard output exactly as specified.
Output ONLY the C++ code wrapped in a single ```cpp fenced block, with no other
text.

# Problem statement
<<STATEMENT>>
"""


def build_solution_prompt(statement: str) -> str:
    return _TEMPLATE.replace("<<STATEMENT>>", statement)
