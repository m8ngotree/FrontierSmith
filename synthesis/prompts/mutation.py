"""Mutation prompt (Stage 1, paper Section 3.1).

Instructs the LLM to (a) extract the seed problem's formulation as the tuple
(O, C_I, C_O) -- computational goal, admissible inputs, output constraints --
and (b) apply one or more of the three mutation types to produce an open-ended
variant with no efficiently-certifiable optimum and a continuous quality score.
"""

from __future__ import annotations

_TEMPLATE = r"""You are an expert competitive-programming problem setter. Your task is to
transform a CLOSED-ended problem (one with a single correct answer or a known
efficient optimum) into an OPEN-ended optimization problem (one with no
efficiently-certifiable optimum, where solutions are scored on a continuous
quality scale).

# Formulation model
Represent any problem as a tuple (O, C_I, C_O):
  - O   = the computational goal (a required output, decision, property, or quantity to optimize)
  - C_I = the admissible problem instances (constraints on the input domain)
  - C_O = the constraints on valid program outputs

# Mutation types (apply ONE OR MORE)
1. Changing goals (O -> O'): replace a decision/exact-answer goal with an
   optimization-oriented goal that admits graded performance.
   e.g. 2-SAT (decide satisfiability) -> Min-True 2-SAT (minimize true variables).
2. Restricting outputs (C_O -> C_O'): add/tighten output constraints while
   keeping the goal, making exact solutions intractable at scale.
   e.g. Minimum Spanning Tree -> Degree-Constrained Spanning Tree (NP-hard).
3. Generalizing inputs (C_I -> C_I'): relax structural assumptions on inputs.
   e.g. Max Independent Set on bipartite graphs (poly) -> on general graphs (NP-hard).

# Requirements for the mutated problem
- Exact optimum must be intractable to certify at the stated scale.
- A continuous quality measure must exist (a scoring function that ranks submissions).
- The problem must remain self-contained: a text input, a text output, no external services.
- Provide a baseline notion so a verifier can normalize scores later.

# Output format
The problem_statement MUST follow this FrontierCS markdown structure exactly,
wrapped in a ```markdown fenced block:
  # <Title>
  ## Problem        (narrative description of the open-ended objective)
  ## Input          (precise input format)
  ## Output         (precise output format)
  ## Feasibility requirements   (what makes an output valid; infeasible -> score 0)
  ## Objective      (the quantity O to optimize, "minimize"/"maximize")
  ## Scoring        (continuous test score relative to a baseline)
  ## Constraints    (sizes; time limit; memory limit)
  ## Example        (one worked input/output with explanation)

Respond with a SINGLE JSON object and nothing else:
{
  "original_formulation": {"O": "...", "C_I": "...", "C_O": "..."},
  "mutations_applied": ["changing_goals" | "restricting_outputs" | "generalizing_inputs", ...],
  "mutated_formulation": {"O_prime": "...", "C_I_prime": "...", "C_O_prime": "..."},
  "problem_statement": "```markdown\n# Title\n...\n```"
}

# Seed problem statement
<<STATEMENT>>
"""


def build_mutation_prompt(seed_statement: str) -> str:
    return _TEMPLATE.replace("<<STATEMENT>>", seed_statement)
