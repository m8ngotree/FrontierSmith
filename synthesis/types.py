"""Data structures that flow between the synthesis pipeline stages.

A ``SeedProblem`` is a closed-ended competitive-programming problem drawn from
the seed pool (Stage 0). ``Mutate`` (Stage 1) turns it into one or more
``Candidate`` objects, which are then progressively enriched by Stages 2-4
(coarse filter result, sampled solutions, divergence scores, generated test
infrastructure). ``PairwiseJudgment`` records a single LLM-as-a-Judge verdict
used by the LLM-based idea-divergence estimate (Stage 3).

Every dataclass is JSON-serializable via ``to_dict``/``from_dict`` so the
orchestrator can checkpoint and resume the pipeline at any stage.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class SeedProblem:
    """A closed-ended seed problem from the HardTests corpus (or a validated
    open-ended problem fed back into the pool on a later iteration)."""

    problem_id: str          # e.g. "hard_codeforces_143_b"
    tier: str                # e.g. "hard"
    folder_name: str         # directory name under the problems root
    statement: str           # full text of statement.txt
    origin: str = "seed"     # "seed" (HardTests) or "synthesized" (fed back in)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SeedProblem":
        return cls(
            problem_id=d["problem_id"],
            tier=d.get("tier", ""),
            folder_name=d.get("folder_name", ""),
            statement=d.get("statement", ""),
            origin=d.get("origin", "seed"),
        )


@dataclass
class PairwiseJudgment:
    """One LLM-as-a-Judge verdict on whether two sampled solutions use the
    same core algorithmic strategy. ``different`` is the indicator used in
    Eq. (2): 1 if strategies differ, 0 if they are the same."""

    solution_i: int          # index of the first solution
    solution_j: int          # index of the second solution
    different: bool           # True == "different strategy"
    reason: str = ""          # short rationale from the judge (for debugging)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PairwiseJudgment":
        return cls(
            solution_i=d["solution_i"],
            solution_j=d["solution_j"],
            different=bool(d["different"]),
            reason=d.get("reason", ""),
        )


@dataclass
class Candidate:
    """A mutated, open-ended problem candidate as it moves through the pipeline.

    Fields populated at creation (Stage 1):
        candidate_id, seed_problem_id, mutated_statement, mutation_types,
        original_formulation, mutated_formulation

    Fields populated by later stages are optional and default to empty/None.
    """

    # --- Stage 1: mutation ---
    candidate_id: str
    seed_problem_id: str
    mutated_statement: str
    mutation_types: List[str] = field(default_factory=list)
    original_formulation: Dict[str, Any] = field(default_factory=dict)  # {O, C_I, C_O}
    mutated_formulation: Dict[str, Any] = field(default_factory=dict)    # {O', C_I', C_O'}

    # --- Stage 2: coarse LLM-as-a-judge filter ---
    coarse_filter_passed: Optional[bool] = None
    coarse_filter_reasons: Dict[str, Any] = field(default_factory=dict)

    # --- Stage 3: idea-divergence (LLM-based) ---
    solutions: List[str] = field(default_factory=list)          # extracted C++ source per solution
    pairwise_judgments: List[PairwiseJudgment] = field(default_factory=list)
    llm_divergence_score: Optional[float] = None               # d_hat from Eq. (2)

    # --- Stage 4: testing infrastructure + execution divergence ---
    generator_code: Optional[str] = None       # gen.cpp (testlib generator)
    checker_code: Optional[str] = None          # chk.cc (testlib verifier)
    test_inputs: List[str] = field(default_factory=list)        # materialized testdata/*.in contents
    score_matrix: List[List[float]] = field(default_factory=list)  # [n_solutions][m] of per-test scoreRatio
    exec_divergence_score: Optional[float] = None              # d_hat from Eq. (3)
    validated: bool = False                                     # passed cross-validation
    build_log: List[str] = field(default_factory=list)          # per-round notes for debugging

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # asdict already recursively converts the nested PairwiseJudgment list.
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Candidate":
        judgments = [PairwiseJudgment.from_dict(j) for j in d.get("pairwise_judgments", [])]
        return cls(
            candidate_id=d["candidate_id"],
            seed_problem_id=d["seed_problem_id"],
            mutated_statement=d["mutated_statement"],
            mutation_types=list(d.get("mutation_types", [])),
            original_formulation=dict(d.get("original_formulation", {})),
            mutated_formulation=dict(d.get("mutated_formulation", {})),
            coarse_filter_passed=d.get("coarse_filter_passed"),
            coarse_filter_reasons=dict(d.get("coarse_filter_reasons", {})),
            solutions=list(d.get("solutions", [])),
            pairwise_judgments=judgments,
            llm_divergence_score=d.get("llm_divergence_score"),
            generator_code=d.get("generator_code"),
            checker_code=d.get("checker_code"),
            test_inputs=list(d.get("test_inputs", [])),
            score_matrix=[list(row) for row in d.get("score_matrix", [])],
            exec_divergence_score=d.get("exec_divergence_score"),
            validated=bool(d.get("validated", False)),
            build_log=list(d.get("build_log", [])),
        )
