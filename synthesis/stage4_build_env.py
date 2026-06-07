"""Stage 4 - Testing infrastructure + execution divergence (lines 10-13; Sec 3.3).

For each top-N_div candidate, a test-case agent and a verifier agent build a
problem environment, cross-validating each other against the sampled solutions:

  * test-case agent  -> gen.cpp  -> /exec/gen materializes testdata/*.in
  * verifier agent   -> chk.cc   (continuous score in [0,1], Eq. 4)
  * install the package, run every sampled solution, read per-test scoreRatio
  * cross-validate:
      - gen.cpp/chk.cc fails to compile, or solutions crash      -> revise the offender
      - scores collapse to a narrow band / are all-zero / degenerate -> revise the verifier
  * iterate up to ``cross_validation_max_rounds``; discard candidates that
    never converge (paper: ~10% survive).

On success, the execution-grounded divergence (Eq. 3) is computed from the
score matrix:

    d_hat(c) = (1 / C(n,2)) * sum_{i<j} (1/sqrt(m)) * || q_i - q_j ||_2
"""

from __future__ import annotations

import math
import re
from itertools import combinations
from pathlib import Path
from typing import List, Optional, Tuple

from .config import PipelineConfig
from .judge_client import JudgeClient, JudgeError
from .llm import Clients
from .prompts import build_test_case_prompt, build_verifier_prompt
from .types import Candidate
from .utils import extract_cpp, write_problem_directory

# A near-constant score spread across solutions on every test means the verifier
# fails to discriminate quality -- treat as a flawed scoring program.
_COLLAPSE_EPS = 1e-6


def _safe_pid(candidate_id: str) -> str:
    return "synth_" + re.sub(r"[^0-9A-Za-z_]", "_", candidate_id)


def compute_exec_divergence(score_matrix: List[List[float]]) -> float:
    """Average normalized pairwise L2 distance between solution score vectors."""
    n = len(score_matrix)
    if n < 2:
        return 0.0
    m = len(score_matrix[0]) if score_matrix[0] else 0
    if m == 0:
        return 0.0
    inv_sqrt_m = 1.0 / math.sqrt(m)
    total = 0.0
    n_pairs = 0
    for i, j in combinations(range(n), 2):
        qi, qj = score_matrix[i], score_matrix[j]
        dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(qi, qj)))
        total += inv_sqrt_m * dist
        n_pairs += 1
    return total / n_pairs if n_pairs else 0.0


def _run_solutions(
    judge: JudgeClient, pid: str, solutions: List[str], n_tests: int
) -> Tuple[List[List[float]], List[str]]:
    """Run every solution; return (score_matrix, error_notes).

    Each row is a length-``n_tests`` score vector. Crashes/timeouts/unparseable
    output score 0 (judge convention, Eq. 4). A solution whose submission errors
    at the judge level (e.g. checker crash) records a note for cross-validation.
    """
    matrix: List[List[float]] = []
    notes: List[str] = []
    for si, code in enumerate(solutions):
        try:
            result = judge.submit_and_wait(pid, code)
            scores = JudgeClient.case_scores(result)
        except JudgeError as e:
            notes.append(f"solution {si}: judge error: {e}")
            scores = []
        # Pad/truncate to exactly n_tests so the matrix is rectangular.
        if len(scores) < n_tests:
            scores = scores + [0.0] * (n_tests - len(scores))
        else:
            scores = scores[:n_tests]
        matrix.append(scores)
    return matrix, notes


def _cross_validate(matrix: List[List[float]], notes: List[str]) -> Tuple[bool, str, str]:
    """Inspect the score matrix. Returns (ok, test_feedback, verifier_feedback)."""
    if notes:
        # Judge-level errors usually mean the checker failed to compile/ran badly.
        return False, "", "The checker errored on submissions: " + "; ".join(notes)

    if not matrix or not matrix[0]:
        return False, "No test cases produced any scores.", ""

    m = len(matrix[0])
    n = len(matrix)

    # All-zero across the board: baseline too strong, infeasible parsing, or a
    # broken checker. Route to the verifier (and mention tests).
    all_zero = all(all(v == 0.0 for v in row) for row in matrix)
    if all_zero:
        return (
            False,
            "Every solution scored 0 on every test; inputs may be infeasible or trivially baseline.",
            "Every solution scored 0; the baseline may be too strong or feasibility parsing is wrong.",
        )

    # Per-test spread: if scores barely vary across solutions on EVERY test, the
    # verifier does not discriminate quality -> revise the verifier.
    collapsed = True
    for t in range(m):
        col = [matrix[i][t] for i in range(n)]
        if (max(col) - min(col)) > _COLLAPSE_EPS:
            collapsed = False
            break
    if collapsed:
        return (
            False,
            "",
            "Scores collapse to a near-constant band across solutions; the scoring "
            "function does not discriminate solution quality. Make scores reflect "
            "relative objective quality per Eq. 4.",
        )

    return True, "", ""


def _build_one(
    candidate: Candidate, clients: Clients, judge: JudgeClient, config: PipelineConfig
) -> Candidate:
    """Run the cross-validation loop for a single candidate."""
    pid = _safe_pid(candidate.candidate_id)
    build_dir = Path(config.output_dir).parent / "_build" / pid
    test_feedback = ""
    verifier_feedback = ""

    for round_idx in range(config.cross_validation_max_rounds):
        # 1) Test-case agent -> gen.cpp -> materialize inputs via /exec/gen.
        gen_code = extract_cpp(
            clients.solver.generate(
                build_test_case_prompt(
                    candidate.mutated_statement,
                    candidate.solutions,
                    config.n_test_cases,
                    feedback=test_feedback,
                )
            )
        )
        if not gen_code:
            candidate.build_log.append(f"round {round_idx}: empty gen.cpp")
            continue
        try:
            test_inputs = judge.exec_gen(gen_code, list(range(1, config.n_test_cases + 1)))
        except JudgeError as e:
            test_feedback = f"gen.cpp failed: {e}"
            candidate.build_log.append(f"round {round_idx}: {test_feedback}")
            continue
        test_inputs = [t for t in test_inputs if t.strip()]
        if len(test_inputs) < 2:
            test_feedback = "gen.cpp produced too few non-empty inputs."
            candidate.build_log.append(f"round {round_idx}: {test_feedback}")
            continue

        # 2) Verifier agent -> chk.cc.
        chk_code = extract_cpp(
            clients.solver.generate(
                build_verifier_prompt(
                    candidate.mutated_statement,
                    candidate.solutions,
                    feedback=verifier_feedback,
                )
            )
        )
        if not chk_code:
            verifier_feedback = "Checker was empty; produce a complete chk.cc."
            candidate.build_log.append(f"round {round_idx}: empty chk.cc")
            continue

        # 3) Write package + install in the judge.
        write_problem_directory(
            str(build_dir),
            statement=candidate.mutated_statement,
            checker_code=chk_code,
            generator_code=gen_code,
            test_inputs=test_inputs,
        )
        try:
            judge.add_problem(pid, str(build_dir))
        except Exception as e:  # noqa: BLE001 - surface install failure as feedback
            verifier_feedback = f"Problem failed to install in judge: {e}"
            candidate.build_log.append(f"round {round_idx}: {verifier_feedback}")
            continue

        # 4) Run solutions -> per-test score matrix.
        matrix, notes = _run_solutions(judge, pid, candidate.solutions, len(test_inputs))

        # 5) Cross-validate.
        ok, test_feedback, verifier_feedback = _cross_validate(matrix, notes)
        if ok:
            candidate.generator_code = gen_code
            candidate.checker_code = chk_code
            candidate.test_inputs = test_inputs
            candidate.score_matrix = matrix
            candidate.exec_divergence_score = compute_exec_divergence(matrix)
            candidate.validated = True
            candidate.build_log.append(f"round {round_idx}: validated")
            return candidate
        candidate.build_log.append(
            f"round {round_idx}: not valid (test='{test_feedback[:60]}' "
            f"verifier='{verifier_feedback[:60]}')"
        )

    candidate.validated = False
    return candidate


def build_environment(
    candidates: List[Candidate], clients: Clients, judge: JudgeClient, config: PipelineConfig
) -> List[Candidate]:
    """Build + validate test infrastructure for each candidate (sequential).

    Kept sequential because each candidate is judge-bound (many compile/run
    submissions); the judge serializes work anyway, and this keeps logs readable.
    Returns only validated candidates, each with an execution-divergence score.
    """
    validated: List[Candidate] = []
    for idx, cand in enumerate(candidates):
        result = _build_one(cand, clients, judge, config)
        if result.validated:
            validated.append(result)
        print(
            f"[stage4] ({idx + 1}/{len(candidates)}) {cand.candidate_id}: "
            f"{'VALIDATED' if result.validated else 'discarded'}"
            + (
                f" exec_div={result.exec_divergence_score:.4f}"
                if result.exec_divergence_score is not None
                else ""
            )
        )
        # Surface the per-round cross-validation reasons so a discard is
        # diagnosable (compile error vs. degenerate scores vs. install failure).
        if not result.validated:
            for line in result.build_log:
                print(f"[stage4]      {line}")
    print(f"[stage4] {len(validated)}/{len(candidates)} candidates produced validated (T_c, V_c).")
    return validated
