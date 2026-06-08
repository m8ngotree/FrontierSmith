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
) -> Tuple[List[List[float]], List[int], List[str], List[str], List[List[dict]]]:
    """Run every solution; return (matrix, kept_indices, compile_errors, checker_errors, case_details).

    Only solutions that actually ran contribute a row to ``matrix`` (each a
    length-``n_tests`` score vector); ``kept_indices`` records their original
    positions for logging. A submission that raises a judge error is *excluded*
    from the matrix (so it can't pollute the all-zero / collapse checks with a
    spurious zero row) and recorded as a note instead, classified as:

      * ``compile_errors``  - the solution itself failed to compile (the
        solution's fault, not the generator's or checker's);
      * ``checker_errors``  - any other judge-level failure (e.g. the checker
        crashed), which should drive a verifier revision.

    ``case_details`` is parallel to ``matrix``: one list per solution that ran,
    each containing a dict per test case with keys ``status``, ``msg``, ``time_s``.
    """
    matrix: List[List[float]] = []
    kept_indices: List[int] = []
    compile_errors: List[str] = []
    checker_errors: List[str] = []
    case_details: List[List[dict]] = []
    for si, code in enumerate(solutions):
        try:
            result = judge.submit_and_wait(pid, code)
            scores = JudgeClient.case_scores(result)
        except JudgeError as e:
            msg = str(e).lower()
            note = f"solution {si}: {e}"
            if "compile" in msg or "compilation" in msg:
                compile_errors.append(note)
            else:
                checker_errors.append(note)
            continue  # do NOT add a zero row for a submission that never ran
        # Pad/truncate to exactly n_tests so the matrix is rectangular.
        if len(scores) < n_tests:
            scores = scores + [0.0] * (n_tests - len(scores))
        else:
            scores = scores[:n_tests]
        matrix.append(scores)
        kept_indices.append(si)
        # Collect per-case details for logging.
        details = []
        for case in (result.get("cases") or []):
            details.append({
                "status": case.get("status", "?"),
                "msg": (case.get("msg") or "").strip(),
                "time_s": round(case.get("time", 0) / 1e6, 2),
            })
        case_details.append(details)
    return matrix, kept_indices, compile_errors, checker_errors, case_details


def _cross_validate(matrix: List[List[float]], checker_errors: List[str]) -> Tuple[bool, str, str]:
    """Inspect the score matrix. Returns (ok, test_feedback, verifier_feedback).

    ``matrix`` contains only solutions that actually ran (solution compile
    failures are dropped upstream in ``_run_solutions``). ``checker_errors`` are
    judge-level failures attributable to the checker.
    """
    if checker_errors:
        return False, "", "The checker errored on submissions: " + "; ".join(checker_errors)

    if not matrix or not matrix[0]:
        return False, "No solutions ran successfully on the generated tests.", ""

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
        s4 = f"[stage4] [{pid}] round {round_idx}"
        print(f"{s4} --- start "
              f"(solutions={len(candidate.solutions)}, "
              f"test_fb={bool(test_feedback)}, verifier_fb={bool(verifier_feedback)})")
        if test_feedback:
            print(f"{s4}   test_feedback:     {test_feedback[:200]}")
        if verifier_feedback:
            print(f"{s4}   verifier_feedback: {verifier_feedback[:200]}")

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
            print(f"{s4}   gen.cpp: EMPTY")
            candidate.build_log.append(f"round {round_idx}: empty gen.cpp")
            continue
        print(f"{s4}   gen.cpp: {len(gen_code)} chars, {gen_code.count(chr(10))} lines")

        try:
            test_inputs = judge.exec_gen(gen_code, list(range(1, config.n_test_cases + 1)))
        except JudgeError as e:
            test_feedback = f"gen.cpp failed: {e}"
            print(f"{s4}   exec_gen FAILED: {e}")
            candidate.build_log.append(f"round {round_idx}: {test_feedback}")
            continue
        test_inputs = [t for t in test_inputs if t.strip()]
        if len(test_inputs) < 2:
            test_feedback = "gen.cpp produced too few non-empty inputs."
            print(f"{s4}   exec_gen: only {len(test_inputs)} non-empty inputs")
            candidate.build_log.append(f"round {round_idx}: {test_feedback}")
            continue
        input_sizes = [len(t) for t in test_inputs]
        print(f"{s4}   exec_gen: {len(test_inputs)} inputs, "
              f"sizes={[f'{s}b' for s in input_sizes]}")

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
            print(f"{s4}   chk.cc: EMPTY")
            candidate.build_log.append(f"round {round_idx}: empty chk.cc")
            continue
        print(f"{s4}   chk.cc:  {len(chk_code)} chars, {chk_code.count(chr(10))} lines")

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
            print(f"{s4}   add_problem FAILED: {e}")
            candidate.build_log.append(f"round {round_idx}: {verifier_feedback}")
            continue

        # 4) Run solutions -> per-test score matrix.
        print(f"{s4}   running {len(candidate.solutions)} solutions ...")
        matrix, kept_indices, compile_errors, checker_errors, case_details = _run_solutions(
            judge, pid, candidate.solutions, len(test_inputs)
        )

        # Print score matrix with per-case judge feedback.
        print(f"{s4}   score matrix ({len(matrix)} solutions x {len(test_inputs)} tests):")
        for row_idx, row in enumerate(matrix):
            row_str = "  ".join(f"{v:.3f}" for v in row)
            print(f"{s4}     sol[{kept_indices[row_idx]}]: [{row_str}]")
            for ti, detail in enumerate(case_details[row_idx]):
                status = detail["status"]
                msg = detail["msg"][:120]
                t = detail["time_s"]
                print(f"{s4}       test{ti+1}: {status:15s}  {t:.2f}s  {msg}")
        for note in compile_errors:
            print(f"{s4}   solution dropped (compile): {note}")
        for note in checker_errors:
            print(f"{s4}   checker error: {note}")

        # 5) Cross-validate.
        ok, test_feedback, verifier_feedback = _cross_validate(matrix, checker_errors)
        if ok:
            div = compute_exec_divergence(matrix)
            print(f"{s4}   VALIDATED  exec_div={div:.4f}")
            candidate.generator_code = gen_code
            candidate.checker_code = chk_code
            candidate.test_inputs = test_inputs
            candidate.score_matrix = matrix
            candidate.case_details = case_details
            candidate.exec_divergence_score = div
            candidate.validated = True
            candidate.build_log.append(f"round {round_idx}: validated")
            return candidate

        if checker_errors:
            verdict = "checker-errors"
        elif not matrix or not matrix[0]:
            verdict = "no-solutions-ran"
        else:
            all_zero = all(all(v == 0 for v in r) for r in matrix)
            has_spread = any(
                (max(matrix[i][t] for i in range(len(matrix)))
                 - min(matrix[i][t] for i in range(len(matrix)))) > _COLLAPSE_EPS
                for t in range(len(matrix[0]))
            )
            verdict = "all-zero" if all_zero else "collapsed" if not has_spread else "other"
        print(f"{s4}   cross-validation FAILED ({verdict})")
        if test_feedback:
            print(f"{s4}     → test_feedback:     {test_feedback[:200]}")
        if verifier_feedback:
            print(f"{s4}     → verifier_feedback: {verifier_feedback[:200]}")
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
