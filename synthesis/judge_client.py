"""HTTP client for the FrontierCS judge (Node service, default localhost:8082).

Wraps the endpoints exercised by Stage 4:
  POST /submit             - compile + run a C++ solution, returns a submission id
  GET  /result/{sid}       - poll; on "done" returns score AND per-case detail
  POST /exec/gen           - compile a testlib gen.cpp and run it per testId
  POST /problem/add-problem - install a generated problem package (tar.gz)

The crucial difference from verl's reward client is that we keep the FULL
result dict: ``cases[k].scoreRatio`` (in [0,1]) gives the per-test score that
feeds the execution-grounded divergence estimate (Eq. 3).
"""

from __future__ import annotations

import io
import os
import time
import zipfile as zipfile_mod
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from .config import PipelineConfig


def _jlog(msg: str) -> None:
    print(f"[judge {time.strftime('%H:%M:%S')}] {msg}")


class JudgeError(RuntimeError):
    """Raised when the judge returns an unrecoverable error for a request."""


class JudgeClient:
    def __init__(self, config: PipelineConfig) -> None:
        self.url = config.judge_url.rstrip("/")
        self.max_wait = config.judge_max_wait
        self.poll_interval = config.judge_poll_interval

    # ------------------------------------------------------------------ submit
    def submit(self, pid: str, code: str, lang: str = "cpp") -> Optional[str]:
        """Submit C++ source for problem ``pid``; return the submission id."""
        _jlog(f"submit pid={pid} code={len(code)} chars")
        r = requests.post(
            f"{self.url}/submit",
            files={"code": ("sol.cpp", code)},
            data={"pid": pid, "lang": lang},
            timeout=30,
        )
        r.raise_for_status()
        sid = r.json().get("sid")
        _jlog(f"submit → sid={sid}")
        return sid

    def poll_result(self, sid: str) -> Dict[str, Any]:
        """Poll ``/result/{sid}`` until done/error or the wait budget expires.

        Returns the full result dict. On "done" this includes ``score`` (mean
        over cases, 0-100) and ``cases`` (a list with per-case ``scoreRatio``).
        Raises ``JudgeError`` on judge-side error or timeout.
        """
        _jlog(f"polling sid={sid} ...")
        start = time.time()
        while time.time() - start < self.max_wait:
            r = requests.get(f"{self.url}/result/{sid}", timeout=10)
            if r.status_code == 404:
                time.sleep(self.poll_interval)
                continue
            r.raise_for_status()
            res = r.json()
            status = res.get("status")
            if status == "done":
                scores = self.case_scores(res)
                score_str = ", ".join(f"{s:.3f}" for s in scores)
                _jlog(f"sid={sid} DONE  score={res.get('score')}  cases=[{score_str}]")
                return res
            if status == "error":
                _jlog(f"sid={sid} ERROR: {res.get('error', '?')}")
                raise JudgeError(res.get("error", "judge error"))
            time.sleep(self.poll_interval)
        raise JudgeError(f"timed out after {self.max_wait}s polling sid={sid}")

    def submit_and_wait(self, pid: str, code: str) -> Dict[str, Any]:
        """Convenience: submit then poll, returning the full result dict."""
        sid = self.submit(pid, code)
        if not sid:
            raise JudgeError("submit returned no sid")
        return self.poll_result(sid)

    # ------------------------------------------------------------ per-case util
    @staticmethod
    def case_scores(result: Dict[str, Any]) -> List[float]:
        """Extract the per-test ``scoreRatio`` vector (in [0,1]) from a result.

        Solutions that crash / time out / produce unparseable output yield a
        scoreRatio of 0 from the judge, matching Eq. (4)'s convention.
        """
        cases = result.get("cases") or []
        scores: List[float] = []
        for c in cases:
            try:
                scores.append(float(c.get("scoreRatio", 0.0)))
            except (TypeError, ValueError):
                scores.append(0.0)
        return scores

    # --------------------------------------------------------------- exec/gen
    def exec_gen(self, gen_code: str, test_ids: List[int], timeout_ms: int = 30000) -> List[str]:
        """Compile a testlib generator and run it for each ``testId``.

        Returns one input string per test id, in order. Raises ``JudgeError``
        if compilation fails.
        """
        _jlog(f"exec_gen  {len(test_ids)} test ids  code={len(gen_code)} chars")
        r = requests.post(
            f"{self.url}/exec/gen",
            json={"code": gen_code, "testIds": test_ids, "timeoutMs": timeout_ms},
            timeout=max(60, timeout_ms // 1000 + 30),
        )
        r.raise_for_status()
        payload = r.json()
        if payload.get("compileStatus") != "ok":
            _jlog(f"exec_gen  COMPILE ERROR: {payload.get('compileStderr', '')[:200]}")
            raise JudgeError(f"gen.cpp compile error: {payload.get('compileStderr', '')[:2000]}")
        results = payload.get("results", [])
        inputs: List[str] = []
        for res in results:
            if isinstance(res, str):
                inputs.append(res)
            elif isinstance(res, dict):
                inputs.append(res.get("stdout") or res.get("output") or res.get("input") or "")
            else:
                inputs.append("")
        non_empty = sum(1 for i in inputs if i.strip())
        _jlog(f"exec_gen  OK  {non_empty}/{len(inputs)} non-empty inputs generated")
        return inputs

    # ------------------------------------------------------------ add-problem
    def _zip_package(self, package_dir: str) -> io.BytesIO:
        """Build an in-memory zip of ``package_dir`` for upload."""
        buf = io.BytesIO()
        base = Path(package_dir)
        with zipfile_mod.ZipFile(buf, "w", zipfile_mod.ZIP_DEFLATED) as zf:
            for path in sorted(base.rglob("*")):
                if path.is_file():
                    zf.write(str(path), arcname=str(path.relative_to(base)))
        buf.seek(0)
        return buf

    def add_problem(self, pid: str, package_dir: str) -> Dict[str, Any]:
        """Install (or re-install) a problem package into the judge.

        Uses POST /problem/add-problem on first install and POST /problem/setup
        on subsequent rounds so the "problem already exists" check doesn't fire
        when the cross-validation loop revises chk.cc or gen.cpp.

        The judge's multer middleware expects field name ``zipfile`` and only
        accepts .zip files (application/zip), not tar.gz.
        """
        _jlog(f"add_problem pid={pid}  dir={package_dir}")

        # Try add-problem first; if the problem already exists, fall back to setup.
        buf = self._zip_package(package_dir)
        r = requests.post(
            f"{self.url}/problem/add-problem",
            files={"zipfile": (f"{pid}.zip", buf, "application/zip")},
            data={"pid": pid},
            timeout=120,
        )
        if r.status_code == 500 and "already exists" in r.text:
            _jlog(f"add_problem pid={pid}  already exists, using /problem/setup")
            buf = self._zip_package(package_dir)
            r = requests.post(
                f"{self.url}/problem/setup",
                files={"zipfile": (f"{pid}.zip", buf, "application/zip")},
                data={"pid": pid},
                timeout=120,
            )
        if not r.ok:
            _jlog(f"add_problem pid={pid}  ERROR {r.status_code}: {r.text[:500]}")
        r.raise_for_status()
        result = r.json()
        _jlog(f"add_problem pid={pid}  → {result}")
        return result

    # --------------------------------------------------------------- health
    def health(self) -> bool:
        try:
            r = requests.get(f"{self.url}/health", timeout=5)
            return r.status_code == 200 and bool(r.json().get("ok"))
        except Exception:
            return False
