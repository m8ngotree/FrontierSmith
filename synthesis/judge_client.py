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
import tarfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from .config import PipelineConfig


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
        r = requests.post(
            f"{self.url}/submit",
            files={"code": ("sol.cpp", code)},
            data={"pid": pid, "lang": lang},
            timeout=30,
        )
        r.raise_for_status()
        return r.json().get("sid")

    def poll_result(self, sid: str) -> Dict[str, Any]:
        """Poll ``/result/{sid}`` until done/error or the wait budget expires.

        Returns the full result dict. On "done" this includes ``score`` (mean
        over cases, 0-100) and ``cases`` (a list with per-case ``scoreRatio``).
        Raises ``JudgeError`` on judge-side error or timeout.
        """
        start = time.time()
        while time.time() - start < self.max_wait:
            r = requests.get(f"{self.url}/result/{sid}", timeout=10)
            if r.status_code == 404:
                # Not scheduled yet; keep polling.
                time.sleep(self.poll_interval)
                continue
            r.raise_for_status()
            res = r.json()
            status = res.get("status")
            if status == "done":
                return res
            if status == "error":
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
        r = requests.post(
            f"{self.url}/exec/gen",
            json={"code": gen_code, "testIds": test_ids, "timeoutMs": timeout_ms},
            timeout=max(60, timeout_ms // 1000 + 30),
        )
        r.raise_for_status()
        payload = r.json()
        if payload.get("compileStatus") != "ok":
            raise JudgeError(f"gen.cpp compile error: {payload.get('compileStderr', '')[:2000]}")
        results = payload.get("results", [])
        # Each result is expected to carry the generator's stdout. The judge
        # returns objects; pull the stdout-like field defensively.
        inputs: List[str] = []
        for res in results:
            if isinstance(res, str):
                inputs.append(res)
            elif isinstance(res, dict):
                inputs.append(res.get("stdout") or res.get("output") or res.get("input") or "")
            else:
                inputs.append("")
        return inputs

    # ------------------------------------------------------------ add-problem
    def add_problem(self, pid: str, package_dir: str) -> Dict[str, Any]:
        """Install a problem package (statement/config/chk.cc/gen.cpp/testdata)
        into the judge by uploading a tar.gz of ``package_dir``.
        """
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            base = Path(package_dir)
            for path in sorted(base.rglob("*")):
                if path.is_file():
                    tar.add(str(path), arcname=str(path.relative_to(base)))
        buf.seek(0)
        r = requests.post(
            f"{self.url}/problem/add-problem",
            files={"file": (f"{pid}.tar.gz", buf, "application/gzip")},
            data={"pid": pid},
            timeout=120,
        )
        r.raise_for_status()
        return r.json()

    # --------------------------------------------------------------- health
    def health(self) -> bool:
        try:
            r = requests.get(f"{self.url}/health", timeout=5)
            return r.status_code == 200 and bool(r.json().get("ok"))
        except Exception:
            return False
