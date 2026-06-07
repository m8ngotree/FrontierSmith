"""Per-run logging setup.

Everything a single pipeline invocation produces is collected under one
timestamped directory ``synthesis/logs/run_<YYYYmmdd_HHMMSS>/``:

* ``pipeline.log``    - a verbatim copy of everything printed to the terminal
                        (stage progress, filter decisions, stage-4 discard
                        reasons, the live ``[llm] -->/<--`` lines, errors).
* ``api_calls.jsonl`` - one JSON object per LLM call (prompt, response, model,
                        provider, role, stage, latency, ok) - see ``llm._log_call``.
* ``config.json``     - the resolved ``PipelineConfig`` for this run.

``pipeline.log`` is produced by *teeing* ``sys.stdout``/``sys.stderr`` so we
capture all existing ``print(...)`` output to file without rewriting any stage.
"""

from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path
from typing import List, Optional, TextIO

from . import llm
from .config import PipelineConfig, REPO_ROOT


class _Tee(io.TextIOBase):
    """Write-through stream that forwards to several underlying streams."""

    def __init__(self, streams: List[TextIO]) -> None:
        self._streams = streams

    def write(self, s: str) -> int:  # type: ignore[override]
        for st in self._streams:
            st.write(s)
            st.flush()
        return len(s)

    def flush(self) -> None:  # type: ignore[override]
        for st in self._streams:
            st.flush()


class RunLogger:
    """Owns the per-run directory, the tee, and the log file handle."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self._log_fh: Optional[TextIO] = None
        self._orig_stdout: Optional[TextIO] = None
        self._orig_stderr: Optional[TextIO] = None

    def _start_tee(self) -> None:
        self._log_fh = open(self.run_dir / "pipeline.log", "a", encoding="utf-8")
        self._orig_stdout, self._orig_stderr = sys.stdout, sys.stderr
        sys.stdout = _Tee([self._orig_stdout, self._log_fh])
        sys.stderr = _Tee([self._orig_stderr, self._log_fh])

    def close(self) -> None:
        if self._orig_stdout is not None:
            sys.stdout = self._orig_stdout
        if self._orig_stderr is not None:
            sys.stderr = self._orig_stderr
        if self._log_fh is not None:
            self._log_fh.close()
            self._log_fh = None

    def __enter__(self) -> "RunLogger":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def init_run_logging(config: PipelineConfig, *, label: str = "run") -> RunLogger:
    """Create the per-run log directory, tee terminal output into it, and point
    the LLM API-call log at it. Returns a ``RunLogger`` (use as a context
    manager, or call ``.close()`` to restore stdout/stderr)."""
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(REPO_ROOT) / "synthesis" / "logs" / f"{label}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    logger = RunLogger(run_dir)
    logger._start_tee()

    # Route the structured API-call log into this run's directory.
    llm.set_log_file(run_dir / "api_calls.jsonl")

    # Snapshot the resolved config for reproducibility.
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config.to_dict(), f, indent=2)

    print(f"[pipeline] run logs → {run_dir}")
    return logger
