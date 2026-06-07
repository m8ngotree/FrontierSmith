"""Shared utilities: C++ extraction, checkpoint I/O, and package writing."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .config import PipelineConfig, REPO_ROOT


# ---------------------------------------------------------------------------
# C++ extraction: reuse the exact implementation from the VERL reward module.
# We load that single file by path with importlib so we do NOT import the whole
# ``verl`` package (which pulls in ray/torch and is slow to initialize).
# ---------------------------------------------------------------------------
_FRONTIERCS_REWARD = REPO_ROOT / "verl" / "verl" / "utils" / "reward_score" / "frontiercs.py"
_spec = importlib.util.spec_from_file_location("_fs_frontiercs_reward", _FRONTIERCS_REWARD)
_mod = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
_spec.loader.exec_module(_mod)  # type: ignore[union-attr]

extract_cpp: Callable[[str], str] = _mod.extract_cpp
strip_think: Callable[[str], str] = _mod.strip_think


# ---------------------------------------------------------------------------
# JSON checkpointing
# ---------------------------------------------------------------------------
def checkpoint_path(config: PipelineConfig, stage_name: str) -> str:
    """Absolute path of the JSON checkpoint file for a given stage name."""
    return os.path.join(config.checkpoint_dir, f"{stage_name}.json")


def save_checkpoint(config: PipelineConfig, stage_name: str, data: Any) -> str:
    """Serialize ``data`` (any JSON-able structure) to the stage checkpoint."""
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    path = checkpoint_path(config, stage_name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return path


def load_checkpoint(config: PipelineConfig, stage_name: str) -> Optional[Any]:
    """Load a stage checkpoint if it exists, else return ``None``."""
    path = checkpoint_path(config, stage_name)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Problem-package writing (FrontierCS format, matching frontiersmith_1/)
# ---------------------------------------------------------------------------
def render_config_yaml(n_cases: int, *, time_s: int = 8, memory: str = "512m") -> str:
    """Build the config.yaml content for a default-type, custom-checker problem.

    Mirrors Frontier-CS/algorithmic/problems/frontiersmith_1/config.yaml:
        checker: chk.cc
        memory: 512m
        subtasks:
        - n_cases: <n>
          score: 100
        time: 8s
        type: default
    """
    return (
        "checker: chk.cc\n"
        f"memory: {memory}\n"
        "subtasks:\n"
        f"- n_cases: {n_cases}\n"
        "  score: 100\n"
        f"time: {time_s}s\n"
        "type: default\n"
    )


def write_problem_directory(
    out_dir: str,
    *,
    statement: str,
    checker_code: str,
    generator_code: str,
    test_inputs: List[str],
    time_s: int = 8,
    memory: str = "512m",
) -> str:
    """Write a complete FrontierCS problem package to ``out_dir``.

    Layout (no .ans files -- the custom checker scores from input + output):
        out_dir/statement.txt
        out_dir/config.yaml
        out_dir/chk.cc
        out_dir/gen.cpp
        out_dir/testdata/1.in ... N.in
    Returns the path to ``out_dir``.
    """
    out = Path(out_dir)
    testdata = out / "testdata"
    testdata.mkdir(parents=True, exist_ok=True)

    (out / "statement.txt").write_text(statement, encoding="utf-8")
    (out / "config.yaml").write_text(
        render_config_yaml(len(test_inputs), time_s=time_s, memory=memory),
        encoding="utf-8",
    )
    (out / "chk.cc").write_text(checker_code, encoding="utf-8")
    (out / "gen.cpp").write_text(generator_code, encoding="utf-8")

    for i, content in enumerate(test_inputs, start=1):
        # Ensure a trailing newline, as most testlib readers expect one.
        text = content if content.endswith("\n") else content + "\n"
        (testdata / f"{i}.in").write_text(text, encoding="utf-8")

    return str(out)


# ---------------------------------------------------------------------------
# Robust JSON parsing of LLM responses
# ---------------------------------------------------------------------------
def parse_json_response(text: str) -> Optional[Dict[str, Any]]:
    """Best-effort extraction of a single JSON object from an LLM response.

    Handles responses that wrap JSON in ```json fences or include surrounding
    prose. Returns ``None`` if no parseable object is found.
    """
    if not text:
        return None
    text = strip_think(text).strip()

    # Try fenced ```json ... ``` first.
    import re
    fence = re.search(r"```(?:json)?\s*\n(.*?)```", text, re.DOTALL)
    if fence:
        candidate = fence.group(1).strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    # Fall back to the outermost {...} span.
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    return None
