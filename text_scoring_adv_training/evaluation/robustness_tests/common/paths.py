#!/usr/bin/env python
from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Paths:
    repo_root: Path
    data: Path
    checkpoints: Path

def _infer_repo_root() -> Path:
    here = Path(__file__).resolve()

    for p in here.parents:
        if (p / "text_scoring_adv_training").exists():
            return p
    return Path.cwd()

def _component_root(from_file: str) -> Path:
    return Path(from_file).resolve().parents[1]

def local_runs_dir(from_file: str) -> Path:
    env = os.getenv("EVAL_RUNS_DIR")
    if env:
        return Path(env)
    return _component_root(from_file) / "runs"

def local_eval_out_dir(from_file: str) -> Path:
    env = os.getenv("EVAL_OUT_DIR")
    if env:
        return Path(env)
    return _component_root(from_file) / "evaluation_outputs"

def _p(env: str, default: Path) -> Path:
    val = os.getenv(env, "")
    return Path(val) if val else default
    
_REPO = _infer_repo_root()
PATHS = Paths(
    repo_root=_REPO,
    data=_p("EVAL_DATA_DIR", _REPO / "data_files"),
    checkpoints=_p("EVAL_CHECKPOINTS_DIR", _REPO / "checkpoints"),
)
