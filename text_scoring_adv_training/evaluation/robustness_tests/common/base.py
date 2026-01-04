#!/usr/bin/env python
from __future__ import annotations
import json, random
from pathlib import Path
from typing import Any, List, Optional, Tuple
import torch
from .paths import PATHS
from dataclasses import dataclass, asdict
import argparse

def seed_everything(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def log_result(jsonl_path: Path, sample_id: str, success: bool, edits: int):
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"sample_id": sample_id,
                            "success": bool(success),
                            "edits": int(edits)}) + "\n")

def log_injection_trial(
    jsonl_path: Path,
    sample_id: str,
    trial_id: int,
    success: bool,
    num_injections: int | None = None,
):
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "sample_id": sample_id,
        "trial_id": int(trial_id),
        "success": bool(success),
    }
    if num_injections is not None:
        record["num_injections"] = int(num_injections)
    with jsonl_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

def load_existing_results(
    jsonl_path: Path,
    *,
    stats: Optional["RunningEditStats"] = None,
) -> set[str]:
    done: set[str] = set()
    if not jsonl_path.exists():
        return done

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue

            sid = rec.get("sample_id")
            if not sid:
                continue
            sid_str = str(sid)
            done.add(sid_str)

            if stats is None:
                continue

            if "success" in rec and "edits" in rec:
                try:
                    success = bool(rec["success"])
                    edits = int(rec["edits"])
                except Exception:
                    continue

                if success:
                    stats.record(True, edits)
                else:
                    stats.record(False, 0)

    return done

def build_stats_path(out_dir: Path, script_slug: str, model_name: str, dataset: str, run_tag: str) -> Path:
    return out_dir / f"{script_slug}.{model_name}.{dataset}.{run_tag}.jsonl"

def dataset_slug_from_file(p: Path | str, default: str = "targets") -> str:
    try:
        return Path(p).stem or default
    except Exception:
        return default

def progress_line(msg: str):
    try:
        from tqdm import tqdm as _tqdm
        _tqdm.write(msg)
    except Exception:
        print(msg, flush=True)

def short_text(s: str, n: int = 120) -> str:
    if not isinstance(s, str):
        return "<non-str>"
    return s if len(s) <= n else (s[: max(0, n - 1)] + "…")

def resolve_model_source(model_name: str, models_dir: Optional[Path] = None) -> Tuple[str, bool]:
    p = Path(model_name)
    if p.exists() and p.is_dir():
        return str(p), True
    if models_dir is not None:
        p2 = (models_dir / model_name)
        if p2.exists() and p2.is_dir():
            return str(p2), True
    return model_name, False

def decode_token(tok, tid: int) -> str:
    try:
        return tok.decode([tid], skip_special_tokens=False)
    except Exception:
        return f"<{tid}>"

def decode_span_text(ids: torch.Tensor, s: int, e: int, tok) -> str:
    try:
        return tok.decode(ids[s:e].tolist(), skip_special_tokens=True)
    except Exception:
        try:
            return "".join(decode_token(tok, int(t)) for t in ids[s:e])
        except Exception:
            return "<decode_failed>"

class RunningEditStats:
    def __init__(self, max_steps: int, success_phrase: str = "succeed"):
        self.max_steps = max_steps
        self.success_phrase = success_phrase
        self.attempts  = 0
        self.successes = 0
        self.edit_counts_success: List[int] = []
        self.edit_counts_all: List[int] = []

    def record(self, success: bool, edits_if_success: int):
        self.attempts += 1
        if success:
            self.successes += 1
            self.edit_counts_success.append(edits_if_success)
            self.edit_counts_all.append(edits_if_success)
        else:
            self.edit_counts_all.append(self.max_steps)

    def summary_strings(self, model_name: str) -> List[str]:
        sr = (self.successes / self.attempts * 100.0) if self.attempts else 0.0
        outs = []
        if self.edit_counts_success:
            avg_edits = sum(self.edit_counts_success) / len(self.edit_counts_success)
            outs.append(f"→ AVERAGE EDITS to {self.success_phrase} for {model_name}: {avg_edits:.2f} "
                        f"(successes: {self.successes}/{self.attempts}, max_steps={self.max_steps})")
        else:
            outs.append(f"→ No successes for {model_name} within {self.max_steps} edits.")
        if self.attempts:
            avg_with_fail = sum(self.edit_counts_all) / self.attempts
            outs.append(f"→ AVERAGE EDITS (counting failures as max_steps): {avg_with_fail:.2f} "
                        f"(attempts: {self.attempts}, max_steps={self.max_steps})")
        outs.append(f"→ SUCCESS RATE for {model_name}: {self.successes}/{self.attempts} = {sr:.2f}% "
                    f"(max_steps={self.max_steps})")
        return outs


def enable_tf32() -> None:
    torch.backends.cuda.matmul.allow_tf32 = True

def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"

def slug_for_file(__file__: str) -> str:
    return Path(__file__).stem

def override_models_from_cli(cfg: "ExperimentConfig") -> "ExperimentConfig":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--models",
        nargs="+",
        metavar="MODEL",
        help="Override list of model names to evaluate for this script.",
    )
    args, _ = parser.parse_known_args()

    if getattr(args, "models", None):
        cfg.models = list(args.models)

@dataclass
class ExperimentConfig:
    models: list[str]
    models_dir: Path = PATHS.checkpoints
    device: str = default_device()
    batch_size: int = 256
    seed: int = 42
    max_steps: int = 512
    num_beams: int = 16
    test_variants_per_beam: int = 16
    diversity_last_n: int = 3
    elitism_keep: int = 8
    out_dir: Path = PATHS.repo_root / "evaluation/evaluation_outputs/per_sample_stats"
    run_tag: str = "test"
    script_slug: Optional[str] = None

    def materialize(self, __file__: str) -> "ExperimentConfig":
        if not self.script_slug:
            self.script_slug = slug_for_file(__file__)
        seed_everything(self.seed)
        enable_tf32()
        return self

    def stats_path(self, model_name: str, dataset_slug: str) -> Path:
        return build_stats_path(self.out_dir, self.script_slug or "script", model_name, dataset_slug, self.run_tag)

    def to_json(self) -> str:
        def _ser(x: Any) -> Any:
            return str(x) if isinstance(x, Path) else x
        return json.dumps({k:_ser(v) for k,v in asdict(self).items()}, indent=2)