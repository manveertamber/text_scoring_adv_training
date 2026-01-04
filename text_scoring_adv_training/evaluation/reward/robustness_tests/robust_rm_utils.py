from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Any, TypedDict

import torch
from text_scoring_adv_training.evaluation.robustness_tests.common.pointwise import make_pointwise_scorer
from text_scoring_adv_training.evaluation.robustness_tests.common.chat_spans import _token_span_by_substring

def load_reward_bench_jsonl(path: Path, max_items=None) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            if not ln.strip():
                continue
            rec = json.loads(ln)
            prompt = rec["prompt"]
            chosen = rec["chosen"]
            if isinstance(chosen, list):
                if not chosen:
                    continue
                chosen = chosen[0]
            rejected = list(rec.get("rejected", []))
            out.append({
                "id": rec.get("id"),
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
            })

    if max_items is not None:
        out = out[:max_items]

    return out

def _listify(x) -> List[str]:
    if x is None:
        return []
    if isinstance(x, str):
        return [x]
    if isinstance(x, list):
        out = []
        for y in x:
            if isinstance(y, str):
                out.append(y)
            elif isinstance(y, list):
                out.extend([z for z in y if isinstance(z, str)])
        return out
    return []

class RewardInj(TypedDict):
    chosen: List[str]
    rejected: List[List[str]]
    
def load_reward_injections_jsonl(path: Path) -> Dict[str, RewardInj]:
    inj: Dict[str, Dict[str, List[str]]] = {}
    with Path(path).open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            rec = json.loads(line)
            rid = str(rec["id"])
            chosen_variants = _listify(rec.get("chosen"))
            rejected_variants = rec.get("rejected")
            if isinstance(rejected_variants, list):
                rejected_variants = [ _listify(x) for x in rejected_variants ]
            else:
                rejected_variants = []
            inj[rid] = {"chosen": chosen_variants, "rejected": rejected_variants}
    return inj

def make_reward_scorer(
    model_name: str,
    *,
    device: str = "cuda",
    batch_size: int = 256,
    models_dir: Optional[Path] = None,
    dtype = torch.bfloat16,
    use_flash_attention: bool = True,
):
    def _build_text(tok, prompt: str, resp: str) -> str:
        return tok.apply_chat_template(
            [{"role":"user","content":prompt},{"role":"assistant","content":resp}],
            tokenize=False, add_generation_prompt=False, enable_thinking=False
        )
    def _locate_span(tok, prompt: str, resp: str):
        full = _build_text(tok, prompt, resp)
        
        s, e = _token_span_by_substring(tok, full, resp, find_last=True)
        
        return full, int(s), int(e)

    return make_pointwise_scorer(
        model_name,
        build_text=_build_text,
        locate_span=_locate_span,
        device=device,
        batch_size=batch_size,
        models_dir=models_dir,
        dtype=dtype,
        use_flash_attention=use_flash_attention,
    )
