from __future__ import annotations

from pathlib import Path
from typing import Tuple, Optional, Callable, List, Dict

import torch
from text_scoring_adv_training.evaluation.robustness_tests.common.pointwise import make_pointwise_scorer
from text_scoring_adv_training.evaluation.robustness_tests.common.chat_spans import _token_span_by_substring

PROMPT_PREFIX_TMPL = "How relevant is the following document to the query?\n\nQuery: {q}\n\nDocument: "

def make_reranker_scorer(
    model_name: str,
    *,
    device: str = "cuda",
    batch_size: int = 256,
    models_dir: Optional[Path] = None,
    dtype = torch.bfloat16,
    use_flash_attention: bool = True,
):
    def _build_text(tok, q: str, d: str) -> str:
        return tok.apply_chat_template(
            [{"role":"user","content": PROMPT_PREFIX_TMPL.format(q=q) + d}],
            tokenize=False, add_generation_prompt=False, enable_thinking=False
        )
    def _locate_span(tok, q: str, d: str):
        prefix = PROMPT_PREFIX_TMPL.format(q=q)
        full_text = tok.apply_chat_template(
            [{"role":"user","content": prefix + d}],
            tokenize=False, add_generation_prompt=False, enable_thinking=False
        )
        s_char = full_text.rfind(d)
        if s_char == -1:
             return full_text, 0, 0 

        s, e = _token_span_by_substring(tok, full_text, d, find_last=True)
        
        return full_text, int(s), int(e)

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

def baseline_top_score_for_query(
    score_fn: Callable[[List[Tuple[str, str]]], torch.Tensor],
    query_text: str,
    run_full: List[Tuple[str, int, float]],
    corpus: Dict[str, str]
) -> Tuple[Optional[str], float]:
    """
    Returns (baseline_pid, top_score) for the first available ranked doc in corpus.
    """
    baseline_pid = next((pid for pid, _, _ in run_full if pid in corpus), None)
    if baseline_pid is None:
        return None, float("-inf")
    top_score = float(score_fn([(query_text, corpus[baseline_pid])]).tolist()[0])
    return baseline_pid, top_score

