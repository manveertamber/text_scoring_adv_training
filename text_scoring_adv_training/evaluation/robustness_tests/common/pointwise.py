from __future__ import annotations
from typing import Callable, List, Tuple, Any, Optional, Set
import torch
from .base import resolve_model_source
from text_scoring_adv_training.models.pointwise_scorer import PointwiseScorer

BuildTextFn = Callable[[Any, str, str], str]
LocateSpanFn = Callable[[Any, str, str], Tuple[str, int, int]]

def _collect_specials(tok) -> Set[int]:
    sp = set(x for x in tok.all_special_ids if x is not None)
    for maybe in (tok.pad_token_id, tok.eos_token_id, tok.bos_token_id):
        if maybe is not None: sp.add(maybe)
    return sp

def make_pointwise_scorer(
    model_name: str,
    *,
    build_text: BuildTextFn,
    locate_span: Optional[LocateSpanFn] = None,
    device: str = "cuda",
    batch_size: int = 256,
    models_dir=None,
    dtype=torch.bfloat16,
    use_flash_attention: bool = True,
):
    src, _ = resolve_model_source(model_name, models_dir)
    scorer = PointwiseScorer(src, dtype=dtype, use_flash_attention=use_flash_attention)
    tok = scorer.tokenizer
    tok.padding_side = "left"
    dev = torch.device(device)
    scorer.model = scorer.model.to(dev).eval()
    specials = _collect_specials(tok)

    @torch.inference_mode()
    def score_fn(pairs: List[Tuple[str, str]]) -> torch.Tensor:
        if not pairs: return torch.empty(0)
        outs = []
        max_len = tok.model_max_length
        for i in range(0, len(pairs), batch_size):
            chunk = pairs[i:i+batch_size]
            texts = [build_text(tok, a, b) for a,b in chunk]
            enc = tok(texts, return_tensors="pt", add_special_tokens=False,
                      truncation=True, max_length=max_len, padding=True)
            enc = {k: v.to(dev) for k, v in enc.items()}
            logits = scorer(**enc)
            if isinstance(logits, (tuple, list)): logits = logits[0]
            if getattr(logits, "ndim", 1) > 1: logits = logits.squeeze(-1)
            outs.append(logits.detach().float().cpu())
        return torch.cat(outs, dim=0)

    def format_ids_fn(left: str, right: str):
        if locate_span is None:
            text = build_text(tok, left, right)
            enc = tok(text, return_tensors="pt", add_special_tokens=False,
                      truncation=True, max_length=tok.model_max_length)
            ids, att = enc["input_ids"][0].cpu(), enc["attention_mask"][0].cpu()
            s, e = 0, int(att.sum().item())
            return ids, att, s, e
        text, s, e = locate_span(tok, left, right)
        enc = tok(text, return_tensors="pt", add_special_tokens=False,
                  truncation=True, max_length=tok.model_max_length)
        return enc["input_ids"][0].cpu(), enc["attention_mask"][0].cpu(), s, e

    return tok, scorer, score_fn, format_ids_fn, specials
