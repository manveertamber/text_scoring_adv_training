#!/usr/bin/env python
from __future__ import annotations
import random
from typing import Dict, List, Tuple, Optional
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

def load_mlm(model_name: str = "answerdotai/ModernBERT-base",
             *, device: str = "cuda", dtype=torch.bfloat16):
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForMaskedLM.from_pretrained(model_name, torch_dtype=dtype).to(device).eval()
    specials = set(tok.all_special_ids)
    return tok, mdl, tok.mask_token_id, specials

@torch.inference_mode()
def build_replacement_map(
    ids: torch.Tensor,
    mask_id: int,
    specials: set,
    mlm_model,
    *,
    batch_size: int = 256,
    top_k: int = 64,
    prob_min: float = 1e-3,
    positions: Optional[List[int]] = None,
) -> Dict[int, List[int]]:
    ids_list = ids.tolist()
    if positions is None:
        pos = [i for i, tid in enumerate(ids_list) if tid not in specials]
    else:
        pos = [
            int(p)
            for p in positions
            if 0 <= int(p) < len(ids_list) and ids_list[int(p)] not in specials
        ]
    if not pos:
        return {}

    repl: Dict[int, List[int]] = {}
    for s in range(0, len(pos), batch_size):
        sub = pos[s : s + batch_size]
        templ = ids.repeat(len(sub), 1)
        row = torch.arange(len(sub), device=ids.device)
        templ[row, sub] = mask_id
        logits = mlm_model(templ).logits
        masked = logits[row, sub]
        top_v, top_i = torch.topk(masked, top_k)
        logZ = torch.logsumexp(masked, dim=-1, keepdim=True)
        probs = (top_v - logZ).exp()
        for r, p in enumerate(sub):
            keep = probs[r] >= prob_min
            cand = [
                int(x)
                for x in top_i[r][keep].tolist()
                if x not in specials and x != int(ids[p])
            ]
            if cand:
                repl[p] = cand
    return repl


@torch.inference_mode()
def enumerate_swaps_dynamic(
    mlm_tok,
    ids_cur: torch.Tensor,
    mask_id: int,
    specials: set,
    mlm_model,
    *,
    num_positions: int,
    limit: int,
    batch_size: int = 256,
    top_k: int = 64,
    prob_min: float = 1e-3,
    allowed_positions: Optional[List[int]] = None,
) -> List[Tuple[str, torch.Tensor]]:
    if num_positions <= 0 or limit <= 0:
        return []

    ids_dev = ids_cur.to(mlm_model.device)
    ids_list = ids_dev.tolist()

    if allowed_positions is not None and len(allowed_positions) > 0:
        cand_pos = [
            int(p)
            for p in allowed_positions
            if 0 <= int(p) < len(ids_list) and ids_list[int(p)] not in specials
        ]
    else:
        cand_pos = [i for i, tid in enumerate(ids_list) if tid not in specials]

    if not cand_pos:
        return []

    if len(cand_pos) <= num_positions:
        sample_pos = cand_pos
    else:
        sample_pos = random.sample(cand_pos, num_positions)

    repl_map = build_replacement_map(
        ids_dev,
        mask_id,
        specials,
        mlm_model,
        batch_size=batch_size,
        top_k=top_k,
        prob_min=prob_min,
        positions=sample_pos,
    )
    if not repl_map:
        return []

    pairs = [
        (pos, cid)
        for pos, cand_ids in repl_map.items()
        for cid in cand_ids
        if ids_list[pos] != cid
    ]
    random.shuffle(pairs)

    out: List[Tuple[str, torch.Tensor]] = []
    for pos, cid in pairs[:limit]:
        ids_new = ids_dev.clone()
        ids_new[pos] = cid
        t = mlm_tok.decode(
            ids_new,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        out.append((t, ids_new.cpu()))
    return out
