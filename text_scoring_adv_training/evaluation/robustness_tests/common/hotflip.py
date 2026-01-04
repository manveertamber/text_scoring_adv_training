#!/usr/bin/env python
from __future__ import annotations
import heapq, random
from typing import Any, Dict, List, Optional, Tuple
import torch

def _sample_positions(ids_1d: torch.Tensor, span: Tuple[int,int], specials: set) -> List[int]:
    s, e = span
    pos = []
    for i in range(s, e):
        tid = int(ids_1d[i])
        if specials and tid in specials:
            continue
        pos.append(i)
    return pos

@torch.no_grad()
def _topk_per_position(E: torch.Tensor, grad_vec: torch.Tensor, old_id: int, specials: set, k: int) -> Tuple[List[int], List[float]]:
    V = E.shape[0]
    base = torch.dot(grad_vec, E[old_id])
    gains = -(torch.mv(E, grad_vec) - base)
    if specials:
        gains[list(specials)] = float("-inf")
    if 0 <= old_id < V:
        gains[old_id] = float("-inf")
    k = min(k, V)
    top_vals, top_idx = torch.topk(gains, k)
    return top_idx.tolist(), top_vals.tolist()

def hotflip_causal_lm(
    *,
    ids: torch.Tensor,
    att: torch.Tensor,
    edit_span: Tuple[int,int],
    asst_span: Tuple[int,int],
    model, tokenizer,
    specials: set,
    n_sample_pos: int,
    top_k_overall: int,
    device: str,
) -> Tuple[List[torch.Tensor], List[Dict], Dict]:
    ids2 = ids.to(device).unsqueeze(0)
    att2 = att.to(device).unsqueeze(0)
    emb = model.get_input_embeddings()
    E = emb.weight.detach().to(torch.float32)
    s_a, e_a = asst_span

    with torch.enable_grad():
        emb_in = emb(ids2).detach().clone().requires_grad_(True)
        labels = ids2.clone()
        labels[:, :s_a] = -100
        if e_a < labels.shape[1]:
            labels[:, e_a:] = -100
        out = model(inputs_embeds=emb_in, attention_mask=att2, labels=labels)
        loss = out.loss
        (grad,) = torch.autograd.grad(loss, emb_in, retain_graph=False)
    g = grad.squeeze(0).to(torch.float32)
    ids1 = ids2.squeeze(0)

    positions = _sample_positions(ids1, edit_span, specials)
    if not positions:
        return [], [], {"best_pos": -1, "best_old_id": -1, "sampled_positions": []}

    kpos = min(max(1, int(n_sample_pos)), len(positions))
    sampled = random.sample(positions, kpos)

    pooled: List[Dict[str, Any]] = []
    for pos in sampled:
        old = int(ids1[pos])
        cand_ids, cand_gains = _topk_per_position(E, g[pos], old, specials, min(top_k_overall, E.shape[0]))
        for nid, gain in zip(cand_ids, cand_gains):
            if gain == float("-inf"):
                continue
            pooled.append({"pos": pos, "old_id": old, "new_id": int(nid), "gain": float(gain)})

    if not pooled:
        return [], [], {"best_pos": -1, "best_old_id": -1, "sampled_positions": sampled}

    top_global = heapq.nlargest(min(top_k_overall, len(pooled)), pooled, key=lambda x: x["gain"])
    out_ids, swaps = [], []
    for cand in top_global:
        t = ids1.clone(); t[cand["pos"]] = cand["new_id"]
        out_ids.append(t.detach().cpu()); swaps.append(cand)

    meta = {"best_pos": top_global[0]["pos"], "best_old_id": top_global[0]["old_id"], "sampled_positions": sampled}
    return out_ids, swaps, meta

def hotflip_pointwise(
    *,
    ids: torch.Tensor,
    att: torch.Tensor,
    span: Tuple[int,int],
    scorer,
    specials: set,
    n_sample_pos: int,
    top_k_overall: int,
    token_embeddings,
    device: str,
) -> Tuple[List[torch.Tensor], List[Dict], Dict]:
    ids2 = ids.to(device).unsqueeze(0)
    att2 = att.to(device).unsqueeze(0)
    E = token_embeddings.weight.detach().to(torch.float32)

    with torch.enable_grad():
        emb_in = token_embeddings(ids2).detach().clone().requires_grad_(True)
        logits = scorer(inputs_embeds=emb_in, attention_mask=att2)
        if isinstance(logits, (tuple, list)): logits = logits[0]
        if getattr(logits, "ndim", 1) > 1: logits = logits.squeeze(-1)
        (grad,) = torch.autograd.grad(logits.sum(), emb_in, retain_graph=False)
    g = grad.squeeze(0).to(torch.float32)
    ids1 = ids2.squeeze(0)

    positions = _sample_positions(ids1, span, specials)
    if not positions:
        return [], [], {"best_pos": -1, "best_old_id": -1, "sampled_positions": []}

    kpos = min(max(1, int(n_sample_pos)), len(positions))
    sampled = random.sample(positions, kpos)

    pooled: List[Dict] = []
    for pos in sampled:
        old = int(ids1[pos])
        cand_ids, cand_gains = _topk_per_position(
            E, -g[pos], old, specials, min(top_k_overall, E.shape[0])
        )
        for nid, gain in zip(cand_ids, cand_gains):
            if gain == float("-inf"): continue
            pooled.append({"pos": pos, "old_id": old, "new_id": int(nid), "gain": float(gain)})

    if not pooled:
        return [], [], {"best_pos": -1, "best_old_id": -1, "sampled_positions": sampled}

    top_global = heapq.nlargest(min(top_k_overall, len(pooled)), pooled, key=lambda x: x["gain"])
    out_ids, swaps = [], []
    for cand in top_global:
        t = ids1.clone(); t[cand["pos"]] = cand["new_id"]
        out_ids.append(t.detach().cpu()); swaps.append(cand)
    meta = {"best_pos": top_global[0]["pos"], "best_old_id": top_global[0]["old_id"], "sampled_positions": sampled}
    return out_ids, swaps, meta
