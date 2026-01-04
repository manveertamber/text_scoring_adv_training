#!/usr/bin/env python
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Dict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from text_scoring_adv_training.evaluation.robustness_tests.common import resolve_model_source
from text_scoring_adv_training.evaluation.robustness_tests.common.chat_spans import format_user_assistant_spans

from pathlib import Path


def load_targets_jsonl(path: Path) -> List[Dict[str, Any]]:

    out: List[Dict[str, Any]] = []
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"targets jsonl not found: {p}")

    with p.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                rec = json.loads(ln)
            except json.JSONDecodeError:
                continue

            pr = rec.get("prompt")
            if not isinstance(pr, str):
                continue

            raw_targets = rec.get("targets", rec.get("target"))
            if isinstance(raw_targets, str):
                targets = [raw_targets]
            elif isinstance(raw_targets, (list, tuple)):
                targets = [t for t in raw_targets if isinstance(t, str) and t.strip()]
            else:
                targets = []

            if not targets:
                continue

            out.append({
                "id": rec.get("id"),
                "prompt": pr,
                "targets": targets,
                "prompt_frozen": rec.get("prompt_frozen") or "",
            })
    return out



@dataclass
class GenModel:
    tokenizer: Any
    model: Any
    specials: set

def make_generative_model(
    model_name: str,
    *,
    device: str = "cuda",
    models_dir: Optional[Path] = None,
    dtype = torch.bfloat16,
    use_flash_attention: bool = True,
) -> GenModel:
    src, _ = resolve_model_source(model_name, models_dir)
    tok = AutoTokenizer.from_pretrained(src)

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    tok.padding_side = "left"

    n_gpus = torch.cuda.device_count()

    if device.startswith("cuda") and n_gpus > 1:
        mdl = AutoModelForCausalLM.from_pretrained(
            src, torch_dtype=dtype, attn_implementation=("flash_attention_2" if use_flash_attention else "eager"), device_map="auto"
        ).eval()
    else:
        mdl = AutoModelForCausalLM.from_pretrained(
            src,
            torch_dtype=dtype,
            attn_implementation=("flash_attention_2" if use_flash_attention else "eager"),
        ).to(device).eval()

    specials = set(x for x in tok.all_special_ids if x is not None)
    for maybe in (getattr(tok, "pad_token_id", None), getattr(tok, "eos_token_id", None), getattr(tok, "bos_token_id", None)):
        if maybe is not None: specials.add(maybe)
    return GenModel(tok, mdl, specials)

@torch.inference_mode()
def score_target_logprob(
    gm: GenModel,
    pairs: List[Tuple[str, Any]],
    *,
    reduction: str = "sum",
    batch_size: int = 64,
) -> torch.Tensor:

    if not pairs:
        return torch.empty(0)
    tok, mdl = gm.tokenizer, gm.model
    dev = mdl.device
    neg_big = torch.finfo(torch.float32).min
    
    flat: List[Tuple[str, str, int]] = []
    for idx, (prompt, tg) in enumerate(pairs):
        if not isinstance(prompt, str):
            continue
        if isinstance(tg, str):
            tgt_list = [tg]
        elif isinstance(tg, (list, tuple)):
            tgt_list = [t for t in tg if isinstance(t, str) and t.strip()]
        else:
            tgt_list = []
        if not tgt_list:
            continue
        for t in tgt_list:
            flat.append((prompt, t, idx))
            
    if not flat:
        return torch.full((len(pairs),), neg_big, dtype=torch.float32)
        
    flat_scores: List[float] = []
    flat_owners: List[int] = []
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
    
    for i in range(0, len(flat), batch_size):
        chunk = flat[i:i + batch_size]
        texts = []
        labels_list = []
        owners_batch: List[int] = []
        
        for prompt, target, owner in chunk:
            full_text, (_, _), (s_a, e_a) = format_user_assistant_spans(tok, prompt, target)
            enc = tok(
                full_text,
                add_special_tokens=False,
                truncation=True,
                max_length=4096,
                return_tensors=None,
            )
            ids = torch.tensor(enc["input_ids"], dtype=torch.long)
            labels = ids.clone()
            labels[:s_a] = -100
            if e_a < labels.numel():
                labels[e_a:] = -100

            texts.append(full_text)
            labels_list.append(labels)
            owners_batch.append(owner)

        enc_batch = tok(
            texts,
            add_special_tokens=False,
            truncation=True,
            max_length=4096,
            padding=True,
            return_tensors="pt",
        )
        pad_ids = enc_batch["input_ids"]
        pad_att = enc_batch["attention_mask"]

        max_len = pad_ids.shape[1]
        padded_labels = []
        for lab in labels_list:
            n_pad = max_len - len(lab)
            pads = torch.full((n_pad,), -100, dtype=torch.long, device=dev)
            padded_labels.append(torch.cat([pads, lab.to(dev)]))

        pad_lab = torch.stack(padded_labels)

        pad_ids = pad_ids.to(dev)
        pad_att = pad_att.to(dev)
        pad_lab = pad_lab.to(dev)

        
        out = mdl(input_ids=pad_ids, attention_mask=pad_att, labels=pad_lab)
        logits = out.logits
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = pad_lab[:, 1:].contiguous()
        
        token_losses = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
        token_losses = token_losses.view(shift_labels.shape)
        mask = (shift_labels != -100)
        per_ex_sum = (token_losses * mask).sum(dim=1)
        per_ex_cnt = mask.sum(dim=1).clamp_min(1)
        valid = (mask.sum(dim=1) > 0)
        
        if reduction == "sum":
            scores = -per_ex_sum
        else:
            scores = -(per_ex_sum / per_ex_cnt)
            
        scores = scores.detach().float().cpu()
        valid = valid.detach().cpu()
        
        for sc, owner, ok in zip(scores.tolist(), owners_batch, valid.tolist()):
            flat_scores.append(sc if ok else neg_big)
            flat_owners.append(owner)
            
    num_pairs = len(pairs)
    out_scores = torch.full((num_pairs,), neg_big, dtype=torch.float32)
    sum_scores = [0.0] * num_pairs
    counts = [0] * num_pairs
    for sc, owner in zip(flat_scores, flat_owners):
        sum_scores[owner] += sc
        counts[owner] += 1
    for i in range(num_pairs):
        if counts[i] > 0:
            out_scores[i] = sum_scores[i] / float(counts[i])
    return out_scores

@torch.inference_mode()
def generate_and_check(
    gm: GenModel,
    prompt: str,
    targets,
    *,
    max_new_tokens: Optional[int] = None,
    temperature: float = 0.0,
    top_p: float = 1.0,
    match_mode: str = "text_prefix",
) -> Tuple[bool, str]:

    tok, mdl = gm.tokenizer, gm.model
    dev = mdl.device

    if isinstance(targets, str):
        targets_list = [targets]
    else:
        targets_list = [t for t in (targets or []) if isinstance(t, str) and t.strip()]

    if not targets_list:
        return False, ""

    chat = [{"role": "user", "content": prompt}, {"role": "assistant", "content": ""}]
    
    inp = tok.apply_chat_template(
        chat, tokenize=True, add_generation_prompt=True, return_tensors="pt", 
        enable_thinking=False 
    )

    if isinstance(inp, torch.Tensor):
        model_inputs = {
            "input_ids": inp.to(dev),
            "attention_mask": torch.ones_like(inp, device=dev),
        }
    else:
        model_inputs = {k: v.to(dev) for k, v in inp.items()}


    if max_new_tokens is None:
        max_target_len = 0
        for t in targets_list:
            t_ids = tok(t, add_special_tokens=False)["input_ids"]
            max_target_len = max(max_target_len, len(t_ids))
        
        max_new_tokens = max_target_len + 5

    gen_kwargs = dict(
        max_new_tokens=int(max_new_tokens),
        do_sample=(temperature > 0.0),
        pad_token_id=(tok.eos_token_id if tok.eos_token_id is not None else tok.pad_token_id),
        eos_token_id=tok.eos_token_id,
    )
    if temperature > 0.0:
        gen_kwargs["temperature"] = float(temperature)
        gen_kwargs["top_p"] = float(top_p)

    gen = mdl.generate(**model_inputs, **gen_kwargs)
    
    out_ids = gen[0][model_inputs["input_ids"].shape[1]:]
    
    gen_text = tok.decode(out_ids, skip_special_tokens=True).strip()

    if match_mode == "text_prefix":
        for t in targets_list:
            if gen_text.startswith(t.strip()):
                return True, gen_text
                
    elif match_mode == "exact":
        for t in targets_list:
            if gen_text == t.strip():
                return True, gen_text

    return False, gen_text


@torch.inference_mode()
def generate_text(
    gm: GenModel,
    prompt: str,
    *,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    top_p: float = 1.0,
) -> str:
    tok, mdl = gm.tokenizer, gm.model
    dev = mdl.device
    chat = [{"role": "user", "content": prompt}, {"role": "assistant", "content": ""}]
    inp = tok.apply_chat_template(
        chat, tokenize=True, add_generation_prompt=True, return_tensors="pt", enable_thinking=False
    )
    if isinstance(inp, torch.Tensor):
        model_inputs = {
            "input_ids": inp.to(dev),
            "attention_mask": torch.ones_like(inp, device=dev),
        }
    else:
        model_inputs = {k: (v.to(dev) if isinstance(v, torch.Tensor) else v)
                        for k, v in dict(inp).items()}
    gen_kwargs = dict(
        max_new_tokens=int(max_new_tokens),
        do_sample=(temperature > 0.0),
        pad_token_id=(tok.eos_token_id if tok.eos_token_id is not None else tok.pad_token_id),
        eos_token_id=tok.eos_token_id,
        temperature=float(temperature) if temperature > 0.0 else None,
        top_p=float(top_p) if temperature > 0.0 else None,
    )
    gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}
    gen = mdl.generate(**model_inputs, **gen_kwargs)
    out_ids = gen[0][model_inputs["input_ids"].shape[1]:].tolist()
    return tok.decode(out_ids, skip_special_tokens=True)

