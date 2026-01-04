#!/usr/bin/env python
from __future__ import annotations
from typing import Optional, Tuple, List
from transformers import PreTrainedTokenizerBase
import torch

def _token_span_by_substring(
    tok: PreTrainedTokenizerBase,
    full_text: str,
    sub: str,
    *,
    find_last: bool = False,
) -> Tuple[int, int]:

    if not sub:
        return 0, 0

    try:
        enc_full = tok(
            full_text,
            return_tensors="pt",
            add_special_tokens=False,
            truncation=True,
            max_length=4096,
            return_offsets_mapping=True,
        )
    except Exception:
        return 0, 0

    full_ids = enc_full["input_ids"][0]

    def find_subarray(main_arr, sub_arr):
        n = len(main_arr)
        m = len(sub_arr)
        if m == 0: return []
        sub_t = torch.tensor(sub_arr, device=main_arr.device)
        occurrences = []
        for i in range(n - m + 1):
            if torch.equal(main_arr[i : i + m], sub_t):
                occurrences.append((i, i + m))
        return occurrences

    sub_ids = tok(sub, add_special_tokens=False)["input_ids"]
    matches = find_subarray(full_ids, sub_ids)

    if not matches and not sub.startswith(" "):
        sub_ids_space = tok(" " + sub, add_special_tokens=False)["input_ids"]
        matches = find_subarray(full_ids, sub_ids_space)

    if matches:
        if find_last:
            return matches[-1]
        return matches[0]
    try:
        if find_last:
            start_char = full_text.rfind(sub)
        else:
            start_char = full_text.find(sub)
        if start_char == -1:
            return 0, 0
        end_char = start_char + len(sub)
    except Exception:
        return 0, 0

    if "offset_mapping" not in enc_full:
        return 0, len(full_ids)
    offs = enc_full["offset_mapping"][0].tolist()
    touched = [
        i for i, (a, b) in enumerate(offs)
        if not (b <= start_char or a >= end_char)
    ]
    if not touched:
        return 0, 0
    return int(touched[0]), int(touched[-1] + 1)


def format_user_assistant_spans(tok: PreTrainedTokenizerBase, prompt: str, target: str) -> Tuple[str, Tuple[int,int], Tuple[int,int]]:
    full_text = tok.apply_chat_template(
        [{"role": "user", "content": prompt},
         {"role": "assistant", "content": target}],
        tokenize=False, add_generation_prompt=False, enable_thinking=False
    )
    s_u, e_u = _token_span_by_substring(tok, full_text, prompt, find_last=False)
    s_a, e_a = _token_span_by_substring(tok, full_text, target, find_last=True)
    return full_text, (s_u, e_u), (s_a, e_a)

def compute_spans_with_frozen(
    tok: PreTrainedTokenizerBase, prompt: str, target: str, prompt_frozen: Optional[str] = None
) -> Tuple[str, Tuple[int,int], Tuple[int,int], Tuple[int,int]]:
    full_text, user_span, asst_span = format_user_assistant_spans(tok, prompt, target)
    s_u, e_u = user_span
    edit_span = (s_u, e_u)
    if isinstance(prompt_frozen, str) and prompt_frozen and prompt.startswith(prompt_frozen):
        s_f, e_f = _token_span_by_substring(tok, full_text, prompt_frozen)
        e_f = min(max(e_f, s_u), e_u)
        edit_span = (e_f, e_u)
    return full_text, user_span, asst_span, edit_span
