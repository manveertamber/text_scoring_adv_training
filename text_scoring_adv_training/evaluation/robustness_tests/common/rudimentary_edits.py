from __future__ import annotations
import random, string
from typing import List

LETTER_CHARS = string.ascii_letters
DIGIT_CHARS  = string.digits
PUNC_CHARS   = string.punctuation

def _apply_rudimentary(text: str) -> str:
    if not text.strip(): return text
    
    if random.choice([True, False]):
        words = text.split(' ')
        if not words: 
            return text
        i = random.randint(0, len(words)-1)
        ops = ["repeat"]
        if len(words) > 1: ops.append("delete")
        if i < len(words)-1 and words[i] != words[i+1]: ops.append("swap")
        op = random.choice(ops)
        if   op == "delete": words.pop(i)
        elif op == "swap":   words[i], words[i+1] = words[i+1], words[i]
        else:                words.insert(i+1, words[i])
        return " ".join(words)
    
    t = list(text)
    if not t: return text
    i = random.randint(0, len(t)-1)
    pool = LETTER_CHARS + DIGIT_CHARS + PUNC_CHARS
    ops = ["sub","ins"]
    if len(t)>1: ops.append("del")
    if i < len(t)-1 and t[i] != t[i+1]: ops.append("swap")
    op = random.choice(ops)
    if   op == "sub":
        c = random.choice(pool)
        while c == t[i]: c = random.choice(pool)
        t[i] = c
    elif op == "ins": t.insert(i, random.choice(pool))
    elif op == "del": t.pop(i)
    else:             t[i], t[i+1] = t[i+1], t[i]
    return "".join(t)

def sample_variants(text: str, max_variants: int) -> List[str]:
    seen, out = {text}, []
    attempts = max_variants * 4 if max_variants else 0
    for _ in range(attempts):
        t = _apply_rudimentary(text)
        if t not in seen:
            seen.add(t); out.append(t)
            if len(out) >= max_variants: break
    return out

def sample_variants_with_frozen(full_text: str, max_variants: int, frozen_prefix: str) -> List[str]:
    if frozen_prefix and full_text.startswith(frozen_prefix):
        prefix = frozen_prefix
        editable = full_text[len(prefix):]
    else:
        prefix, editable = "", full_text
    base = full_text
    seen, out = {base}, []
    attempts = max_variants * 4 if max_variants else 0
    for _ in range(attempts):
        t = prefix + _apply_rudimentary(editable)
        if t not in seen:
            seen.add(t); out.append(t)
            if len(out) >= max_variants: break
    return out
