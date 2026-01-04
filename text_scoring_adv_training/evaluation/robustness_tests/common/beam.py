from __future__ import annotations
from typing import Any, Callable, Iterable, List, Optional, Tuple

def _last_n_key_default(history: List[Tuple[int,int]] | List[str] | None, last_n: int) -> Optional[Tuple]:
    if not history or last_n is None or last_n <= 0:
        return None
    n = min(int(last_n), len(history))
    return tuple(history[-n:])

def prune_beams_diverse(
    candidates: List[Any],
    *,
    beam_size: int,
    last_n: int = 5,
    key_fn: Optional[Callable[[Any], Optional[Tuple]]] = None,
    pinned: Optional[Iterable[Any]] = None,
) -> List[Any]:
    if not candidates and not pinned:
        return []

    def score(b):
        return getattr(b, "score", getattr(b, "sim", float("-inf")))

    if key_fn is None:
        key_fn = lambda c: _last_n_key_default(getattr(c, "history_text", None), last_n)

    if pinned:
        merged = list(candidates)
        for b in pinned:
            if b not in merged:
                merged.append(b)
        candidates = merged

    used, out = set(), []

    for b in sorted(candidates, key=score, reverse=True):
        k = key_fn(b)
        if k is None or k not in used:
            if b not in out:
                out.append(b)
                if k is not None:
                    used.add(k)
                if len(out) >= beam_size:
                    out.sort(key=score, reverse=True)
                    return out

    for b in sorted(candidates, key=score, reverse=True):
        if len(out) >= beam_size:
            break
        if b in out:
            continue
        out.append(b)

    out.sort(key=score, reverse=True)
    return out

def select_next_beam(
    expansions: List[Any],
    prev_beam: Optional[Iterable[Any]],
    *,
    beam_size: int,
    last_n: int = 5,
    elitism_keep: int = 0,
) -> List[Any]:
    pinned = []
    if prev_beam and elitism_keep > 0:
        def _val(b):
            return getattr(b, "score", getattr(b, "sim", float("-inf")))
        pinned = sorted(prev_beam, key=_val, reverse=True)[:elitism_keep]

    return prune_beams_diverse(
        expansions,
        beam_size=beam_size,
        last_n=last_n,
        pinned=pinned or None,
    )