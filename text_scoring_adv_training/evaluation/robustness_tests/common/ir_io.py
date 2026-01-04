#!/usr/bin/env python
from __future__ import annotations

import json
import collections
from pathlib import Path
from typing import Dict, List, Tuple, Set
import random

def load_queries_tsv(path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    with Path(path).open(encoding="utf-8") as fh:
        for raw in fh:
            ln = raw.rstrip("\n")
            if not ln or ln.startswith("#"):
                continue
            parts = ln.split("\t", 1)
            if len(parts) != 2:
                continue
            qid, q = parts
            out[str(qid)] = q
    return out

def load_qrels(path: Path, min_rel: int = 2) -> Dict[str, Dict[str, int]]:
    out: Dict[str, Dict[str, int]] = collections.defaultdict(dict)
    with Path(path).open(encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            parts = raw.split()
            if len(parts) < 4:
                continue
            qid, _, pid, rel_str = parts[:4]
            try:
                rel = int(rel_str)
            except ValueError:
                continue
            if rel >= min_rel:
                out[str(qid)][str(pid)] = rel
    return out

def load_corpus_jsonl(path: Path) -> Dict[str, str]:
    corpus: Dict[str, str] = {}
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            pid = str(obj["id"])
            txt = obj["text"]
            corpus[pid] = txt
    return corpus

def load_run(path: Path, top_k: int) -> Dict[str, List[Tuple[str, int, float]]]:
    runs: Dict[str, List[Tuple[str, int, float]]] = collections.defaultdict(list)
    with Path(path).open(encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln:
                continue
            parts = ln.split()
            if len(parts) >= 6:
                qid, _Q0, pid, rank_s, score_s = parts[0], parts[1], parts[2], parts[3], parts[4]
                try:
                    rank = int(rank_s)
                    score = float(score_s)
                except ValueError:
                    continue
            else:
                if len(parts) < 3:
                    continue
                qid, pid, rank_s = parts[:3]
                score = float(parts[3]) if len(parts) > 3 else 0.0
                try:
                    rank = int(rank_s)
                except ValueError:
                    continue
            if rank <= int(top_k):
                runs[str(qid)].append((str(pid), rank, float(score)))
    for q in runs:
        runs[q].sort(key=lambda t: t[1])
        dedup: Dict[str, Tuple[str, int, float]] = {}
        for pid, r, sc in runs[q]:
            if pid not in dedup:
                dedup[pid] = (pid, r, sc)
        runs[q] = list(dedup.values())
    return runs

def dataset_files(data_root: Path, which: str) -> Tuple[Path, Path]:
    queries = Path(data_root) / "queries" / f"{which}-queries.tsv"
    qrels   = Path(data_root) / "qrels"   / f"{which}-qrels.txt"
    return queries, qrels

def annotated_nonrel_pids(qrels_all: Dict[str, Dict[str, int]], qid: str, max_rel: int = 0) -> Set[str]:
    judg = qrels_all.get(str(qid), {})
    return {pid for pid, rel in judg.items() if int(rel) <= int(max_rel)}


def choose_work_set_from_qrels(
    corpus: Dict[str, str],
    nonrel: Set[str],
    n_passages_per_query: int,
    seed: int = 42
) -> List[Tuple[str, int]]:

    out: List[Tuple[str, int]] = []
    candidates = sorted(list(nonrel))    
    rng = random.Random(seed)
    rng.shuffle(candidates)
    for pid in candidates:
        if pid in corpus:
            out.append((pid, 0))
        if len(out) >= int(n_passages_per_query):
            break
    return out

