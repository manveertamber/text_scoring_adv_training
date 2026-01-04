#!/usr/bin/env python
from __future__ import annotations
import json, collections
from pathlib import Path
from typing import Dict, List, Tuple

def load_injections_jsonl(path: Path) -> Tuple[Dict[str, List[str]], Dict[Tuple[str, str], List[str]]]:
    by_qid = collections.defaultdict(list)
    by_qid_pid = collections.defaultdict(list)
    if path is None:
        raise ValueError("injected_jsonl_path is None but use_generated_injected_texts=True")

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            qid = str(rec["qid"])
            pid = str(rec.get("pid", "")) or ""
            sn  = rec.get("sample_num")
            txt = rec["text"]
            by_qid[qid].append((sn, txt))
            if pid:
                by_qid_pid[(qid, pid)].append((sn, txt))

    def _sorted_vals(d):
        out = {}
        for k, lst in d.items():
            lst_sorted = sorted(lst, key=lambda x: (x[0] is None, x[0]))
            out[k] = [t for _, t in lst_sorted]
        return out

    return _sorted_vals(by_qid), _sorted_vals(by_qid_pid)
