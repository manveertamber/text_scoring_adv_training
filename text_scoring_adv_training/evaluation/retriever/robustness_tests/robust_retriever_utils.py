from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
from transformers import AutoTokenizer
import contextlib

from text_scoring_adv_training.models.embedding_model import EmbeddingModel
from text_scoring_adv_training.evaluation.robustness_tests.common import resolve_model_source

Q_PREFIX = "query: "
P_PREFIX = "passage: "


def make_embedder(model_name: str,
                  device: Optional[str] = None,
                  batch_size: int = 8192,
                  models_dir: Optional[Path] = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    src, is_local = resolve_model_source(model_name, models_dir=models_dir)

    tok = AutoTokenizer.from_pretrained(src, local_files_only=is_local)
    model = EmbeddingModel(src).to(device).eval() 

    @torch.inference_mode()
    def embed(texts: List[str]) -> torch.Tensor:
        all_embeddings = []
        use_cuda = (device.startswith("cuda") and torch.cuda.is_available())
        autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_cuda else contextlib.nullcontext()
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            inputs = tok(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(device)
            
            with autocast_ctx:
                embeddings = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
            all_embeddings.append(embeddings)
            
        return torch.cat(all_embeddings, dim=0)

    return tok, model, embed

def prefix_token_len(tok, prefix: str) -> int:
    ids = tok(prefix, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
    n_prefix = ids.numel()
    has_leading_special = (tok.cls_token_id is not None) or (tok.bos_token_id is not None)
    return n_prefix + (1 if has_leading_special else 0)

@torch.inference_mode()
def sims_on_prefixed(embed, q_vec: torch.Tensor, prefixed_texts: List[str], batch: int = 8192) -> torch.Tensor:
    outs = []
    for i in range(0, len(prefixed_texts), batch):
        e = embed(prefixed_texts[i:i+batch])
        scores = e @ q_vec
        outs.append(scores.cpu())
    return torch.cat(outs, dim=0).view(-1) if outs else torch.empty(0)

def baseline_top_sim_for_query(
    embed,
    query_text: str,
    run_full: List[Tuple[str, int, float]],
    corpus: Dict[str, str]
) -> Tuple[Optional[str], float, torch.Tensor]:

    q_vec = embed([Q_PREFIX + query_text])[0]
    baseline_pid = next((pid for pid, _, _ in run_full if pid in corpus), None)
    if baseline_pid is None:
        return None, float("-inf"), q_vec

    passage_vec = embed([P_PREFIX + corpus[baseline_pid]])[0]
    top_sim = (passage_vec @ q_vec).item()
    return baseline_pid, top_sim, q_vec
