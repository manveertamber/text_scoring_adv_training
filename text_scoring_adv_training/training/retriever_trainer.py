from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import deepspeed
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import AutoTokenizer

from text_scoring_adv_training.data.collators import RetrieverCollator
from text_scoring_adv_training.data.datasets import RankingDataset
from text_scoring_adv_training.training.losses import hinge_loss, softmax_nll_loss, mse_loss, kl_divergence_loss
from text_scoring_adv_training.models.embedding_model import EmbeddingModel
from text_scoring_adv_training.utils.seed import seed_everything

torch.backends.cuda.matmul.allow_tf32 = True


@dataclass
class RetrieverConfig:
    train_file: str
    dev_file: str
    output_dir: str
    paraphrased_path: str = None
    injected_path: str = None
    max_dev_samples: int = None

    num_epochs: int = 2
    eval_every: int = None
    save_model: bool = True
    save_every_epoch: bool = False

    temperature: float = 0.01
    negatives_per_query: int = 10

    distill_with_kl: bool = False
    distill_teacher_temperature: float = 1.0
    distill_student_temperature: float = 1.0
    teacher_scores_train: Optional[str] = None
    teacher_scores_dev: Optional[str] = None

    use_paraphrased: bool = False
    use_injected: bool = False
    use_rudimentary_manipulations: bool = False
    use_hotflip_swaps: bool = False
    use_pgd: bool = False

    paraphrase_weight: float = 0.0
    injection_weight: float = 0.0
    rudimentary_weight: float = 0.0
    hotflip_weight: float = 0.0
    pgd_weight: float = 0.0

    eps_max: float = 0.01
    pgd_steps: int = 1
    pgd_lr_factor: float = 1.0
    eps_warmup_steps: int = 500

    query_maxlen: int = 1024
    passage_maxlen: int = 1024

    model_name: str = "intfloat/e5-base-unsupervised"
    dropout: float = 0.0
    query_prefix: str = "query: "
    passage_prefix: str = "passage: "

    train_num_workers: int = 8
    dev_num_workers: int = 8

    seed: int = 42


class _EpsilonScheduler:
    def __init__(self, eps_start: float, eps_end: float,
                 warmup_steps: int, acc_steps: int):
        self.eps_start = eps_start
        self.eps = eps_start
        self.eps_end = eps_end
        self.warmup = warmup_steps
        self.acc = acc_steps
        self.micro = 0
        self.opt_step = 0

    def step(self) -> float:
        self.micro += 1
        if self.micro >= self.acc:
            self.micro = 0
            self.opt_step += 1
            if self.opt_step <= self.warmup:
                r = self.opt_step / float(self.warmup)
                self.eps = self.eps_start + r * (self.eps_end - self.eps_start)
            else:
                self.eps = self.eps_end
        return self.eps


def _proj_l2(delta: torch.Tensor, eps: float) -> torch.Tensor:
    norm = delta.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-8)
    return delta * torch.where(norm > eps, eps / norm, torch.ones_like(norm))


def _sample_l2(shape, eps, device, dtype):
    noise = torch.randn(*shape, device=device, dtype=dtype)
    norm = noise.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-8)
    unit = noise / norm
    u = torch.rand_like(norm)
    d = shape[-1]
    radius = eps * u.pow(1.0 / d)
    return unit * radius


def _pgd_attack(
    ids: torch.Tensor,
    mask: torch.Tensor,
    model_engine,
    eps: float,
    steps: int,
    lr: float,
    loss_fn: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    embed_layer = model_engine.module.encoder.get_input_embeddings()
    base = embed_layer(ids)
    delta = _sample_l2(base.shape, eps, base.device, base.dtype)
    delta.requires_grad_(True)

    for _ in range(steps):
        emb_adv = model_engine(inputs_embeds=base + delta, attention_mask=mask)
        loss = loss_fn(emb_adv)
        (grad,) = torch.autograd.grad(loss, delta, retain_graph=False)
        with torch.no_grad():
            delta += lr * grad / grad.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-8)
            delta[:] = _proj_l2(delta, eps)
        delta.detach_().requires_grad_(True)

    return delta.detach().requires_grad_(False)


def _hotflip_swap(
    ids: torch.Tensor,
    mask: torch.Tensor,
    q_emb: torch.Tensor,
    tokenizer: AutoTokenizer,
    model_engine,
) -> torch.Tensor:
    device = ids.device
    N, T = ids.size()
    embedder = model_engine.module.encoder.get_input_embeddings()
    W = embedder.weight
    V, H = W.size()

    special = torch.tensor(tokenizer.all_special_ids, device=device)
    valid = mask.bool() & ~torch.isin(ids, special)

    pick = torch.full((N,), -1, device=device, dtype=torch.long)
    for i in range(N):
        cand = torch.nonzero(valid[i], as_tuple=False).squeeze(1)
        if not cand.numel():
            continue
        pick[i] = cand[torch.randint(0, cand.numel(), (), device=device)]


    embeds = embedder(ids).detach().clone().requires_grad_(True) 
    out = model_engine(inputs_embeds=embeds, attention_mask=mask)
    sims = (q_emb * out).sum(-1)
    grads, = torch.autograd.grad(sims.sum(), embeds)

    grads = grads[torch.arange(N, device=device), pick.clamp_min(0)]
    orig_id = ids[torch.arange(N, device=device), pick.clamp_min(0)]
    orig_E = W[orig_id]

    delta = grads @ W.T
    delta -= (grads * orig_E).sum(-1, keepdim=True)
    delta[torch.arange(N), orig_id] = -float("inf")
    delta[:, tokenizer.all_special_ids] = -float("inf")

    best_id = delta.argmax(1)
    ids_adv = ids.clone()
    ok_rows = (pick >= 0)
    ids_adv[ok_rows, pick[ok_rows]] = best_id[ok_rows]

    return ids_adv


def _batched_hotflip_swap(
    pos_rows: List[int],
    neg_rows: List[int],
    *,
    pos_ids: torch.Tensor,
    pos_mask: torch.Tensor,
    neg_ids: torch.Tensor,
    neg_mask: torch.Tensor,
    q_emb: torch.Tensor,
    K: int,
    tokenizer: AutoTokenizer,
    model_engine,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:

    if not pos_rows and not neg_rows:
        return None, None

    ids_cat, mask_cat, q_cat = [], [], []

    if len(pos_rows) > 0:
        ids_cat.append(pos_ids[pos_rows])
        mask_cat.append(pos_mask[pos_rows])
        q_cat.append(q_emb[pos_rows])

    if len(neg_rows) > 0:
        ids_cat.append(neg_ids[neg_rows])
        mask_cat.append(neg_mask[neg_rows])
        q_rep = q_emb.repeat_interleave(K, 0)
        q_cat.append(q_rep[neg_rows])

    ids_all  = torch.cat(ids_cat,  0)
    mask_all = torch.cat(mask_cat, 0)
    q_all    = torch.cat(q_cat,    0)

    swapped_all = _hotflip_swap(
        ids_all,
        mask_all,
        q_all,
        tokenizer=tokenizer,
        model_engine=model_engine,
    )

    cursor = 0
    pos_ids_adv = neg_ids_adv = None
    if len(pos_rows) > 0:
        take = len(pos_rows)
        pos_ids_adv = swapped_all[cursor: cursor + take]
        cursor += take
    if len(neg_rows) > 0:
        neg_ids_adv = swapped_all[cursor:]

    return pos_ids_adv, neg_ids_adv


def _combined_loss(
    q_emb: torch.Tensor,
    p_emb: torch.Tensor,
    n_emb: torch.Tensor,
    adv_pos_emb: Optional[torch.Tensor],
    adv_neg_emb: Optional[torch.Tensor],
    manipulation_info,
    config: RetrieverConfig,
    *,
    teacher_scores: Optional[torch.Tensor] = None,
    return_details: bool = False,
) ->  Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
    loss = 0
    B, K = q_emb.size(0), config.negatives_per_query
    device = q_emb.device
    details = defaultdict(lambda: torch.tensor(0.0, device=device))

    pos_scores = (q_emb * p_emb).sum(-1, keepdim=True)
    neg_scores = torch.bmm(
        n_emb.reshape(B, K, -1),
        q_emb.unsqueeze(-1)
    ).squeeze(-1)

    student_all = torch.cat([pos_scores, neg_scores], 1)
    
    if config.distill_with_kl and (teacher_scores is not None):
        kll = kl_divergence_loss(student_all, teacher_scores, teacher_temperature=config.distill_teacher_temperature, student_temperature=config.distill_student_temperature)
        details["kl_distill"] = kll
        loss += kll
    else:
        softmax_nll_loss_val = softmax_nll_loss(student_all, temperature=config.temperature)
        details["softmax_nll_loss"] = softmax_nll_loss_val
        loss += softmax_nll_loss_val

    pos_tags = [info.get("pos", "") for info in manipulation_info]
    neg_tags = [t for info in manipulation_info for t in info["negs"]]

    assert len(neg_tags) == n_emb.shape[0], f"Mismatch: {len(neg_tags)} vs {n_emb.shape[0]}"

    def _mask(tags: List[str], cond_str: str) -> torch.Tensor:
        return torch.tensor([t == cond_str for t in tags], device=device)

    pos_para_mask = _mask(pos_tags, "paraphrase")
    neg_para_mask = _mask(neg_tags, "paraphrase")

    pos_inj_sent_mask = _mask(pos_tags, "injection_sentence")
    neg_inj_sent_mask = _mask(neg_tags, "injection_sentence")
    neg_inj_query_mask = _mask(neg_tags, "injection_query")
    
    pos_rud_mask = _mask(pos_tags, "rudimentary")
    neg_rud_mask = _mask(neg_tags, "rudimentary")
    
    pos_hot_mask = _mask(pos_tags, "hotflip")
    neg_hot_mask = _mask(neg_tags, "hotflip")

    q_rep = q_emb.repeat_interleave(K, 0)
        
    if config.use_paraphrased:
        adv, clean = [], []
        if adv_pos_emb is not None and pos_para_mask.any():
            adv.append(adv_pos_emb[pos_para_mask])
            clean.append(p_emb[pos_para_mask])
        if adv_neg_emb is not None and neg_para_mask.any():
            adv.append(adv_neg_emb[neg_para_mask])
            clean.append(n_emb[neg_para_mask])
        if adv:
            pl = mse_loss(torch.cat(adv), torch.cat(clean))
            loss += config.paraphrase_weight * pl
            details["paraphrase"] = pl

    if config.use_injected:
        h_pos, h_neg = [], []
        if adv_pos_emb is not None and pos_inj_sent_mask.any():
            qs, ps, aps = q_emb[pos_inj_sent_mask], p_emb[pos_inj_sent_mask], adv_pos_emb[pos_inj_sent_mask]
            h_pos.append((qs * ps).sum(-1)); h_neg.append((qs * aps).sum(-1))
        if adv_neg_emb is not None and neg_inj_sent_mask.any():
            qs, ns, ans = q_rep[neg_inj_sent_mask], n_emb[neg_inj_sent_mask], adv_neg_emb[neg_inj_sent_mask]
            h_pos.append((qs * ns).sum(-1)); h_neg.append((qs * ans).sum(-1))
        if h_pos:
            isl = hinge_loss(torch.cat(h_pos), torch.cat(h_neg), margin=0.0, squared=True)
            loss += config.injection_weight * isl
            details["injection_sentence"] = isl

    if config.use_rudimentary_manipulations:
        h_pos, h_neg = [], []
        if adv_pos_emb is not None and pos_rud_mask.any():
            qs, ps, aps = q_emb[pos_rud_mask], p_emb[pos_rud_mask], adv_pos_emb[pos_rud_mask]
            h_pos.append((qs * ps).sum(-1)); h_neg.append((qs * aps).sum(-1))
        if adv_neg_emb is not None and neg_rud_mask.any():
            qs, ns, ans = q_rep[neg_rud_mask], n_emb[neg_rud_mask], adv_neg_emb[neg_rud_mask]
            h_pos.append((qs * ns).sum(-1)); h_neg.append((qs * ans).sum(-1))
        if h_pos:
            rl = hinge_loss(torch.cat(h_pos), torch.cat(h_neg), margin=0.0, squared=True)
            loss += config.rudimentary_weight * rl
            details["rudimentary"] = rl
    
    if config.use_hotflip_swaps:
        h_pos, h_neg = [], []
        if adv_pos_emb is not None and pos_hot_mask.any():
            qs, ps, aps = q_emb[pos_hot_mask], p_emb[pos_hot_mask], adv_pos_emb[pos_hot_mask]
            h_pos.append((qs * ps).sum(-1)); h_neg.append((qs * aps).sum(-1))
        if adv_neg_emb is not None and neg_hot_mask.any():
            qs, ns, ans = q_rep[neg_hot_mask], n_emb[neg_hot_mask], adv_neg_emb[neg_hot_mask]
            h_pos.append((qs * ns).sum(-1)); h_neg.append((qs * ans).sum(-1))
        if h_pos:
            hl = hinge_loss(torch.cat(h_pos), torch.cat(h_neg), margin=0.0, squared=True)
            loss += config.hotflip_weight * hl
            details["hotflip"] = hl

    if config.use_injected and adv_neg_emb is not None and neg_inj_query_mask.any():
        qs = q_rep[neg_inj_query_mask]
        ps = p_emb.repeat_interleave(K, 0)[neg_inj_query_mask]
        ans = adv_neg_emb[neg_inj_query_mask]
        iql = hinge_loss((qs * ps).sum(-1), (qs * ans).sum(-1), margin=0.0, squared=True)
        loss += config.injection_weight / 16.0 * iql
        details["injection_query"] = iql

    if return_details:
        return loss, details
    return loss

class RetrieverTrainer:
    def __init__(self, config: RetrieverConfig, deepspeed_args: Optional[Any] = None):
        self.config = config
        self.deepspeed_args = deepspeed_args
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name, trust_remote_code=True
        )

        self.model_engine = None
        self.best_dev_loss = float("inf")
        self.global_step = 0
        self.steps_since_eval = 0

    def is_main_process(self) -> bool:
        if self.model_engine is None:
            return not (dist.is_initialized() and dist.get_rank() != 0)
        return self.model_engine.global_rank == 0

    def log(self, *args, **kwargs):
        if self.is_main_process():
            print(*args, **kwargs, flush=True)

    def setup_model(self):
        model = EmbeddingModel(self.config.model_name, dropout=self.config.dropout)

        params = (p for p in model.parameters() if p.requires_grad)

        self.model_engine, _, _, _ = deepspeed.initialize(
            args=self.deepspeed_args,
            model=model,
            model_parameters=params,
        )        

        self.train_micro_bs = self.model_engine.train_micro_batch_size_per_gpu()
        self.device = self.model_engine.device

        rank = self.model_engine.global_rank if dist.is_initialized() else 0
        seed_everything(self.config.seed + rank)

        self._eps_sched = _EpsilonScheduler(
            1e-8, self.config.eps_max,
            self.config.eps_warmup_steps,
            self.model_engine.gradient_accumulation_steps()
        )

    def setup_data(self) -> Tuple[DataLoader, DataLoader]:
        train_ds, dev_ds = RankingDataset.build_train_dev(
            train_jsonl_path=self.config.train_file,
            dev_jsonl_path=self.config.dev_file,
            paraphrased_path=self.config.paraphrased_path,
            injected_path=self.config.injected_path,
            max_dev_samples=self.config.max_dev_samples,
            teacher_train_scores_path=self.config.teacher_scores_train,
            teacher_dev_scores_path=self.config.teacher_scores_dev,            
        )
        collator_kwargs = dict(
            tokenizer=self.tokenizer,
            query_prefix=self.config.query_prefix,
            passage_prefix=self.config.passage_prefix,
            query_max_length=self.config.query_maxlen,
            passage_max_length=self.config.passage_maxlen,
            negatives_per_query=self.config.negatives_per_query,
            use_paraphrased=self.config.use_paraphrased,
            use_injected=self.config.use_injected,
            use_rudimentary_manipulations=self.config.use_rudimentary_manipulations,
            use_hotflip_swaps=self.config.use_hotflip_swaps,
        )

        emit_teacher = bool(self.config.distill_with_kl)
        train_collator = RetrieverCollator(sample_negatives=True, emit_teacher_scores=emit_teacher, **collator_kwargs)
        dev_collator   = RetrieverCollator(sample_negatives=False, emit_teacher_scores=emit_teacher, **collator_kwargs)

        def _loader(ds, sampler, collator, workers):
            return DataLoader(
                ds,
                batch_size=self.train_micro_bs,
                sampler=sampler,
                drop_last=True,
                num_workers=workers,
                pin_memory=True,
                collate_fn=collator,
            )

        samp_train = DistributedSampler(train_ds, drop_last=True, seed=self.config.seed)
        samp_dev = DistributedSampler(dev_ds, shuffle=False, drop_last=True, seed=self.config.seed)

        return (
            _loader(train_ds, samp_train, train_collator, self.config.train_num_workers),
            _loader(dev_ds, samp_dev, dev_collator, self.config.dev_num_workers),
        )

    def _split(self, e, slices):
        q   = e[slices["q"]]
        p   = e[slices["p"]]
        n   = e[slices["n"]]
        ap  = e[slices["adv_p"]] if "adv_p" in slices else None
        an  = e[slices["adv_n"]] if "adv_n" in slices else None
        return q, p, n, ap, an
    
    def _prepare_batch_for_step(self, batch, is_eval=False):
        K = self.config.negatives_per_query
        device = self.device

        query_data = batch.get("query")
        pos_data = batch.get("pos_passage")
        neg_data = batch.get("neg_passages")

        adv_pos_data = batch.get("adversarial_pos_passage")
        adv_neg_data = batch.get("adversarial_negative_passages")

        teacher_scores = batch.get("teacher_scores", None)

        def _to_device(d):
            if d is None: return
            for k in ("input_ids", "attention_mask"):
                if k in d and torch.is_tensor(d[k]):
                    d[k] = d[k].to(device, non_blocking=True)

        for d in (query_data, pos_data, neg_data, adv_pos_data, adv_neg_data):
            _to_device(d)

        if isinstance(teacher_scores, torch.Tensor):
            teacher_scores = teacher_scores.to(device, non_blocking=True)

        if query_data is None or pos_data is None or neg_data is None:
            return None

        manipulation_info = batch.get("manipulation_info", [])
        B = query_data["input_ids"].size(0)

        with torch.set_grad_enabled(not is_eval):
            q_emb = self.model_engine(
                input_ids=query_data["input_ids"],
                attention_mask=query_data["attention_mask"]
            )

        if self.config.use_hotflip_swaps:
            pos_tags = [info.get("pos", "") for info in manipulation_info]
            neg_tags = [t for info in manipulation_info for t in info.get("negs", [])]
            hot_pos_rows = [i for i, t in enumerate(pos_tags) if t == "hotflip"]
            hot_neg_rows = [i for i, t in enumerate(neg_tags) if t == "hotflip"]
            
            local_has_hotflips = torch.tensor(1.0 if (hot_pos_rows or hot_neg_rows) else 0.0, device=self.device)
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(local_has_hotflips, op=dist.ReduceOp.SUM)

            if local_has_hotflips.item() > 0:
                if hot_pos_rows or hot_neg_rows:
                    hp_ids, hn_ids = _batched_hotflip_swap(
                        hot_pos_rows, hot_neg_rows,
                        pos_ids=pos_data["input_ids"].detach(),
                        pos_mask=pos_data["attention_mask"].detach(),
                        neg_ids=neg_data["input_ids"].detach(),
                        neg_mask=neg_data["attention_mask"].detach(),
                        q_emb=q_emb.detach(),
                        K=K, tokenizer=self.tokenizer, model_engine=self.model_engine,
                    )
                    
                    if hp_ids is not None and adv_pos_data is not None:
                        adv_pos_data["input_ids"][hot_pos_rows] = hp_ids
                    if hn_ids is not None and adv_neg_data is not None:
                        adv_neg_data["input_ids"][hot_neg_rows] = hn_ids
                else:
                    pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
                    dummy_ids  = torch.full((1, 1), pad_id, device=self.device, dtype=torch.long)
                    dummy_mask = torch.ones_like(dummy_ids)

                    emb_layer = self.model_engine.module.encoder.get_input_embeddings()
                    dummy_embeds = emb_layer(dummy_ids).detach().clone().requires_grad_(True)

                    dummy_logits = self.model_engine(inputs_embeds=dummy_embeds,
                                                        attention_mask=dummy_mask)
                    ( _ , ) = torch.autograd.grad(dummy_logits.sum(), dummy_embeds)
        
        ids_list, mask_list, slices = [], [], {}
        cursor = 0
        def _add(tag: str, item) -> None:
            nonlocal cursor
            if item is None or item["input_ids"].size(0) == 0: return
            sz = item["input_ids"].size(0)
            slices[tag] = slice(cursor, cursor + sz)
            ids_list.append(item["input_ids"])
            mask_list.append(item["attention_mask"])
            cursor += sz

        _add("p", pos_data)
        _add("n", neg_data)
        if adv_pos_data: _add("adv_p", adv_pos_data)
        if adv_neg_data: _add("adv_n", adv_neg_data)

        if not ids_list:
            return None

        other_ids = torch.cat(ids_list, 0)
        other_mask = torch.cat(mask_list, 0)
        
        with torch.set_grad_enabled(not is_eval):
            other_embs = self.model_engine(input_ids=other_ids, attention_mask=other_mask)
        
        p_emb = other_embs[slices["p"]]
        n_emb = other_embs[slices["n"]]
        adv_pos_emb = other_embs[slices["adv_p"]] if "adv_p" in slices else None
        adv_neg_emb = other_embs[slices["adv_n"]] if "adv_n" in slices else None

        clean_ids_list, clean_mask_list, clean_slices = [], [], {}
        cur_clean = 0

        def _add_clean(tag: str, d):
            nonlocal cur_clean
            if d is None or d["input_ids"].size(0) == 0:
                return
            sz = d["input_ids"].size(0)
            clean_slices[tag] = slice(cur_clean, cur_clean + sz)
            clean_ids_list.append(d["input_ids"])
            clean_mask_list.append(d["attention_mask"])
            cur_clean += sz

        _add_clean("q", query_data)
        _add_clean("p", pos_data)
        _add_clean("n", neg_data)

        clean_ids  = torch.cat(clean_ids_list, 0) if clean_ids_list else None
        clean_mask = torch.cat(clean_mask_list, 0) if clean_mask_list else None

        full_ids, full_mask, full_slices = [], [], {}
        cur = 0

        def _add_full(tag, d):
            nonlocal cur
            if d is None or d["input_ids"].size(0) == 0:
                return
            sz = d["input_ids"].size(0)
            full_slices[tag] = slice(cur, cur + sz)
            full_ids.append(d["input_ids"])
            full_mask.append(d["attention_mask"])
            cur += sz

        _add_full("q",         query_data)
        _add_full("p",         pos_data)
        _add_full("n",         neg_data)
        _add_full("adv_p",     adv_pos_data)
        _add_full("adv_n",     adv_neg_data)

        full_ids  = torch.cat(full_ids,  0)
        full_mask = torch.cat(full_mask, 0)

        return {
            "q_emb": q_emb, "p_emb": p_emb, "n_emb": n_emb,
            "adv_pos_emb": adv_pos_emb, "adv_neg_emb": adv_neg_emb,
            "manipulation_info": manipulation_info, "B": B,
            "query_data": query_data, "pos_data": pos_data, "neg_data": neg_data,
            "all_ids":    full_ids,
            "all_mask":   full_mask,
            "all_slices": full_slices,
            "clean_ids":  clean_ids,
            "clean_mask": clean_mask,
            "clean_slices": clean_slices,
            "teacher_scores": teacher_scores,
        }

    def train_step(self, batch) -> float:        
        prepared = self._prepare_batch_for_step(batch, is_eval=False)
        if prepared is None:
            return 0.0

        loss = _combined_loss(
            q_emb=prepared["q_emb"], p_emb=prepared["p_emb"], n_emb=prepared["n_emb"],
            adv_pos_emb=prepared["adv_pos_emb"], adv_neg_emb=prepared["adv_neg_emb"],
            manipulation_info=prepared["manipulation_info"], config=self.config,
            teacher_scores=(prepared.get("teacher_scores") if self.config.distill_with_kl else None),
        )

        if self.config.use_pgd and prepared["clean_ids"] is not None:
            eps_now = self._eps_sched.step()
            lr_now  = eps_now / self.config.pgd_lr_factor
            cslices = prepared["clean_slices"]

            def _pgd_loss_fn(emb_adv_clean: torch.Tensor) -> torch.Tensor:

                q_adv = emb_adv_clean[cslices["q"]]
                p_adv = emb_adv_clean[cslices["p"]]
                n_adv = emb_adv_clean[cslices["n"]]
                return _combined_loss(
                    q_emb=q_adv, p_emb=p_adv, n_emb=n_adv,
                    adv_pos_emb=None, adv_neg_emb=None,
                    manipulation_info=prepared["manipulation_info"], config=self.config,
                    teacher_scores=(prepared.get("teacher_scores") if self.config.distill_with_kl else None),
                )

            delta = _pgd_attack(
                ids=prepared["clean_ids"], mask=prepared["clean_mask"], model_engine=self.model_engine,
                eps=eps_now, steps=self.config.pgd_steps, lr=lr_now, loss_fn=_pgd_loss_fn
            )

            embed_layer = self.model_engine.module.encoder.get_input_embeddings()
            emb_adv_clean = self.model_engine(
                inputs_embeds=embed_layer(prepared["clean_ids"]) + delta,
                attention_mask=prepared["clean_mask"]
            )

            q_adv = emb_adv_clean[cslices["q"]]
            p_adv = emb_adv_clean[cslices["p"]]
            n_adv = emb_adv_clean[cslices["n"]]
            pl = _combined_loss(
                q_emb=q_adv, p_emb=p_adv, n_emb=n_adv,
                adv_pos_emb=None, adv_neg_emb=None,
                manipulation_info=prepared["manipulation_info"], config=self.config,
                teacher_scores=(prepared.get("teacher_scores") if self.config.distill_with_kl else None),
            )
            loss += self.config.pgd_weight * pl

        self.model_engine.backward(loss)
        self.model_engine.step()
        return loss.item()

    def eval_step(self, batch) -> Dict[str, float]:
        prepared = self._prepare_batch_for_step(batch, is_eval=True)
        if prepared is None:
            return {}

        loss_dict = {}
        with torch.no_grad():
            base_loss, comps = _combined_loss(
                q_emb=prepared["q_emb"], p_emb=prepared["p_emb"], n_emb=prepared["n_emb"],
                adv_pos_emb=prepared["adv_pos_emb"], adv_neg_emb=prepared["adv_neg_emb"],
                manipulation_info=prepared["manipulation_info"], config=self.config,
                teacher_scores=(prepared.get("teacher_scores") if self.config.distill_with_kl else None),
                return_details=True,
            )
        loss_dict["combined"] = base_loss.item()
        for k, v in comps.items():
            loss_dict[k] = v.item()

        pos_tags = [info.get("pos", "") for info in prepared["manipulation_info"]]
        neg_tags = [t for info in prepared["manipulation_info"] for t in info.get("negs", [])]

        counts = {
            "paraphrase__count": float(
                sum(t == "paraphrase" for t in pos_tags) +
                sum(t == "paraphrase" for t in neg_tags)
            ),
            "injection_sentence__count": float(
                sum(t == "injection_sentence" for t in pos_tags) +
                sum(t == "injection_sentence" for t in neg_tags)
            ),
            "injection_query__count": float(
                sum(t == "injection_query" for t in neg_tags)
            ),
            "rudimentary__count": float(
                sum(t == "rudimentary" for t in pos_tags) +
                sum(t == "rudimentary" for t in neg_tags)
            ),
            "hotflip__count": float(
                sum(t == "hotflip" for t in pos_tags) +
                sum(t == "hotflip" for t in neg_tags)
            ),
        }

        for k in list(loss_dict.keys()):
            if k in ("combined", "softmax_nll_loss", "kl_distill", "pgd_combined"):
                continue
            ckey = f"{k}__count"
            if ckey in counts:
                loss_dict[ckey] = counts[ckey]

        if self.config.use_pgd and prepared.get("clean_ids") is not None:
            eps_now = self.config.eps_max
            lr_now  = eps_now / self.config.pgd_lr_factor
            cslices = prepared["clean_slices"]

            def _pgd_loss_fn(emb_adv_clean: torch.Tensor) -> torch.Tensor:
                q_adv = emb_adv_clean[cslices["q"]]
                p_adv = emb_adv_clean[cslices["p"]]
                n_adv = emb_adv_clean[cslices["n"]]
                return _combined_loss(
                    q_emb=q_adv, p_emb=p_adv, n_emb=n_adv,
                    adv_pos_emb=None, adv_neg_emb=None,
                    manipulation_info=prepared["manipulation_info"], config=self.config,
                    teacher_scores=(prepared.get("teacher_scores") if self.config.distill_with_kl else None),
                )

            delta = _pgd_attack(
                ids=prepared["clean_ids"], mask=prepared["clean_mask"], model_engine=self.model_engine,
                eps=eps_now, steps=self.config.pgd_steps, lr=lr_now, loss_fn=_pgd_loss_fn
            )

            with torch.no_grad():
                embed_layer = self.model_engine.module.encoder.get_input_embeddings()
                emb_adv_clean = self.model_engine(
                    inputs_embeds=embed_layer(prepared["clean_ids"]) + delta,
                    attention_mask=prepared["clean_mask"],
                ) 
                q_adv = emb_adv_clean[cslices["q"]]
                p_adv = emb_adv_clean[cslices["p"]]
                n_adv = emb_adv_clean[cslices["n"]]
                pgd_loss = _combined_loss(
                    q_emb=q_adv, p_emb=p_adv, n_emb=n_adv,
                    adv_pos_emb=None, adv_neg_emb=None,
                    manipulation_info=prepared["manipulation_info"], config=self.config,
                    teacher_scores=(prepared.get("teacher_scores") if self.config.distill_with_kl else None),
                )
                loss_dict["pgd_combined"] = pgd_loss.item()

        loss_dict["n"] = float(prepared["B"])
        return loss_dict
        
    def evaluate(self, dataloader: DataLoader, verbose: bool = True) -> Dict[str, float]:
        self.model_engine.eval()
        totals = defaultdict(float)
        counts = defaultdict(float)

        eval_iter = tqdm(
            dataloader,
            desc="Evaluating",
            disable=not (verbose and self.is_main_process()),
        )

        for batch in eval_iter:
            res = self.eval_step(batch)
            if not res:
                continue
            n = float(res.pop("n", 0.0))
            if n == 0.0:
                continue
            
            special_counts = {k[:-8]: float(v) for k, v in list(res.items()) if k.endswith("__count")}
            for k in list(special_counts.keys()):
                res.pop(f"{k}__count", None)

            for key, value in res.items():
                weight = special_counts.get(key, n)
                totals[key] += value * weight
                counts[key] += weight


        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            local_keys = list(totals.keys())
            gathered = [None] * world_size
            dist.all_gather_object(gathered, local_keys)
            all_keys = sorted({k for ks in gathered for k in ks})
        else:
            all_keys = sorted(totals.keys())

        avg_losses = {}
        for k in all_keys:
            tot = torch.tensor(totals.get(k, 0.0), device=self.device)
            cnt = torch.tensor(counts.get(k, 0.0),  device=self.device)

            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(tot, op=dist.ReduceOp.SUM)
                dist.all_reduce(cnt, op=dist.ReduceOp.SUM)

            if cnt.item() > 0:
                avg_losses[k] = (tot / cnt).item()

        self.model_engine.train()
        return avg_losses


    def model_save(self, dev_loss, epoch=None):
        if self.config.save_every_epoch and (epoch is not None) and self.config.save_model and self.is_main_process():
            p = Path(self.config.output_dir) / f"epoch-{epoch + 1:04d}"
            p.mkdir(parents=True, exist_ok=True)
            self.model_engine.module.encoder.save_pretrained(p)
            self.tokenizer.save_pretrained(p)
            self.log(f"  ✔ Saved checkpoint → {p}")

        if dev_loss < self.best_dev_loss:
            self.best_dev_loss = dev_loss
            if self.config.save_model and self.is_main_process():
                p = Path(self.config.output_dir)
                p.mkdir(parents=True, exist_ok=True)
                self.model_engine.module.encoder.save_pretrained(p)
                self.tokenizer.save_pretrained(p)
                self.log(f"  ✔ Saved checkpoint → {p}")
    

    def train(self):
        self.setup_model()
        train_loader, dev_loader = self.setup_data()

        for epoch in range(self.config.num_epochs):
            self.log(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")
            train_loader.sampler.set_epoch(epoch)

            train_loss_sum = 0.0
            train_steps = 0

            train_iter = tqdm(
                train_loader,
                desc="Training",
                disable=not self.is_main_process()
            )

            for batch in train_iter:
                if (self.config.eval_every and (self.steps_since_eval >= self.config.eval_every)) or (self.global_step == 0):
                    dev_losses = self.evaluate(dev_loader)
                    loss_str = ", ".join([f"{k}: {v:.8f}" for k, v in dev_losses.items()])
                    self.log(f"[Step {self.global_step:>7}] Dev losses | {loss_str}")

                    self.model_save(dev_losses.get("combined", float("inf")))

                    self.steps_since_eval = 0

                loss_val = self.train_step(batch)
                train_loss_sum += loss_val
                train_steps += 1
                self.global_step += 1

                if self.model_engine.is_gradient_accumulation_boundary():
                    self.steps_since_eval += 1

                if self.is_main_process():
                    train_iter.set_postfix({"loss": f"{loss_val:.2e}"})

            if train_steps:
                avg_train_loss = train_loss_sum / train_steps
                loss_tensor = torch.tensor(avg_train_loss, device=self.model_engine.device)
                if dist.is_available() and dist.is_initialized():
                    dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                    avg_train_loss = loss_tensor.item() / self.model_engine.world_size
                self.log(f"  Train loss: {avg_train_loss:.6f}")

            dev_losses = self.evaluate(dev_loader)
            loss_str = ", ".join([f"{k}: {v:.8f}" for k, v in dev_losses.items()])
            self.log(f"  End of Epoch Dev losses | {loss_str}")

            self.steps_since_eval = 0

            self.model_save(dev_losses.get("combined", float("inf")), epoch)

        self.log("\nTraining complete! 🎉")

