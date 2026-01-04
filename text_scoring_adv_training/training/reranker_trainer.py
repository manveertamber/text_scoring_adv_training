from __future__ import annotations


from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import contextlib

import deepspeed
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import AutoTokenizer
from deepspeed import zero

from text_scoring_adv_training.data.collators import RerankerCollator
from text_scoring_adv_training.data.datasets import RankingDataset
from text_scoring_adv_training.training.losses import hinge_loss, softmax_nll_loss, mse_loss
from text_scoring_adv_training.models.pointwise_scorer import PointwiseScorer
from text_scoring_adv_training.utils.seed import seed_everything

torch.backends.cuda.matmul.allow_tf32 = True


@dataclass
class RerankerConfig:
    train_file:  str
    dev_file:    str
    output_dir:  str
    paraphrased_path: str = None
    injected_path:    str = None
    max_dev_samples: int = 30000

    num_epochs:  int  = 1
    eval_every:  int  = None
    save_model:  bool = True

    temperature: float = 1.0
    negatives_per_query: int = 7

    use_paraphrased:               bool = False
    use_injected:                  bool = False
    use_rudimentary_manipulations: bool = False
    use_hotflip_swaps:             bool = False
    use_pgd:                       bool = False

    paraphrase_weight:    float = 0.0
    injection_weight:     float = 0.0
    rudimentary_weight:   float = 0.0
    hotflip_weight:       float = 0.0
    pgd_weight:           float = 0.0

    eps_max:          float = 0.01
    pgd_steps:        int   = 1
    pgd_lr_factor:    float = 1.0
    eps_warmup_steps: int   = 500

    max_length: int = 1024
    pad_to_multiple_of: int  = 8
    padding_side:       str  = "left"

    model_name: str = "Qwen/Qwen3-0.6B"
    dropout:    float = 0.0
    gradient_checkpointing:  bool = False

    train_num_workers: int = 8
    dev_num_workers:   int = 8

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
    embed_layer = model_engine.module.model.get_input_embeddings()
    base = embed_layer(ids)
    delta = _sample_l2(base.shape, eps, base.device, base.dtype)
    delta.requires_grad_(True)

    for _ in range(steps):
        perturbed_logits = model_engine(inputs_embeds=base + delta, attention_mask=mask)
        if perturbed_logits.ndim == 2 and perturbed_logits.size(-1) == 1:
            perturbed_logits = perturbed_logits.squeeze(-1)
        loss = loss_fn(perturbed_logits)
        (grad,) = torch.autograd.grad(loss, delta, retain_graph=False)
        with torch.no_grad():
            delta += lr * grad / grad.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-8)
            delta[:] = _proj_l2(delta, eps)
        delta.detach_().requires_grad_(True)

    return delta.detach().requires_grad_(False)


def _hotflip_swap(
    ids: torch.Tensor,
    mask: torch.Tensor,
    tokenizer: AutoTokenizer,
    model_engine,
    modifiable_start_idx: Optional[torch.Tensor] = None,
    *,
    weight_is_gathered: bool = False,
) -> torch.Tensor:

    device = ids.device
    N, T = ids.size()
    token_embeddings = model_engine.module.model.get_input_embeddings()

    special = torch.tensor(tokenizer.all_special_ids, device=device)
    valid = mask.bool() & ~torch.isin(ids, special)

    if modifiable_start_idx is not None:
        first_nonpad = (mask > 0).float().argmax(dim=1).to(torch.long)
        doc_start = (first_nonpad + modifiable_start_idx).clamp(max=T)
        arangeT = torch.arange(T, device=device).unsqueeze(0).expand(N, T)
        valid = valid & (arangeT >= doc_start.unsqueeze(1))

    pick = torch.full((N,), -1, device=device, dtype=torch.long)
    for i in range(N):
        cand = torch.nonzero(valid[i], as_tuple=False).squeeze(1)
        if not cand.numel():
            continue
        pick[i] = cand[torch.randint(0, cand.numel(), (), device=device)]

    embeds = token_embeddings(ids).detach().clone().requires_grad_(True)
    logits = model_engine(inputs_embeds=embeds, attention_mask=mask)
    grads, = torch.autograd.grad(logits.sum(), embeds)

    grads = grads[torch.arange(N, device=device), pick.clamp_min(0)]
    orig_id = ids[torch.arange(N, device=device), pick.clamp_min(0)]

    ctx = (contextlib.nullcontext() if weight_is_gathered
           else zero.GatheredParameters([token_embeddings.weight], modifier_rank=None))
    with ctx:
        with torch.no_grad():
            W = token_embeddings.weight
            orig_E = W[orig_id]
            delta  = grads @ W.t()
            delta -= (grads * orig_E).sum(-1, keepdim=True)
            delta[torch.arange(N), orig_id] = -float("inf")
            if tokenizer.all_special_ids:
                delta[:, tokenizer.all_special_ids] = -float("inf")
            best_id = delta.argmax(1)

    torch.cuda.synchronize(device=grads.device)

    ids_adv = ids.clone()
    ok_rows = (pick >= 0)
    ids_adv[ok_rows, pick[ok_rows]] = best_id[ok_rows]

    return ids_adv


def _batched_hotflip_swap(
    pos_rows,
    neg_rows,
    *,
    pos_ids,
    pos_mask,
    neg_ids,
    neg_mask,
    tokenizer,
    model_engine,
    pos_start_idx: Optional[torch.Tensor] = None,
    neg_start_idx: Optional[torch.Tensor] = None,
):

    if not pos_rows and not neg_rows:
        return None, None

    ids_cat, mask_cat, idx_cat = [], [], []

    if pos_rows:
        ids_cat.append(pos_ids[pos_rows])
        mask_cat.append(pos_mask[pos_rows])
        if pos_start_idx is not None: idx_cat.append(pos_start_idx[pos_rows])
    if neg_rows:
        ids_cat.append(neg_ids[neg_rows])
        mask_cat.append(neg_mask[neg_rows])
        if neg_start_idx is not None: idx_cat.append(neg_start_idx[neg_rows])

    ids_all  = torch.cat(ids_cat, 0)
    mask_all = torch.cat(mask_cat, 0)
    idx_all  = torch.cat(idx_cat, 0) if idx_cat else None

    swapped_all = _hotflip_swap(
        ids_all,
        mask_all,
        tokenizer=tokenizer,
        model_engine=model_engine,
        modifiable_start_idx=idx_all,
        weight_is_gathered=True,
    )

    cursor = 0
    hp_ids = hn_ids = None
    if pos_rows:
        hp_ids = swapped_all[cursor : cursor + len(pos_rows)]
        cursor += len(pos_rows)
    if neg_rows:
        hn_ids = swapped_all[cursor:]
    return hp_ids, hn_ids

def _combined_loss(
    pos_logits:         torch.Tensor,
    neg_logits:         torch.Tensor,
    adv_pos_logits:     Optional[torch.Tensor],
    adv_neg_logits:     Optional[torch.Tensor],
    manipulation_info:  List[Dict[str, Any]],
    config:             RerankerConfig,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

    loss = 0
    B, K   = pos_logits.size(0), config.negatives_per_query
    device = pos_logits.device
    details = defaultdict(lambda: torch.tensor(0.0, device=device))

    neg_logits_reshaped = neg_logits.reshape(B, K)
    pos_logits_reshaped = pos_logits.unsqueeze(1)
    all_logits = torch.cat([pos_logits_reshaped, neg_logits_reshaped], dim=1)

    softmax_nll_loss_val = softmax_nll_loss(all_logits, temperature=config.temperature)
    details["softmax_nll_loss"] = softmax_nll_loss_val

    loss += softmax_nll_loss_val

    pos_tags = [info.get("pos", "none") for info in manipulation_info]
    neg_tags = [tag for info in manipulation_info for tag in info.get("negs", [])]

    def _mask(tags, cond) -> torch.Tensor:
        return torch.tensor([cond(t) for t in tags],
                            dtype=torch.bool, device=device)

    pos_para  = _mask(pos_tags, lambda t: t == "paraphrase")
    neg_para  = _mask(neg_tags, lambda t: t == "paraphrase")

    pos_inj_s = _mask(pos_tags, lambda t: t == "injection_sentence")
    neg_inj_s = _mask(neg_tags, lambda t: t == "injection_sentence")
    neg_inj_q = _mask(neg_tags, lambda t: t == "injection_query")

    pos_rud   = _mask(pos_tags, lambda t: t == "rudimentary")
    neg_rud   = _mask(neg_tags, lambda t: t == "rudimentary")

    pos_hot   = _mask(pos_tags, lambda t: t == "hotflip")
    neg_hot   = _mask(neg_tags, lambda t: t == "hotflip")

    if config.use_paraphrased:
        adv, clean = [], []
        if adv_pos_logits is not None and pos_para.any():
            adv.append(adv_pos_logits[pos_para])
            clean.append(pos_logits[pos_para])
        if adv_neg_logits is not None and neg_para.any():
            adv.append(adv_neg_logits[neg_para])
            clean.append(neg_logits[neg_para])
        if adv:
            pl = mse_loss(torch.cat(adv), torch.cat(clean))
            loss += config.paraphrase_weight * pl
            details["paraphrase"] = pl


    if config.use_injected:
        hinge_pos, hinge_neg = [], []
        if adv_pos_logits is not None and pos_inj_s.any():
            hinge_pos.append(pos_logits[pos_inj_s])
            hinge_neg.append(adv_pos_logits[pos_inj_s])
        if adv_neg_logits is not None and neg_inj_s.any():
            hinge_pos.append(neg_logits[neg_inj_s])
            hinge_neg.append(adv_neg_logits[neg_inj_s])
        if hinge_pos:
            isl = hinge_loss(torch.cat(hinge_pos), torch.cat(hinge_neg),
                             margin=0.0, squared=True)
            loss += config.injection_weight * isl
            details["injection_sentence"] = isl


    if config.use_rudimentary_manipulations:
        hinge_pos, hinge_neg = [], []
        if adv_pos_logits is not None and pos_rud.any():
            hinge_pos.append(pos_logits[pos_rud])
            hinge_neg.append(adv_pos_logits[pos_rud])
        if adv_neg_logits is not None and neg_rud.any():
            hinge_pos.append(neg_logits[neg_rud])
            hinge_neg.append(adv_neg_logits[neg_rud])
        if hinge_pos:
            rl = hinge_loss(torch.cat(hinge_pos), torch.cat(hinge_neg),
                            margin=0.0, squared=True)
            loss += config.rudimentary_weight * rl
            details["rudimentary"] = rl

    if config.use_hotflip_swaps:
        hinge_pos, hinge_neg = [], []
        if adv_pos_logits is not None and pos_hot.any():
            hinge_pos.append(pos_logits[pos_hot])
            hinge_neg.append(adv_pos_logits[pos_hot])
        if adv_neg_logits is not None and neg_hot.any():
            hinge_pos.append(neg_logits[neg_hot])
            hinge_neg.append(adv_neg_logits[neg_hot])
        if hinge_pos:
            hl = hinge_loss(torch.cat(hinge_pos), torch.cat(hinge_neg),
                            margin=0.0, squared=True)
            loss += config.hotflip_weight * hl
            details["hotflip"] = hl

    if config.use_injected and adv_neg_logits is not None and neg_inj_q.any():
        row_idx = torch.nonzero(neg_inj_q, as_tuple=False).squeeze(1)
        ref_pos = pos_logits[row_idx // K]
        iql = hinge_loss(ref_pos, adv_neg_logits[neg_inj_q],
                         margin=0.0, squared=True)
        iql *= config.injection_weight / 16.0
        loss += iql
        details["injection_query"] = iql

    return loss, details


def _extract_split_logits(
        logits_all: torch.Tensor,
        slices: Dict[str, Tuple[int, int]]
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        pos_logits = logits_all[slice(*slices["pos"])]
        neg_logits = logits_all[slice(*slices["neg"])]
        adv_pos_logits = logits_all[slice(*slices["adv_pos"])] if "adv_pos" in slices else None
        adv_neg_logits = logits_all[slice(*slices["adv_neg"])] if "adv_neg" in slices else None
        return pos_logits, neg_logits, adv_pos_logits, adv_neg_logits


class RerankerTrainer:
    def __init__(self, config: RerankerConfig, deepspeed_args: Optional[Any] = None):
        self.config            = config
        self.deepspeed_args = deepspeed_args
        self.tokenizer      = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
        self.tokenizer.padding_side = config.padding_side

        self.model_engine   = None
        self.best_dev_loss  = float("inf")
        self.global_step    = 0
        self.steps_since_eval = 0

    def is_main_process(self) -> bool:
        if self.model_engine is None:
            return not (dist.is_initialized() and dist.get_rank() != 0)
        return self.model_engine.global_rank == 0

    def log(self,*a,**k):
        if self.is_main_process(): print(*a,**k,flush=True)

    def setup_model(self):
        scorer = PointwiseScorer(
            self.config.model_name,
            use_flash_attention=True,
            gradient_checkpointing=self.config.gradient_checkpointing
        )
        params = (p for p in scorer.parameters() if p.requires_grad)

        self.model_engine, _, _, _ = deepspeed.initialize(
            args=self.deepspeed_args,
            model=scorer,
            model_parameters=params,
        )

        self.device          = self.model_engine.device
        self.train_micro_bs  = self.model_engine.train_micro_batch_size_per_gpu()

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
        )

        col_kwargs = dict(
            tokenizer=self.tokenizer,
            max_length=self.config.max_length,
            negatives_per_query=self.config.negatives_per_query,
            use_paraphrased=self.config.use_paraphrased,
            use_injected=self.config.use_injected,
            use_rudimentary_manipulations=self.config.use_rudimentary_manipulations,
            use_hotflip_swaps=self.config.use_hotflip_swaps,
            padding_side=self.config.padding_side,
        )

        train_coll = RerankerCollator(**col_kwargs, sample_negatives=True)
        dev_coll   = RerankerCollator(**col_kwargs, sample_negatives=False)

        def _loader(ds,samp,coll,workers):
            return DataLoader(
                ds, batch_size=self.train_micro_bs, sampler=samp, drop_last=True,
                num_workers=workers, pin_memory=True, collate_fn=coll,
            )

        return (
            _loader(train_ds, DistributedSampler(train_ds, drop_last=True, seed=self.config.seed), train_coll, self.config.train_num_workers),
            _loader(dev_ds,   DistributedSampler(dev_ds, shuffle=False, drop_last=True, seed=self.config.seed), dev_coll, self.config.dev_num_workers),
        )

    def _prepare_batch_for_step(self, batch, is_eval=False):
        """Deduplicates logic for train_step and eval_step."""
        pos     = batch.get("positive_pairs")
        neg     = batch.get("negative_pairs")
        adv_pos = batch.get("adversarial_positive_pairs")
        adv_neg = batch.get("adversarial_negative_pairs")
        info    = batch.get("manipulation_info", [])

        pos_start_idx_list = batch.get("positive_doc_start_indices")
        neg_start_idx_list = batch.get("negative_doc_start_indices")

        def _to_device(d):
            if d is None: return
            for k in ("input_ids", "attention_mask"):
                if k in d and torch.is_tensor(d[k]):
                    d[k] = d[k].to(self.device, non_blocking=True)

        for d in (pos, neg, adv_pos, adv_neg):
            _to_device(d)

        clean_ids_list, clean_mask_list, clean_slices = [], [], {}
        cur_clean = 0
        def _add_clean(tag, data):
            nonlocal cur_clean
            if data is None or data["input_ids"].size(0) == 0: return
            n = data["input_ids"].size(0)
            clean_slices[tag] = (cur_clean, cur_clean + n)
            clean_ids_list.append(data["input_ids"])
            clean_mask_list.append(data["attention_mask"])
            cur_clean += n

        _add_clean("pos", pos)
        _add_clean("neg", neg)

        clean_ids  = torch.cat(clean_ids_list, 0) if clean_ids_list else None
        clean_mask = torch.cat(clean_mask_list, 0) if clean_mask_list else None

        pos_start_idx = torch.tensor(pos_start_idx_list, device=self.device, dtype=torch.long) if pos_start_idx_list else None
        neg_start_idx = torch.tensor(neg_start_idx_list, device=self.device, dtype=torch.long) if neg_start_idx_list else None

        if self.config.use_hotflip_swaps:
            pos_tags = [item.get("pos", "none") for item in info]
            neg_tags = [tag for item in info for tag in item.get("negs", [])]
            hot_pos_rows = [i for i, t in enumerate(pos_tags) if t == "hotflip"]
            hot_neg_rows = [i for i, t in enumerate(neg_tags) if t == "hotflip"]

            any_local = 1 if (hot_pos_rows or hot_neg_rows) else 0
            any_local_t = torch.tensor(any_local, device=self.device)
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(any_local_t, op=dist.ReduceOp.SUM)
            any_global = any_local_t.item() > 0

            hp_ids, hn_ids = None, None
            
            if any_global:
                token_embeddings = self.model_engine.module.model.get_input_embeddings()

                with zero.GatheredParameters([token_embeddings.weight], modifier_rank=None):

                    if not (hot_pos_rows or hot_neg_rows):
                        if pos is not None and pos["input_ids"].size(0) > 0:
                            dummy_ids  = pos["input_ids"][:1].detach()
                            dummy_mask = pos["attention_mask"][:1].detach()
                        else:
                            dummy_ids  = neg["input_ids"][:1].detach()
                            dummy_mask = neg["attention_mask"][:1].detach()

                        _ = _hotflip_swap(
                            dummy_ids,
                            dummy_mask,
                            tokenizer=self.tokenizer,
                            model_engine=self.model_engine,
                            modifiable_start_idx=None,
                            weight_is_gathered=True,
                        )
                    else:
                        hp_ids, hn_ids = _batched_hotflip_swap(
                            hot_pos_rows, hot_neg_rows,
                            pos_ids=pos["input_ids"].detach(),
                            pos_mask=pos["attention_mask"].detach(),
                            neg_ids=neg["input_ids"].detach(),
                            neg_mask=neg["attention_mask"].detach(),
                            tokenizer=self.tokenizer,
                            model_engine=self.model_engine,
                            pos_start_idx=pos_start_idx,
                            neg_start_idx=neg_start_idx,
                        )

                if hp_ids is not None and adv_pos is not None and hot_pos_rows:
                    adv_pos["input_ids"][hot_pos_rows] = hp_ids
                if hn_ids is not None and adv_neg is not None and hot_neg_rows:
                    adv_neg["input_ids"][hot_neg_rows] = hn_ids

        ids_list, mask_list, slices = [], [], {}
        cur = 0
        def _add(tag, data):
            nonlocal cur
            if data is None or data["input_ids"].size(0) == 0: return
            n = data["input_ids"].size(0)
            slices[tag] = (cur, cur + n)
            ids_list.append(data["input_ids"])
            mask_list.append(data["attention_mask"])
            cur += n

        _add("pos", pos)
        _add("neg", neg)
        if any([self.config.use_paraphrased, self.config.use_injected,
                self.config.use_rudimentary_manipulations, self.config.use_hotflip_swaps]):
            _add("adv_pos", adv_pos)
            _add("adv_neg", adv_neg)

        big_ids  = torch.cat(ids_list, 0)
        big_mask = torch.cat(mask_list, 0)

        with torch.set_grad_enabled(not is_eval):
            big_logits = self.model_engine(input_ids=big_ids, attention_mask=big_mask)
            if big_logits.ndim == 2 and big_logits.size(-1) == 1:
                big_logits = big_logits.squeeze(-1)

        pos_logits, neg_logits, adv_pos_logits, adv_neg_logits = _extract_split_logits(big_logits, slices)

        return {
            "pos_logits": pos_logits, "neg_logits": neg_logits,
            "adv_pos_logits": adv_pos_logits, "adv_neg_logits": adv_neg_logits,
            "info": info, "big_ids": big_ids, "big_mask": big_mask, "slices": slices,
            "clean_ids": clean_ids, "clean_mask": clean_mask, "clean_slices": clean_slices,
        }

    def train_step(self, batch) -> float:
        prepared = self._prepare_batch_for_step(batch, is_eval=False)

        loss, _ = _combined_loss(
            prepared["pos_logits"], prepared["neg_logits"],
            prepared["adv_pos_logits"], prepared["adv_neg_logits"],
            prepared["info"], self.config,
        )

        if self.config.use_pgd and prepared["clean_ids"] is not None:
            eps_now = self._eps_sched.step()
            lr_now  = eps_now / self.config.pgd_lr_factor

            def _pgd_loss_fn(perturbed_logits: torch.Tensor) -> torch.Tensor:
                pos_l, neg_l, _, _ = _extract_split_logits(perturbed_logits, prepared["clean_slices"])
                loss_val, _ = _combined_loss(pos_l, neg_l, None, None, prepared["info"], self.config)
                return loss_val

            d = _pgd_attack(
                ids=prepared["clean_ids"], mask=prepared["clean_mask"], model_engine=self.model_engine,
                eps=eps_now, steps=self.config.pgd_steps, lr=lr_now, loss_fn=_pgd_loss_fn
            )

            emb_layer = self.model_engine.module.model.get_input_embeddings()

            pgd_logits = self.model_engine(inputs_embeds=emb_layer(prepared["clean_ids"]) + d,
                                           attention_mask=prepared["clean_mask"])
            if pgd_logits.ndim == 2 and pgd_logits.size(-1) == 1:
                pgd_logits = pgd_logits.squeeze(-1)
            pos_l, neg_l, _, _ = _extract_split_logits(pgd_logits, prepared["clean_slices"])
            pl, _ = _combined_loss(pos_l, neg_l, None, None, prepared["info"], self.config)

            loss += self.config.pgd_weight * pl

        self.model_engine.backward(loss)
        self.model_engine.step()
        return loss.item()

    def eval_step(self, batch) -> Dict[str,float]:
        prepared = self._prepare_batch_for_step(batch, is_eval=True)

        B = prepared["pos_logits"].size(0)

        out = {}

        loss, details = _combined_loss(
            prepared["pos_logits"], prepared["neg_logits"],
            prepared["adv_pos_logits"], prepared["adv_neg_logits"],
            prepared["info"], self.config
        )
        out['combined'] = loss.item()
        for k, v in details.items():
            out[k] = v.item()

        if self.config.use_pgd and prepared["clean_ids"] is not None:
            eps_now = self.config.eps_max
            lr_now  = eps_now / self.config.pgd_lr_factor

            def _pgd_loss_fn(perturbed_logits: torch.Tensor) -> torch.Tensor:
                pos_l, neg_l, _, _ = _extract_split_logits(perturbed_logits, prepared["clean_slices"])
                loss_val, _ = _combined_loss(pos_l, neg_l, None, None, prepared["info"], self.config)
                return loss_val

            delta = _pgd_attack(
                ids=prepared["clean_ids"], mask=prepared["clean_mask"], model_engine=self.model_engine,
                eps=eps_now, steps=self.config.pgd_steps, lr=lr_now, loss_fn=_pgd_loss_fn
            )

            with torch.no_grad():
                embed_layer = self.model_engine.module.model.get_input_embeddings()
                perturbed_logits = self.model_engine(
                    inputs_embeds=embed_layer(prepared["clean_ids"]) + delta,
                    attention_mask=prepared["clean_mask"]
                )
                if perturbed_logits.ndim == 2 and perturbed_logits.size(-1) == 1:
                    perturbed_logits = perturbed_logits.squeeze(-1)
                pgd_split_logits = _extract_split_logits(perturbed_logits, prepared["clean_slices"])
                pgd_loss, _ = _combined_loss(*pgd_split_logits, prepared["info"], self.config)
                out['pgd_combined'] = pgd_loss.item()

        pos_tags = [it.get("pos","none") for it in prepared["info"]]
        neg_tags = [t for it in prepared["info"] for t in it.get("negs", [])]
        counts = {
            "paraphrase__count": float(sum(t == "paraphrase"         for t in pos_tags) +
                                       sum(t == "paraphrase"         for t in neg_tags)),
            "injection_sentence__count": float(sum(t == "injection_sentence" for t in pos_tags) +
                                               sum(t == "injection_sentence" for t in neg_tags)),
            "injection_query__count": float(sum(t == "injection_query" for t in neg_tags)),
            "rudimentary__count": float(sum(t == "rudimentary" for t in pos_tags) +
                                        sum(t == "rudimentary" for t in neg_tags)),
            "hotflip__count": float(sum(t == "hotflip" for t in pos_tags) +
                                    sum(t == "hotflip" for t in neg_tags)),
        }
        for k in list(details.keys()):
            ckey = f"{k}__count"
            if ckey in counts:
                out[ckey] = counts[ckey]

        out["n"] = float(B)
        return out

    def train(self):
        self.setup_model()
        train_loader, dev_loader = self.setup_data()

        for epoch in range(self.config.num_epochs):
            self.log(f"\nEpoch {epoch+1}/{self.config.num_epochs}")
            train_loader.sampler.set_epoch(epoch)

            train_loss_sum = 0.0
            train_steps    = 0

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

                    self.model_save(dev_losses["combined"])

                    self.steps_since_eval = 0

                loss_val = self.train_step(batch)
                train_loss_sum += loss_val
                train_steps    += 1
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
            self.model_save(dev_losses["combined"])

        self.log("\nTraining complete! 🎉")

    def evaluate(self, dl):
        self.model_engine.eval()
        totals = defaultdict(float)
        counts = defaultdict(float)

        for batch in tqdm(dl, disable=not self.is_main_process(), desc="eval"):
            res = self.eval_step(batch)
            n = float(res.pop("n", 0.0))
            special_counts = {k[:-8]: float(v) for k,v in list(res.items()) if k.endswith("__count")}
            for k in list(special_counts.keys()):
                res.pop(f"{k}__count", None)
            for k, v in res.items():
                weight = special_counts.get(k, n)
                totals[k] += v * weight
                counts[k] += weight

        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            local_keys = list(totals.keys())
            gathered = [None] * world_size
            dist.all_gather_object(gathered, local_keys)
            all_keys = sorted({k for ks in gathered for k in ks})
        else:
            all_keys = sorted(totals.keys())

        synced_avgs = {}
        for k in all_keys:
            total_loss_tensor = torch.tensor(totals.get(k, 0.0), device=self.device)
            item_count_tensor = torch.tensor(counts.get(k, 0.0), device=self.device)

            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(item_count_tensor, op=dist.ReduceOp.SUM)

            n = item_count_tensor.item()
            if n > 0:
                synced_avgs[k] = total_loss_tensor.item() / n

        self.model_engine.train()
        return synced_avgs

    def model_save(self, dev_loss):
        if dev_loss < self.best_dev_loss:
            self.best_dev_loss = dev_loss
            if self.config.save_model and self.is_main_process():
                p = Path(self.config.output_dir); p.mkdir(parents=True, exist_ok=True)
                self.model_engine.module.model.save_pretrained(p)
                self.tokenizer.save_pretrained(p)
                self.log(f"  ✔ Saved checkpoint → {p}")
