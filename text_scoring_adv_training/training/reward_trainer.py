from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import deepspeed
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import AutoTokenizer
from deepspeed import zero

from text_scoring_adv_training.data.collators import PreferenceCollator
from text_scoring_adv_training.data.datasets import PreferenceDataset
from text_scoring_adv_training.training.losses import hinge_loss, softmax_nll_loss, mse_loss
from text_scoring_adv_training.models.pointwise_scorer import PointwiseScorer
from text_scoring_adv_training.utils.seed import seed_everything

torch.backends.cuda.matmul.allow_tf32 = True


@dataclass
class RewardConfig:
    train_file:  str
    dev_file:    str
    output_dir:  str
    paraphrased_path: str = None
    injected_path:    str = None
    max_dev_samples:  int = None

    num_epochs:  int  = 1
    eval_every:  int  = None
    save_model:  bool = True

    temperature: float = 1.0

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
    eps_warmup_steps: int   = 50

    max_length:         int  = 4096
    pad_to_multiple_of: int  = 8
    padding_side:       str  = "left"

    model_name: str = "Llama-3.2-3B-Instruct"
    gradient_checkpointing: bool = False

    train_num_workers: int = 8
    dev_num_workers:   int = 8

    seed: int = 42


class _EpsilonScheduler:
    def __init__(self, eps_start: float, eps_end: float, warmup_steps: int, acc_steps: int):
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
        adv_logits_all = model_engine(inputs_embeds=base + delta, attention_mask=mask)
        if adv_logits_all.ndim == 2 and adv_logits_all.size(-1) == 1:
            adv_logits_all = adv_logits_all.squeeze(-1)
        loss = loss_fn(adv_logits_all)
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
    
    
    with zero.GatheredParameters([token_embeddings.weight], modifier_rank=None):
        with torch.no_grad():
            W = token_embeddings.weight
            orig_E = W[orig_id]
            delta  = grads @ W.t()
  
            delta -= (grads * orig_E).sum(-1, keepdim=True)
            delta[torch.arange(N), orig_id] = -float("inf")
            delta[:, tokenizer.all_special_ids] = -float("inf")

            best_id = delta.argmax(1)
    
        torch.cuda.synchronize(device=grads.device)

    ids_adv = ids.clone()
    ok_rows = (pick >= 0)
    ids_adv[ok_rows, pick[ok_rows]] = best_id[ok_rows]
    return ids_adv


def _batched_hotflip_swap(a_rows, b_rows, *, a_ids, a_mask, b_ids, b_mask, tokenizer, model_engine,
                            a_start_idx: Optional[torch.Tensor] = None,
                            b_start_idx: Optional[torch.Tensor] = None):
    
    ids_cat, mask_cat, idx_cat = [], [], []

    if a_rows:
        ids_cat.append(a_ids[a_rows]); mask_cat.append(a_mask[a_rows])
        if a_start_idx is not None: idx_cat.append(a_start_idx[a_rows])
    if b_rows:
        ids_cat.append(b_ids[b_rows]); mask_cat.append(b_mask[b_rows])
        if b_start_idx is not None: idx_cat.append(b_start_idx[b_rows])

    ids_all  = torch.cat(ids_cat, 0)
    mask_all = torch.cat(mask_cat, 0)
    idx_all = torch.cat(idx_cat, 0) if idx_cat else None
    swapped_all = _hotflip_swap(ids_all, mask_all, tokenizer=tokenizer, model_engine=model_engine, modifiable_start_idx=idx_all)

    cursor = 0
    a_out = b_out = None
    if a_rows:
        a_out = swapped_all[cursor:cursor+len(a_rows)]; cursor += len(a_rows)
    if b_rows:
        b_out = swapped_all[cursor:]
    return a_out, b_out


def _extract_split_logits_reward(
    logits_all: torch.Tensor,
    slices: Dict[str, Tuple[int, int]]
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    chosen_logits   = logits_all[slice(*slices["chosen"])]
    rejected_logits = logits_all[slice(*slices["rejected"])]
    adv_chosen_logits   = logits_all[slice(*slices["adv_chosen"])] if "adv_chosen" in slices else None
    adv_rejected_logits = logits_all[slice(*slices["adv_rejected"])] if "adv_rejected" in slices else None
    return chosen_logits, rejected_logits, adv_chosen_logits, adv_rejected_logits


def _combined_reward_loss(
    chosen_logits:        torch.Tensor,
    rejected_logits:      torch.Tensor,
    adv_chosen_logits:    Optional[torch.Tensor],
    adv_rejected_logits:  Optional[torch.Tensor],
    manipulation_info:    List[Dict[str, Any]],
    config:               RewardConfig,
    sample_weights:       Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

    device = chosen_logits.device
    details = defaultdict(lambda: torch.tensor(0.0, device=device))

    loss = 0

    pair_logits = torch.stack([chosen_logits, rejected_logits], dim=1)
    core = softmax_nll_loss(pair_logits, temperature=config.temperature, sample_weights=sample_weights)
    details["softmax_nll_loss"] = core

    loss += core
    
    with torch.no_grad():
        details["acc"] = (chosen_logits > rejected_logits).float().mean()

    ch_tags = [info.get("chosen", "none")   for info in manipulation_info]
    re_tags = [info.get("rejected", "none") for info in manipulation_info]

    def _mask(tags, cond) -> torch.Tensor:
        return torch.tensor([cond(t) for t in tags], dtype=torch.bool, device=device)

    ch_para = _mask(ch_tags, lambda t: t == "paraphrase")
    re_para = _mask(re_tags, lambda t: t == "paraphrase")

    ch_inj = _mask(ch_tags, lambda t: t.startswith("injection"))
    re_inj = _mask(re_tags, lambda t: t.startswith("injection"))

    ch_rud = _mask(ch_tags, lambda t: t == "rudimentary")
    re_rud = _mask(re_tags, lambda t: t == "rudimentary")

    ch_hot = _mask(ch_tags, lambda t: t == "hotflip")
    re_hot = _mask(re_tags, lambda t: t == "hotflip")

    if config.use_paraphrased and (adv_chosen_logits is not None or adv_rejected_logits is not None):
        parts = []
        if adv_chosen_logits is not None and ch_para.any():
            parts.append(mse_loss(adv_chosen_logits[ch_para], chosen_logits[ch_para]))
        if adv_rejected_logits is not None and re_para.any():
            parts.append(mse_loss(adv_rejected_logits[re_para], rejected_logits[re_para]))
        if parts:
            pl = torch.stack(parts).mean()
            loss += config.paraphrase_weight * pl
            details["paraphrase"] = pl

    if config.use_injected and (adv_chosen_logits is not None or adv_rejected_logits is not None):
        parts = []
        if adv_chosen_logits is not None and ch_inj.any():
            parts.append(hinge_loss(chosen_logits[ch_inj],   adv_chosen_logits[ch_inj],   margin=0.0, squared=True))
        if adv_rejected_logits is not None and re_inj.any():
            parts.append(hinge_loss(rejected_logits[re_inj], adv_rejected_logits[re_inj], margin=0.0, squared=True))
        if parts:
            il = torch.stack(parts).mean()
            loss += config.injection_weight * il
            details["injection"] = il

    if config.use_rudimentary_manipulations and (adv_chosen_logits is not None or adv_rejected_logits is not None):
        parts = []
        if adv_chosen_logits is not None and ch_rud.any():
            parts.append(hinge_loss(chosen_logits[ch_rud],   adv_chosen_logits[ch_rud],   margin=0.0, squared=True))
        if adv_rejected_logits is not None and re_rud.any():
            parts.append(hinge_loss(rejected_logits[re_rud], adv_rejected_logits[re_rud], margin=0.0, squared=True))
        if parts:
            rl = torch.stack(parts).mean()
            loss += config.rudimentary_weight * rl
            details["rudimentary"] = rl

    if config.use_hotflip_swaps and (adv_chosen_logits is not None or adv_rejected_logits is not None):
        parts = []
        if adv_chosen_logits is not None and ch_hot.any():
            parts.append(hinge_loss(chosen_logits[ch_hot],   adv_chosen_logits[ch_hot],   margin=0.0, squared=True))
        if adv_rejected_logits is not None and re_hot.any():
            parts.append(hinge_loss(rejected_logits[re_hot], adv_rejected_logits[re_hot], margin=0.0, squared=True))
        if parts:
            hl = torch.stack(parts).mean()
            loss += config.hotflip_weight * hl
            details["hotflip"] = hl

    return loss, details


class RewardTrainer:
    def __init__(self, config: RewardConfig, deepspeed_args: Optional[Any] = None):
        self.config         = config
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

    def log(self, *a, **k):
        if self.is_main_process(): print(*a, **k, flush=True)

    def setup_model(self):
        scorer = PointwiseScorer(self.config.model_name, gradient_checkpointing=self.config.gradient_checkpointing, use_flash_attention=True, torch_compile_model=False)
        params = (p for p in scorer.parameters() if p.requires_grad)
        self.model_engine, _, _, _ = deepspeed.initialize(args=self.deepspeed_args, model=scorer, model_parameters=params)
        self.device         = self.model_engine.device
        
        rank = self.model_engine.global_rank if dist.is_initialized() else 0
        seed_everything(self.config.seed + rank)

        self.train_micro_bs = self.model_engine.train_micro_batch_size_per_gpu()
        self._eps_sched = _EpsilonScheduler(1e-8, self.config.eps_max, self.config.eps_warmup_steps, self.model_engine.gradient_accumulation_steps())

    def setup_data(self) -> Tuple[DataLoader, DataLoader]:
        train_ds = PreferenceDataset(self.config.train_file, paraphrased_path=self.config.paraphrased_path, injected_path=self.config.injected_path)
        dev_ds   = PreferenceDataset(self.config.dev_file,   paraphrased_path=self.config.paraphrased_path, injected_path=self.config.injected_path, max_samples=self.config.max_dev_samples)

        coll_kwargs = dict(
            tokenizer=self.tokenizer,
            max_length=self.config.max_length,
            pad_to_multiple_of=self.config.pad_to_multiple_of,
            use_paraphrased=self.config.use_paraphrased,
            use_injected=self.config.use_injected,
            use_rudimentary_manipulations=self.config.use_rudimentary_manipulations,
            use_hotflip_swaps=self.config.use_hotflip_swaps,
            padding_side=self.config.padding_side,
        )
        train_coll = PreferenceCollator(**coll_kwargs)
        dev_coll   = PreferenceCollator(**coll_kwargs)

        def _loader(ds, samp, coll, workers, drop_last=True):
            return DataLoader(ds, batch_size=self.train_micro_bs, sampler=samp, drop_last=drop_last, num_workers=workers, pin_memory=True, collate_fn=coll,)

        return (
            _loader(train_ds, DistributedSampler(train_ds, drop_last=True, seed=self.config.seed), train_coll, self.config.train_num_workers),
            _loader(dev_ds,   DistributedSampler(dev_ds, shuffle=False, drop_last=False, seed=self.config.seed), dev_coll, self.config.dev_num_workers, drop_last=False),
        )

    def _prepare_batch_for_step(self, batch, is_eval: bool = False):
        chosen       = batch.get("chosen")
        rejected     = batch.get("rejected")
        adv_chosen   = batch.get("adversarial_chosen")
        adv_rejected = batch.get("adversarial_rejected")
        info         = batch.get("manipulation_info", [])
        weights      = batch.get("weights")

        ch_start_idx_list = batch.get("chosen_start_indices")
        re_start_idx_list = batch.get("rejected_start_indices")

        if weights is not None:
            weights = weights.to(self.device, non_blocking=True)

        def _to_device(d):
            if d is None: return
            for k in ("input_ids", "attention_mask"):
                if k in d and torch.is_tensor(d[k]):
                    d[k] = d[k].to(self.device, non_blocking=True)

        for d in (chosen, rejected, adv_chosen, adv_rejected):
            _to_device(d)

        ch_start_idx = torch.tensor(ch_start_idx_list, device=self.device, dtype=torch.long) if ch_start_idx_list else None
        re_start_idx = torch.tensor(re_start_idx_list, device=self.device, dtype=torch.long) if re_start_idx_list else None
        
        if self.config.use_hotflip_swaps:
            ch_tags = [it.get("chosen", "none") for it in info]
            re_tags = [it.get("rejected", "none") for it in info]
            hot_ch_rows = [i for i, t in enumerate(ch_tags) if t == "hotflip"]
            hot_re_rows = [i for i, t in enumerate(re_tags) if t == "hotflip"]
            
            local_has_hotflips = torch.tensor(1.0 if (hot_ch_rows or hot_re_rows) else 0.0, device=self.device)
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(local_has_hotflips, op=dist.ReduceOp.SUM)

            if local_has_hotflips.item() > 0:
                if hot_ch_rows or hot_re_rows:
                    hc_ids, hr_ids = _batched_hotflip_swap(
                        hot_ch_rows, hot_re_rows,
                        a_ids=chosen["input_ids"].detach(),
                        a_mask=chosen["attention_mask"].detach(),
                        b_ids=rejected["input_ids"].detach(),
                        b_mask=rejected["attention_mask"].detach(),
                        tokenizer=self.tokenizer,
                        model_engine=self.model_engine,
                        a_start_idx=ch_start_idx,
                        b_start_idx=re_start_idx,
                    )
                    if hc_ids is not None and adv_chosen is not None:
                        adv_chosen["input_ids"][hot_ch_rows] = hc_ids
                    if hr_ids is not None and adv_rejected is not None:
                        adv_rejected["input_ids"][hot_re_rows] = hr_ids
                else:
                    pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
                    dummy_ids  = torch.full((1, 1), pad_id, device=self.device, dtype=torch.long)
                    dummy_mask = torch.ones_like(dummy_ids)

                    emb_layer = self.model_engine.module.model.get_input_embeddings()
                    dummy_embeds = emb_layer(dummy_ids).detach().clone().requires_grad_(True)

                    dummy_logits = self.model_engine(inputs_embeds=dummy_embeds,
                                                        attention_mask=dummy_mask)
                    ( _ , ) = torch.autograd.grad(dummy_logits.sum(), dummy_embeds)

                    token_embeddings = self.model_engine.module.model.get_input_embeddings()
                    with zero.GatheredParameters([token_embeddings.weight], modifier_rank=None):
                        pass

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

        _add_clean("chosen",   chosen)
        _add_clean("rejected", rejected)

        clean_ids  = torch.cat(clean_ids_list, 0) if clean_ids_list else None
        clean_mask = torch.cat(clean_mask_list, 0) if clean_mask_list else None
        
        ids_list, mask_list, slices = [], [], {}
        cur = 0
        def _add(tag, data):
            nonlocal cur
            if data is None or data["input_ids"].size(0) == 0: return
            n = data["input_ids"].size(0)
            slices[tag] = (cur, cur+n)
            ids_list.append(data["input_ids"]); mask_list.append(data["attention_mask"])
            cur += n

        _add("chosen", chosen)
        _add("rejected", rejected)
        if any([self.config.use_paraphrased, self.config.use_injected, self.config.use_rudimentary_manipulations, self.config.use_hotflip_swaps]):
            _add("adv_chosen",   adv_chosen)
            _add("adv_rejected", adv_rejected)

        big_ids  = torch.cat(ids_list, 0)
        big_mask = torch.cat(mask_list, 0)

        with torch.set_grad_enabled(not is_eval):
            big_logits = self.model_engine(input_ids=big_ids, attention_mask=big_mask)
            if big_logits.ndim == 2 and big_logits.size(-1) == 1:
                big_logits = big_logits.squeeze(-1)

        c, r, ac, ar = _extract_split_logits_reward(big_logits, slices)
        return {"chosen_logits": c, "rejected_logits": r, "adv_chosen_logits": ac, "adv_rejected_logits": ar, "info": info, "big_ids": big_ids, "big_mask": big_mask, "slices": slices, "weights": weights,  "clean_ids": clean_ids, "clean_mask": clean_mask, "clean_slices": clean_slices,}

    def train_step(self, batch) -> float:
        prepared = self._prepare_batch_for_step(batch, is_eval=False)
        loss, _ = _combined_reward_loss(
            prepared["chosen_logits"], prepared["rejected_logits"],
            prepared["adv_chosen_logits"], prepared["adv_rejected_logits"],
            prepared["info"], self.config,
            sample_weights=prepared.get("weights"),
        )

        if self.config.use_pgd and prepared["clean_ids"] is not None:
            eps_now = self._eps_sched.step()
            lr_now  = eps_now / self.config.pgd_lr_factor

            def _pgd_loss_fn(adv_logits_all: torch.Tensor) -> torch.Tensor:
                c, r, _, _ = _extract_split_logits_reward(adv_logits_all, prepared["clean_slices"])
                l, _ = _combined_reward_loss(
                    c, r, None, None, prepared["info"], self.config,
                    sample_weights=prepared.get("weights"),
                )
                return l

            d = _pgd_attack(
                prepared["clean_ids"], prepared["clean_mask"],
                self.model_engine, eps_now, self.config.pgd_steps, lr_now, _pgd_loss_fn
            )

            emb = self.model_engine.module.model.get_input_embeddings()
            adv_logits_all = self.model_engine(
                inputs_embeds=emb(prepared["clean_ids"]) + d,
                attention_mask=prepared["clean_mask"]
            )
            if adv_logits_all.ndim == 2 and adv_logits_all.size(-1) == 1:
                adv_logits_all = adv_logits_all.squeeze(-1)
            c2, r2, _, _ = _extract_split_logits_reward(adv_logits_all, prepared["clean_slices"])
            pl, _ = _combined_reward_loss(
                c2, r2, None, None, prepared["info"], self.config,
                sample_weights=prepared.get("weights"),
            )
            loss += self.config.pgd_weight * pl

        self.model_engine.backward(loss)
        self.model_engine.step()
        return loss.item()

    def eval_step(self, batch) -> Dict[str, float]:
        prepared = self._prepare_batch_for_step(batch, is_eval=True)
        n = prepared["chosen_logits"].size(0)
        out = {"n": float(n)}
        loss, details = _combined_reward_loss(
            prepared["chosen_logits"], prepared["rejected_logits"],
            prepared["adv_chosen_logits"], prepared["adv_rejected_logits"],
            prepared["info"], self.config,
            sample_weights=prepared.get("weights"),
        )
        out["combined"] = loss.item()
        for k, v in details.items(): out[k] = v.item()

        info = prepared["info"]
        ch_tags = [it.get("chosen", "none") for it in info]
        re_tags = [it.get("rejected", "none") for it in info]

        if prepared["adv_chosen_logits"] is not None:
            ch_para = sum(t == "paraphrase"        for t in ch_tags)
            ch_inj  = sum(t.startswith("injection") for t in ch_tags)
            ch_rud  = sum(t == "rudimentary"        for t in ch_tags)
            ch_hot  = sum(t == "hotflip"            for t in ch_tags)
        else:
            ch_para = ch_inj = ch_rud = ch_hot = 0

        if prepared["adv_rejected_logits"] is not None:
            re_para = sum(t == "paraphrase"        for t in re_tags)
            re_inj  = sum(t.startswith("injection") for t in re_tags)
            re_rud  = sum(t == "rudimentary"        for t in re_tags)
            re_hot  = sum(t == "hotflip"            for t in re_tags)
        else:
            re_para = re_inj = re_rud = re_hot = 0

        counts = {
            "paraphrase__count": float(ch_para + re_para),
            "injection__count":  float(ch_inj  + re_inj),
            "rudimentary__count":float(ch_rud  + re_rud),
            "hotflip__count":    float(ch_hot  + re_hot),
        }
        for k in list(out.keys()):
            if k in ("combined","softmax_nll_loss","acc","pgd_combined"):
                continue
            ckey = f"{k}__count"
            if ckey in counts:
                out[ckey] = counts[ckey]

        if self.config.use_pgd and prepared["clean_ids"] is not None:
            eps_now = self.config.eps_max
            lr_now  = eps_now / self.config.pgd_lr_factor

            def _pgd_loss_fn(adv_logits_all: torch.Tensor) -> torch.Tensor:
                c, r, _, _ = _extract_split_logits_reward(adv_logits_all, prepared["clean_slices"])
                l, _ = _combined_reward_loss(
                    c, r, None, None, prepared["info"], self.config,
                    sample_weights=prepared.get("weights"),
                )
                return l

            delta = _pgd_attack(
                prepared["clean_ids"], prepared["clean_mask"],
                self.model_engine, eps_now, self.config.pgd_steps, lr_now, _pgd_loss_fn
            )
            with torch.no_grad():
                emb = self.model_engine.module.model.get_input_embeddings()
                adv_logits_all = self.model_engine(
                    inputs_embeds=emb(prepared["clean_ids"]) + delta,
                    attention_mask=prepared["clean_mask"]
                )
                if adv_logits_all.ndim == 2 and adv_logits_all.size(-1) == 1:
                    adv_logits_all = adv_logits_all.squeeze(-1)
                c2, r2, _, _ = _extract_split_logits_reward(adv_logits_all, prepared["clean_slices"])
                pgd_loss, _ = _combined_reward_loss(
                    c2, r2, None, None, prepared["info"], self.config,
                    sample_weights=prepared.get("weights"),
                )
                out["pgd_combined"] = pgd_loss.item()

        return out

    def train(self):
        self.setup_model()
        train_loader, dev_loader = self.setup_data()
        for epoch in range(self.config.num_epochs):
            self.log(f"\nEpoch {epoch+1}/{self.config.num_epochs}")
            train_loader.sampler.set_epoch(epoch)

            train_loss_sum = 0.0
            train_steps = 0
            train_iter = tqdm(train_loader, desc="Training", disable=not self.is_main_process())

            for batch in train_iter:
                if (self.config.eval_every and (self.steps_since_eval >= self.config.eval_every)) or (self.global_step == 0):
                    dev_losses = self.evaluate(dev_loader)
                    loss_str = ", ".join([f"{k}: {v:.8f}" for k, v in dev_losses.items()])
                    self.log(f"[Step {self.global_step:>7}] Dev losses | {loss_str}")
                    self.model_save(dev_losses.get("combined", float("inf")))
                    self.steps_since_eval = 0

                loss_val = self.train_step(batch)
                train_loss_sum += loss_val; train_steps += 1; self.global_step += 1

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
            self.model_save(dev_losses.get("combined", float("inf")))

        self.log("\nTraining complete! 🎉")

    def evaluate(self, dl):
        self.model_engine.eval()
        totals = defaultdict(float)
        counts = defaultdict(float)

        for batch in tqdm(dl, disable=not self.is_main_process(), desc="eval"):
            res = self.eval_step(batch)
            n = float(res.pop("n", 0.0))
            if n == 0.0:
                continue

            special_counts = {k[:-8]: float(v) for k, v in list(res.items()) if k.endswith("__count")}
            for k in list(special_counts.keys()):
                res.pop(f"{k}__count", None)

            for k, v in res.items():
                weight = special_counts.get(k, n)
                totals[k] += v * weight
                counts[k] += weight

        if dist.is_available() and dist.is_initialized():
            local_keys = list(totals.keys())
            gathered = [None] * dist.get_world_size()
            dist.all_gather_object(gathered, local_keys)
            all_keys = sorted({k for ks in gathered for k in ks})
        else:
            all_keys = sorted(totals.keys())

        synced_avgs = {}
        for k in all_keys:
            tot = torch.tensor(totals.get(k, 0.0), device=self.device)
            cnt = torch.tensor(counts.get(k, 0.0), device=self.device)
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(tot, op=dist.ReduceOp.SUM)
                dist.all_reduce(cnt, op=dist.ReduceOp.SUM)
            if cnt.item() > 0:
                synced_avgs[k] = tot.item() / cnt.item()

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
        