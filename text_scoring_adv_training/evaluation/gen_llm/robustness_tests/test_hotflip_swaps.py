#!/usr/bin/env python
from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path
from typing import List, Set, Tuple

import torch
from tqdm import tqdm

from text_scoring_adv_training.evaluation.robustness_tests.common import (
    dataset_slug_from_file, log_result,
    progress_line, short_text, decode_span_text, ExperimentConfig, RunningEditStats,
)
from text_scoring_adv_training.evaluation.robustness_tests.common.beam import select_next_beam
from text_scoring_adv_training.evaluation.robustness_tests.common.paths import PATHS, local_eval_out_dir

from robust_gen_llm_utils import (
    make_generative_model, load_targets_jsonl,
    score_target_logprob, generate_and_check, generate_text,
)
from text_scoring_adv_training.evaluation.robustness_tests.common.chat_spans import compute_spans_with_frozen
from text_scoring_adv_training.evaluation.robustness_tests.common.hotflip import hotflip_causal_lm


DATA_PATH = Path("targets_sample.jsonl")
EVAL_OUT_ROOT = local_eval_out_dir(__file__)

GEN_MAX_NEW_TOKENS = None
GEN_MAX_NEW_TOKENS_FULL = 1024
VERBOSE = True
PROGRESS_EVERY = 1

N_SAMPLE_POS = 1
HOTFLIP_TOP_K_OVERALL = 256


CFG = ExperimentConfig(
    models=[
        "google/gemma-3-27b-it",
    ],
    models_dir=PATHS.checkpoints,
    device=("cuda" if torch.cuda.is_available() else "cpu"),
    batch_size=64,
    max_steps=2048,
    num_beams=16,
    test_variants_per_beam=16,
    diversity_last_n=3,
    elitism_keep=8,
    out_dir=EVAL_OUT_ROOT / "per_sample_stats",
    run_tag="test",
    script_slug="gen_hotflip_beam",
).materialize(__file__)

@dataclass(eq=False)
class BeamItem:
    ids: torch.Tensor
    att: torch.Tensor
    user_span: Tuple[int, int]
    edit_span: Tuple[int, int]
    asst_span: Tuple[int, int]
    score: float
    prompt_text: str
    history: List[Tuple[int, int]]
    history_text: List[str]


def main():
    dev = CFG.device
    items = load_targets_jsonl(DATA_PATH)
    ds_slug = dataset_slug_from_file(DATA_PATH)

    for MODEL in CFG.models:
        progress_line(f"\n=== {MODEL} ===")
        gm = make_generative_model(MODEL, device=dev, models_dir=CFG.models_dir, use_flash_attention=True)
        tok, mdl, specials = gm.tokenizer, gm.model, gm.specials

        stats = RunningEditStats(max_steps=CFG.max_steps)
        stats_path = CFG.stats_path(MODEL, ds_slug)

        for idx, it in enumerate(tqdm(items, desc="items")):
            rid = str(it.get("id", idx))
            prompt = it["prompt"]
            targets = it["targets"]
            prompt_frozen = it.get("prompt_frozen") or ""

            primary_target = targets[0]

            base_score = float(
                score_target_logprob(
                    gm,
                    [(prompt, targets)],
                    batch_size=CFG.batch_size,
                )[0]
            )

            full_text, user_span, asst_span, edit_span = compute_spans_with_frozen(
                tok, prompt, primary_target, prompt_frozen
            )

            enc = tok(
                full_text, return_tensors="pt", add_special_tokens=False,
                truncation=True, max_length=4096
            )
            ids = enc["input_ids"][0].cpu()
            att = enc["attention_mask"][0].cpu()

            beam: List[BeamItem] = [BeamItem(
                ids=ids, att=att, user_span=user_span, edit_span=edit_span,
                asst_span=asst_span, score=base_score, prompt_text=prompt,
                history=[], history_text=[prompt]
            )]
            visited: Set[str] = {prompt}
            depth = 0
            success = False

            if VERBOSE:
                preview = " || ".join(short_text(t, 40) for t in targets)
                progress_line(f"[{rid}] baseline score={base_score:.4f} | targets={preview}")

            while depth < CFG.max_steps:
                prev_top = beam[0].score if beam else float("-inf")
                expansions: List[BeamItem] = []

                for bi in beam:
                    per_beam_k = min(CFG.test_variants_per_beam, HOTFLIP_TOP_K_OVERALL)
                    variants, swaps, _meta = hotflip_causal_lm(
                        ids=bi.ids, att=bi.att, edit_span=bi.edit_span, asst_span=bi.asst_span,
                        model=mdl, tokenizer=tok, specials=specials,
                        n_sample_pos=N_SAMPLE_POS, top_k_overall=per_beam_k, device=dev
                    )
                    if not variants:
                        continue

                    cands: List[Tuple[str, torch.Tensor, dict]] = []
                    for v_ids, sw in zip(variants, swaps):
                        ptxt = decode_span_text(v_ids, bi.user_span[0], bi.user_span[1], tok)
                        if not ptxt:
                            continue
                        if ptxt in visited:
                            continue

                        visited.add(ptxt)
                        cands.append((ptxt, v_ids, sw))

                    if not cands:
                        continue

                    scores = score_target_logprob(
                        gm,
                        [(p, targets) for (p, _vid, _sw) in cands],
                        batch_size=CFG.batch_size,
                    ).tolist()

                    for (ptxt, _v_ids, sw), sc in zip(cands, scores):
                        full_text2, user_span2, asst_span2, edit_span2 = compute_spans_with_frozen(
                            tok, ptxt, primary_target, prompt_frozen
                        )
                        enc2 = tok(
                            full_text2, return_tensors="pt", add_special_tokens=False,
                            truncation=True, max_length=4096
                        )
                        ids2 = enc2["input_ids"][0].cpu()
                        att2 = enc2["attention_mask"][0].cpu()
                        new_hist = bi.history + [(int(sw["pos"]), int(sw["new_id"]))]
                        expansions.append(BeamItem(
                            ids=ids2, att=att2, user_span=user_span2, edit_span=edit_span2,
                            asst_span=asst_span2, score=float(sc), prompt_text=ptxt,
                            history=new_hist, history_text=bi.history_text + [ptxt]
                        ))

                if not expansions:
                    if VERBOSE:
                        progress_line(f"[{rid}] no more expansions at step {depth}.")
                    break

                beam = select_next_beam(
                    expansions,
                    beam,
                    beam_size=CFG.num_beams,
                    last_n=int(CFG.diversity_last_n),
                    elitism_keep=int(CFG.elitism_keep or 0),
                )

                for bi in beam:
                    visited.add(bi.prompt_text)
                depth += 1

                ok, out_text = generate_and_check(
                    gm,
                    beam[0].prompt_text,
                    targets,
                    max_new_tokens=GEN_MAX_NEW_TOKENS,
                    temperature=0.0,
                    match_mode="text_prefix",
                )

                if VERBOSE and (depth % max(1, int(PROGRESS_EVERY)) == 0):
                    delta = (beam[0].score - prev_top) if prev_top != float("-inf") else 0.0
                    progress_line(f"[{rid}] step {depth}/{CFG.max_steps} | best={beam[0].score:.4f} (Δ{delta:+.4f})")
                    progress_line(f"[{rid}] CURRENT PROMPT:\n{beam[0].prompt_text}")
                    progress_line(f"[{rid}] MODEL OUTPUT (check pass):\n{out_text}")

                if ok:
                    success = True
                    full_out = generate_text(
                        gm, beam[0].prompt_text,
                        max_new_tokens=int(GEN_MAX_NEW_TOKENS_FULL or 512),
                        temperature=0.0
                    )
                    progress_line(f"[SUCCESS id={rid}] Prompt (full):\n{beam[0].prompt_text}")
                    progress_line(f"[SUCCESS id={rid}] Model output (FULL):\n{full_out}")
                    break

            edits = depth if success else CFG.max_steps
            log_result(stats_path, f"{rid}", success, edits)
            stats.record(success, depth)

        for line in stats.summary_strings(MODEL):
            progress_line(line)

        del gm
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    main()
